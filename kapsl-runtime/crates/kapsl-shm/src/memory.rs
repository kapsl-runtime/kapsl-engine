use shared_memory::*;
#[cfg(windows)]
use std::os::windows::raw::HANDLE;
const MAGIC_NUMBER: u32 = 0x41494D4F; // "AIMO"
const VERSION: u32 = 1;

/// Header structure at the beginning of shared memory
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ShmHeader {
    pub magic: u32,
    pub version: u32,
    pub request_queue_offset: u64,
    pub response_queue_offset: u64,
    pub tensor_pool_offset: u64,
    pub max_tensor_size: u64,
    pub notify_read_fd: i32,  // Read end of notification pipe (all platforms)
    pub notify_write_fd: i32, // Write end of notification pipe (all platforms)
}

/// Tensor header in shared memory
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TensorHeader {
    pub ndim: u32,
    pub dtype: u8,
    pub _padding: [u8; 3],
    pub shape: [i64; 8], // Max 8 dimensions
    pub data_size: u64,
}

/// Shared memory manager for zero-copy tensor communication
pub struct ShmManager {
    shmem: Shmem,
    size: usize,
}

// SAFETY: Shmem uses OS-level shared memory which is inherently thread-safe.
// Multiple threads/processes can safely access the same shared memory region.
unsafe impl Send for ShmManager {}
unsafe impl Sync for ShmManager {}

impl ShmManager {
    /// Create a new shared memory region
    pub fn create(name: &str, size: usize) -> Result<Self, ShmError> {
        // Force remove any existing shared memory with this name
        let _ = ShmemConf::new().os_id(name).force_create_flink();

        // Try to create, if it exists, remove owner and try again
        let shmem = match ShmemConf::new().size(size).os_id(name).create() {
            Ok(s) => s,
            Err(_e) => {
                // Try removing existing owner and creating again
                let _ = ShmemConf::new().os_id(name).force_create_flink();
                std::thread::sleep(std::time::Duration::from_millis(100));
                ShmemConf::new().size(size).os_id(name).create()?
            }
        };

        // Create notification pipe (works on all platforms)
        let (notify_read_fd, notify_write_fd) = unsafe {
            let mut fds = [0i32; 2];

            #[cfg(unix)]
            let ret = libc::pipe(fds.as_mut_ptr());

            #[cfg(windows)]
            let ret = libc::pipe(fds.as_mut_ptr(), 8192, libc::O_BINARY);

            if ret == 0 {
                // Set non-blocking mode on both ends
                #[cfg(unix)]
                {
                    libc::fcntl(fds[0], libc::F_SETFL, libc::O_NONBLOCK);
                    libc::fcntl(fds[1], libc::F_SETFL, libc::O_NONBLOCK);
                }

                log::info!(
                    "Created notification pipe: read_fd={}, write_fd={}",
                    fds[0],
                    fds[1]
                );
                (fds[0], fds[1])
            } else {
                log::warn!("Failed to create pipe, falling back to polling");
                (-1, -1)
            }
        };

        // Initialize header
        unsafe {
            let ptr = shmem.as_ptr() as *mut ShmHeader;
            std::ptr::write(
                ptr,
                ShmHeader {
                    magic: MAGIC_NUMBER,
                    version: VERSION,
                    request_queue_offset: 4096, // Start after header page
                    response_queue_offset: 4096 + 64 * 1024, // 64KB for request queue
                    tensor_pool_offset: 128 * 1024, // Start tensor pool at 128KB
                    max_tensor_size: (size - 128 * 1024) as u64,
                    notify_read_fd,
                    notify_write_fd,
                },
            );
        }

        log::info!(
            "Created shared memory region '{}' of size {} bytes",
            name,
            size
        );

        Ok(Self { shmem, size })
    }

    /// Connect to an existing shared memory region
    pub fn connect(name: &str) -> Result<Self, ShmError> {
        let shmem = ShmemConf::new().os_id(name).open()?;

        let size = shmem.len();

        // Verify header
        unsafe {
            let ptr = shmem.as_ptr() as *const ShmHeader;
            let header = std::ptr::read(ptr);

            if header.magic != MAGIC_NUMBER {
                return Err(ShmError::InvalidMagic);
            }
            if header.version != VERSION {
                return Err(ShmError::VersionMismatch);
            }
        }

        log::info!(
            "Connected to shared memory region '{}' of size {} bytes",
            name,
            size
        );

        Ok(Self { shmem, size })
    }

    /// Get the base pointer to shared memory
    pub fn as_ptr(&self) -> *mut u8 {
        self.shmem.as_ptr()
    }

    /// Get the size of the shared memory region
    pub fn size(&self) -> usize {
        self.size
    }

    /// Helper to read the header
    fn header(&self) -> ShmHeader {
        unsafe { std::ptr::read(self.as_ptr() as *const ShmHeader) }
    }

    /// Get request queue offset
    pub fn request_queue_offset(&self) -> usize {
        let header = self.header();
        header.request_queue_offset as usize
    }

    /// Get response queue offset
    pub fn response_queue_offset(&self) -> usize {
        let header = self.header();
        header.response_queue_offset as usize
    }

    /// Get tensor pool offset
    pub fn tensor_pool_offset(&self) -> usize {
        let header = self.header();
        header.tensor_pool_offset as usize
    }

    /// Get maximum tensor size
    pub fn max_tensor_size(&self) -> usize {
        let header = self.header();
        header.max_tensor_size as usize
    }

    /// Get notification pipe read fd
    pub fn notify_read_fd(&self) -> i32 {
        let header = self.header();
        header.notify_read_fd
    }

    /// Get notification pipe write fd
    pub fn notify_write_fd(&self) -> i32 {
        let header = self.header();
        header.notify_write_fd
    }

    #[cfg(windows)]
    pub fn notify_read_handle(&self) -> HANDLE {
        let header = self.header();
        header.notify_read_fd as HANDLE
    }

    #[cfg(windows)]
    pub fn notify_write_handle(&self) -> HANDLE {
        let header = self.header();
        header.notify_write_fd as HANDLE
    }
}

#[derive(Debug)]
pub enum ShmError {
    ShmemError(shared_memory::ShmemError),
    InvalidMagic,
    VersionMismatch,
}

impl From<shared_memory::ShmemError> for ShmError {
    fn from(e: shared_memory::ShmemError) -> Self {
        ShmError::ShmemError(e)
    }
}

impl std::fmt::Display for ShmError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ShmError::ShmemError(e) => write!(f, "Shared memory error: {}", e),
            ShmError::InvalidMagic => write!(f, "Invalid magic number in shared memory"),
            ShmError::VersionMismatch => write!(f, "Shared memory version mismatch"),
        }
    }
}

impl std::error::Error for ShmError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_connect() {
        let name = format!("/test_shm_create_connect_{}", std::process::id());
        let size = 1024 * 1024; // 1MB

        // Create
        let manager1 = match ShmManager::create(&name, size) {
            Ok(manager) => manager,
            Err(ShmError::ShmemError(shared_memory::ShmemError::MapCreateFailed(_))) => {
                eprintln!("Skipping shared memory test (mapping creation failed)");
                return;
            }
            Err(err) => panic!("Failed to create shared memory: {}", err),
        };
        assert_eq!(manager1.size(), size);

        // Connect
        let manager2 = ShmManager::connect(&name).unwrap();
        assert_eq!(manager2.size(), size);

        // Verify offsets match
        assert_eq!(
            manager1.request_queue_offset(),
            manager2.request_queue_offset()
        );
        assert_eq!(
            manager1.response_queue_offset(),
            manager2.response_queue_offset()
        );
        assert_eq!(manager1.tensor_pool_offset(), manager2.tensor_pool_offset());
    }
}

#[cfg(test)]
#[path = "memory_tests.rs"]
mod memory_tests;
