use std::sync::atomic::{AtomicUsize, Ordering};

/// Lock-free MPSC ring buffer queue for request/response metadata.
///
/// Uses a two-phase head protocol to ensure consumers never observe a slot
/// before its producer has finished writing:
///
/// - `head_reserved`: producers advance this with CAS to claim a write slot.
/// - `head_committed`: producers advance this *after* the write completes;
///   consumers gate their reads on this, not on `head_reserved`.
/// - `tail`: the consumer read position.
///
/// The first three T-sized slots in the backing buffer are used for the three
/// control atomics; data slots follow from offset 3 onwards.
///
/// # Safety invariants
/// - T must be at least `size_of::<AtomicUsize>()` bytes and aligned to
///   `align_of::<AtomicUsize>()` so the control atomics can live in-band.
/// - The backing memory must outlive every `LockFreeRingBuffer` that points
///   into it (shared-memory lifetime is managed by the caller).
pub struct LockFreeRingBuffer<T: Copy> {
    buffer: *mut T,
    capacity: usize,
    head_reserved: *const AtomicUsize,
    head_committed: *const AtomicUsize,
    tail: *const AtomicUsize,
}

unsafe impl<T: Copy> Send for LockFreeRingBuffer<T> {}
unsafe impl<T: Copy> Sync for LockFreeRingBuffer<T> {}

impl<T: Copy> LockFreeRingBuffer<T> {
    /// Create a new ring buffer, initialising control atomics to zero.
    ///
    /// # Safety
    /// `buffer` must point to valid, writable memory of at least
    /// `capacity * size_of::<T>()` bytes that outlives this struct.
    pub unsafe fn new(buffer: *mut T, capacity: usize) -> Self {
        assert!(capacity >= 4, "capacity must be at least 4");
        assert!(
            std::mem::size_of::<T>() >= std::mem::size_of::<AtomicUsize>(),
            "T must be at least AtomicUsize sized"
        );
        assert!(
            std::mem::align_of::<T>() >= std::mem::align_of::<AtomicUsize>(),
            "T must be aligned for AtomicUsize"
        );

        let head_reserved_ptr = buffer as *mut AtomicUsize;
        let head_committed_ptr = buffer.add(1) as *mut AtomicUsize;
        let tail_ptr = buffer.add(2) as *mut AtomicUsize;

        std::ptr::write(head_reserved_ptr, AtomicUsize::new(0));
        std::ptr::write(head_committed_ptr, AtomicUsize::new(0));
        std::ptr::write(tail_ptr, AtomicUsize::new(0));

        Self {
            buffer: buffer.add(3),
            capacity: capacity - 3,
            head_reserved: head_reserved_ptr,
            head_committed: head_committed_ptr,
            tail: tail_ptr,
        }
    }

    /// Connect to an existing ring buffer that was previously initialised with
    /// [`new`](Self::new). Does **not** re-initialise the control atomics.
    ///
    /// # Safety
    /// Same lifetime requirements as `new`; the buffer must already have been
    /// initialised.
    pub unsafe fn connect(buffer: *mut T, capacity: usize) -> Self {
        assert!(capacity >= 4, "capacity must be at least 4");
        assert!(
            std::mem::size_of::<T>() >= std::mem::size_of::<AtomicUsize>(),
            "T must be at least AtomicUsize sized"
        );
        assert!(
            std::mem::align_of::<T>() >= std::mem::align_of::<AtomicUsize>(),
            "T must be aligned for AtomicUsize"
        );

        Self {
            buffer: buffer.add(3),
            capacity: capacity - 3,
            head_reserved: buffer as *const AtomicUsize,
            head_committed: buffer.add(1) as *const AtomicUsize,
            tail: buffer.add(2) as *const AtomicUsize,
        }
    }

    /// Push `item` into the queue.
    ///
    /// Returns `Err(QueueError::Full)` immediately if there is no space.
    ///
    /// Multiple concurrent producers are supported. Each producer:
    /// 1. Claims a slot by CAS-advancing `head_reserved`.
    /// 2. Writes data into the claimed slot.
    /// 3. Spin-waits until all earlier producers have committed, then
    ///    CAS-advances `head_committed`, making the slot visible to consumers.
    ///
    /// The spin in step 3 is bounded by earlier producers completing their
    /// (very short) writes; it is *not* a lock.
    pub fn push(&self, item: T) -> Result<(), QueueError> {
        loop {
            let reserved = unsafe { (*self.head_reserved).load(Ordering::Relaxed) };
            let next_reserved = (reserved + 1) % self.capacity;
            let tail = unsafe { (*self.tail).load(Ordering::Acquire) };

            if next_reserved == tail {
                return Err(QueueError::Full);
            }

            // Phase 1: claim the slot.
            if unsafe {
                (*self.head_reserved)
                    .compare_exchange_weak(
                        reserved,
                        next_reserved,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    )
                    .is_ok()
            } {
                // Phase 2: write to the claimed slot. The slot is exclusively
                // ours until we advance head_committed.
                unsafe { std::ptr::write(self.buffer.add(reserved), item) };

                // Phase 3: commit in reservation order so consumers always see
                // a contiguous prefix of written slots.
                loop {
                    match unsafe {
                        (*self.head_committed).compare_exchange_weak(
                            reserved,
                            next_reserved,
                            Ordering::Release,
                            Ordering::Relaxed,
                        )
                    } {
                        Ok(_) => break,
                        Err(_) => std::hint::spin_loop(),
                    }
                }
                return Ok(());
            }
            // CAS failed: another producer took this slot; retry.
        }
    }

    /// Pop an item from the queue.
    ///
    /// Returns `None` if the queue is empty. A single consumer is expected;
    /// multiple concurrent consumers are also safe (MPMC consumer CAS).
    pub fn pop(&self) -> Option<T> {
        loop {
            let tail = unsafe { (*self.tail).load(Ordering::Relaxed) };
            // Gate on head_committed, not head_reserved, so we only read slots
            // whose producers have finished writing.
            let committed = unsafe { (*self.head_committed).load(Ordering::Acquire) };

            if tail == committed {
                return None;
            }

            let next_tail = (tail + 1) % self.capacity;
            if unsafe {
                (*self.tail)
                    .compare_exchange_weak(
                        tail,
                        next_tail,
                        Ordering::Release,
                        Ordering::Relaxed,
                    )
                    .is_ok()
            } {
                let item = unsafe { std::ptr::read(self.buffer.add(tail)) };
                return Some(item);
            }
            // CAS failed: another consumer took this slot; retry.
        }
    }

    /// Returns `true` if there are no committed items ready to be popped.
    ///
    /// This is an approximate snapshot — a concurrent push may have committed
    /// between the two loads.
    pub fn is_empty(&self) -> bool {
        let committed = unsafe { (*self.head_committed).load(Ordering::Acquire) };
        let tail = unsafe { (*self.tail).load(Ordering::Acquire) };
        committed == tail
    }

    /// Returns the approximate number of items currently ready to pop.
    pub fn len(&self) -> usize {
        let committed = unsafe { (*self.head_committed).load(Ordering::Acquire) };
        let tail = unsafe { (*self.tail).load(Ordering::Acquire) };
        if committed >= tail {
            committed - tail
        } else {
            self.capacity - tail + committed
        }
    }

    /// Returns the maximum number of items the queue can hold.
    pub fn capacity(&self) -> usize {
        // One slot is kept empty to distinguish full from empty.
        self.capacity - 1
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueError {
    Full,
}

impl std::fmt::Display for QueueError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            QueueError::Full => write!(f, "Queue is full"),
        }
    }
}

impl std::error::Error for QueueError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::{alloc, dealloc, Layout};

    #[test]
    fn test_push_pop() {
        unsafe {
            let layout = Layout::array::<u64>(1024).unwrap();
            let buffer = alloc(layout) as *mut u64;

            let queue = LockFreeRingBuffer::new(buffer, 1024);

            assert!(queue.push(42).is_ok());
            assert!(queue.push(43).is_ok());
            assert!(queue.push(44).is_ok());

            assert_eq!(queue.pop(), Some(42));
            assert_eq!(queue.pop(), Some(43));
            assert_eq!(queue.pop(), Some(44));
            assert_eq!(queue.pop(), None);

            dealloc(buffer as *mut u8, layout);
        }
    }

    #[test]
    fn test_queue_full() {
        unsafe {
            // capacity=10 → 3 control slots → 7 data slots → 6 max items (1 sentinel)
            let capacity = 10;
            let layout = Layout::array::<u64>(capacity).unwrap();
            let buffer = alloc(layout) as *mut u64;

            let queue = LockFreeRingBuffer::new(buffer, capacity);
            assert_eq!(queue.capacity(), capacity - 4);

            for i in 0..(capacity - 4) {
                assert!(queue.push(i as u64).is_ok());
            }

            assert!(queue.push(999).is_err());

            dealloc(buffer as *mut u8, layout);
        }
    }
}

#[cfg(test)]
#[path = "ring_buffer_tests.rs"]
mod ring_buffer_tests;
