//! The runtime's own per-process GPU footprint (CUDA context and the like),
//! measured once, right after the CUDA context exists and before any device
//! pool or model does. It is attribution only: it explains bytes the device
//! already holds, and never changes a domain's budget or availability.
//!
//! The figure comes from the driver's per-process accounting (NVML), so it is
//! independent of the memory ledger. Taking it before any model is resident
//! means it cannot absorb per-model under-reporting, which is what a
//! ledger-vs-driver closure check is there to catch.

/// Convert a CUDA PCI bus id (`0000:01:00.0`) to the form NVML expects
/// (`00000000:01:00.0`). Already-long ids and anything unrecognised pass
/// through unchanged.
#[cfg(any(test, feature = "gpu-device-pool"))]
pub(super) fn nvml_pci_bus_id(cuda_bus_id: &str) -> String {
    let trimmed = cuda_bus_id.trim_end_matches('\0').trim();
    match trimmed.split_once(':') {
        Some((domain, rest))
            if domain.len() < 8 && domain.chars().all(|c| c.is_ascii_hexdigit()) =>
        {
            format!("{domain:0>8}:{rest}")
        }
        _ => trimmed.to_string(),
    }
}

/// Whether two PCI bus ids name the same device, ignoring case, the NUL the CUDA
/// runtime appends and the 4- versus 8-digit domain (`0000:01:00.0` is
/// `00000000:01:00.0`).
#[cfg(any(test, feature = "gpu-device-pool"))]
pub(super) fn same_pci_bus_id(a: &str, b: &str) -> bool {
    nvml_pci_bus_id(a).eq_ignore_ascii_case(&nvml_pci_bus_id(b))
}

/// Check that the CUDA device opened for engine device `device_id` is the GPU the
/// inventory (NVML, via `kapsl-hal`) calls `device_id`. The runtime keys budgets,
/// pools and leases by that id and opens the CUDA device with the same number,
/// which is only right when CUDA's ordering matches NVML's (`CUDA_DEVICE_ORDER`
/// unset defaults to fastest-first, and `CUDA_VISIBLE_DEVICES` renumbers). A
/// mismatch must stop the load: continuing would charge one GPU's budget for
/// another GPU's memory. A missing id on either side cannot be checked and is
/// accepted. With the PCI-based ordinal map in place this is a defensive check.
#[cfg(any(test, feature = "gpu-device-pool"))]
pub(super) fn verify_cuda_identity(
    device_id: usize,
    inventory_pci: Option<&str>,
    cuda_pci: Option<&str>,
) -> Result<(), String> {
    match (inventory_pci, cuda_pci) {
        (Some(inventory), Some(cuda)) if !same_pci_bus_id(inventory, cuda) => Err(format!(
            "CUDA device {device_id} is PCI {cuda} but the device inventory lists device {device_id} as PCI {inventory}: \
             the opened CUDA device is not the inventory device, so it is not managed rather than \
             accounting one GPU's memory against another"
        )),
        _ => Ok(()),
    }
}

/// The PCI bus id the CUDA driver reports for an opened device.
#[cfg(feature = "gpu-device-pool")]
pub(super) fn cuda_pci_bus_id(cuda: &cudarc::driver::CudaDevice) -> Option<String> {
    let mut buffer = [0 as std::os::raw::c_char; 32];
    // SAFETY: the buffer outlives the call, its length is passed, and the device
    // handle comes from a live `CudaDevice`.
    let status = unsafe {
        cudarc::driver::sys::lib().cuDeviceGetPCIBusId(
            buffer.as_mut_ptr(),
            buffer.len() as std::os::raw::c_int,
            *cuda.cu_device(),
        )
    };
    if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        return None;
    }
    // SAFETY: the driver NUL-terminates the id within the buffer.
    Some(
        unsafe { std::ffi::CStr::from_ptr(buffer.as_ptr()) }
            .to_string_lossy()
            .into_owned(),
    )
}

/// Bytes the driver attributes to this process on `cuda`'s device, or `None`
/// when NVML is unavailable, does not list this process (for example inside a
/// PID namespace that hides it), or does not report per-process memory.
#[cfg(feature = "gpu-device-pool")]
pub(super) fn sample_process_gpu_bytes(cuda: &cudarc::driver::CudaDevice) -> Option<usize> {
    use nvml_wrapper::enums::device::UsedGpuMemory;

    let mut buffer = [0 as std::os::raw::c_char; 32];
    // SAFETY: the buffer outlives the call, its length is passed, and the
    // device handle comes from a live `CudaDevice`.
    let status = unsafe {
        cudarc::driver::sys::lib().cuDeviceGetPCIBusId(
            buffer.as_mut_ptr(),
            buffer.len() as std::os::raw::c_int,
            *cuda.cu_device(),
        )
    };
    if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        log::debug!("[device-memory] runtime overhead: cuDeviceGetPCIBusId failed: {status:?}");
        return None;
    }
    // SAFETY: the driver NUL-terminates the id within the buffer.
    let cuda_bus_id = unsafe { std::ffi::CStr::from_ptr(buffer.as_ptr()) }
        .to_string_lossy()
        .into_owned();

    let nvml = match nvml_wrapper::Nvml::init() {
        Ok(nvml) => nvml,
        Err(error) => {
            log::debug!("[device-memory] runtime overhead: NVML unavailable: {error}");
            return None;
        }
    };
    let device = nvml
        .device_by_pci_bus_id(nvml_pci_bus_id(&cuda_bus_id))
        .ok()?;
    let pid = std::process::id();
    let used = device
        .running_compute_processes()
        .ok()?
        .into_iter()
        .find(|process| process.pid == pid)
        .map(|process| process.used_gpu_memory);
    match used {
        Some(UsedGpuMemory::Used(bytes)) => usize::try_from(bytes).ok(),
        _ => {
            log::debug!(
                "[device-memory] runtime overhead: NVML does not report this process on {cuda_bus_id}"
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{nvml_pci_bus_id, same_pci_bus_id, verify_cuda_identity};

    #[test]
    fn cuda_bus_ids_are_widened_to_the_nvml_form() {
        assert_eq!(nvml_pci_bus_id("0000:01:00.0"), "00000000:01:00.0");
        assert_eq!(nvml_pci_bus_id("0000:af:00.0\0"), "00000000:af:00.0");
    }

    #[test]
    fn long_or_unrecognised_bus_ids_pass_through() {
        assert_eq!(nvml_pci_bus_id("00000000:01:00.0"), "00000000:01:00.0");
        assert_eq!(nvml_pci_bus_id("not-a-bus-id"), "not-a-bus-id");
    }

    #[test]
    fn pci_bus_ids_compare_across_cuda_and_nvml_spellings() {
        assert!(same_pci_bus_id("0000:01:00.0", "00000000:01:00.0"));
        assert!(same_pci_bus_id("0000:AF:00.0\0", "00000000:af:00.0"));
        assert!(!same_pci_bus_id("0000:01:00.0", "0000:41:00.0"));
    }

    #[test]
    fn matching_cuda_and_inventory_ids_pass_and_unknown_ids_are_accepted() {
        assert!(verify_cuda_identity(0, Some("00000000:01:00.0"), Some("0000:01:00.0")).is_ok());
        assert!(verify_cuda_identity(0, None, Some("0000:01:00.0")).is_ok());
        assert!(verify_cuda_identity(0, Some("00000000:01:00.0"), None).is_ok());
        assert!(verify_cuda_identity(0, None, None).is_ok());
    }

    #[test]
    fn a_device_that_is_not_the_inventory_device_is_reported_so_it_is_not_managed() {
        // Defensive check behind the PCI ordinal map: if the opened CUDA device is
        // not the inventory device, the caller leaves it unmanaged and says why.
        let error = verify_cuda_identity(0, Some("00000000:01:00.0"), Some("0000:41:00.0"))
            .expect_err("CUDA 0 is not the inventory's device 0");
        assert!(error.contains("not managed"), "{error}");
        assert!(error.contains("0000:41:00.0") && error.contains("00000000:01:00.0"));
    }
}
