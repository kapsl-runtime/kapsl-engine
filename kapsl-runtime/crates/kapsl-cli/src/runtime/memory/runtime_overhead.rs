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
    use super::nvml_pci_bus_id;

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
}
