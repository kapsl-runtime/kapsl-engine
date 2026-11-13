//! The runtime's own per-process GPU footprint (CUDA context and the like),
//! measured once, right after the CUDA context exists and before any device
//! pool or model does. It is attribution only: it explains bytes the device
//! already holds, and never changes a domain's budget or availability.
//!
//! The figure comes from the driver, so it is independent of the memory ledger.
//! Taking it before any model is resident means it cannot absorb per-model
//! under-reporting, which is what a ledger-vs-driver closure check is there to
//! catch. Two sources, in order of preference:
//!
//! 1. NVML's per-process accounting for this PID (`nvml-process`).
//! 2. When NVML cannot list this PID (a PID namespace hides it, which is most
//!    containers), the growth of the device's used memory across context
//!    creation (`device-delta`). NVML's device-level figure is visible from any
//!    namespace and the difference cancels the driver-reserved memory.
//!
//! Neither source sees what ORT, cuDNN and cuBLAS add once a model creates its
//! first session (run 7: 34 to 182 MiB beyond a 256 MiB context, varying with
//! the model family). That stays outside the row on purpose: it is what the
//! closure check reports.

#![cfg_attr(not(any(test, feature = "gpu-device-pool")), allow(dead_code))]

use std::time::Duration;

const MIB: usize = 1024 * 1024;

/// A CUDA context is never smaller than this. A device delta below it means
/// another process freed memory during the window, not a measurement.
pub(super) const MIN_PLAUSIBLE_OVERHEAD_BYTES: usize = 32 * MIB;
/// A device delta above this means another process allocated during the window.
pub(super) const MAX_PLAUSIBLE_OVERHEAD_BYTES: usize = 2048 * MIB;
/// Readings of the device's used memory taken back to back must agree within
/// this, or something else is allocating and the delta cannot be trusted.
pub(super) const STABLE_READING_TOLERANCE_BYTES: usize = 8 * MIB;
/// Back-to-back readings per side of the window, and the pause between them.
#[cfg(feature = "gpu-device-pool")]
pub(super) const STABLE_READINGS: usize = 3;
#[cfg(feature = "gpu-device-pool")]
pub(super) const STABLE_READING_PAUSE: Duration = Duration::from_millis(5);

/// Where a device's runtime-overhead figure came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum OverheadSource {
    NvmlProcess,
    DeviceDelta,
}

impl OverheadSource {
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::NvmlProcess => "nvml-process",
            Self::DeviceDelta => "device-delta",
        }
    }
}

/// What the startup probe read around one device's context creation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct OverheadReadings {
    /// NVML's per-process figure for this PID, once the context exists.
    pub(super) process_bytes: Option<usize>,
    /// The device's used memory (NVML, device level) before the context.
    pub(super) device_before: Option<usize>,
    /// The same, after the context exists.
    pub(super) device_after: Option<usize>,
}

impl OverheadReadings {
    /// Growth of the device's used memory across the window, when both sides
    /// were read.
    pub(super) fn device_delta(&self) -> Option<usize> {
        match (self.device_before, self.device_after) {
            (Some(before), Some(after)) => after.checked_sub(before),
            _ => None,
        }
    }
}

/// The runtime row for one device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RuntimeOverhead {
    pub(super) bytes: usize,
    pub(super) source: OverheadSource,
    /// The device delta when it was readable, kept for comparison even when
    /// the process figure is the one used.
    pub(super) device_delta: Option<usize>,
}

impl RuntimeOverhead {
    /// The line the engine logs (and the harness parses) once per device. The
    /// leading sentence is unchanged from the NVML-only version.
    pub(super) fn log_line(&self, device_id: usize) -> String {
        let delta = self
            .device_delta
            .map_or_else(|| "n/a".to_string(), |bytes| bytes.to_string());
        format!(
            "CUDA device {device_id}: runtime process overhead after init = {} bytes (source={}, device-delta={delta})",
            self.bytes,
            self.source.label()
        )
    }
}

/// Choose the row from the readings: the process figure when NVML listed this
/// PID, otherwise the device delta when it is plausible. `Err` carries why
/// there is no row.
pub(super) fn resolve_overhead(readings: &OverheadReadings) -> Result<RuntimeOverhead, String> {
    let delta = readings.device_delta();
    if let Some(bytes) = readings.process_bytes.filter(|bytes| *bytes > 0) {
        return Ok(RuntimeOverhead {
            bytes,
            source: OverheadSource::NvmlProcess,
            device_delta: delta,
        });
    }
    match (readings.device_before, readings.device_after, delta) {
        (Some(before), Some(after), None) => Err(format!(
            "device used memory fell from {before} to {after} bytes across context creation \
             (another process freed memory); no device delta"
        )),
        (_, _, Some(bytes)) if bytes < MIN_PLAUSIBLE_OVERHEAD_BYTES => Err(format!(
            "device used memory grew by only {bytes} bytes across context creation, below the \
             {MIN_PLAUSIBLE_OVERHEAD_BYTES}-byte minimum for a CUDA context; not a measurement"
        )),
        (_, _, Some(bytes)) if bytes > MAX_PLAUSIBLE_OVERHEAD_BYTES => Err(format!(
            "device used memory grew by {bytes} bytes across context creation, above the \
             {MAX_PLAUSIBLE_OVERHEAD_BYTES}-byte ceiling for a CUDA context (another process \
             allocated during the window)"
        )),
        (_, _, Some(bytes)) => Ok(RuntimeOverhead {
            bytes,
            source: OverheadSource::DeviceDelta,
            device_delta: Some(bytes),
        }),
        _ => Err(
            "NVML lists no figure for this process and device-level memory could not be \
                  read on both sides of context creation"
                .to_string(),
        ),
    }
}

/// Read the device's used memory `reads` times back to back. `None` when any
/// read fails or the reads differ by more than `tolerance` (something else is
/// allocating). Returns the last read, the one closest to what follows.
pub(super) fn stable_reading(
    mut sample: impl FnMut() -> Option<usize>,
    reads: usize,
    tolerance: usize,
    pause: Duration,
) -> Option<usize> {
    let mut lowest = usize::MAX;
    let mut highest = 0usize;
    let mut last = None;
    for read in 0..reads.max(1) {
        if read > 0 && !pause.is_zero() {
            std::thread::sleep(pause);
        }
        let value = sample()?;
        lowest = lowest.min(value);
        highest = highest.max(value);
        last = Some(value);
    }
    (highest - lowest <= tolerance).then_some(last).flatten()
}

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
        log::warn!("[device-memory] runtime overhead: cuDeviceGetPCIBusId failed: {status:?}");
        return None;
    }
    // SAFETY: the driver NUL-terminates the id within the buffer.
    let cuda_bus_id = unsafe { std::ffi::CStr::from_ptr(buffer.as_ptr()) }
        .to_string_lossy()
        .into_owned();

    let nvml = match crate::runtime::resources::init_nvml() {
        Ok(nvml) => nvml,
        Err(error) => {
            log::warn!("[device-memory] runtime overhead: NVML unavailable: {error}");
            return None;
        }
    };
    // Warnings, not debug lines: without this sample the runtime row is absent
    // and the ledger is short by the CUDA context, so the reason must be visible.
    let device = match nvml.device_by_pci_bus_id(nvml_pci_bus_id(&cuda_bus_id)) {
        Ok(device) => device,
        Err(error) => {
            log::warn!("[device-memory] runtime overhead: NVML cannot open {cuda_bus_id}: {error}");
            return None;
        }
    };
    let pid = std::process::id();
    let processes = match device.running_compute_processes() {
        Ok(processes) => processes,
        Err(error) => {
            log::warn!(
                "[device-memory] runtime overhead: NVML cannot list compute processes on {cuda_bus_id}: {error}"
            );
            return None;
        }
    };
    let listed: Vec<u32> = processes.iter().map(|process| process.pid).collect();
    let used = processes
        .into_iter()
        .find(|process| process.pid == pid)
        .map(|process| process.used_gpu_memory);
    match used {
        Some(UsedGpuMemory::Used(bytes)) => usize::try_from(bytes).ok(),
        Some(_) => {
            log::warn!(
                "[device-memory] runtime overhead: NVML lists pid {pid} on {cuda_bus_id} without a memory figure"
            );
            None
        }
        None => {
            log::warn!(
                "[device-memory] runtime overhead: NVML does not list pid {pid} on {cuda_bus_id} \
                 (it lists {:?}; a PID namespace can hide this process)",
                listed
            );
            None
        }
    }
}

/// The device's used bytes as NVML reports them for the inventory device
/// (`pci_bus_id` when the inventory has one, else NVML index `device_id`).
/// Works from any PID namespace; includes memory the driver reserves, which
/// cancels in a before/after difference.
#[cfg(feature = "gpu-device-pool")]
pub(super) fn sample_device_used_bytes(
    device_id: usize,
    pci_bus_id: Option<&str>,
) -> Option<usize> {
    let nvml = crate::runtime::resources::init_nvml().ok()?;
    let device = match pci_bus_id {
        Some(pci) => nvml.device_by_pci_bus_id(nvml_pci_bus_id(pci)).ok()?,
        None => nvml.device_by_index(u32::try_from(device_id).ok()?).ok()?,
    };
    usize::try_from(device.memory_info().ok()?.used).ok()
}

/// The row for one device: read the device before and after `create_context`,
/// ask NVML for this process's figure, and resolve. The context is created by
/// the caller's closure so the readings bracket exactly that.
#[cfg(feature = "gpu-device-pool")]
pub(super) fn probe_runtime_overhead<T>(
    device_id: usize,
    pci_bus_id: Option<&str>,
    create_context: impl FnOnce() -> Result<T, String>,
    process_bytes: impl FnOnce(&T) -> Option<usize>,
) -> Result<(T, Result<RuntimeOverhead, String>), String> {
    let device_before = stable_reading(
        || sample_device_used_bytes(device_id, pci_bus_id),
        STABLE_READINGS,
        STABLE_READING_TOLERANCE_BYTES,
        STABLE_READING_PAUSE,
    );
    let context = create_context()?;
    let device_after = stable_reading(
        || sample_device_used_bytes(device_id, pci_bus_id),
        STABLE_READINGS,
        STABLE_READING_TOLERANCE_BYTES,
        STABLE_READING_PAUSE,
    );
    let readings = OverheadReadings {
        process_bytes: process_bytes(&context),
        device_before,
        device_after,
    };
    let resolved = resolve_overhead(&readings);
    Ok((context, resolved))
}

#[cfg(test)]
mod tests {
    use super::*;
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

    fn readings(
        process: Option<usize>,
        before: Option<usize>,
        after: Option<usize>,
    ) -> OverheadReadings {
        OverheadReadings {
            process_bytes: process,
            device_before: before,
            device_after: after,
        }
    }

    #[test]
    fn a_listed_process_uses_its_own_figure_and_still_records_the_delta() {
        // Run 3 to 5 hosts: NVML lists the PID, the row is the process figure.
        let row = resolve_overhead(&readings(
            Some(256 * MIB),
            Some(1000 * MIB),
            Some(1290 * MIB),
        ))
        .unwrap();
        assert_eq!(row.source, OverheadSource::NvmlProcess);
        assert_eq!(row.bytes, 256 * MIB);
        assert_eq!(
            row.device_delta,
            Some(290 * MIB),
            "kept to compare the two sources"
        );
        let line = row.log_line(1);
        assert_eq!(
            line,
            "CUDA device 1: runtime process overhead after init = 268435456 bytes \
             (source=nvml-process, device-delta=304087040)"
        );
    }

    #[test]
    fn an_unlisted_process_falls_back_to_the_device_delta() {
        // Run 7 host: NVML does not list the PID, so there is no process figure.
        let row = resolve_overhead(&readings(None, Some(500 * MIB), Some(790 * MIB))).unwrap();
        assert_eq!(row.source, OverheadSource::DeviceDelta);
        assert_eq!(row.bytes, 290 * MIB);
        assert_eq!(
            row.log_line(0),
            "CUDA device 0: runtime process overhead after init = 304087040 bytes \
             (source=device-delta, device-delta=304087040)"
        );
        // A listed process with a zero figure is the same as unlisted.
        let zero = resolve_overhead(&readings(Some(0), Some(500 * MIB), Some(790 * MIB))).unwrap();
        assert_eq!(zero.source, OverheadSource::DeviceDelta);
    }

    #[test]
    fn an_implausible_or_missing_delta_is_no_row_with_a_reason() {
        // Another process freed memory: the device got emptier.
        let fell = resolve_overhead(&readings(None, Some(900 * MIB), Some(700 * MIB))).unwrap_err();
        assert!(
            fell.contains("fell from") && fell.contains("freed"),
            "{fell}"
        );
        // Too small to be a context.
        let tiny = resolve_overhead(&readings(None, Some(900 * MIB), Some(910 * MIB))).unwrap_err();
        assert!(tiny.contains("below") && tiny.contains("minimum"), "{tiny}");
        // Too large: another process allocated in the window.
        let huge = resolve_overhead(&readings(None, Some(0), Some(3000 * MIB))).unwrap_err();
        assert!(huge.contains("ceiling"), "{huge}");
        // NVML unavailable on either side.
        for missing in [
            readings(None, None, Some(800 * MIB)),
            readings(None, Some(800 * MIB), None),
            readings(None, None, None),
        ] {
            let error = resolve_overhead(&missing).unwrap_err();
            assert!(error.contains("could not be read"), "{error}");
        }
        // The same delta is accepted at the bounds.
        assert!(
            resolve_overhead(&readings(None, Some(0), Some(MIN_PLAUSIBLE_OVERHEAD_BYTES))).is_ok()
        );
        assert!(
            resolve_overhead(&readings(None, Some(0), Some(MAX_PLAUSIBLE_OVERHEAD_BYTES))).is_ok()
        );
    }

    #[test]
    fn a_listed_process_wins_even_when_the_delta_is_implausible() {
        let row =
            resolve_overhead(&readings(Some(300 * MIB), Some(900 * MIB), Some(700 * MIB))).unwrap();
        assert_eq!(row.source, OverheadSource::NvmlProcess);
        assert_eq!(row.device_delta, None);
        assert!(row.log_line(0).ends_with("device-delta=n/a)"));
    }

    #[test]
    fn back_to_back_readings_must_agree_or_there_is_no_reading() {
        let steady = [800 * MIB, 802 * MIB, 801 * MIB];
        let mut reads = steady.into_iter();
        assert_eq!(
            stable_reading(
                || reads.next(),
                3,
                STABLE_READING_TOLERANCE_BYTES,
                Duration::ZERO
            ),
            Some(801 * MIB),
            "the last read is returned"
        );
        // Something else allocated 100 MiB between reads.
        let moving = [800 * MIB, 900 * MIB, 1000 * MIB];
        let mut reads = moving.into_iter();
        assert_eq!(
            stable_reading(
                || reads.next(),
                3,
                STABLE_READING_TOLERANCE_BYTES,
                Duration::ZERO
            ),
            None
        );
        // A failed read is no reading.
        let mut calls = 0;
        assert_eq!(
            stable_reading(
                || {
                    calls += 1;
                    (calls < 2).then_some(MIB)
                },
                3,
                STABLE_READING_TOLERANCE_BYTES,
                Duration::ZERO
            ),
            None
        );
    }
}
