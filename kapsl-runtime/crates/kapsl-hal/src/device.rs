use serde::{Deserialize, Serialize};
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use sys_info;
use thiserror::Error;

const DEFAULT_PROBE_TIMEOUT: Duration = Duration::from_millis(800);
const SYSTEM_PROFILER_TIMEOUT: Duration = Duration::from_secs(3);
const COMMAND_POLL_INTERVAL: Duration = Duration::from_millis(25);

static PROBE_CACHE: OnceLock<DeviceInfo> = OnceLock::new();

#[derive(Debug, Error)]
pub enum DeviceProbeError {
    #[error("sys_info error: {0}")]
    SysInfo(#[from] sys_info::Error),
}

/// Backend/provider for a device.
///
/// The serialized form is always a lowercase string (e.g. "cuda").
/// Unknown strings round-trip via `Custom`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DeviceBackend {
    Cpu,
    Cuda,
    Metal,
    Rocm,
    DirectML,
    OpenCL,
    Vulkan,
    WebGpu,
    OneApi,
    Custom(String),
}

impl DeviceBackend {
    fn parse(raw: &str) -> Self {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Self::Custom(String::new());
        }

        match trimmed.to_ascii_lowercase().as_str() {
            "cpu" => Self::Cpu,
            "cuda" => Self::Cuda,
            "metal" => Self::Metal,
            "rocm" => Self::Rocm,
            "directml" => Self::DirectML,
            "opencl" => Self::OpenCL,
            "vulkan" => Self::Vulkan,
            "webgpu" => Self::WebGpu,
            "oneapi" => Self::OneApi,
            other => Self::Custom(other.to_string()),
        }
    }
}

impl std::fmt::Display for DeviceBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceBackend::Cpu => write!(f, "cpu"),
            DeviceBackend::Cuda => write!(f, "cuda"),
            DeviceBackend::Metal => write!(f, "metal"),
            DeviceBackend::Rocm => write!(f, "rocm"),
            DeviceBackend::DirectML => write!(f, "directml"),
            DeviceBackend::OpenCL => write!(f, "opencl"),
            DeviceBackend::Vulkan => write!(f, "vulkan"),
            DeviceBackend::WebGpu => write!(f, "webgpu"),
            DeviceBackend::OneApi => write!(f, "oneapi"),
            DeviceBackend::Custom(s) => write!(f, "{s}"),
        }
    }
}

impl Serialize for DeviceBackend {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for DeviceBackend {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        Ok(Self::parse(&raw))
    }
}

// NOTE: Keep this struct as a simple JSON-friendly record: all optional fields are `Option<...>`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Device {
    pub id: usize,
    pub name: String,
    pub backend: DeviceBackend,

    pub memory_mb: u64,
    pub compute_units: u32,

    pub pci_bus_id: Option<String>,

    /// Driver version string (when available).
    pub driver_version: Option<String>,

    /// CUDA version string (e.g. "12.0") for CUDA-capable devices.
    pub cuda_version: Option<String>,

    /// CUDA compute capability (e.g. "8.6") for CUDA devices.
    pub compute_capability: Option<String>,

    pub utilization_gpu_pct: Option<u32>,
    pub temperature_c: Option<u32>,

    pub supports_fp16: bool,
    pub supports_int8: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    // CPU info
    pub cpu_cores: u32,
    pub total_memory: u64,
    pub os_type: String,
    pub os_release: String,

    pub has_cuda: bool,
    pub has_metal: bool,
    pub has_rocm: bool,
    pub has_directml: bool,

    // All detected devices (CPU + GPUS)
    pub devices: Vec<Device>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuPreference {
    /// Match by provider backend and device id, e.g. "cuda:1".
    BackendId { backend: String, id: usize },
    /// Match by PCI bus id, e.g. "00000000:01:00.0".
    PciBusId(String),
    /// Match by case-insensitive substring in `Device.name`.
    NameContains(String),
}

impl GpuPreference {
    pub fn parse(spec: &str) -> Option<Self> {
        let trimmed = spec.trim();
        if trimmed.is_empty() {
            return None;
        }

        let lowered = trimmed.to_ascii_lowercase();

        if let Some((backend, id)) = lowered.split_once(':') {
            if let Ok(id) = id.trim().parse::<usize>() {
                return Some(Self::BackendId {
                    backend: backend.trim().to_string(),
                    id,
                });
            }
        }

        // Heuristic: if it contains ':' and '.', it looks like a PCI bus id.
        if trimmed.contains(':') && trimmed.contains('.') {
            return Some(Self::PciBusId(trimmed.to_string()));
        }

        Some(Self::NameContains(trimmed.to_string()))
    }
}

impl DeviceInfo {
    /// Probe device information (cached).
    pub fn probe() -> Self {
        if let Some(cached) = PROBE_CACHE.get() {
            return cached.clone();
        }

        let probed = Self::try_probe_with_timeout(DEFAULT_PROBE_TIMEOUT).unwrap_or_else(|_| {
            // Best-effort fallback; keeps the old API non-panicking.
            Self::fallback()
        });

        let _ = PROBE_CACHE.set(probed.clone());
        probed
    }

    /// Probe device information (not cached).
    pub fn try_probe() -> Result<Self, DeviceProbeError> {
        Self::try_probe_with_timeout(DEFAULT_PROBE_TIMEOUT)
    }

    /// Probe device information with a timeout applied to external commands.
    pub fn try_probe_with_timeout(timeout: Duration) -> Result<Self, DeviceProbeError> {
        let cpu_cores = sys_info::cpu_num()?;
        let total_memory = sys_info::mem_info()?.total;
        let os_type = sys_info::os_type().unwrap_or_else(|_| "unknown".to_string());
        let os_release = sys_info::os_release().unwrap_or_else(|_| "unknown".to_string());

        let mut devices = Vec::new();
        devices.push(Device {
            id: 0,
            name: "CPU".to_string(),
            backend: DeviceBackend::Cpu,
            memory_mb: total_memory / 1024,
            compute_units: cpu_cores,
            pci_bus_id: None,
            driver_version: None,
            cuda_version: None,
            compute_capability: None,
            utilization_gpu_pct: None,
            temperature_c: None,
            supports_fp16: true,
            supports_int8: true,
        });

        let cuda_version = Self::detect_cuda_version(timeout);

        if let Some(nvml_devices) = Self::detect_cuda_gpus_nvml(cuda_version.as_deref()) {
            devices.extend(nvml_devices);
        } else {
            devices.extend(Self::detect_cuda_gpus(timeout, cuda_version.as_deref()));
        }

        devices.extend(Self::detect_rocm_gpus(timeout, &os_release));
        devices.extend(Self::detect_metal(SYSTEM_PROFILER_TIMEOUT, &os_release));
        devices.extend(Self::detect_directml(timeout, &os_release));
        devices.extend(Self::detect_oneapi(timeout));
        devices.extend(Self::detect_webgpu());

        let (has_cuda, has_metal, has_rocm, has_directml) = Self::provider_flags(&devices);

        Ok(Self {
            cpu_cores,
            total_memory,
            os_type,
            os_release,
            has_cuda,
            has_metal,
            has_rocm,
            has_directml,
            devices,
        })
    }

    fn fallback() -> Self {
        Self {
            cpu_cores: 1,
            total_memory: 0,
            os_type: "unknown".to_string(),
            os_release: "unknown".to_string(),
            has_cuda: false,
            has_metal: false,
            has_rocm: false,
            has_directml: false,
            devices: vec![Device {
                id: 0,
                name: "CPU".to_string(),
                backend: DeviceBackend::Cpu,
                memory_mb: 0,
                compute_units: 1,
                pci_bus_id: None,
                driver_version: None,
                cuda_version: None,
                compute_capability: None,
                utilization_gpu_pct: None,
                temperature_c: None,
                supports_fp16: true,
                supports_int8: true,
            }],
        }
    }

    fn provider_flags(devices: &[Device]) -> (bool, bool, bool, bool) {
        let has_cuda = devices
            .iter()
            .any(|d| matches!(d.backend, DeviceBackend::Cuda));
        let has_metal = devices
            .iter()
            .any(|d| matches!(d.backend, DeviceBackend::Metal));
        let has_rocm = devices
            .iter()
            .any(|d| matches!(d.backend, DeviceBackend::Rocm));
        let has_directml = devices
            .iter()
            .any(|d| matches!(d.backend, DeviceBackend::DirectML));
        (has_cuda, has_metal, has_rocm, has_directml)
    }

    fn run_command_with_timeout(
        program: &str,
        args: &[&str],
        timeout: Duration,
    ) -> Option<Vec<u8>> {
        let mut child = Command::new(program)
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .ok()?;

        let start = Instant::now();
        loop {
            if start.elapsed() >= timeout {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }

            match child.try_wait().ok()? {
                Some(_) => break,
                None => std::thread::sleep(COMMAND_POLL_INTERVAL),
            }
        }

        let out = child.wait_with_output().ok()?;
        if !out.status.success() {
            return None;
        }
        Some(out.stdout)
    }

    fn parse_cuda_version_from_smi_summary(stdout: &str) -> Option<String> {
        // Example header line:
        // | NVIDIA-SMI 535.54.03 Driver Version: 535.54.03 CUDA Version: 12.2 |
        let needle = "CUDA Version:";
        let idx = stdout.find(needle)?;
        let rest = &stdout[idx + needle.len()..];
        let version = rest
            .trim_start()
            .split(|c: char| c.is_whitespace() || c == '|')
            .next()?
            .trim();
        if version.is_empty() {
            None
        } else {
            Some(version.to_string())
        }
    }

    fn detect_cuda_version(timeout: Duration) -> Option<String> {
        let stdout = Self::run_command_with_timeout("nvidia-smi", &[], timeout)?;
        let text = String::from_utf8_lossy(&stdout);
        Self::parse_cuda_version_from_smi_summary(&text)
    }

    fn parse_compute_capability(value: &str) -> Option<(u32, u32)> {
        let trimmed = value.trim();
        if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("n/a") {
            return None;
        }
        let (major, minor) = trimmed.split_once('.')?;
        Some((major.parse().ok()?, minor.parse().ok()?))
    }

    fn detect_cuda_gpus(timeout: Duration, cuda_version: Option<&str>) -> Vec<Device> {
        let mut devices = Vec::new();

        let query = "index,name,memory.total,utilization.gpu,temperature.gpu,pci.bus_id,driver_version,compute_cap";
        let args = ["--query-gpu", query, "--format=csv,noheader,nounits"];

        let stdout = match Self::run_command_with_timeout("nvidia-smi", &args, timeout) {
            Some(s) => s,
            None => return devices,
        };

        let text = String::from_utf8_lossy(&stdout);
        for line in text.lines() {
            let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            if parts.len() < 7 {
                continue;
            }

            let id = parts[0].parse::<usize>().unwrap_or(0);
            let name = parts.get(1).copied().unwrap_or("NVIDIA GPU").to_string();
            let memory_mb = parts
                .get(2)
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(0);
            let utilization_gpu_pct = parts.get(3).and_then(|v| v.parse::<u32>().ok());
            let temperature_c = parts.get(4).and_then(|v| v.parse::<u32>().ok());

            let pci_bus_id = parts
                .get(5)
                .map(|v| v.trim())
                .filter(|v| !v.is_empty() && !v.eq_ignore_ascii_case("n/a"))
                .map(|v| v.to_string());

            let driver_version = parts
                .get(6)
                .map(|v| v.trim())
                .filter(|v| !v.is_empty() && !v.eq_ignore_ascii_case("n/a"))
                .map(|v| v.to_string());

            let compute_capability = parts
                .get(7)
                .map(|v| v.trim())
                .filter(|v| !v.is_empty() && !v.eq_ignore_ascii_case("n/a"))
                .map(|v| v.to_string());

            let (supports_fp16, supports_int8) = match compute_capability
                .as_deref()
                .and_then(Self::parse_compute_capability)
            {
                Some((major, _minor)) => (major >= 5, major >= 6),
                None => (true, true),
            };

            devices.push(Device {
                id,
                name,
                backend: DeviceBackend::Cuda,
                memory_mb,
                compute_units: 0,
                pci_bus_id,
                driver_version,
                cuda_version: cuda_version.map(|s| s.to_string()),
                compute_capability,
                utilization_gpu_pct,
                temperature_c,
                supports_fp16,
                supports_int8,
            });
        }

        devices
    }

    fn detect_rocm_gpus(timeout: Duration, os_release: &str) -> Vec<Device> {
        let mut devices = Vec::new();

        let stdout = match Self::run_command_with_timeout("rocm-smi", &["-i"], timeout) {
            Some(s) => s,
            None => return devices,
        };

        let text = String::from_utf8_lossy(&stdout);

        // Prefer bracket form: GPU[0]
        let mut ids = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            if let Some(start) = line.find("GPU[") {
                let rest = &line[start + 4..];
                if let Some(end) = rest.find(']') {
                    if let Ok(id) = rest[..end].parse::<usize>() {
                        if !ids.contains(&id) {
                            ids.push(id);
                        }
                    }
                }
            }
        }

        if ids.is_empty() {
            // Fallback: table format where first token is GPU index.
            for line in text.lines() {
                let line = line.trim_start();
                if line.is_empty() {
                    continue;
                }
                let first = line.split_whitespace().next().unwrap_or("");
                if let Ok(id) = first.parse::<usize>() {
                    if !ids.contains(&id) {
                        ids.push(id);
                    }
                }
            }
        }

        ids.sort_unstable();
        for id in ids {
            devices.push(Device {
                id,
                name: format!("AMD ROCm GPU {id}"),
                backend: DeviceBackend::Rocm,
                memory_mb: 0,
                compute_units: 0,
                pci_bus_id: None,
                driver_version: Some(os_release.to_string()),
                cuda_version: None,
                compute_capability: None,
                utilization_gpu_pct: None,
                temperature_c: None,
                supports_fp16: true,
                supports_int8: true,
            });
        }

        devices
    }

    fn parse_memory_mb(value: &str) -> Option<u64> {
        let lowered = value.trim().to_ascii_lowercase();
        if lowered.is_empty() {
            return None;
        }

        let mut number = String::new();
        for ch in lowered.chars() {
            if ch.is_ascii_digit() || ch == '.' {
                number.push(ch);
            } else if !number.is_empty() {
                break;
            }
        }

        let num: f64 = number.parse().ok()?;
        if lowered.contains("gb") {
            Some((num * 1024.0) as u64)
        } else if lowered.contains("mb") {
            Some(num as u64)
        } else if lowered.contains("kb") {
            Some((num / 1024.0) as u64)
        } else {
            None
        }
    }

    fn detect_metal(timeout: Duration, os_release: &str) -> Vec<Device> {
        #[cfg(target_os = "macos")]
        {
            let mut devs = Vec::new();
            let stdout = match Self::run_command_with_timeout(
                "system_profiler",
                &["SPDisplaysDataType", "-json"],
                timeout,
            ) {
                Some(s) => s,
                None => return devs,
            };

            let value: serde_json::Value = match serde_json::from_slice(&stdout) {
                Ok(v) => v,
                Err(_) => return devs,
            };

            let displays = match value.get("SPDisplaysDataType").and_then(|v| v.as_array()) {
                Some(v) => v,
                None => return devs,
            };

            for (idx, item) in displays.iter().enumerate() {
                let name = item
                    .get("spdisplays_chipset_model")
                    .and_then(|v| v.as_str())
                    .or_else(|| item.get("sppci_model").and_then(|v| v.as_str()))
                    .or_else(|| item.get("_name").and_then(|v| v.as_str()))
                    .unwrap_or("Apple Metal GPU")
                    .to_string();

                let vram_text = item
                    .get("spdisplays_vram")
                    .and_then(|v| v.as_str())
                    .or_else(|| item.get("spdisplays_vram_shared").and_then(|v| v.as_str()))
                    .unwrap_or("");

                let memory_mb = Self::parse_memory_mb(vram_text).unwrap_or(0);

                devs.push(Device {
                    id: idx,
                    name,
                    backend: DeviceBackend::Metal,
                    memory_mb,
                    compute_units: 0,
                    pci_bus_id: None,
                    driver_version: Some(os_release.to_string()),
                    cuda_version: None,
                    compute_capability: None,
                    utilization_gpu_pct: None,
                    temperature_c: None,
                    supports_fp16: true,
                    supports_int8: true,
                });
            }

            devs
        }

        #[cfg(not(target_os = "macos"))]
        {
            let _ = (timeout, os_release);
            Vec::new()
        }
    }

    fn detect_directml(_timeout: Duration, os_release: &str) -> Vec<Device> {
        #[cfg(target_os = "windows")]
        {
            // Best-effort: DirectML runs on DX12 adapters; detailed enumeration would require
            // Windows APIs. Provide a placeholder device for feature-gating higher layers.
            vec![Device {
                id: 0,
                name: "DirectML GPU".into(),
                backend: DeviceBackend::DirectML,
                memory_mb: 0,
                compute_units: 0,
                pci_bus_id: None,
                driver_version: Some(os_release.to_string()),
                cuda_version: None,
                compute_capability: None,
                utilization_gpu_pct: None,
                temperature_c: None,
                supports_fp16: true,
                supports_int8: true,
            }]
        }

        #[cfg(not(target_os = "windows"))]
        {
            let _ = os_release;
            Vec::new()
        }
    }

    fn detect_oneapi(_timeout: Duration) -> Vec<Device> {
        // OneAPI/Level-Zero enumeration is intentionally stubbed here to keep `kapsl-hal`
        // dependency-light. Higher layers can treat the backend as available when they
        // can actually create a OneAPI engine.
        Vec::new()
    }

    fn detect_webgpu() -> Vec<Device> {
        // WebGPU is only meaningful for wasm/browser builds.
        Vec::new()
    }

    #[cfg(feature = "nvml")]
    fn detect_cuda_gpus_nvml(cuda_version: Option<&str>) -> Option<Vec<Device>> {
        use nvml_wrapper::Nvml;

        let nvml = Nvml::init().ok()?;
        let driver_version = nvml.sys_driver_version().ok();
        let count = nvml.device_count().ok()?;

        let mut devices = Vec::with_capacity(count as usize);
        for index in 0..count {
            let dev = nvml.device_by_index(index).ok()?;
            let name = dev.name().ok().unwrap_or_else(|| "NVIDIA GPU".to_string());
            let memory_mb = dev
                .memory_info()
                .ok()
                .map(|m| m.total / (1024 * 1024))
                .unwrap_or(0);
            let pci_bus_id = dev
                .pci_info()
                .ok()
                .map(|p| p.bus_id)
                .filter(|s| !s.trim().is_empty());
            let utilization_gpu_pct = dev.utilization_rates().ok().map(|u| u.gpu);
            let temperature_c = dev
                .temperature(nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu)
                .ok();
            let cc = dev
                .cuda_compute_capability()
                .ok()
                .map(|(maj, min)| format!("{maj}.{min}"));

            let (supports_fp16, supports_int8) =
                match cc.as_deref().and_then(Self::parse_compute_capability) {
                    Some((major, _minor)) => (major >= 5, major >= 6),
                    None => (true, true),
                };

            devices.push(Device {
                id: index as usize,
                name,
                backend: DeviceBackend::Cuda,
                memory_mb,
                compute_units: 0,
                pci_bus_id,
                driver_version: driver_version.clone(),
                cuda_version: cuda_version.map(|s| s.to_string()),
                compute_capability: cc,
                utilization_gpu_pct,
                temperature_c,
                supports_fp16,
                supports_int8,
            });
        }

        Some(devices)
    }

    #[cfg(not(feature = "nvml"))]
    fn detect_cuda_gpus_nvml(_cuda_version: Option<&str>) -> Option<Vec<Device>> {
        None
    }

    /// Select the "best" GPU using a simple heuristic.
    ///
    /// Primary key: `memory_mb`.
    /// Tie-breaker: `compute_capability` when present.
    pub fn best_gpu(&self) -> Option<&Device> {
        self.devices
            .iter()
            .filter(|d| !matches!(d.backend, DeviceBackend::Cpu))
            .max_by(|a, b| {
                let by_mem = a.memory_mb.cmp(&b.memory_mb);
                if by_mem != std::cmp::Ordering::Equal {
                    return by_mem;
                }

                let a_cc = a
                    .compute_capability
                    .as_deref()
                    .and_then(Self::parse_compute_capability)
                    .unwrap_or((0, 0));
                let b_cc = b
                    .compute_capability
                    .as_deref()
                    .and_then(Self::parse_compute_capability)
                    .unwrap_or((0, 0));
                a_cc.cmp(&b_cc)
            })
    }

    pub fn best_gpu_with_preference(&self, preference: &GpuPreference) -> Option<&Device> {
        match preference {
            GpuPreference::BackendId { backend, id } => self.devices.iter().find(|d| {
                !matches!(d.backend, DeviceBackend::Cpu)
                    && d.backend.to_string().eq_ignore_ascii_case(backend)
                    && d.id == *id
            }),
            GpuPreference::PciBusId(bus_id) => self.devices.iter().find(|d| {
                !matches!(d.backend, DeviceBackend::Cpu)
                    && d.pci_bus_id
                        .as_deref()
                        .is_some_and(|v| v.eq_ignore_ascii_case(bus_id))
            }),
            GpuPreference::NameContains(needle) => {
                let needle = needle.to_ascii_lowercase();
                self.devices.iter().find(|d| {
                    !matches!(d.backend, DeviceBackend::Cpu)
                        && d.name.to_ascii_lowercase().contains(&needle)
                })
            }
        }
    }

    pub fn cuda_devices(&self) -> Vec<&Device> {
        self.devices
            .iter()
            .filter(|d| matches!(d.backend, DeviceBackend::Cuda))
            .collect()
    }

    /// Get the best available execution provider.
    pub fn get_best_provider(&self) -> String {
        if self.has_cuda {
            "cuda".to_string()
        } else if self.has_metal {
            "metal".to_string()
        } else if self.has_rocm {
            "rocm".to_string()
        } else if self
            .devices
            .iter()
            .any(|d| matches!(d.backend, DeviceBackend::OneApi))
        {
            "oneapi".to_string()
        } else if self.has_directml {
            "directml".to_string()
        } else {
            "cpu".to_string()
        }
    }

    /// Check if a specific provider is available.
    pub fn has_provider(&self, provider: &str) -> bool {
        let key = provider.trim().to_ascii_lowercase();
        match key.as_str() {
            "cuda" => self.has_cuda,
            "metal" | "coreml" => self.has_metal,
            "rocm" => self.has_rocm,
            "directml" => self.has_directml,
            "oneapi" => self
                .devices
                .iter()
                .any(|d| matches!(d.backend, DeviceBackend::OneApi)),
            "webgpu" => self
                .devices
                .iter()
                .any(|d| matches!(d.backend, DeviceBackend::WebGpu)),
            "cpu" => true,
            other => self
                .devices
                .iter()
                .any(|d| d.backend.to_string().eq_ignore_ascii_case(other)),
        }
    }
}
