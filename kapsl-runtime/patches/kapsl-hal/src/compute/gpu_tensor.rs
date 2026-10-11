//! GPU tensor types wrapping cudarc allocations.
//!
//! These are thin wrappers around `CudaSlice<T>` that carry shape metadata,
//! used by CUDA kernels and cuBLAS calls in the native backend.

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice};
#[cfg(feature = "cuda")]
use std::sync::Arc;

/// An owned tensor living on a CUDA device.
#[cfg(feature = "cuda")]
pub struct GpuTensor<T: cudarc::driver::DeviceRepr> {
    pub data: CudaSlice<T>,
    pub shape: Vec<usize>,
}

#[cfg(feature = "cuda")]
impl<T: cudarc::driver::DeviceRepr> GpuTensor<T> {
    /// Allocate a zeroed tensor on `device`.
    pub fn zeros(
        device: &Arc<CudaDevice>,
        shape: &[usize],
    ) -> Result<Self, cudarc::driver::DriverError>
    where
        T: cudarc::driver::ValidAsZeroBits,
    {
        let numel: usize = shape.iter().product();
        let data = device.alloc_zeros::<T>(numel)?;
        Ok(Self {
            data,
            shape: shape.to_vec(),
        })
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

#[cfg(feature = "cuda")]
impl<T: cudarc::driver::DeviceRepr> std::fmt::Debug for GpuTensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GpuTensor<{}>({:?})",
            std::any::type_name::<T>(),
            self.shape
        )
    }
}
