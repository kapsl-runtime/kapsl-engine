use kapsl_hal::{
    kernel::{KernelError, MlpKernel},
    tensor::{TensorView, TensorViewMut},
};

#[derive(Debug)]
pub struct CpuMlp;
pub struct CudaMlp;
pub struct RocmMlp;

use rayon::prelude::*;

impl MlpKernel for CpuMlp {
    fn fused_swiglu(
        &self,
        gate: &TensorView<f32>,
        up: &TensorView<f32>,
        out: &mut TensorViewMut<f32>,
    ) -> Result<(), KernelError> {
        // Validation
        if gate.shape != up.shape || gate.shape != out.shape {
            return Err(KernelError::InvalidInput(
                "Shape mismatch for fused_swiglu".to_string(),
            ));
        }

        let n = gate.data.len();
        if n != up.data.len() || n != out.data.len() {
            return Err(KernelError::InvalidInput(
                "Length mismatch for fused_swiglu".to_string(),
            ));
        }

        // SwiGLU = (x * sigmoid(x)) * y
        // where x = gate, y = up
        // Parallel execution using Rayon
        out.data
            .par_iter_mut()
            .zip(gate.data.par_iter())
            .zip(up.data.par_iter())
            .for_each(|((o, g), u)| {
                let x = *g;
                let y = *u;
                let sigmoid_x = 1.0 / (1.0 + (-x).exp());
                *o = (x * sigmoid_x) * y;
            });

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_swiglu_cpu() {
        let gate_data = vec![1.0, 0.0, -1.0];
        let up_data = vec![2.0, 3.0, 4.0];
        let mut out_data = vec![0.0; 3];
        let shape = vec![3];

        let gate = TensorView::new(&gate_data, &shape);
        let up = TensorView::new(&up_data, &shape);
        let mut out = TensorViewMut::new(&mut out_data, &shape);

        let mlp = CpuMlp;
        mlp.fused_swiglu(&gate, &up, &mut out).unwrap();

        // Expected values:
        // x=1.0: 1.0 * sig(1.0) * 2.0 = 1.0 * 0.73105 * 2.0 = 1.4621
        // x=0.0: 0.0 * sig(0.0) * 3.0 = 0.0
        // x=-1.0: -1.0 * sig(-1.0) * 4.0 = -1.0 * 0.26894 * 4.0 = -1.07576

        let expected = [1.4621172, 0.0, -1.0757664];

        for (a, b) in out_data.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "Expected {}, got {}", b, a);
        }
    }
}

// Future: CudaMlp, RocmMlp implementations
