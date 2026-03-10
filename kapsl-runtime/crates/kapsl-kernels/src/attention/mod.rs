use kapsl_hal::kernel::{AttentionConfig, AttentionKernel, KernelError};
use kapsl_hal::tensor::{TensorView, TensorViewMut};

#[derive(Debug)]
pub struct CpuAttention;

impl AttentionKernel for CpuAttention {
    fn flash_attention_v2(
        &self,
        query: &TensorView<f32>,
        key: &TensorView<f32>,
        value: &TensorView<f32>,
        output: &mut TensorViewMut<f32>,
        config: AttentionConfig,
    ) -> Result<(), KernelError> {
        let dims = AttentionDims::from_views(query, key, value, output)?;
        let scale = config.scale_for(dims.head_dim);

        for b in 0..dims.batch {
            for h in 0..dims.heads {
                let q_base = dims.q_base(b, h);
                let k_base = dims.k_base(b, h);
                let v_base = dims.v_base(b, h);
                let out_base = dims.out_base(b, h);

                let mut out_acc = vec![0.0f32; dims.head_dim];
                for q_pos in 0..dims.q_len {
                    let q_offset = q_base + q_pos * dims.head_dim;
                    let max_k = if config.causal {
                        let offset = dims.k_len.saturating_sub(dims.q_len);
                        let limit = offset + q_pos + 1;
                        dims.k_len.min(limit)
                    } else {
                        dims.k_len
                    };

                    if max_k == 0 {
                        let out_offset = out_base + q_pos * dims.head_dim;
                        output.data[out_offset..out_offset + dims.head_dim].fill(0.0);
                        continue;
                    }

                    let mut max_logit = f32::NEG_INFINITY;
                    for k_pos in 0..max_k {
                        let k_offset = k_base + k_pos * dims.head_dim;
                        let mut dot = 0.0f32;
                        for d in 0..dims.head_dim {
                            dot += query.data[q_offset + d] * key.data[k_offset + d];
                        }
                        let logit = dot * scale;
                        if logit > max_logit {
                            max_logit = logit;
                        }
                    }

                    out_acc.fill(0.0);
                    let mut sum = 0.0f32;
                    for k_pos in 0..max_k {
                        let k_offset = k_base + k_pos * dims.head_dim;
                        let v_offset = v_base + k_pos * dims.head_dim;
                        let mut dot = 0.0f32;
                        for d in 0..dims.head_dim {
                            dot += query.data[q_offset + d] * key.data[k_offset + d];
                        }
                        let weight = (dot * scale - max_logit).exp();
                        sum += weight;
                        let v_slice = &value.data[v_offset..v_offset + dims.head_dim];
                        for (acc, v) in out_acc.iter_mut().zip(v_slice.iter()) {
                            *acc += weight * *v;
                        }
                    }

                    let out_offset = out_base + q_pos * dims.head_dim;
                    if sum > 0.0 {
                        let inv = 1.0 / sum;
                        let out_slice = &mut output.data[out_offset..out_offset + dims.head_dim];
                        for (out, acc) in out_slice.iter_mut().zip(out_acc.iter()) {
                            *out = *acc * inv;
                        }
                    } else {
                        output.data[out_offset..out_offset + dims.head_dim].fill(0.0);
                    }
                }
            }
        }

        Ok(())
    }

    fn paged_attention_v1(
        &self,
        query: &TensorView<f32>,
        key: &TensorView<f32>,
        value: &TensorView<f32>,
        output: &mut TensorViewMut<f32>,
        config: AttentionConfig,
    ) -> Result<(), KernelError> {
        self.flash_attention_v2(query, key, value, output, config)
    }
}

// Future: CudaAttention, RocmAttention implementations

struct AttentionDims {
    batch: usize,
    heads: usize,
    q_len: usize,
    k_len: usize,
    head_dim: usize,
    q_stride: usize,
    k_stride: usize,
    v_stride: usize,
    out_stride: usize,
}

impl AttentionDims {
    fn from_views(
        query: &TensorView<f32>,
        key: &TensorView<f32>,
        value: &TensorView<f32>,
        output: &TensorViewMut<f32>,
    ) -> Result<Self, KernelError> {
        let (batch, heads, q_len, head_dim) = parse_shape(query, "query")?;
        let (k_batch, k_heads, k_len, k_head_dim) = parse_shape(key, "key")?;
        let (v_batch, v_heads, v_len, v_head_dim) = parse_shape(value, "value")?;
        let (o_batch, o_heads, o_len, o_head_dim) = parse_shape_mut(output, "output")?;

        if batch != k_batch || batch != v_batch || batch != o_batch {
            return Err(KernelError::InvalidInput(
                "Mismatched batch size for attention tensors".to_string(),
            ));
        }
        if heads != k_heads || heads != v_heads || heads != o_heads {
            return Err(KernelError::InvalidInput(
                "Mismatched head count for attention tensors".to_string(),
            ));
        }
        if head_dim != k_head_dim || head_dim != v_head_dim || head_dim != o_head_dim {
            return Err(KernelError::InvalidInput(
                "Mismatched head_dim for attention tensors".to_string(),
            ));
        }
        if q_len != o_len {
            return Err(KernelError::InvalidInput(
                "Output sequence length must match query length".to_string(),
            ));
        }
        if k_len != v_len {
            return Err(KernelError::InvalidInput(
                "Key/value sequence lengths must match".to_string(),
            ));
        }

        let q_stride = q_len * head_dim;
        let k_stride = k_len * head_dim;
        let v_stride = v_len * head_dim;
        let out_stride = o_len * head_dim;

        Ok(Self {
            batch,
            heads,
            q_len,
            k_len,
            head_dim,
            q_stride,
            k_stride,
            v_stride,
            out_stride,
        })
    }

    #[inline]
    fn q_base(&self, batch: usize, head: usize) -> usize {
        (batch * self.heads + head) * self.q_stride
    }

    #[inline]
    fn k_base(&self, batch: usize, head: usize) -> usize {
        (batch * self.heads + head) * self.k_stride
    }

    #[inline]
    fn v_base(&self, batch: usize, head: usize) -> usize {
        (batch * self.heads + head) * self.v_stride
    }

    #[inline]
    fn out_base(&self, batch: usize, head: usize) -> usize {
        (batch * self.heads + head) * self.out_stride
    }
}

fn parse_shape<T>(
    view: &TensorView<T>,
    label: &str,
) -> Result<(usize, usize, usize, usize), KernelError> {
    parse_shape_impl(view.data.len(), view.shape, label)
}

fn parse_shape_mut<T>(
    view: &TensorViewMut<T>,
    label: &str,
) -> Result<(usize, usize, usize, usize), KernelError> {
    parse_shape_impl(view.data.len(), view.shape, label)
}

fn parse_shape_impl(
    data_len: usize,
    shape: &[usize],
    label: &str,
) -> Result<(usize, usize, usize, usize), KernelError> {
    if shape.len() != 4 {
        return Err(KernelError::InvalidInput(format!(
            "{} shape must be rank-4 [batch, heads, seq_len, head_dim], got {:?}",
            label, shape
        )));
    }
    let dims = (shape[0], shape[1], shape[2], shape[3]);
    let expected = dims.0 * dims.1 * dims.2 * dims.3;
    if data_len != expected {
        return Err(KernelError::InvalidInput(format!(
            "{} data length {} does not match shape product {}",
            label, data_len, expected
        )));
    }
    if dims.3 == 0 || dims.1 == 0 || dims.2 == 0 || dims.0 == 0 {
        return Err(KernelError::InvalidInput(format!(
            "{} shape contains zero dimension: {:?}",
            label, shape
        )));
    }
    Ok(dims)
}
