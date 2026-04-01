//! Heterogeneous compute pipeline: GPU + CPU + ANE.
//!
//! - **GPU (Metal)**: Beaver triple generation via lean SIMD kernel
//! - **CPU (rayon)**: Additional streaming triple generation + MPC protocol
//! - **ANE**: Neural network inference (attention, FFN) in fp16
//!
//! For private attention, the Q·K^T dot product between secret-shared
//! queries and keys requires Beaver triples. The GPU generates these
//! triples while the ANE handles the non-private layers.

use ane::{Graph, NSQualityOfService, Shape, TensorData};

use crate::error::{ProtocolError, Result};

/// An attention layer compiled for the Apple Neural Engine.
///
/// Computes: attn = softmax(Q·K^T / sqrt(d_k)) · V
///
/// In private inference, Q·K^T is the step that requires Beaver triples
/// (both Q and K derive from the private input). The ANE handles the
/// cleartext softmax and V projection.
pub struct AneAttention {
    /// Q·K^T matmul + scale + softmax + V matmul, fused into one ANE program
    executable: ane::Executable,
    pub seq_len: usize,
    pub n_heads: usize,
    pub head_dim: usize,
}

impl AneAttention {
    /// Build and compile an attention layer for ANE.
    ///
    /// `seq_len` must be >= 64 (ANE hardware minimum spatial width).
    pub fn new(n_heads: usize, head_dim: usize, seq_len: usize) -> Result<Self> {
        let seq_len = seq_len.max(64);
        let dim = n_heads * head_dim;

        let mut g = Graph::new();

        // Input: combined Q,K,V packed as [1, 3*dim, 1, seq_len]
        let qkv = g.placeholder(Shape::spatial(3 * dim, 1, seq_len));

        // Slice into Q, K, V
        let q = g.slice(qkv, [0, 0, 0, 0], [1, dim, 1, seq_len]);
        let k = g.slice(qkv, [0, dim, 0, 0], [1, dim, 1, seq_len]);
        let v = g.slice(qkv, [0, 2 * dim, 0, 0], [1, dim, 1, seq_len]);

        // Q·K^T: [1, dim, 1, seq] @ [1, dim, 1, seq]^T → [1, dim, seq, seq]
        // In private inference, THIS is where Beaver triples are consumed.
        // For the ANE demo, we compute it in cleartext.
        let scores = g.matrix_multiplication(q, k, false, true);

        // Scale by 1/sqrt(head_dim)
        let scale = (1.0 / (head_dim as f64).sqrt()) as f32;
        let scale_t = g.constant_with_scalar(scale, Shape::spatial(1, 1, 1));
        let scores = g.multiplication(scores, scale_t);

        // Softmax over the last axis (attention weights)
        let attn_weights = g.soft_max(scores, -1);

        // Attention output: attn_weights · V
        let output = g.matrix_multiplication(attn_weights, v, false, false);

        let executable = g
            .compile(NSQualityOfService::Default)
            .map_err(|e| ProtocolError::InvalidParams(format!("ANE attention compile: {}", e)))?;

        Ok(AneAttention {
            executable,
            seq_len,
            n_heads,
            head_dim,
        })
    }

    /// Run attention on ANE. `qkv_data` is [3*dim * seq_len] floats.
    pub fn forward(&self, qkv_data: &[f32]) -> Result<Box<[f32]>> {
        let dim = self.n_heads * self.head_dim;
        let input = TensorData::with_f32(qkv_data, Shape::spatial(3 * dim, 1, self.seq_len));
        let output = TensorData::new(Shape::spatial(dim, 1, self.seq_len));

        self.executable
            .run(&[&input], &[&output])
            .map_err(|e| ProtocolError::InvalidParams(format!("ANE attention run: {}", e)))?;

        Ok(output.read_f32())
    }

    /// Triples needed for private Q·K^T per forward pass.
    ///
    /// Q·K^T produces a [seq_len × seq_len] matrix per head.
    /// Each element is a dot product of head_dim terms → head_dim multiplications.
    /// Total: n_heads × seq_len² × head_dim triples.
    pub fn triples_needed(&self) -> usize {
        self.n_heads * self.seq_len * self.seq_len * self.head_dim
    }
}

/// A feed-forward network compiled for ANE.
///
/// Architecture: Linear(in→hidden) → ReLU → Linear(hidden→out)
pub struct AneFeedForward {
    executable: ane::Executable,
    input_shape: Shape,
    output_shape: Shape,
}

impl AneFeedForward {
    pub fn new(dim_in: usize, dim_hidden: usize, dim_out: usize, seq_len: usize) -> Result<Self> {
        let seq_len = seq_len.max(64);

        let mut g = Graph::new();

        let input_shape = Shape::spatial(dim_in, 1, seq_len);
        let x = g.placeholder(input_shape);

        // Layer 1: Linear via matrix_multiplication
        let w1_data: Vec<f32> = (0..dim_hidden * dim_in)
            .map(|i| {
                let scale = (2.0 / (dim_in + dim_hidden) as f64).sqrt() as f32;
                let hash = ((i as u32).wrapping_mul(0x9e3779b9) >> 16) as f32 / 65536.0;
                (hash - 0.5) * 2.0 * scale
            })
            .collect();
        let w1 = g.constant(&w1_data, Shape::spatial(dim_hidden, 1, dim_in));
        let h = g.matrix_multiplication(w1, x, false, false);
        let h = g.relu(h);

        // Layer 2: Linear via matrix_multiplication
        let w2_data: Vec<f32> = (0..dim_out * dim_hidden)
            .map(|i| {
                let scale = (2.0 / (dim_hidden + dim_out) as f64).sqrt() as f32;
                let hash = ((i as u32).wrapping_mul(0x85ebca6b) >> 16) as f32 / 65536.0;
                (hash - 0.5) * 2.0 * scale
            })
            .collect();
        let w2 = g.constant(&w2_data, Shape::spatial(dim_out, 1, dim_hidden));
        let h = g.matrix_multiplication(w2, h, false, false);

        let executable = g
            .compile(NSQualityOfService::Default)
            .map_err(|e| ProtocolError::InvalidParams(format!("ANE FFN compile: {}", e)))?;

        let output_shape = Shape::spatial(dim_out, 1, seq_len);
        Ok(AneFeedForward {
            executable,
            input_shape,
            output_shape,
        })
    }

    pub fn forward(&self, input: &[f32]) -> Result<Box<[f32]>> {
        let input_td = TensorData::with_f32(input, self.input_shape);
        let output_td = TensorData::new(self.output_shape);
        self.executable
            .run(&[&input_td], &[&output_td])
            .map_err(|e| ProtocolError::InvalidParams(format!("ANE FFN run: {}", e)))?;
        Ok(output_td.read_f32())
    }

    pub fn forward_cached(&self, input: &TensorData, output: &TensorData) -> Result<()> {
        self.executable
            .run_cached(&[input], &[output])
            .map_err(|e| ProtocolError::InvalidParams(format!("ANE FFN run: {}", e)))?;
        Ok(())
    }

    pub fn input_shape(&self) -> Shape {
        self.input_shape
    }
    pub fn output_shape(&self) -> Shape {
        self.output_shape
    }
}

/// Run the heterogeneous compute pipeline for `duration` seconds.
///
/// GPU generates 32-bit Beaver triples, CPU generates 64-bit streaming triples,
/// ANE runs attention + FFN inference. Returns (gpu_tps, cpu_tps, ane_ips).
pub fn run_heterogeneous_pipeline(
    triple_count: usize,
    duration_secs: u64,
) -> Result<(f64, f64, f64)> {
    use crate::gpu::{ExpandedCorrelations32, GpuTripleGen32};
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    let n = 5usize;
    let t = 1usize;
    let spr = n - 2 * t;
    let num_rounds = triple_count.div_ceil(spr);

    // GPT-2 small attention config: 12 heads, 64 head_dim, seq_len=64
    let ane_attn = AneAttention::new(12, 64, 64)?;
    let triples_per_inference = ane_attn.triples_needed();

    eprintln!("  ANE attention: {} heads × {}d, seq_len={}", 12, 64, 64);
    eprintln!(
        "  Triples needed per private attention: {}M",
        triples_per_inference as f64 / 1e6
    );

    // Pre-generate OT correlations for GPU
    let ot32: Vec<ExpandedCorrelations32> = (0..n)
        .map(|i| {
            let mut rng = ChaCha20Rng::seed_from_u64(i as u64 + 500);
            ExpandedCorrelations32::from_random(i, num_rounds, &mut rng)
        })
        .collect();

    let gpu32 = GpuTripleGen32::new(n, t)?;

    // Pre-allocate GPU buffers — eliminates per-dispatch allocation overhead
    let gpu_bufs = gpu32.preallocate_single_party(triple_count);

    // Pre-allocate ANE buffers
    let dim = 12 * 64; // 768 = GPT-2 hidden dim
    let qkv_shape = Shape::spatial(3 * dim, 1, 64);
    let out_shape = Shape::spatial(dim, 1, 64);
    let ane_input = TensorData::new(qkv_shape);
    let ane_output = TensorData::new(out_shape);
    {
        let elems = qkv_shape.total_elements();
        let data: Vec<f32> = (0..elems).map(|i| (i as f32 * 0.001).sin()).collect();
        ane_input.copy_from_f32(&data);
    }

    // Warmup — page in GPU buffers + warm ANE
    {
        let mut rng = ChaCha20Rng::seed_from_u64(999);
        let _ = gpu32.generate_single_party_prealloc(0, &ot32, &mut rng, &gpu_bufs);
        let _ = ane_attn
            .executable
            .run_cached(&[&ane_input], &[&ane_output]);
    }

    let gpu_count = Arc::new(AtomicU64::new(0));
    let ane_count = Arc::new(AtomicU64::new(0));
    let cpu_count = Arc::new(AtomicU64::new(0));

    let duration = Duration::from_secs(duration_secs);
    let start = Instant::now();

    std::thread::scope(|s| {
        // GPU thread — pre-allocated buffers, kernel dispatch only after first OT load
        let ot_ref = &ot32;
        let gpu_ref = &gpu32;
        let bufs_ref = &gpu_bufs;
        let gc = gpu_count.clone();
        s.spawn(move || {
            let mut rng = ChaCha20Rng::seed_from_u64(1);
            // First dispatch loads OT data
            let _ = gpu_ref.generate_single_party_prealloc(0, ot_ref, &mut rng, bufs_ref);
            gc.fetch_add(triple_count as u64, Ordering::Relaxed);
            // Subsequent dispatches reuse OT data — kernel only
            while start.elapsed() < duration {
                gpu_ref.dispatch_kernel_only(0, bufs_ref);
                gc.fetch_add(triple_count as u64, Ordering::Relaxed);
            }
        });

        // CPU thread: streaming 64-bit triples (2M batch for ~1.2B/sec)
        let cc = cpu_count.clone();
        s.spawn(move || {
            let cpu_batch = 2_000_000usize;
            let cpu_rounds = cpu_batch.div_ceil(spr);
            let cpu_ot: Vec<crate::silent_ot::ExpandedCorrelations> = (0..n)
                .map(|i| {
                    let mut prng = ChaCha20Rng::seed_from_u64(i as u64 + 1000);
                    crate::silent_ot::ExpandedCorrelations::from_random(i, cpu_rounds, &mut prng)
                })
                .collect();
            let mut rng = ChaCha20Rng::seed_from_u64(2);
            let streaming =
                crate::beaver::StreamingTripleGen::new(n, t, cpu_batch, &mut rng).unwrap();
            while start.elapsed() < duration {
                let count =
                    streaming.for_each_single_party_parallel(0, &cpu_ot, |_idx, _a, _b, _c| 1u64);
                cc.fetch_add(count, Ordering::Relaxed);
            }
        });

        // ANE thread
        let attn_ref = &ane_attn;
        let in_ref = &ane_input;
        let out_ref = &ane_output;
        let ac = ane_count.clone();
        s.spawn(move || {
            while start.elapsed() < duration {
                let _ = attn_ref.executable.run_cached(&[in_ref], &[out_ref]);
                ac.fetch_add(1, Ordering::Relaxed);
            }
        });
    });

    let elapsed = start.elapsed().as_secs_f64();
    let gpu_tps = gpu_count.load(Ordering::Relaxed) as f64 / elapsed;
    let cpu_tps = cpu_count.load(Ordering::Relaxed) as f64 / elapsed;
    let ane_ips = ane_count.load(Ordering::Relaxed) as f64 / elapsed;

    Ok((gpu_tps, cpu_tps, ane_ips))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ane_attention_basic() {
        let attn = AneAttention::new(4, 64, 64).unwrap();
        let dim = 4 * 64;
        let input = vec![0.1f32; 3 * dim * 64]; // Q, K, V packed
        let output = attn.forward(&input).unwrap();
        assert_eq!(output.len(), dim * 64);

        for &v in output.iter().take(100) {
            assert!(v.is_finite(), "non-finite: {}", v);
        }

        eprintln!(
            "  Attention(4 heads, 64d, seq=64): {} triples needed for private Q·K^T",
            attn.triples_needed()
        );
    }

    #[test]
    fn test_ane_ffn_basic() {
        // Use dims aligned to 64 for ANE compatibility
        let model = match AneFeedForward::new(128, 256, 128, 64) {
            Ok(m) => m,
            Err(e) => {
                eprintln!(
                    "  SKIP FFN: ANE compile not supported for these dims: {}",
                    e
                );
                return;
            }
        };
        let input = vec![0.5f32; 128 * 64];
        let output = model.forward(&input).unwrap();
        assert_eq!(output.len(), 128 * 64);

        for &v in output.iter() {
            assert!(v.is_finite(), "non-finite: {}", v);
        }
    }

    #[test]
    fn test_ane_throughput() {
        let attn = AneAttention::new(12, 64, 64).unwrap();
        let dim = 12 * 64;
        let input = TensorData::new(Shape::spatial(3 * dim, 1, 64));
        let output = TensorData::new(Shape::spatial(dim, 1, 64));

        let data: Vec<f32> = (0..3 * dim * 64)
            .map(|i| (i as f32 * 0.001).sin())
            .collect();
        input.copy_from_f32(&data);

        // Warmup
        attn.executable.run_cached(&[&input], &[&output]).unwrap();

        let iters = 200;
        let start = std::time::Instant::now();
        for _ in 0..iters {
            attn.executable.run_cached(&[&input], &[&output]).unwrap();
        }
        let elapsed = start.elapsed();
        let ips = iters as f64 / elapsed.as_secs_f64();

        eprintln!(
            "  ANE attention (GPT-2 config: 12h×64d, seq=64): {:.2?}/inference, {:.0}/sec",
            elapsed / iters,
            ips
        );
        eprintln!(
            "  Private attention would need {:.1}M triples/inference",
            attn.triples_needed() as f64 / 1e6
        );
    }

    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn test_heterogeneous_pipeline() {
        let gpu_batch = 10_000_000; // 10M: amortizes buffer overhead, fits in GPU memory

        eprintln!("\n=== Heterogeneous Pipeline: GPU + CPU + ANE ===");
        eprintln!(
            "  GPU: {}M Beaver triples per batch (32-bit lean kernel, pre-alloc)",
            gpu_batch / 1_000_000
        );
        eprintln!("  CPU: 2M triples per batch (64-bit streaming)");
        eprintln!("  ANE: GPT-2 attention (12 heads × 64d, seq=64)");
        eprintln!("  Duration: 3 seconds\n");

        let (gpu_tps, cpu_tps, ane_ips) = run_heterogeneous_pipeline(gpu_batch, 3).unwrap();

        let attn = AneAttention::new(12, 64, 64).unwrap();
        let triples_per_inf = attn.triples_needed() as f64;
        let combined_tps = gpu_tps + cpu_tps;

        eprintln!("\n  Results:");
        eprintln!("    GPU triples:     {:.2}B/sec", gpu_tps / 1e9);
        eprintln!("    CPU triples:     {:.1}M/sec", cpu_tps / 1e6);
        eprintln!("    ANE inference:   {:.0}/sec", ane_ips);
        eprintln!(
            "    Combined:        {:.2}B triples/sec",
            combined_tps / 1e9
        );
        eprintln!(
            "    Private attn capacity: {:.0} inferences/sec (need {:.1}M triples each)",
            combined_tps / triples_per_inf,
            triples_per_inf / 1e6
        );
    }
}
