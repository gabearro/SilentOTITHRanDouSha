//! Cleartext GPT-OSS-20B inference session.
//!
//! ANE: QKV projection, output projection, router, final norm, LM head, residual adds
//! CPU: Attention (Q·K^T, softmax, attn·V), expert FFN (NEON matmul), MoE top-k

use ane::{Shape, TensorData};
use rayon::prelude::*;

use super::config::LayerType;
use super::executables::DECODE_SPATIAL_WIDTH;
use super::kv_cache::{KvCache, TurboQuantConfig};
use super::model::OssModel;
use super::rope::RopeTable;
use super::weights;

/// NEON-accelerated matrix-vector multiply: y = A @ x
/// A is fp16 [rows, cols] row-major, x is f32 [cols], y is f32 [rows].
/// Uses FCVTL for fp16→f32 conversion, FMA for dot product.
/// cols must be a multiple of 16 (hs=2880, inter=2880: both ÷16 = 0).
#[cfg(target_arch = "aarch64")]
fn matvec_f16xf32(a_f16: &[u16], x: &[f32], y: &mut [f32], rows: usize, cols: usize) {
    use std::arch::aarch64::*;
    unsafe {
        let x_ptr = x.as_ptr();
        for r in 0..rows {
            let a_ptr = a_f16.as_ptr().add(r * cols);
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let mut acc2 = vdupq_n_f32(0.0);
            let mut acc3 = vdupq_n_f32(0.0);

            let mut k = 0usize;
            while k + 16 <= cols {
                let w0: float16x8_t = std::mem::transmute(vld1q_u16(a_ptr.add(k)));
                let w1: float16x8_t = std::mem::transmute(vld1q_u16(a_ptr.add(k + 8)));

                acc0 = vfmaq_f32(
                    acc0,
                    vld1q_f32(x_ptr.add(k)),
                    vcvt_f32_f16(vget_low_f16(w0)),
                );
                acc1 = vfmaq_f32(
                    acc1,
                    vld1q_f32(x_ptr.add(k + 4)),
                    vcvt_f32_f16(vget_high_f16(w0)),
                );
                acc2 = vfmaq_f32(
                    acc2,
                    vld1q_f32(x_ptr.add(k + 8)),
                    vcvt_f32_f16(vget_low_f16(w1)),
                );
                acc3 = vfmaq_f32(
                    acc3,
                    vld1q_f32(x_ptr.add(k + 12)),
                    vcvt_f32_f16(vget_high_f16(w1)),
                );

                k += 16;
            }
            while k + 8 <= cols {
                let w: float16x8_t = std::mem::transmute(vld1q_u16(a_ptr.add(k)));
                acc0 = vfmaq_f32(acc0, vld1q_f32(x_ptr.add(k)), vcvt_f32_f16(vget_low_f16(w)));
                acc1 = vfmaq_f32(
                    acc1,
                    vld1q_f32(x_ptr.add(k + 4)),
                    vcvt_f32_f16(vget_high_f16(w)),
                );
                k += 8;
            }
            y[r] = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn matvec_f16xf32(a_f16: &[u16], x: &[f32], y: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let a_row = &a_f16[r * cols..(r + 1) * cols];
        let mut acc = 0.0f32;
        for k in 0..cols {
            acc += x[k] * half::f16::from_bits(a_row[k]).to_f32();
        }
        y[r] = acc;
    }
}

/// CPU expert SwiGLU FFN. Operates directly on fp16 weights in memory.
/// Eliminates all IOSurface overhead from the ANE path.
fn cpu_expert_ffn(
    input: &[f32],           // [hs]
    gu_w: &[u16],            // [2*inter, hs] fp16 row-major (gate then up, reordered)
    gu_bias: &[u16],         // [2*inter] fp16
    d_w: &[u16],             // [hs, inter] fp16 row-major
    d_bias: &[u16],          // [hs] fp16
    output: &mut [f32],      // [hs]
    gate_up_buf: &mut [f32], // [2*inter] scratch
    gated_buf: &mut [f32],   // [inter] scratch
    hs: usize,
    inter: usize,
) {
    let limit: f32 = 7.0;
    let alpha: f32 = 1.702;

    // 1. gate_up = input @ gu_w^T + bias → [2*inter]
    matvec_f16xf32(gu_w, input, gate_up_buf, 2 * inter, hs);
    for j in 0..2 * inter {
        gate_up_buf[j] += half::f16::from_bits(gu_bias[j]).to_f32();
    }

    // 2. Split + clamp + SwiGLU: glu = gate * sigmoid(gate * alpha), out = (up+1) * glu
    for j in 0..inter {
        let gate = gate_up_buf[j].min(limit);
        let up = gate_up_buf[inter + j].clamp(-limit, limit);
        let glu = gate / (1.0 + (-gate * alpha).exp());
        gated_buf[j] = (up + 1.0) * glu;
    }

    // 3. output = gated @ d_w^T + bias → [hs]
    matvec_f16xf32(d_w, gated_buf, output, hs, inter);
    for j in 0..hs {
        let v = output[j] + half::f16::from_bits(d_bias[j]).to_f32();
        output[j] = v.clamp(-65504.0, 65504.0);
    }
}

/// Default cached experts per layer.
/// Can be overridden via `GPT_OSS_EXPERT_CACHE_SLOTS`.
const DEFAULT_EXPERT_CACHE_SLOTS: usize = 4;

fn parse_expert_cache_slots(top_k: usize, num_experts: usize) -> usize {
    let parsed = std::env::var("GPT_OSS_EXPERT_CACHE_SLOTS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_EXPERT_CACHE_SLOTS);
    parsed.max(top_k).min(num_experts)
}

fn parse_active_experts_per_tok(default_top_k: usize, num_experts: usize) -> usize {
    std::env::var("GPT_OSS_ACTIVE_EXPERTS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default_top_k)
        .min(default_top_k)
        .min(num_experts)
}

fn parse_bool_env(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn parse_f32_env(name: &str, default: f32, min: f32, max: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<f32>().ok())
        .map(|v| v.clamp(min, max))
        .unwrap_or(default)
}

#[derive(Clone, Copy)]
struct AdaptiveMoeTopKConfig {
    min_k: usize,
    top1_min_prob: f32,
    top1_min_margin: f32,
    top2_min_mass: f32,
}

fn parse_adaptive_moe_topk(max_k: usize) -> Option<AdaptiveMoeTopKConfig> {
    if max_k <= 1 || !parse_bool_env("GPT_OSS_ADAPTIVE_MOE_TOPK") {
        return None;
    }
    let min_k = std::env::var("GPT_OSS_ADAPTIVE_MIN_K")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(if max_k >= 4 {
            3
        } else if max_k >= 2 {
            2
        } else {
            1
        })
        .min(max_k);
    let top1_min_prob = parse_f32_env("GPT_OSS_ADAPTIVE_TOP1_MIN", 0.90, 0.0, 1.0);
    let top1_min_margin = parse_f32_env("GPT_OSS_ADAPTIVE_TOP1_MARGIN_MIN", 0.35, 0.0, 1.0);
    let top2_min_mass = parse_f32_env("GPT_OSS_ADAPTIVE_TOP2_MASS_MIN", 0.98, 0.0, 1.0);
    Some(AdaptiveMoeTopKConfig {
        min_k,
        top1_min_prob,
        top1_min_margin,
        top2_min_mass,
    })
}

fn parse_lm_head_eval_retries() -> usize {
    std::env::var("GPT_OSS_LM_HEAD_EVAL_RETRIES")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(2)
}

fn parse_verify_output_proj_router_layer(num_layers: usize) -> Option<usize> {
    if num_layers == 0 {
        return None;
    }
    let enabled = std::env::var("GPT_OSS_VERIFY_OUTPUT_PROJ_ROUTER")
        .ok()
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false);
    if !enabled {
        return None;
    }
    let layer = std::env::var("GPT_OSS_VERIFY_OUTPUT_PROJ_ROUTER_LAYER")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(0);
    Some(layer.min(num_layers - 1))
}

/// Cached dequantized fp16 weights for one expert (heap-allocated).
/// TensorData/IOSurface cache was tried but IOSurface kernel object allocation
/// during inference is 10-20x slower than heap memcpy. Vec<u16> + copy_from_f16 wins.
struct CachedExpertWeights {
    expert_idx: usize,
    last_used: u64,
    gu_f16: Vec<u16>,
    gu_bias_f16: Vec<u16>,
    d_f16: Vec<u16>,
    d_bias_f16: Vec<u16>,
}

pub struct Session<'model> {
    model: &'model OssModel,
    rope: RopeTable,
    kv_cache: KvCache,
    position: usize,

    // Hidden state lives on ANE as TensorData [hs, 1, sw]
    ane_hidden: TensorData,

    // RoPE cos/sin TensorData — updated per position, fed to QKV graph
    ane_q_cos: TensorData,
    ane_q_sin: TensorData,
    ane_k_cos: TensorData,
    ane_k_sin: TensorData,
    // Pre-allocated fp16 buffers for cos/sin
    rope_q_cos_f16: Vec<u16>,
    rope_q_sin_f16: Vec<u16>,
    rope_k_cos_f16: Vec<u16>,
    rope_k_sin_f16: Vec<u16>,

    // Pre-allocated CPU buffers to avoid per-layer Vec allocations
    q_buf: Vec<f32>,
    k_buf: Vec<f32>,
    v_buf: Vec<f32>,
    key_query_mse_buf: Vec<f32>, // [head_dim] = Pi*q or q (full-precision mode)
    key_query_qjl_buf: Vec<f32>, // [head_dim] = S*q (TurboQuant mode)
    value_acc_buf: Vec<f32>,     // [head_dim] rotated accumulation for V
    attn_scores_buf: Vec<f32>,   // [max_seq + 1] reused softmax scores (+1 sink)
    attn_out_buf: Vec<f32>,
    moe_out_buf: Vec<f32>,

    // Pre-allocated ANE tensor buffers
    ane_qkv_out: TensorData,
    ane_proj_in: TensorData,    // attn_out written here for output projection
    ane_hidden_out: TensorData, // output projection result (hidden + proj) — swapped with ane_hidden
    ane_router_norm_out: TensorData, // [num_experts + hidden_size, 1, sw]
    ane_moe_out: TensorData,    // [hs, 1, sw] — accumulated weighted expert output for residual add

    // CPU expert FFN buffers (replaces ANE expert path — avoids IOSurface overhead)
    expert_input_f32: Vec<f32>, // [hs] — extracted from router output

    // Per-layer Vec<u16> cache: up to `expert_cache_slots` entries per layer.
    // Cache hit skips MXFP4 dequant and large heap allocations.
    active_experts_per_tok: usize,
    adaptive_moe_topk: Option<AdaptiveMoeTopKConfig>,
    expert_cache_slots: usize,
    cache_epoch: u64,
    expert_layer_cache: Vec<Vec<CachedExpertWeights>>,
    expert_cache_slot_buf: Vec<usize>,
    expert_output_scratch: Vec<Vec<f32>>,  // [top_k][hs]
    expert_gate_up_scratch: Vec<Vec<f32>>, // [top_k][2*inter]
    expert_gated_scratch: Vec<Vec<f32>>,   // [top_k][inter]

    // LM head ANE chunked matmul
    lm_input: TensorData,  // [1, 1, 1, hs] — normalized hidden for LM head
    lm_output: TensorData, // [1, 1, 1, chunk_size] — logit chunk output
    lm_weight_chunks: Box<[TensorData]>, // pre-loaded BF16→fp16 weight chunks
    lm_head_eval_retries: usize,
    verify_output_proj_router_layer: Option<usize>,
    verify_output_proj_router_done: bool,
    logits: Vec<f32>,
}

impl<'model> Session<'model> {
    pub fn new(model: &'model OssModel, max_seq: usize) -> Self {
        let hs = model.config.hidden_size;
        let qd = model.config.q_dim();
        let kvd = model.config.kv_dim();
        let nkv = model.config.num_key_value_heads;
        let hd = model.config.head_dim;
        let ne = model.config.num_local_experts;
        let top_k = parse_active_experts_per_tok(model.config.num_experts_per_tok, ne);
        let inter = model.config.intermediate_size;
        let vs = model.config.vocab_size;
        let nl = model.config.num_hidden_layers;
        let sw = DECODE_SPATIAL_WIDTH;
        let expert_cache_slots = parse_expert_cache_slots(top_k, ne);
        let adaptive_moe_topk = parse_adaptive_moe_topk(top_k);
        let lm_head_eval_retries = parse_lm_head_eval_retries();
        let verify_output_proj_router_layer = parse_verify_output_proj_router_layer(nl);
        eprintln!(
            "  Expert cache: {} slots/layer (set GPT_OSS_EXPERT_CACHE_SLOTS to override)",
            expert_cache_slots
        );
        eprintln!(
            "  Active experts/token: {} (set GPT_OSS_ACTIVE_EXPERTS to override, model default={})",
            top_k, model.config.num_experts_per_tok
        );
        if let Some(cfg) = adaptive_moe_topk {
            eprintln!(
                "  Adaptive MoE top-k: on (min_k={}, top1>= {:.2}, margin>= {:.2}, top2_mass>= {:.2}; set GPT_OSS_ADAPTIVE_* to tune)",
                cfg.min_k, cfg.top1_min_prob, cfg.top1_min_margin, cfg.top2_min_mass
            );
        } else {
            eprintln!("  Adaptive MoE top-k: off (set GPT_OSS_ADAPTIVE_MOE_TOPK=1 to enable)");
        }
        eprintln!(
            "  LM head eval retries: {} (set GPT_OSS_LM_HEAD_EVAL_RETRIES to override)",
            lm_head_eval_retries
        );
        if let Some(layer) = verify_output_proj_router_layer {
            eprintln!(
                "  Fusion verify: output_proj+router layer {} (set GPT_OSS_VERIFY_OUTPUT_PROJ_ROUTER_LAYER to override)",
                layer
            );
        }

        let kv_cache = if let Some(cfg) = TurboQuantConfig::from_env() {
            eprintln!(
                "  KV cache: TurboQuant enabled (key={}b, value={}b, seed=0x{:x})",
                cfg.key_bits, cfg.value_bits, cfg.seed
            );
            if cfg.key_bits <= 3 || cfg.value_bits <= 3 {
                eprintln!(
                    "  Warning: <=3-bit TurboQuant can severely degrade quality; recommended key/value bits >= 4"
                );
            }
            KvCache::new_turboquant(nl, nkv, hd, max_seq, cfg)
        } else {
            KvCache::new(nl, nkv, hd, max_seq)
        };

        Session {
            model,
            rope: RopeTable::new(&model.config),
            kv_cache,
            position: 0,
            ane_hidden: TensorData::new(Shape::spatial(hs, 1, sw)),
            ane_q_cos: TensorData::new(Shape::spatial(qd / 2, 1, sw)),
            ane_q_sin: TensorData::new(Shape::spatial(qd / 2, 1, sw)),
            ane_k_cos: TensorData::new(Shape::spatial(kvd / 2, 1, sw)),
            ane_k_sin: TensorData::new(Shape::spatial(kvd / 2, 1, sw)),
            rope_q_cos_f16: vec![0u16; (qd / 2) * sw],
            rope_q_sin_f16: vec![0u16; (qd / 2) * sw],
            rope_k_cos_f16: vec![0u16; (kvd / 2) * sw],
            rope_k_sin_f16: vec![0u16; (kvd / 2) * sw],
            q_buf: vec![0.0f32; qd],
            k_buf: vec![0.0f32; kvd],
            v_buf: vec![0.0f32; kvd],
            key_query_mse_buf: vec![0.0f32; hd],
            key_query_qjl_buf: vec![0.0f32; hd],
            value_acc_buf: vec![0.0f32; hd],
            attn_scores_buf: vec![0.0f32; max_seq + 1],
            attn_out_buf: vec![0.0f32; qd],
            moe_out_buf: vec![0.0f32; hs],
            ane_qkv_out: TensorData::new(Shape::spatial(qd + 2 * kvd, 1, sw)),
            ane_proj_in: TensorData::new(Shape::spatial(qd, 1, sw)),
            ane_hidden_out: TensorData::new(Shape::spatial(hs, 1, sw)),
            ane_router_norm_out: TensorData::new(Shape::spatial(ne + hs, 1, sw)),
            ane_moe_out: TensorData::new(Shape::spatial(hs, 1, sw)),
            expert_input_f32: vec![0.0f32; hs],
            active_experts_per_tok: top_k,
            adaptive_moe_topk,
            expert_cache_slots,
            cache_epoch: 0,
            expert_layer_cache: (0..nl)
                .map(|_| Vec::with_capacity(expert_cache_slots))
                .collect(),
            expert_cache_slot_buf: vec![0usize; top_k],
            expert_output_scratch: (0..top_k).map(|_| vec![0.0f32; hs]).collect(),
            expert_gate_up_scratch: (0..top_k).map(|_| vec![0.0f32; 2 * inter]).collect(),
            expert_gated_scratch: (0..top_k).map(|_| vec![0.0f32; inter]).collect(),
            // LM head: pre-load BF16 weight chunks as fp16 TensorData
            lm_input: TensorData::new(Shape {
                batch: 1,
                channels: 1,
                height: 1,
                width: hs,
            }),
            lm_output: TensorData::new(Shape {
                batch: 1,
                channels: 1,
                height: 1,
                width: model.executables.lm_head_chunk_size,
            }),
            lm_weight_chunks: {
                let chunk = model.executables.lm_head_chunk_size;
                let num_chunks = vs.div_ceil(chunk);
                eprint!("  Loading LM head weights ({} chunks)...", num_chunks);
                let chunks: Box<[TensorData]> = (0..num_chunks)
                    .map(|ci| {
                        let start = ci * chunk;
                        let end = (start + chunk).min(vs);
                        let actual = end - start;
                        let td = TensorData::new(Shape {
                            batch: 1,
                            channels: 1,
                            height: chunk,
                            width: hs,
                        });
                        // Convert BF16 → fp16 for this chunk
                        let mut f16_buf = vec![0u16; chunk * hs];
                        for row in 0..actual {
                            let v = start + row;
                            let bf16_offset = v * hs * 2;
                            let bf16_row =
                                &model.weights.lm_head_bf16[bf16_offset..bf16_offset + hs * 2];
                            for (i, c) in bf16_row.chunks_exact(2).enumerate() {
                                let bf16_val =
                                    half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]]));
                                f16_buf[row * hs + i] =
                                    half::f16::from_f32(bf16_val.to_f32()).to_bits();
                            }
                        }
                        // Pad remaining rows with zeros (for last chunk if vs % chunk != 0)
                        td.copy_from_f16(&f16_buf);
                        td
                    })
                    .collect();
                eprintln!(" done");
                chunks
            },
            lm_head_eval_retries,
            verify_output_proj_router_layer,
            verify_output_proj_router_done: false,
            logits: vec![0.0; vs],
        }
    }

    fn verify_output_proj_router_fusion(&self, layer: usize, attn_out: &[f32], hidden_in: &[f32]) {
        let config = &self.model.config;
        let hs = config.hidden_size;
        let qd = config.q_dim();
        let ne = config.num_local_experts;
        let sw = DECODE_SPATIAL_WIDTH;
        let lw = &self.model.weights.layers[layer];

        let mut hidden_ref = vec![0.0f32; hs];
        for o in 0..hs {
            let mut acc = lw.o_proj_bias[o];
            let row = &lw.o_proj_weight[o * qd..(o + 1) * qd];
            for i in 0..qd {
                acc += row[i] * attn_out[i];
            }
            hidden_ref[o] = hidden_in[o] + acc;
        }

        let mut mean_sq = 0.0f32;
        for &v in &hidden_ref {
            mean_sq += v * v;
        }
        mean_sq /= hs as f32;
        let inv_rms = (mean_sq + config.rms_norm_eps as f32).sqrt().recip();

        let mut norm_ref = vec![0.0f32; hs];
        for d in 0..hs {
            norm_ref[d] = hidden_ref[d] * inv_rms * lw.post_attn_layernorm_weight[d];
        }

        let mut router_logits = vec![0.0f32; ne];
        for e in 0..ne {
            let mut acc = lw.router_bias[e];
            let row = &lw.router_weight[e * hs..(e + 1) * hs];
            for d in 0..hs {
                acc += row[d] * norm_ref[d];
            }
            router_logits[e] = acc;
        }
        let max_logit = router_logits
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0f32;
        let mut router_probs_ref = vec![0.0f32; ne];
        for (i, &v) in router_logits.iter().enumerate() {
            let p = (v - max_logit).exp();
            router_probs_ref[i] = p;
            exp_sum += p;
        }
        if exp_sum > 0.0 {
            for p in &mut router_probs_ref {
                *p /= exp_sum;
            }
        }

        let hidden_out_slice = self.ane_hidden_out.as_f32_slice();
        let router_norm_slice = self.ane_router_norm_out.as_f32_slice();
        let mut hidden_ane = vec![0.0f32; hs];
        for d in 0..hs {
            hidden_ane[d] = hidden_out_slice[d * sw];
        }
        let mut norm_ane = vec![0.0f32; hs];
        for d in 0..hs {
            norm_ane[d] = router_norm_slice[(ne + d) * sw];
        }
        let mut router_probs_ane = vec![0.0f32; ne];
        for e in 0..ne {
            router_probs_ane[e] = router_norm_slice[e * sw];
        }

        let max_abs_diff = |a: &[f32], b: &[f32]| -> (f32, usize) {
            let mut max_diff = 0.0f32;
            let mut max_idx = 0usize;
            for i in 0..a.len().min(b.len()) {
                let d = (a[i] - b[i]).abs();
                if d > max_diff {
                    max_diff = d;
                    max_idx = i;
                }
            }
            (max_diff, max_idx)
        };
        let (hidden_diff, hidden_idx) = max_abs_diff(&hidden_ref, &hidden_ane);
        let (norm_diff, norm_idx) = max_abs_diff(&norm_ref, &norm_ane);
        let (router_diff, router_idx) = max_abs_diff(&router_probs_ref, &router_probs_ane);

        let router_sum: f32 = router_probs_ane.iter().sum();
        let router_min = router_probs_ane
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        let router_max = router_probs_ane
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);

        eprintln!(
            "  Fusion verify L{layer}: hidden max|Δ|={hidden_diff:.3e}@{hidden_idx}, \
             norm max|Δ|={norm_diff:.3e}@{norm_idx}, router max|Δ|={router_diff:.3e}@{router_idx}, \
             router(sum={router_sum:.6}, min={router_min:.6}, max={router_max:.6})"
        );
    }

    fn select_expert_count(&self, sorted_probs_desc: &[(usize, f32)], max_k: usize) -> usize {
        let mut selected = max_k;
        if let Some(cfg) = self.adaptive_moe_topk {
            let p1 = sorted_probs_desc.first().map(|(_, p)| *p).unwrap_or(0.0);
            let p2 = sorted_probs_desc.get(1).map(|(_, p)| *p).unwrap_or(0.0);
            let margin = (p1 - p2).max(0.0);
            if max_k >= 1 && p1 >= cfg.top1_min_prob && margin >= cfg.top1_min_margin {
                selected = 1;
            } else if max_k >= 2 && (p1 + p2) >= cfg.top2_min_mass {
                selected = 2;
            }
            selected = selected.max(cfg.min_k).min(max_k);
        }
        selected
    }

    /// Reset decoding position for a new request while reusing allocated buffers
    /// and dequantized expert cache.
    pub fn reset(&mut self) {
        self.position = 0;
        self.kv_cache.position = 0;
    }

    /// Decode one token and update KV cache without materializing logits.
    /// Used for prompt prefill where only the final token's logits are needed.
    pub fn decode_step_prefill(&mut self, token: u32) {
        self.decode_step_inner(token, false);
    }

    /// Decode one token and return logits for next-token sampling.
    pub fn decode_step(&mut self, token: u32) -> &[f32] {
        self.decode_step_inner(token, true);
        &self.logits
    }

    fn decode_step_inner(&mut self, token: u32, compute_logits: bool) {
        let config = &self.model.config;
        let hs = config.hidden_size;
        let qd = config.q_dim();
        let kvd = config.kv_dim();
        let nh = config.num_attention_heads;
        let nkv = config.num_key_value_heads;
        let hd = config.head_dim;
        let gqa = config.gqa_ratio();
        let nl = config.num_hidden_layers;
        let ne = config.num_local_experts;
        let top_k = self.active_experts_per_tok;
        let sw = DECODE_SPATIAL_WIDTH;
        let pos = self.position;
        let num_keys = pos + 1;
        // 1. Embedding → ANE hidden (BF16→f32→fp16 into IOSurface)
        {
            let mut surf = self.ane_hidden.as_f32_slice_mut();
            let ti = token as usize;
            let offset = ti * hs * 2;
            let bytes = &self.model.weights.embed_tokens_bf16[offset..offset + hs * 2];
            for (d, chunk) in bytes.chunks_exact(2).enumerate() {
                surf[d * sw] =
                    half::bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();
            }
        }

        // Pre-compute cos/sin for this position ONCE (same for all layers)
        {
            let half = hd / 2;

            for i in 0..half {
                let theta = pos as f64 * self.rope.inv_freq()[i];
                let cos_val = half::f16::from_f32(theta.cos() as f32 * self.rope.attn_scale());
                let sin_val = half::f16::from_f32(theta.sin() as f32 * self.rope.attn_scale());
                let cos_bits = cos_val.to_bits();
                let sin_bits = sin_val.to_bits();

                for h in 0..nh {
                    self.rope_q_cos_f16[(h * half + i) * sw] = cos_bits;
                    self.rope_q_sin_f16[(h * half + i) * sw] = sin_bits;
                }
                for h in 0..nkv {
                    self.rope_k_cos_f16[(h * half + i) * sw] = cos_bits;
                    self.rope_k_sin_f16[(h * half + i) * sw] = sin_bits;
                }
            }
            self.ane_q_cos.copy_from_f16(&self.rope_q_cos_f16);
            self.ane_q_sin.copy_from_f16(&self.rope_q_sin_f16);
            self.ane_k_cos.copy_from_f16(&self.rope_k_cos_f16);
            self.ane_k_sin.copy_from_f16(&self.rope_k_sin_f16);
        }

        for layer in 0..nl {
            // ── Attention block ──

            // 2b. ANE: Fused RMSNorm + QKV + RoPE
            self.model.executables.layers[layer]
                .qkv
                .run(
                    &[
                        &self.ane_hidden,
                        &self.ane_q_cos,
                        &self.ane_q_sin,
                        &self.ane_k_cos,
                        &self.ane_k_sin,
                    ],
                    &[&self.ane_qkv_out],
                )
                .unwrap_or_else(|e| panic!("qkv layer {layer}: {e}"));

            // 2c-2e. Extract Q,K,V + KV cache + attention (CPU)
            let q = &mut self.q_buf;
            let k = &mut self.k_buf;
            let v = &mut self.v_buf;
            {
                let qkv_slice = self.ane_qkv_out.as_f32_slice();
                let half = hd / 2;
                let half_qd = qd / 2;
                let half_kvd = kvd / 2;

                // Q: reorder from [all_first_halves, all_second_halves] → interleaved heads
                for h in 0..nh {
                    for d in 0..half {
                        q[h * hd + d] = qkv_slice[(h * half + d) * sw]; // first half
                        q[h * hd + half + d] = qkv_slice[(half_qd + h * half + d) * sw];
                        // second half
                    }
                }
                // K: same reorder
                for h in 0..nkv {
                    for d in 0..half {
                        k[h * hd + d] = qkv_slice[(qd + h * half + d) * sw];
                        k[h * hd + half + d] = qkv_slice[(qd + half_kvd + h * half + d) * sw];
                    }
                }
                // V: no reorder needed (original interleaved layout)
                for d in 0..kvd {
                    v[d] = qkv_slice[(qd + kvd + d) * sw];
                }
            }

            // 2d. CPU: KV cache write
            self.kv_cache.write(layer, pos, &k, &v);

            // 2e. CPU: Attention (Q·K^T, sinks, softmax, attn·V) — stays CPU for Beaver triples
            let scale = 1.0 / (hd as f32).sqrt();
            let layer_type = config.layer_types[layer];
            let window = config.sliding_window;
            let sinks = &self.model.weights.layers[layer].sinks;

            {
                let attn_out = &mut self.attn_out_buf;
                for h in 0..nh {
                    let kv_h = h / gqa;
                    let q_head = &q[h * hd..(h + 1) * hd];
                    self.kv_cache.prepare_key_query_head(
                        q_head,
                        &mut self.key_query_mse_buf,
                        &mut self.key_query_qjl_buf,
                    );

                    let scores = &mut self.attn_scores_buf[..num_keys + 1];
                    for kp in 0..num_keys {
                        let mut dot = self.kv_cache.key_dot_head_from_prepared(
                            layer,
                            kp,
                            kv_h,
                            &self.key_query_mse_buf,
                            &self.key_query_qjl_buf,
                        );
                        dot *= scale;
                        if kp > pos {
                            dot = f32::NEG_INFINITY;
                        } else if layer_type == LayerType::SlidingAttention
                            && pos >= window
                            && kp < pos - window + 1
                        {
                            dot = f32::NEG_INFINITY;
                        }
                        scores[kp] = dot;
                    }
                    scores[num_keys] = sinks[h];

                    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut exp_sum = 0.0f32;
                    for s in scores.iter_mut() {
                        *s = (*s - max_score).exp();
                        exp_sum += *s;
                    }
                    if exp_sum > 0.0 {
                        for s in scores.iter_mut() {
                            *s /= exp_sum;
                        }
                    }

                    self.value_acc_buf.fill(0.0);
                    for kp in 0..num_keys {
                        self.kv_cache.add_weighted_value_head(
                            layer,
                            kp,
                            kv_h,
                            scores[kp],
                            &mut self.value_acc_buf,
                        );
                    }
                    let out_base = h * hd;
                    self.kv_cache.finalize_value_head(
                        &self.value_acc_buf,
                        &mut attn_out[out_base..out_base + hd],
                    );
                }
            }

            // 2f. ANE: Fused output projection + residual + router.
            // Outputs:
            //   ane_hidden_out      = hidden + o_proj(attn_out)
            //   ane_router_norm_out = concat(router_softmax, rms_norm(ane_hidden_out))
            let verify_fused_layer = self.verify_output_proj_router_layer == Some(layer)
                && !self.verify_output_proj_router_done;
            let hidden_before = if verify_fused_layer {
                let hidden_slice = self.ane_hidden.as_f32_slice();
                let mut v = vec![0.0f32; hs];
                for d in 0..hs {
                    v[d] = hidden_slice[d * sw];
                }
                Some(v)
            } else {
                None
            };
            {
                let mut surf = self.ane_proj_in.as_f32_slice_mut();
                for d in 0..qd {
                    surf[d * sw] = self.attn_out_buf[d];
                }
            }
            self.model.executables.layers[layer]
                .output_proj_router
                .run(
                    &[&self.ane_proj_in, &self.ane_hidden],
                    // ANE multi-output requests are observed to bind surfaces in reverse
                    // order for this executable, so pass buffers reversed.
                    &[&self.ane_router_norm_out, &self.ane_hidden_out],
                )
                .unwrap_or_else(|e| panic!("output_proj_router layer {layer}: {e}"));
            if let Some(hidden_in) = hidden_before.as_deref() {
                self.verify_output_proj_router_fusion(layer, &self.attn_out_buf, hidden_in);
                self.verify_output_proj_router_done = true;
            }
            std::mem::swap(&mut self.ane_hidden, &mut self.ane_hidden_out);

            // ── MoE FFN block ──

            // 3b. CPU: Top-k expert selection (softmax already fused in ANE router graph)
            let (expert_indices, expert_weights, selected_k) = {
                let out_slice = self.ane_router_norm_out.as_f32_slice();
                // Router output channels [0..ne] are already softmax probabilities from ANE
                let probs: Vec<f32> = (0..ne).map(|i| out_slice[i * sw]).collect();
                let mut indexed: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
                indexed.sort_unstable_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                let selected_k = if compute_logits {
                    self.select_expert_count(&indexed, top_k)
                } else {
                    top_k
                };
                let top: Vec<(usize, f32)> = indexed[..selected_k].to_vec();
                // Re-normalize top-k weights
                let top_sum: f32 = top.iter().map(|(_, w)| w).sum();
                let indices: Vec<usize> = top.iter().map(|(i, _)| *i).collect();
                let weights: Vec<f32> = if top_sum.is_finite() && top_sum > 0.0 {
                    top.iter().map(|(_, w)| w / top_sum).collect()
                } else {
                    vec![1.0 / selected_k as f32; selected_k]
                };
                (indices, weights, selected_k)
            };

            // 3c. Expert input from ANE-normalized hidden → CPU f32 buffer
            {
                let out_slice = self.ane_router_norm_out.as_f32_slice();
                for d in 0..hs {
                    self.expert_input_f32[d] = out_slice[(ne + d) * sw];
                }
            }

            // 3d. CPU: Run selected experts via NEON matmul (no IOSurface overhead)
            self.moe_out_buf.fill(0.0);
            let inter = config.intermediate_size;
            self.cache_epoch = self.cache_epoch.wrapping_add(1);
            let use_epoch = self.cache_epoch;

            // Phase 1: Ensure selected experts are in cache.
            // Dequantization is the dominant prefill hotspot, so compute missing
            // experts in parallel and merge into the per-layer cache afterward.
            {
                let missing: Vec<usize> = expert_indices
                    .iter()
                    .copied()
                    .filter(|expert_idx| {
                        !self.expert_layer_cache[layer]
                            .iter()
                            .any(|c| c.expert_idx == *expert_idx)
                    })
                    .collect();

                let decoded: Vec<CachedExpertWeights> = missing
                    .par_iter()
                    .map(|&expert_idx| {
                        let expert = &self.model.weights.layers[layer].experts[expert_idx];
                        let mut gu_f16 = vec![0u16; 2 * inter * hs];
                        let mut gu_bias_f16 = vec![0u16; 2 * inter];
                        let mut d_f16 = vec![0u16; hs * inter];
                        let mut d_bias_f16 = vec![0u16; hs];
                        weights::dequant_expert_gate_up_reordered_f16(
                            expert,
                            inter,
                            hs,
                            &mut gu_f16,
                        );
                        weights::reorder_gate_up_bias_f16(
                            &expert.gate_up_bias,
                            inter,
                            &mut gu_bias_f16,
                        );
                        weights::dequant_expert_down_f16(expert, hs, inter, &mut d_f16);
                        weights::f32_to_f16_slice(&expert.down_bias, &mut d_bias_f16);
                        CachedExpertWeights {
                            expert_idx,
                            last_used: use_epoch,
                            gu_f16,
                            gu_bias_f16,
                            d_f16,
                            d_bias_f16,
                        }
                    })
                    .collect();

                for decoded_expert in decoded {
                    let cache = &mut self.expert_layer_cache[layer];
                    if let Some(existing) = cache
                        .iter_mut()
                        .find(|c| c.expert_idx == decoded_expert.expert_idx)
                    {
                        existing.last_used = use_epoch;
                        continue;
                    }
                    if cache.len() < self.expert_cache_slots {
                        cache.push(decoded_expert);
                        continue;
                    }
                    let evict = cache
                        .iter()
                        .enumerate()
                        .filter(|(_, c)| !expert_indices.contains(&c.expert_idx))
                        .min_by_key(|(_, c)| c.last_used)
                        .map(|(i, _)| i)
                        .unwrap_or_else(|| {
                            cache
                                .iter()
                                .enumerate()
                                .min_by_key(|(_, c)| c.last_used)
                                .map(|(i, _)| i)
                                .expect("cache cannot be empty when evicting")
                        });
                    cache[evict] = decoded_expert;
                }

                for &expert_idx in &expert_indices {
                    if let Some(cached) = self.expert_layer_cache[layer]
                        .iter_mut()
                        .find(|c| c.expert_idx == expert_idx)
                    {
                        cached.last_used = use_epoch;
                    }
                }
            }

            // Phase 2: Run top-k experts in parallel with reusable scratch buffers.
            for (slot, &idx) in expert_indices.iter().enumerate() {
                self.expert_cache_slot_buf[slot] = self.expert_layer_cache[layer]
                    .iter()
                    .position(|c| c.expert_idx == idx)
                    .expect("expert must be cached after phase 1");
            }
            let cache = &self.expert_layer_cache[layer];
            let input = &self.expert_input_f32;
            let cache_slot_buf = &self.expert_cache_slot_buf[..selected_k];
            self.expert_output_scratch[..selected_k]
                .par_iter_mut()
                .zip(self.expert_gate_up_scratch[..selected_k].par_iter_mut())
                .zip(self.expert_gated_scratch[..selected_k].par_iter_mut())
                .zip(cache_slot_buf.par_iter())
                .for_each(|(((output, gu_buf), g_buf), &ci)| {
                    let c = &cache[ci];
                    cpu_expert_ffn(
                        input,
                        &c.gu_f16,
                        &c.gu_bias_f16,
                        &c.d_f16,
                        &c.d_bias_f16,
                        output,
                        gu_buf,
                        g_buf,
                        hs,
                        inter,
                    );
                });

            // Phase 3: Weighted accumulation (sequential)
            for (slot, &weight) in expert_weights.iter().enumerate() {
                let output = &self.expert_output_scratch[slot];
                for d in 0..hs {
                    self.moe_out_buf[d] += weight * output[d];
                }
            }

            // 3e. ANE: Residual add (hidden = hidden + moe_out)
            {
                let mut surf = self.ane_moe_out.as_f32_slice_mut();
                for d in 0..hs {
                    surf[d * sw] = self.moe_out_buf[d];
                }
            }
            self.model
                .executables
                .residual_add
                .run(
                    &[&self.ane_hidden, &self.ane_moe_out],
                    &[&self.ane_hidden_out],
                )
                .unwrap_or_else(|e| panic!("moe residual layer {layer}: {e}"));
            std::mem::swap(&mut self.ane_hidden, &mut self.ane_hidden_out);
        }

        if compute_logits {
            // 4. ANE: Final RMSNorm
            self.model
                .executables
                .final_norm
                .run(&[&self.ane_hidden], &[&self.ane_hidden_out])
                .unwrap_or_else(|e| panic!("final_norm: {e}"));

            // 5. ANE: LM head chunked matmul
            // Read normalized hidden from ANE spatial → row vector [1,1,1,hs] for matmul
            {
                let norm_slice = self.ane_hidden_out.as_f32_slice();
                let mut surf = self.lm_input.as_f32_slice_mut();
                for d in 0..hs {
                    surf[d] = norm_slice[d * sw];
                }
            }
            {
                let vs = config.vocab_size;
                let chunk = self.model.executables.lm_head_chunk_size;
                for (ci, weight_td) in self.lm_weight_chunks.iter().enumerate() {
                    let mut run_err: Option<String> = None;
                    for attempt in 1..=self.lm_head_eval_retries {
                        match self
                            .model
                            .executables
                            .lm_head_chunk
                            .run(&[&self.lm_input, weight_td], &[&self.lm_output])
                        {
                            Ok(()) => {
                                run_err = None;
                                break;
                            }
                            Err(e) => {
                                run_err = Some(e.to_string());
                                if attempt < self.lm_head_eval_retries {
                                    std::thread::sleep(std::time::Duration::from_millis(5));
                                }
                            }
                        }
                    }
                    if let Some(e) = run_err {
                        panic!(
                            "lm_head chunk {ci}: {e}. Possible ANE memory pressure/runtime instability. \
                             Try GPT_OSS_EXPERT_CACHE_SLOTS=4 and/or GPT_OSS_LM_HEAD_CHUNK_SIZE=4096."
                        );
                    }

                    let out_slice = self.lm_output.as_f32_slice();
                    let start = ci * chunk;
                    let end = (start + chunk).min(vs);
                    for i in 0..(end - start) {
                        self.logits[start + i] = out_slice[i];
                    }
                }
            }
        }

        self.position += 1;
    }
}
