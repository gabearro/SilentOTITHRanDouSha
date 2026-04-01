//! ANE graph builders for GPT-OSS-20B components.
//!
//! All linear ops (QKV, output proj, router, expert FFN, LM head) run on ANE.
//! Non-linear ops composed from ANE primitives: RMSNorm, SiLU, SwiGLU.
//! Position-dependent ops (RoPE, top-k) handled on CPU.

use ane::graph::Convolution2dDescriptor;
use ane::{Executable, Graph, NSQualityOfService, Shape};

use super::config::GptOssConfig;
use super::weights::LayerWeights;

pub const DECODE_SPATIAL_WIDTH: usize = 64;

fn scalar_shape() -> Shape {
    Shape {
        batch: 1,
        channels: 1,
        height: 1,
        width: 1,
    }
}

/// RMSNorm on ANE via the ANEMLL concat([x, -x]) → layer_norm trick.
///
/// Concatenating [x, -x] forces mean=0, so layer_norm's variance computation
/// equals mean(x²) = RMS². The ANE's fused layer_norm kernel handles the
/// variance computation internally without fp16 overflow.
///
/// Input: [1, dim, 1, sw] (channel-major spatial)
/// Output: [1, dim, 1, sw] (normalized and scaled by gamma)
fn rms_norm(
    graph: &mut Graph,
    input: ane::Tensor,
    gamma: &[f32],
    dim: usize,
    eps: f64,
) -> ane::Tensor {
    let sw = DECODE_SPATIAL_WIDTH;

    // 1. concat [x, -x] along channels → mean becomes exactly 0
    let neg_one = graph.constant_with_scalar(-1.0, scalar_shape());
    let neg = graph.multiplication(input, neg_one);
    let doubled = graph.concat(&[input, neg], 1); // [1, 2*dim, 1, sw]

    // 2. layer_norm over channel axis (axis=1) — fused ANE op, no fp16 overflow
    let normed = graph.layer_norm(doubled, &[1], eps);

    // 3. slice first dim channels (drop the -x mirror half)
    let result = graph.slice(normed, [0, 0, 0, 0], [1, dim, 1, sw]);

    // 4. apply learned gamma
    let gamma_c = graph.constant(
        gamma,
        Shape {
            batch: 1,
            channels: dim,
            height: 1,
            width: 1,
        },
    );
    graph.multiplication(result, gamma_c)
}

/// SiLU activation: x * sigmoid(x)
fn silu(graph: &mut Graph, input: ane::Tensor) -> ane::Tensor {
    let sig = graph.sigmoid(input);
    graph.multiplication(input, sig)
}

/// Reorder projection weight rows: interleaved head dims → first-halves then second-halves.
///
/// Original layout: [h0_d0..h0_d63, h1_d0..h1_d63, ...]
/// Reordered:        [h0_d0..h0_d31, h1_d0..h1_d31, ..., h0_d32..h0_d63, h1_d32..h1_d63, ...]
///
/// This makes first/second halves of all heads contiguous, enabling efficient
/// RoPE via two slices instead of 2×num_heads slices.
fn reorder_weight_for_rope(
    weight: &[f32],
    num_heads: usize,
    head_dim: usize,
    hidden_size: usize,
) -> Vec<f32> {
    let half = head_dim / 2;
    let out_dim = num_heads * head_dim;
    let mut reordered = vec![0.0f32; out_dim * hidden_size];

    for h in 0..num_heads {
        // First half of head h: rows h*head_dim..h*head_dim+half → dest rows h*half
        for d in 0..half {
            let src_row = h * head_dim + d;
            let dst_row = h * half + d;
            let src_off = src_row * hidden_size;
            let dst_off = dst_row * hidden_size;
            reordered[dst_off..dst_off + hidden_size]
                .copy_from_slice(&weight[src_off..src_off + hidden_size]);
        }
        // Second half of head h: rows h*head_dim+half..h*head_dim+head_dim → dest rows num_heads*half + h*half
        for d in 0..half {
            let src_row = h * head_dim + half + d;
            let dst_row = num_heads * half + h * half + d;
            let src_off = src_row * hidden_size;
            let dst_off = dst_row * hidden_size;
            reordered[dst_off..dst_off + hidden_size]
                .copy_from_slice(&weight[src_off..src_off + hidden_size]);
        }
    }
    reordered
}

/// Same reorder for bias vectors.
fn reorder_bias_for_rope(bias: &[f32], num_heads: usize, head_dim: usize) -> Vec<f32> {
    let half = head_dim / 2;
    let out_dim = num_heads * head_dim;
    let mut reordered = vec![0.0f32; out_dim];
    for h in 0..num_heads {
        for d in 0..half {
            reordered[h * half + d] = bias[h * head_dim + d];
            reordered[num_heads * half + h * half + d] = bias[h * head_dim + half + d];
        }
    }
    reordered
}

/// Build fused RMSNorm + QKV projection + RoPE on ANE.
///
/// RoPE trick: Q/K weights are reordered at compile time so that first-half dims
/// of all heads are output first (channels 0..out_dim/2), then second-half dims
/// (channels out_dim/2..out_dim). This makes rotate_half a simple 2-way slice.
///
/// Inputs:
///   0: hidden  [hs, 1, sw]           — raw hidden state
///   1: q_cos   [q_dim/2, 1, sw]      — cos values for Q (expanded per head, one row per position)
///   2: q_sin   [q_dim/2, 1, sw]      — sin values for Q
///   3: k_cos   [kv_dim/2, 1, sw]     — cos values for K
///   4: k_sin   [kv_dim/2, 1, sw]     — sin values for K
///
/// Output: [q_dim + kv_dim + kv_dim, 1, sw]
///   Q with RoPE applied (reordered: first-halves then second-halves per head)
///   K with RoPE applied (same reorder)
///   V unchanged (original interleaved layout)
pub fn build_decode_qkv(
    lw: &LayerWeights,
    config: &GptOssConfig,
) -> Result<Executable, ane::Error> {
    let hs = config.hidden_size;
    let qd = config.q_dim();
    let kvd = config.kv_dim();
    let nh = config.num_attention_heads;
    let nkv = config.num_key_value_heads;
    let hd = config.head_dim;
    let half_qd = qd / 2; // 2048
    let half_kvd = kvd / 2; // 256
    let sw = DECODE_SPATIAL_WIDTH;

    let mut g = Graph::new();

    // Placeholders
    let input = g.placeholder(Shape::spatial(hs, 1, sw));
    let q_cos = g.placeholder(Shape::spatial(half_qd, 1, sw));
    let q_sin = g.placeholder(Shape::spatial(half_qd, 1, sw));
    let k_cos = g.placeholder(Shape::spatial(half_kvd, 1, sw));
    let k_sin = g.placeholder(Shape::spatial(half_kvd, 1, sw));

    // RMSNorm
    let norm = rms_norm(
        &mut g,
        input,
        &lw.input_layernorm_weight,
        hs,
        config.rms_norm_eps,
    );

    // Q projection with reordered weights (first-halves, then second-halves)
    let q_w_reordered = reorder_weight_for_rope(&lw.q_proj_weight, nh, hd, hs);
    let q_b_reordered = reorder_bias_for_rope(&lw.q_proj_bias, nh, hd);
    let q_w = g.constant(&q_w_reordered, Shape::spatial(qd, 1, 1));
    let q = g.convolution_2d_1x1(norm, q_w, None);
    let q_b = g.constant(
        &q_b_reordered,
        Shape {
            batch: 1,
            channels: qd,
            height: 1,
            width: 1,
        },
    );
    let q = g.addition(q, q_b); // [qd, 1, sw] — first half_qd channels are first-halves, next half_qd are second-halves

    // K projection with reordered weights
    let k_w_reordered = reorder_weight_for_rope(&lw.k_proj_weight, nkv, hd, hs);
    let k_b_reordered = reorder_bias_for_rope(&lw.k_proj_bias, nkv, hd);
    let k_w = g.constant(&k_w_reordered, Shape::spatial(kvd, 1, 1));
    let k = g.convolution_2d_1x1(norm, k_w, None);
    let k_b = g.constant(
        &k_b_reordered,
        Shape {
            batch: 1,
            channels: kvd,
            height: 1,
            width: 1,
        },
    );
    let k = g.addition(k, k_b); // [kvd, 1, sw]

    // V projection (no reorder — V doesn't get RoPE)
    let v_w = g.constant(&lw.v_proj_weight, Shape::spatial(kvd, 1, 1));
    let v = g.convolution_2d_1x1(norm, v_w, None);
    let v_b = g.constant(
        &lw.v_proj_bias,
        Shape {
            batch: 1,
            channels: kvd,
            height: 1,
            width: 1,
        },
    );
    let v = g.addition(v, v_b);

    // ── RoPE on Q (all ANE-native: slice, mul, sub, add, concat) ──
    let q_first = g.slice(q, [0, 0, 0, 0], [1, half_qd, 1, sw]);
    let q_second = g.slice(q, [0, half_qd, 0, 0], [1, half_qd, 1, sw]);
    // q_rope_first  = q_first * cos - q_second * sin
    let qf_cos = g.multiplication(q_first, q_cos);
    let qs_sin = g.multiplication(q_second, q_sin);
    let q_rope_first = g.subtraction(qf_cos, qs_sin);
    // q_rope_second = q_first * sin + q_second * cos
    let qf_sin = g.multiplication(q_first, q_sin);
    let qs_cos = g.multiplication(q_second, q_cos);
    let q_rope_second = g.addition(qf_sin, qs_cos);
    let q_rope = g.concat(&[q_rope_first, q_rope_second], 1); // [qd, 1, sw]

    // ── RoPE on K ──
    let k_first = g.slice(k, [0, 0, 0, 0], [1, half_kvd, 1, sw]);
    let k_second = g.slice(k, [0, half_kvd, 0, 0], [1, half_kvd, 1, sw]);
    let kf_cos = g.multiplication(k_first, k_cos);
    let ks_sin = g.multiplication(k_second, k_sin);
    let k_rope_first = g.subtraction(kf_cos, ks_sin);
    let kf_sin = g.multiplication(k_first, k_sin);
    let ks_cos = g.multiplication(k_second, k_cos);
    let k_rope_second = g.addition(kf_sin, ks_cos);
    let k_rope = g.concat(&[k_rope_first, k_rope_second], 1); // [kvd, 1, sw]

    let _out = g.concat(&[q_rope, k_rope, v], 1);
    g.compile(NSQualityOfService::Default)
}

/// Build fused output projection + residual + router path on ANE.
///
/// Inputs:
///   0: attn_out [q_dim, 1, sw]
///   1: hidden   [hs, 1, sw]
///
/// Outputs (in this order):
///   0: hidden_out        [hs, 1, sw]       = hidden + o_proj(attn_out)
///   1: router_norm_out   [ne + hs, 1, sw]  = concat(router_softmax, post-attn RMSNorm(hidden_out))
///
/// This removes one ANE dispatch per layer/token by fusing the old
/// `output_proj` and `router` executables.
pub fn build_decode_output_proj_router(
    lw: &LayerWeights,
    config: &GptOssConfig,
) -> Result<Executable, ane::Error> {
    let hs = config.hidden_size;
    let qd = config.q_dim();
    let ne = config.num_local_experts;
    let sw = DECODE_SPATIAL_WIDTH;

    let mut g = Graph::new();
    let attn_out = g.placeholder(Shape::spatial(qd, 1, sw));
    let hidden = g.placeholder(Shape::spatial(hs, 1, sw));

    // hidden_out = hidden + o_proj(attn_out)
    let o_w = g.constant(&lw.o_proj_weight, Shape::spatial(hs, 1, 1));
    let proj = g.convolution_2d_1x1(attn_out, o_w, None);
    let o_b = g.constant(
        &lw.o_proj_bias,
        Shape {
            batch: 1,
            channels: hs,
            height: 1,
            width: 1,
        },
    );
    let proj = g.addition(proj, o_b);
    let hidden_residual = g.addition(hidden, proj);

    // Create a dedicated leaf output for hidden_out while still consuming
    // hidden_residual for router computation.
    let zero = g.constant_with_scalar(0.0, scalar_shape());
    let _hidden_out = g.addition(hidden_residual, zero);

    // router_norm_out = concat(softmax(router(norm(hidden_out))), norm(hidden_out))
    let norm = rms_norm(
        &mut g,
        hidden_residual,
        &lw.post_attn_layernorm_weight,
        hs,
        config.rms_norm_eps,
    );
    let r_w = g.constant(&lw.router_weight, Shape::spatial(ne, 1, 1));
    let r_proj = g.convolution_2d_1x1(norm, r_w, None);
    let r_b = g.constant(
        &lw.router_bias,
        Shape {
            batch: 1,
            channels: ne,
            height: 1,
            width: 1,
        },
    );
    let router = g.addition(r_proj, r_b);
    let router = g.soft_max(router, 1);
    let _router_norm_out = g.concat(&[router, norm], 1);

    g.compile(NSQualityOfService::Default)
}

/// Build residual add for MoE output: hidden + moe_out → new_hidden.
/// Both inputs [hs, 1, sw]. Used after accumulating weighted expert outputs.
pub fn build_residual_add(config: &GptOssConfig) -> Result<Executable, ane::Error> {
    let hs = config.hidden_size;
    let sw = DECODE_SPATIAL_WIDTH;

    let mut g = Graph::new();
    let hidden = g.placeholder(Shape::spatial(hs, 1, sw));
    let delta = g.placeholder(Shape::spatial(hs, 1, sw));
    let _out = g.addition(hidden, delta);
    g.compile(NSQualityOfService::Default)
}

/// Build fused RMSNorm + MoE router + normalized hidden on ANE.
/// Input: raw hidden state [hs, 1, sw]
/// Output: [num_experts + hs, 1, sw]
///   channels [0..ne]: router logits
///   channels [ne..ne+hs]: normalized hidden (for expert FFN input)
pub fn build_decode_router(
    lw: &LayerWeights,
    config: &GptOssConfig,
) -> Result<Executable, ane::Error> {
    let hs = config.hidden_size;
    let ne = config.num_local_experts;
    let sw = DECODE_SPATIAL_WIDTH;

    let mut g = Graph::new();
    let input = g.placeholder(Shape::spatial(hs, 1, sw));
    let norm = rms_norm(
        &mut g,
        input,
        &lw.post_attn_layernorm_weight,
        hs,
        config.rms_norm_eps,
    );

    let w = g.constant(&lw.router_weight, Shape::spatial(ne, 1, 1));
    let proj = g.convolution_2d_1x1(norm, w, None);
    let b = g.constant(
        &lw.router_bias,
        Shape {
            batch: 1,
            channels: ne,
            height: 1,
            width: 1,
        },
    );
    let router = g.addition(proj, b);

    // Fuse softmax into router graph — eliminates CPU softmax on 32 values
    let router = g.soft_max(router, 1); // softmax over channels (experts)

    // Concat router probs + normalized hidden as single output
    let _out = g.concat(&[router, norm], 1);

    g.compile(NSQualityOfService::Default)
}

/// Build expert SwiGLU FFN on ANE via matrix_multiplication.
///
/// ANE's conv1x1_dynamic requires weight IOSurface width≥64 at the hardware level
/// (kernel shape [OC, IC, 1, 1] has width=1 → rejected by ANE runtime).
/// Matmul with transpose_y=true achieves the same computation without this constraint.
///
/// Placeholders (in order):
///   0: input       [1, 1, 1, hs]               — row vector
///   1: gu_w        [1, 1, 2*inter, hs]          — gate_up weight [OC, IC]
///   2: gu_b        [1, 1, 1, 2*inter]           — gate_up bias
///   3: down_w      [1, 1, hs, inter]            — down weight [OC, IC]
///   4: down_b      [1, 1, 1, hs]               — down bias
///
/// Output: [1, 1, 1, hs]
pub fn build_expert_ffn_dynamic(config: &GptOssConfig) -> Result<Executable, ane::Error> {
    let hs = config.hidden_size;
    let inter = config.intermediate_size;
    let limit = config.swiglu_limit;
    let alpha: f32 = 1.702;

    let mut g = Graph::new();

    // All widths ≥ 64: hs=2880, 2*inter=5760 ✓
    let input = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: 1,
        width: hs,
    });
    let gu_w = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: 2 * inter,
        width: hs,
    });
    let gu_b = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: 1,
        width: 2 * inter,
    });
    let d_w = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: hs,
        width: inter,
    });
    let d_b = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: 1,
        width: hs,
    });

    // gate_up = input @ gu_w^T + bias → [1, 1, 1, 2*inter]
    let gu = g.matrix_multiplication(input, gu_w, false, true);
    let gu = g.addition(gu, gu_b);

    // Split gate/up on width dim (reordered to contiguous at dequant time)
    let gate = g.slice(gu, [0, 0, 0, 0], [1, 1, 1, inter]);
    let up = g.slice(gu, [0, 0, 0, inter], [1, 1, 1, inter]);

    // Clamp: gate max only, up both directions (composed min/max — clip MIL op not supported by private API)
    let pos_limit = g.constant_with_scalar(limit, scalar_shape());
    let neg_limit = g.constant_with_scalar(-limit, scalar_shape());
    let gate = g.minimum(gate, pos_limit);
    let up = g.maximum(up, neg_limit);
    let up = g.minimum(up, pos_limit);

    // glu = gate * sigmoid(gate * alpha)
    let alpha_c = g.constant_with_scalar(alpha, scalar_shape());
    let gate_scaled = g.multiplication(gate, alpha_c);
    let gate_sig = g.sigmoid(gate_scaled);
    let glu = g.multiplication(gate, gate_sig);

    // (up + 1) * glu
    let one = g.constant_with_scalar(1.0, scalar_shape());
    let up_plus_one = g.addition(up, one);
    let gated = g.multiplication(up_plus_one, glu);

    // down = gated @ d_w^T + bias → [1, 1, 1, hs]
    let out = g.matrix_multiplication(gated, d_w, false, true);
    let out = g.addition(out, d_b);

    // Clamp output to fp16 safe range (composed min/max)
    let fp16_max = g.constant_with_scalar(65504.0, scalar_shape());
    let fp16_min = g.constant_with_scalar(-65504.0, scalar_shape());
    let out = g.minimum(out, fp16_max);
    let _out = g.maximum(out, fp16_min);

    g.compile(NSQualityOfService::Default)
}

/// Build expert SwiGLU FFN on ANE via constant-weight conv1x1.
///
/// Uses the same spatial-padding approach as QKV projections: input padded
/// to `[hs, 1, DECODE_SPATIAL_WIDTH]`, conv1x1 with constant FP16 weights
/// baked into the compiled graph.
///
/// Unlike `build_expert_ffn_dynamic` (one graph, dynamic weight upload),
/// this builds a **separate graph per expert** with weights compiled in.
/// The caller must compile 32 × num_layers of these.
///
/// Trade-off: conv1x1 uses the ANE's native fast path (vs matmul fallback),
/// but each compiled graph holds ~47 MB of FP16 weights.
///
/// Placeholder (single):
///   0: input       [hs, 1, sw]    — activation (channel-major, spatial=64)
///
/// Output: [hs, 1, sw]
pub fn build_expert_ffn_conv(
    config: &GptOssConfig,
    gate_up_weight: &[f32], // [2*inter, hs] row-major
    gate_up_bias: &[f32],   // [2*inter]
    down_weight: &[f32],    // [hs, inter] row-major
    down_bias: &[f32],      // [hs]
) -> Result<Executable, ane::Error> {
    let hs = config.hidden_size;
    let inter = config.intermediate_size;
    let limit = config.swiglu_limit;
    let alpha: f32 = 1.702;
    let sw = DECODE_SPATIAL_WIDTH;

    let mut g = Graph::new();

    // Input: spatial-padded activation [hs, 1, sw=64]
    let input = g.placeholder(Shape::spatial(hs, 1, sw));

    // Gate-up: conv1x1 with constant weights [2*inter, hs, 1, 1]
    let gu_w = g.constant(gate_up_weight, Shape::spatial(2 * inter, 1, 1));
    let gu = g.convolution_2d_1x1(input, gu_w, None);
    let gu_b = g.constant(
        gate_up_bias,
        Shape {
            batch: 1,
            channels: 2 * inter,
            height: 1,
            width: 1,
        },
    );
    let gu = g.addition(gu, gu_b);

    // Split gate/up on channel dim: [2*inter, 1, sw] → two [inter, 1, sw]
    let gate = g.slice(gu, [0, 0, 0, 0], [1, inter, 1, sw]);
    let up = g.slice(gu, [0, inter, 0, 0], [1, inter, 1, sw]);

    // Clamp: gate max only, up both directions
    let pos_limit = g.constant_with_scalar(limit, scalar_shape());
    let neg_limit = g.constant_with_scalar(-limit, scalar_shape());
    let gate = g.minimum(gate, pos_limit);
    let up = g.maximum(up, neg_limit);
    let up = g.minimum(up, pos_limit);

    // glu = gate * sigmoid(gate * alpha)
    let alpha_c = g.constant_with_scalar(alpha, scalar_shape());
    let gate_scaled = g.multiplication(gate, alpha_c);
    let gate_sig = g.sigmoid(gate_scaled);
    let glu = g.multiplication(gate, gate_sig);

    // (up + 1) * glu → [inter, 1, sw]
    let one = g.constant_with_scalar(1.0, scalar_shape());
    let up_plus_one = g.addition(up, one);
    let gated = g.multiplication(up_plus_one, glu);

    // Down: conv1x1 with constant weights [hs, inter, 1, 1]
    let d_w = g.constant(down_weight, Shape::spatial(hs, 1, 1));
    let out = g.convolution_2d_1x1(gated, d_w, None);
    let d_b = g.constant(
        down_bias,
        Shape {
            batch: 1,
            channels: hs,
            height: 1,
            width: 1,
        },
    );
    let _out = g.addition(out, d_b);

    g.compile(NSQualityOfService::Default)
}

/// Build final RMSNorm (before LM head) on ANE.
pub fn build_final_norm(
    norm_weight: &[f32],
    config: &GptOssConfig,
) -> Result<Executable, ane::Error> {
    let hs = config.hidden_size;
    let sw = DECODE_SPATIAL_WIDTH;
    let mut g = Graph::new();
    let input = g.placeholder(Shape::spatial(hs, 1, sw));
    let _out = rms_norm(&mut g, input, norm_weight, hs, config.rms_norm_eps);
    g.compile(NSQualityOfService::Default)
}

/// Pre-compiled ANE executables for one transformer layer.
/// RMSNorm fused via layer_norm concat trick.
pub struct LayerExecutables {
    pub qkv: Executable,
    pub output_proj_router: Executable,
}

/// All compiled ANE executables for the model.
pub struct CompiledExecutables {
    pub layers: Box<[LayerExecutables]>,
    pub expert_ffn: Executable,
    pub final_norm: Executable,
    pub residual_add: Executable,
    pub lm_head_chunk: Executable,
    pub lm_head_chunk_size: usize,
}

/// Preferred LM head chunk size for ANE matmul.
/// Larger chunk sizes reduce per-token ANE launch overhead.
pub const LM_HEAD_CHUNK_SIZE_PREFERRED: usize = 8192;

/// Conservative LM head chunk fallback.
pub const LM_HEAD_CHUNK_SIZE_FALLBACK: usize = 4096;

/// Build LM head chunk matmul on ANE.
/// Input: normalized hidden [1,1,1,hs], weight chunk [1,1,chunk_size,hs].
/// Output: logits chunk [1,1,1,chunk_size].
pub fn build_lm_head_chunk(
    config: &GptOssConfig,
    chunk_size: usize,
) -> Result<Executable, ane::Error> {
    let hs = config.hidden_size;
    let mut g = Graph::new();
    let input = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: 1,
        width: hs,
    });
    let weight = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: chunk_size,
        width: hs,
    });
    let _out = g.matrix_multiplication(input, weight, false, true);
    g.compile(NSQualityOfService::Default)
}

#[cfg(test)]
mod tests {
    use super::super::config::GptOssConfig;
    use super::super::weights::{ExpertMxfp4, LayerWeights};
    use super::*;

    #[test]
    fn test_ane_rms_norm_768_compiles() {
        let hs = 768;
        let sw = DECODE_SPATIAL_WIDTH;
        let gamma = vec![1.0f32; hs];
        let mut g = Graph::new();
        let input = g.placeholder(Shape::spatial(hs, 1, sw));
        let _out = rms_norm(&mut g, input, &gamma, hs, 1e-5);
        let exec = g.compile(NSQualityOfService::Default);
        assert!(exec.is_ok(), "RMSNorm 768 compile failed: {:?}", exec.err());
    }

    #[test]
    fn test_ane_rms_norm_steps() {
        // Test RMSNorm step by step to find which op fails
        let hs = 768;
        let sw = DECODE_SPATIAL_WIDTH;
        let bs = scalar_shape();
        let gamma = vec![1.0f32; hs];

        // Step 1: x * x
        {
            let mut g = Graph::new();
            let input = g.placeholder(Shape::spatial(hs, 1, sw));
            let _out = g.multiplication(input, input);
            assert!(g.compile(NSQualityOfService::Default).is_ok(), "x*x failed");
        }

        // Step 2: reduce_sum
        {
            let mut g = Graph::new();
            let input = g.placeholder(Shape::spatial(hs, 1, sw));
            let sq = g.multiplication(input, input);
            let _out = g.reduce_sum(sq, 1);
            assert!(
                g.compile(NSQualityOfService::Default).is_ok(),
                "reduce_sum failed"
            );
        }

        // Step 3: * inv_dim
        {
            let mut g = Graph::new();
            let input = g.placeholder(Shape::spatial(hs, 1, sw));
            let sq = g.multiplication(input, input);
            let sum = g.reduce_sum(sq, 1);
            let inv_dim = g.constant_with_scalar(1.0 / hs as f32, bs);
            let _out = g.multiplication(sum, inv_dim);
            assert!(
                g.compile(NSQualityOfService::Default).is_ok(),
                "mean failed"
            );
        }

        // Step 4: + eps
        {
            let mut g = Graph::new();
            let input = g.placeholder(Shape::spatial(hs, 1, sw));
            let sq = g.multiplication(input, input);
            let sum = g.reduce_sum(sq, 1);
            let inv_dim = g.constant_with_scalar(1.0 / hs as f32, bs);
            let mean = g.multiplication(sum, inv_dim);
            let eps = g.constant_with_scalar(1e-5f32, bs);
            let _out = g.addition(mean, eps);
            assert!(
                g.compile(NSQualityOfService::Default).is_ok(),
                "add eps failed"
            );
        }

        // Step 5a: rsqrt
        {
            let mut g = Graph::new();
            let input = g.placeholder(Shape::spatial(hs, 1, sw));
            let sq = g.multiplication(input, input);
            let sum = g.reduce_sum(sq, 1);
            let inv_dim = g.constant_with_scalar(1.0 / hs as f32, bs);
            let mean = g.multiplication(sum, inv_dim);
            let eps = g.constant_with_scalar(1e-5f32, bs);
            let var_eps = g.addition(mean, eps);
            let _out = g.reciprocal_square_root(var_eps);
            let ok = g.compile(NSQualityOfService::Default).is_ok();
            eprintln!("  rsqrt: {}", if ok { "OK" } else { "FAILED" });
        }

        // Step 5b: power(-0.5) as alternative to rsqrt
        {
            let mut g = Graph::new();
            let input = g.placeholder(Shape::spatial(hs, 1, sw));
            let sq = g.multiplication(input, input);
            let sum = g.reduce_sum(sq, 1);
            let inv_dim = g.constant_with_scalar(1.0 / hs as f32, bs);
            let mean = g.multiplication(sum, inv_dim);
            let eps = g.constant_with_scalar(1e-5f32, bs);
            let var_eps = g.addition(mean, eps);
            let neg_half = g.constant_with_scalar(-0.5, bs);
            let _out = g.power(var_eps, neg_half);
            let ok = g.compile(NSQualityOfService::Default).is_ok();
            eprintln!("  power(-0.5): {}", if ok { "OK" } else { "FAILED" });
            if !ok {
                panic!("Neither rsqrt nor power(-0.5) works");
            }
        }

        // Step 6: x * rstd (broadcasting [1,1,1,sw] × [1,hs,1,sw])
        {
            let mut g = Graph::new();
            let input = g.placeholder(Shape::spatial(hs, 1, sw));
            let sq = g.multiplication(input, input);
            let sum = g.reduce_sum(sq, 1);
            let inv_dim = g.constant_with_scalar(1.0 / hs as f32, bs);
            let mean = g.multiplication(sum, inv_dim);
            let eps = g.constant_with_scalar(1e-5f32, bs);
            let var_eps = g.addition(mean, eps);
            let neg_half = g.constant_with_scalar(-0.5, bs);
            let rstd = g.power(var_eps, neg_half);
            let _out = g.multiplication(input, rstd);
            let ok = g.compile(NSQualityOfService::Default).is_ok();
            eprintln!("  x*rstd (power): {}", if ok { "OK" } else { "FAILED" });
            if !ok {
                panic!("x*rstd failed");
            }
        }

        // Step 7: * gamma
        {
            let mut g = Graph::new();
            let input = g.placeholder(Shape::spatial(hs, 1, sw));
            let sq = g.multiplication(input, input);
            let sum = g.reduce_sum(sq, 1);
            let inv_dim = g.constant_with_scalar(1.0 / hs as f32, bs);
            let mean = g.multiplication(sum, inv_dim);
            let eps = g.constant_with_scalar(1e-5f32, bs);
            let var_eps = g.addition(mean, eps);
            let neg_half = g.constant_with_scalar(-0.5, bs);
            let rstd = g.power(var_eps, neg_half);
            let normalized = g.multiplication(input, rstd);
            let gamma_c = g.constant(
                &gamma,
                Shape {
                    batch: 1,
                    channels: hs,
                    height: 1,
                    width: 1,
                },
            );
            let _out = g.multiplication(normalized, gamma_c);
            let ok = g.compile(NSQualityOfService::Default).is_ok();
            eprintln!("  x*gamma: {}", if ok { "OK" } else { "FAILED" });
            if !ok {
                panic!("x*gamma failed");
            }
        }

        eprintln!("All RMSNorm steps compile individually!");
    }

    #[test]
    #[ignore] // Diagnostic: validates fused output_proj+router numerics against CPU reference.
    fn test_fused_output_proj_router_matches_cpu_reference() {
        use ane::TensorData;

        let config: GptOssConfig = serde_json::from_str(
            r#"{
            "hidden_size": 2880, "num_hidden_layers": 1,
            "num_attention_heads": 64, "num_key_value_heads": 8,
            "head_dim": 64, "intermediate_size": 2880,
            "vocab_size": 1024, "num_local_experts": 32,
            "num_experts_per_tok": 4, "layer_types": ["full_attention"]
        }"#,
        )
        .unwrap();
        let hs = config.hidden_size;
        let qd = config.q_dim();
        let ne = config.num_local_experts;
        let sw = DECODE_SPATIAL_WIDTH;

        let make_data = |len: usize, scale: f32| -> Vec<f32> {
            (0..len)
                .map(|i| ((i as f32 * 0.013).sin() + (i as f32 * 0.007).cos()) * scale)
                .collect()
        };

        let lw = LayerWeights {
            input_layernorm_weight: vec![1.0f32; hs].into_boxed_slice(),
            post_attn_layernorm_weight: make_data(hs, 0.2)
                .into_iter()
                .map(|v| 1.0 + v)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            q_proj_weight: Vec::new().into_boxed_slice(),
            q_proj_bias: Vec::new().into_boxed_slice(),
            k_proj_weight: Vec::new().into_boxed_slice(),
            k_proj_bias: Vec::new().into_boxed_slice(),
            v_proj_weight: Vec::new().into_boxed_slice(),
            v_proj_bias: Vec::new().into_boxed_slice(),
            o_proj_weight: make_data(hs * qd, 0.03).into_boxed_slice(),
            o_proj_bias: make_data(hs, 0.01).into_boxed_slice(),
            sinks: Vec::new().into_boxed_slice(),
            router_weight: make_data(ne * hs, 0.02).into_boxed_slice(),
            router_bias: make_data(ne, 0.01).into_boxed_slice(),
            experts: Vec::<ExpertMxfp4>::new().into_boxed_slice(),
        };

        let exec = build_decode_output_proj_router(&lw, &config).expect("fused compile");
        let attn_td = TensorData::new(Shape::spatial(qd, 1, sw));
        let hidden_td = TensorData::new(Shape::spatial(hs, 1, sw));
        let hidden_out_td = TensorData::new(Shape::spatial(hs, 1, sw));
        let router_norm_td = TensorData::new(Shape::spatial(ne + hs, 1, sw));

        let attn_in = make_data(qd, 0.4);
        let hidden_in = make_data(hs, 0.6);
        {
            let mut s = attn_td.as_f32_slice_mut();
            s.fill(0.0);
            for i in 0..qd {
                s[i * sw] = attn_in[i];
            }
        }
        {
            let mut s = hidden_td.as_f32_slice_mut();
            s.fill(0.0);
            for i in 0..hs {
                s[i * sw] = hidden_in[i];
            }
        }

        exec.run(&[&attn_td, &hidden_td], &[&router_norm_td, &hidden_out_td])
            .expect("fused run");

        let hidden_out_slice = hidden_out_td.as_f32_slice();
        let router_norm_slice = router_norm_td.as_f32_slice();
        let mut hidden_out_ane = vec![0.0f32; hs];
        let mut norm_ane = vec![0.0f32; hs];
        let mut router_ane = vec![0.0f32; ne];
        for i in 0..hs {
            hidden_out_ane[i] = hidden_out_slice[i * sw];
            norm_ane[i] = router_norm_slice[(ne + i) * sw];
        }
        for i in 0..ne {
            router_ane[i] = router_norm_slice[i * sw];
        }

        let run_ref =
            |o_proj_in_out: bool, router_in_out: bool| -> (Vec<f32>, Vec<f32>, Vec<f32>) {
                let mut hidden_ref = vec![0.0f32; hs];
                for o in 0..hs {
                    let mut acc = lw.o_proj_bias[o];
                    for i in 0..qd {
                        let w = if o_proj_in_out {
                            lw.o_proj_weight[i * hs + o]
                        } else {
                            lw.o_proj_weight[o * qd + i]
                        };
                        acc += w * attn_in[i];
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
                for i in 0..hs {
                    norm_ref[i] = hidden_ref[i] * inv_rms * lw.post_attn_layernorm_weight[i];
                }

                let mut router_logits = vec![0.0f32; ne];
                for e in 0..ne {
                    let mut acc = lw.router_bias[e];
                    for i in 0..hs {
                        let w = if router_in_out {
                            lw.router_weight[i * ne + e]
                        } else {
                            lw.router_weight[e * hs + i]
                        };
                        acc += w * norm_ref[i];
                    }
                    router_logits[e] = acc;
                }
                let max_logit = router_logits
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut exp_sum = 0.0f32;
                let mut router_ref = vec![0.0f32; ne];
                for (i, &v) in router_logits.iter().enumerate() {
                    let p = (v - max_logit).exp();
                    router_ref[i] = p;
                    exp_sum += p;
                }
                for p in &mut router_ref {
                    *p /= exp_sum.max(1e-12);
                }
                (hidden_ref, norm_ref, router_ref)
            };

        let max_abs_diff = |a: &[f32], b: &[f32]| -> f32 {
            let mut max_diff = 0.0f32;
            for i in 0..a.len() {
                let d = (a[i] - b[i]).abs();
                if d > max_diff {
                    max_diff = d;
                }
            }
            max_diff
        };

        let (h_oo, n_oo, r_oo) = run_ref(false, false);
        let (h_io, n_io, r_io) = run_ref(true, false);
        let (h_oi, n_oi, r_oi) = run_ref(false, true);
        let (h_ii, n_ii, r_ii) = run_ref(true, true);
        let diff_oo = (
            max_abs_diff(&h_oo, &hidden_out_ane),
            max_abs_diff(&n_oo, &norm_ane),
            max_abs_diff(&r_oo, &router_ane),
        );
        let diff_io = (
            max_abs_diff(&h_io, &hidden_out_ane),
            max_abs_diff(&n_io, &norm_ane),
            max_abs_diff(&r_io, &router_ane),
        );
        let diff_oi = (
            max_abs_diff(&h_oi, &hidden_out_ane),
            max_abs_diff(&n_oi, &norm_ane),
            max_abs_diff(&r_oi, &router_ane),
        );
        let diff_ii = (
            max_abs_diff(&h_ii, &hidden_out_ane),
            max_abs_diff(&n_ii, &norm_ane),
            max_abs_diff(&r_ii, &router_ane),
        );
        let router_sum: f32 = router_ane.iter().sum();
        eprintln!(
            "layout diffs oo={:?} io={:?} oi={:?} ii={:?} router_sum={:.6}",
            diff_oo, diff_io, diff_oi, diff_ii, router_sum
        );

        let best = [diff_oo, diff_io, diff_oi, diff_ii]
            .into_iter()
            .min_by(|a, b| {
                (a.0 + a.1 + a.2)
                    .partial_cmp(&(b.0 + b.1 + b.2))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        assert!(best.0 < 6e-2, "hidden mismatch too large: {:.5e}", best.0);
        assert!(best.1 < 8e-2, "norm mismatch too large: {:.5e}", best.1);
        assert!(best.2 < 8e-2, "router mismatch too large: {:.5e}", best.2);
        assert!(
            (router_sum - 1.0).abs() < 2e-3,
            "router probs do not sum to 1: {router_sum:.6}"
        );
    }

    #[test]
    #[ignore] // instance_norm MIL op rejected by ANE compiler via private API
    fn test_ane_instance_norm_basic() {
        // Test if instance_norm compiles at all on ANE
        let sw = DECODE_SPATIAL_WIDTH;
        let mut g = Graph::new();
        // shape [1, 4, 1, 128]: 4 channels, 128 spatial width
        let input = g.placeholder(Shape {
            batch: 1,
            channels: 4,
            height: 1,
            width: 128,
        });
        let gamma = g.constant(
            &vec![1.0f32; 4],
            Shape {
                batch: 1,
                channels: 4,
                height: 1,
                width: 1,
            },
        );
        let _out = g.instance_norm(input, gamma, 1e-5);
        let exec = g.compile(NSQualityOfService::Default);
        assert!(
            exec.is_ok(),
            "instance_norm basic compile failed: {:?}",
            exec.err()
        );
    }

    #[test]
    #[ignore] // instance_norm MIL op rejected by ANE compiler via private API
    fn test_ane_instance_norm_for_rmsnorm() {
        // Test the transpose + instance_norm pattern for RMSNorm
        let sw = DECODE_SPATIAL_WIDTH; // 64
        let hs = 128; // small test dim
        let mut g = Graph::new();
        let input = g.placeholder(Shape::spatial(hs, 1, sw)); // [1, 128, 1, 64]
                                                              // concat [x, -x]
        let neg_one = g.constant_with_scalar(-1.0, scalar_shape());
        let neg = g.multiplication(input, neg_one);
        let doubled = g.concat(&[input, neg], 1); // [1, 256, 1, 64]
                                                  // transpose: channels ↔ width
        let transposed = g.transpose(doubled, [0, 3, 2, 1]); // [1, 64, 1, 256]
        let gamma = g.constant(
            &vec![1.0f32; sw],
            Shape {
                batch: 1,
                channels: sw,
                height: 1,
                width: 1,
            },
        );
        let normed = g.instance_norm(transposed, gamma, 1e-5);
        let back = g.transpose(normed, [0, 3, 2, 1]); // [1, 256, 1, 64]
        let _result = g.slice(back, [0, 0, 0, 0], [1, hs, 1, sw]);
        let exec = g.compile(NSQualityOfService::Default);
        assert!(
            exec.is_ok(),
            "instance_norm for RMSNorm compile failed: {:?}",
            exec.err()
        );
    }

    #[test]
    fn test_ane_rms_norm_2880_compiles() {
        let hs = 2880;
        let sw = DECODE_SPATIAL_WIDTH;
        let gamma = vec![1.0f32; hs];
        let mut g = Graph::new();
        let input = g.placeholder(Shape::spatial(hs, 1, sw));
        let _out = rms_norm(&mut g, input, &gamma, hs, 1e-5);
        let exec = g.compile(NSQualityOfService::Default);
        assert!(
            exec.is_ok(),
            "RMSNorm 2880 compile failed: {:?}",
            exec.err()
        );
    }

    #[test]
    fn test_ane_reduce_sum_2880() {
        // Test if reduce_sum over 2880 channels works standalone
        let hs = 2880;
        let sw = DECODE_SPATIAL_WIDTH;
        let mut g = Graph::new();
        let input = g.placeholder(Shape::spatial(hs, 1, sw));
        let _out = g.reduce_sum(input, 1);
        let exec = g.compile(NSQualityOfService::Default);
        assert!(
            exec.is_ok(),
            "reduce_sum 2880 compile failed: {:?}",
            exec.err()
        );
    }

    #[test]
    fn test_ane_small_conv_compiles() {
        // Small conv like GPT-2: 768 → 2304
        let sw = DECODE_SPATIAL_WIDTH;
        let w_data = vec![0.01f32; 2304 * 768];
        let mut g = Graph::new();
        let input = g.placeholder(Shape::spatial(768, 1, sw));
        let w = g.constant(&w_data, Shape::spatial(2304, 1, 1));
        let _out = g.convolution_2d_1x1(input, w, None);
        let exec = g.compile(NSQualityOfService::Default);
        assert!(exec.is_ok(), "Small conv compile failed: {:?}", exec.err());
    }

    #[test]
    fn test_ane_large_q_proj_compiles() {
        // GPT-OSS Q projection: 2880 → 4096
        let sw = DECODE_SPATIAL_WIDTH;
        let w_data = vec![0.01f32; 4096 * 2880];
        let mut g = Graph::new();
        let input = g.placeholder(Shape::spatial(2880, 1, sw));
        let w = g.constant(&w_data, Shape::spatial(4096, 1, 1));
        let _out = g.convolution_2d_1x1(input, w, None);
        let exec = g.compile(NSQualityOfService::Default);
        assert!(exec.is_ok(), "Q proj compile failed: {:?}", exec.err());
    }

    #[test]
    fn test_ane_qkv_separate_compiles() {
        // Three separate convolutions like our QKV builder
        let hs = 2880;
        let qd = 4096;
        let kvd = 512;
        let sw = DECODE_SPATIAL_WIDTH;

        let gamma = vec![1.0f32; hs];
        let q_w = vec![0.01f32; qd * hs];
        let q_b = vec![0.0f32; qd];
        let k_w = vec![0.01f32; kvd * hs];
        let k_b = vec![0.0f32; kvd];
        let v_w = vec![0.01f32; kvd * hs];
        let v_b = vec![0.0f32; kvd];

        let mut g = Graph::new();
        let input = g.placeholder(Shape::spatial(hs, 1, sw));
        let norm = rms_norm(&mut g, input, &gamma, hs, 1e-5);

        let qw = g.constant(&q_w, Shape::spatial(qd, 1, 1));
        let q = g.convolution_2d_1x1(norm, qw, None);
        let qb = g.constant(
            &q_b,
            Shape {
                batch: 1,
                channels: qd,
                height: 1,
                width: 1,
            },
        );
        let q = g.addition(q, qb);

        let kw = g.constant(&k_w, Shape::spatial(kvd, 1, 1));
        let k = g.convolution_2d_1x1(norm, kw, None);
        let kb = g.constant(
            &k_b,
            Shape {
                batch: 1,
                channels: kvd,
                height: 1,
                width: 1,
            },
        );
        let k = g.addition(k, kb);

        let vw = g.constant(&v_w, Shape::spatial(kvd, 1, 1));
        let v = g.convolution_2d_1x1(norm, vw, None);
        let vb = g.constant(
            &v_b,
            Shape {
                batch: 1,
                channels: kvd,
                height: 1,
                width: 1,
            },
        );
        let v = g.addition(v, vb);

        let _out = g.concat(&[q, k, v], 1);
        let exec = g.compile(NSQualityOfService::Default);
        assert!(exec.is_ok(), "QKV compile failed: {:?}", exec.err());
    }

    #[test]
    fn test_ane_matmul_dynamic_weight_compiles() {
        // Test: matmul with a placeholder weight tensor
        let hs = 2880;
        let out = 5760;
        let mut g = Graph::new();
        let input = g.placeholder(Shape {
            batch: 1,
            channels: 1,
            height: 1,
            width: hs,
        });
        let weight = g.placeholder(Shape {
            batch: 1,
            channels: 1,
            height: out,
            width: hs,
        });
        let _out = g.matrix_multiplication(input, weight, false, true);
        let exec = g.compile(NSQualityOfService::Default);
        assert!(
            exec.is_ok(),
            "Matmul with dynamic weight failed: {:?}",
            exec.err()
        );
    }

    #[test]
    fn test_ane_expert_ffn_compiles() {
        let config: super::super::config::GptOssConfig = serde_json::from_str(
            r#"{
            "hidden_size": 2880, "num_hidden_layers": 2,
            "num_attention_heads": 64, "num_key_value_heads": 8,
            "head_dim": 64, "intermediate_size": 2880,
            "vocab_size": 201088, "num_local_experts": 32,
            "num_experts_per_tok": 4, "layer_types": ["sliding_attention", "full_attention"]
        }"#,
        )
        .unwrap();
        let exec = super::build_expert_ffn_dynamic(&config);
        assert!(exec.is_ok(), "Expert FFN compile failed: {:?}", exec.err());
    }

    #[test]
    #[ignore] // Diagnostic test — uses old reshape approach, kept for reference
    fn test_ane_expert_ffn_bisect() {
        // Bisect which op causes the expert FFN compilation to fail
        let hs = 2880;
        let inter = 2880;
        let sw = DECODE_SPATIAL_WIDTH;

        // Step 1: matmul + reshape to spatial
        {
            let mut g = Graph::new();
            let input = g.placeholder(Shape {
                batch: 1,
                channels: 1,
                height: 1,
                width: hs,
            });
            let gu_w = g.placeholder(Shape {
                batch: 1,
                channels: 1,
                height: 2 * inter,
                width: hs,
            });
            let gu = g.matrix_multiplication(input, gu_w, false, true);
            let _out = g.reshape(gu, Shape::spatial(2 * inter, 1, sw));
            let ok = g.compile(NSQualityOfService::Default).is_ok();
            eprintln!("  matmul + reshape: {}", if ok { "OK" } else { "FAILED" });
            assert!(ok, "matmul + reshape failed");
        }

        // Step 2: + slice
        {
            let mut g = Graph::new();
            let input = g.placeholder(Shape {
                batch: 1,
                channels: 1,
                height: 1,
                width: hs,
            });
            let gu_w = g.placeholder(Shape {
                batch: 1,
                channels: 1,
                height: 2 * inter,
                width: hs,
            });
            let gu = g.matrix_multiplication(input, gu_w, false, true);
            let gu = g.reshape(gu, Shape::spatial(2 * inter, 1, sw));
            let _gate = g.slice(gu, [0, 0, 0, 0], [1, inter, 1, sw]);
            let _up = g.slice(gu, [0, inter, 0, 0], [1, inter, 1, sw]);
            let ok = g.compile(NSQualityOfService::Default).is_ok();
            eprintln!("  + slice: {}", if ok { "OK" } else { "FAILED" });
            assert!(ok, "slice failed");
        }

        // Step 3: + clamp + sigmoid + multiply (activation)
        {
            let mut g = Graph::new();
            let input = g.placeholder(Shape {
                batch: 1,
                channels: 1,
                height: 1,
                width: hs,
            });
            let gu_w = g.placeholder(Shape {
                batch: 1,
                channels: 1,
                height: 2 * inter,
                width: hs,
            });
            let gu = g.matrix_multiplication(input, gu_w, false, true);
            let gu = g.reshape(gu, Shape::spatial(2 * inter, 1, sw));
            let gate = g.slice(gu, [0, 0, 0, 0], [1, inter, 1, sw]);
            let up = g.slice(gu, [0, inter, 0, 0], [1, inter, 1, sw]);
            let pos_limit = g.constant_with_scalar(7.0, scalar_shape());
            let neg_limit = g.constant_with_scalar(-7.0, scalar_shape());
            let gate = g.minimum(gate, pos_limit);
            let up = g.maximum(up, neg_limit);
            let up = g.minimum(up, pos_limit);
            let alpha_c = g.constant_with_scalar(1.702, scalar_shape());
            let gs = g.multiplication(gate, alpha_c);
            let sig = g.sigmoid(gs);
            let glu = g.multiplication(gate, sig);
            let one = g.constant_with_scalar(1.0, scalar_shape());
            let up1 = g.addition(up, one);
            let _gated = g.multiplication(up1, glu);
            let ok = g.compile(NSQualityOfService::Default).is_ok();
            eprintln!("  + activation: {}", if ok { "OK" } else { "FAILED" });
            assert!(ok, "activation failed");
        }

        // Step 4: + reshape back + down matmul
        {
            let mut g = Graph::new();
            let input = g.placeholder(Shape {
                batch: 1,
                channels: 1,
                height: 1,
                width: hs,
            });
            let gu_w = g.placeholder(Shape {
                batch: 1,
                channels: 1,
                height: 2 * inter,
                width: hs,
            });
            let d_w = g.placeholder(Shape {
                batch: 1,
                channels: 1,
                height: hs,
                width: inter,
            });
            let gu = g.matrix_multiplication(input, gu_w, false, true);
            let gu = g.reshape(gu, Shape::spatial(2 * inter, 1, sw));
            let gate = g.slice(gu, [0, 0, 0, 0], [1, inter, 1, sw]);
            let up = g.slice(gu, [0, inter, 0, 0], [1, inter, 1, sw]);
            let pos_limit = g.constant_with_scalar(7.0, scalar_shape());
            let neg_limit = g.constant_with_scalar(-7.0, scalar_shape());
            let gate = g.minimum(gate, pos_limit);
            let up = g.maximum(up, neg_limit);
            let up = g.minimum(up, pos_limit);
            let alpha_c = g.constant_with_scalar(1.702, scalar_shape());
            let gs = g.multiplication(gate, alpha_c);
            let sig = g.sigmoid(gs);
            let glu = g.multiplication(gate, sig);
            let one = g.constant_with_scalar(1.0, scalar_shape());
            let up1 = g.addition(up, one);
            let gated = g.multiplication(up1, glu);
            let gated_row = g.reshape(
                gated,
                Shape {
                    batch: 1,
                    channels: 1,
                    height: 1,
                    width: inter,
                },
            );
            let _out = g.matrix_multiplication(gated_row, d_w, false, true);
            let ok = g.compile(NSQualityOfService::Default).is_ok();
            eprintln!("  + down matmul: {}", if ok { "OK" } else { "FAILED" });
        }

        // Step 5: + reshape + bias add
        {
            let mut g = Graph::new();
            let input = g.placeholder(Shape {
                batch: 1,
                channels: 1,
                height: 1,
                width: hs,
            });
            let gu_w = g.placeholder(Shape {
                batch: 1,
                channels: 1,
                height: 2 * inter,
                width: hs,
            });
            let gu_b = g.placeholder(Shape::spatial(2 * inter, 1, sw));
            let d_w = g.placeholder(Shape {
                batch: 1,
                channels: 1,
                height: hs,
                width: inter,
            });
            let d_b = g.placeholder(Shape::spatial(hs, 1, sw));
            let gu = g.matrix_multiplication(input, gu_w, false, true);
            let gu = g.reshape(gu, Shape::spatial(2 * inter, 1, sw));
            let gu = g.addition(gu, gu_b);
            let gate = g.slice(gu, [0, 0, 0, 0], [1, inter, 1, sw]);
            let up = g.slice(gu, [0, inter, 0, 0], [1, inter, 1, sw]);
            let pos_limit = g.constant_with_scalar(7.0, scalar_shape());
            let neg_limit = g.constant_with_scalar(-7.0, scalar_shape());
            let gate = g.minimum(gate, pos_limit);
            let up = g.maximum(up, neg_limit);
            let up = g.minimum(up, pos_limit);
            let alpha_c = g.constant_with_scalar(1.702, scalar_shape());
            let gs = g.multiplication(gate, alpha_c);
            let sig = g.sigmoid(gs);
            let glu = g.multiplication(gate, sig);
            let one = g.constant_with_scalar(1.0, scalar_shape());
            let up1 = g.addition(up, one);
            let gated = g.multiplication(up1, glu);
            let gated_row = g.reshape(
                gated,
                Shape {
                    batch: 1,
                    channels: 1,
                    height: 1,
                    width: inter,
                },
            );
            let out = g.matrix_multiplication(gated_row, d_w, false, true);
            let out = g.reshape(out, Shape::spatial(hs, 1, sw));
            let _out = g.addition(out, d_b);
            let ok = g.compile(NSQualityOfService::Default).is_ok();
            eprintln!("  + bias add: {}", if ok { "OK" } else { "FAILED" });
        }
    }

    #[test]
    fn test_ane_dynamic_conv_compiles_with_patch() {
        // Patched ANE crate exempts dynamic conv weight placeholders from spatial width check.
        let hs = 2880;
        let inter = 2880;
        let sw = DECODE_SPATIAL_WIDTH;

        let mut g = Graph::new();
        let input = g.placeholder(Shape::spatial(hs, 1, sw));
        let gu_w = g.placeholder(Shape {
            batch: 2 * inter,
            channels: hs,
            height: 1,
            width: 1,
        });
        let _out = g.convolution_2d_1x1_dynamic(input, gu_w);
        let exec = g.compile(NSQualityOfService::Default);
        assert!(
            exec.is_ok(),
            "Dynamic conv should compile: {:?}",
            exec.err()
        );
    }

    #[test]
    fn test_ane_expert_ffn_conv_compiles() {
        // Constant-weight conv1x1 expert FFN — same approach as QKV projections.
        let config: super::super::config::GptOssConfig = serde_json::from_str(
            r#"{
            "hidden_size": 2880, "num_hidden_layers": 2,
            "num_attention_heads": 64, "num_key_value_heads": 8,
            "head_dim": 64, "intermediate_size": 2880,
            "vocab_size": 201088, "num_local_experts": 32,
            "num_experts_per_tok": 4, "layer_types": ["sliding_attention", "full_attention"]
        }"#,
        )
        .unwrap();
        let hs = config.hidden_size;
        let inter = config.intermediate_size;
        let gate_up_w = vec![0.01f32; 2 * inter * hs];
        let gate_up_b = vec![0.0f32; 2 * inter];
        let down_w = vec![0.01f32; hs * inter];
        let down_b = vec![0.0f32; hs];
        let exec = super::build_expert_ffn_conv(&config, &gate_up_w, &gate_up_b, &down_w, &down_b);
        assert!(
            exec.is_ok(),
            "Expert FFN conv compile failed: {:?}",
            exec.err()
        );
    }

    #[test]
    #[ignore] // ANE compiler rejects dynamic conv with kernel > 1x1.
    fn test_ane_dyn_conv_channel_spatial_basic() {
        // Dynamic conv2d with kernel (1, 64) — the channel→spatial trick.
        // RESULT: ANE hardware only supports dynamic weights for 1x1 conv.
        // Kept as documentation of the failed approach.
        let s = 64;
        let in_channels = 45; // 2880 / 64
        let out_channels = 128;

        let desc = Convolution2dDescriptor::default();
        let mut g = Graph::new();
        let input = g.placeholder(Shape {
            batch: 1,
            channels: in_channels,
            height: 1,
            width: s,
        });
        let weight = g.placeholder(Shape {
            batch: out_channels,
            channels: in_channels,
            height: 1,
            width: s,
        });
        let _out = g.convolution_2d_dynamic(input, weight, &desc);
        let exec = g.compile(NSQualityOfService::Default);
        assert!(
            exec.is_ok(),
            "Dynamic conv (1,64) compile failed: {:?}",
            exec.err()
        );
    }

    #[test]
    fn test_ane_topk_compiles_and_runs() {
        use ane::TensorData;
        let ne = 32;
        let sw = DECODE_SPATIAL_WIDTH;
        let mut g = Graph::new();
        let input = g.placeholder(Shape::spatial(ne, 1, sw));
        let val_shape = Shape::spatial(4, 1, sw);
        let idx_shape = Shape::spatial(4, 1, sw);
        let (_vals, _idxs) = g.topk(input, 4, 1, false, val_shape, idx_shape);
        let exec = g.compile(NSQualityOfService::Default);
        eprintln!(
            "  topk compile: {}",
            if exec.is_ok() { "OK" } else { "FAILED" }
        );
        if let Ok(exec) = exec {
            let input_td = TensorData::new(Shape::spatial(ne, 1, sw));
            let val_td = TensorData::new(val_shape);
            {
                let mut surf = input_td.as_f32_slice_mut();
                for i in 0..ne {
                    surf[i * sw] = (i as f32) * 0.1;
                }
            }
            let result = exec.run(&[&input_td], &[&val_td]);
            eprintln!(
                "  topk run: {}",
                if result.is_ok() { "OK" } else { "FAILED" }
            );
        }
    }

    #[test]
    fn test_ane_rope_fused_compiles() {
        // Test: can we fuse RoPE into the QKV graph by reordering Q/K output
        // to separate first/second halves, then doing elementwise cos/sin multiply?
        let hs = 2880;
        let qd = 4096; // 64 heads × 64 dims
        let half_qd = qd / 2; // 2048
        let sw = DECODE_SPATIAL_WIDTH;

        let mut g = Graph::new();
        let input = g.placeholder(Shape::spatial(hs, 1, sw));

        // Simulate QKV projection output: just use input projection for test
        let w = g.constant(&vec![0.01f32; qd * hs], Shape::spatial(qd, 1, 1));
        let q = g.convolution_2d_1x1(input, w, None);

        // Split Q into first-half dims and second-half dims (contiguous after weight reorder)
        let q_first = g.slice(q, [0, 0, 0, 0], [1, half_qd, 1, sw]);
        let q_second = g.slice(q, [0, half_qd, 0, 0], [1, half_qd, 1, sw]);

        // cos/sin as placeholders [half_qd, 1, sw] — updated per position
        let cos = g.placeholder(Shape::spatial(half_qd, 1, sw));
        let sin = g.placeholder(Shape::spatial(half_qd, 1, sw));

        // RoPE: q_first_rope = q_first * cos - q_second * sin
        //        q_second_rope = q_first * sin + q_second * cos
        let fc = g.multiplication(q_first, cos);
        let ss = g.multiplication(q_second, sin);
        let q_first_rope = g.subtraction(fc, ss);

        let fs = g.multiplication(q_first, sin);
        let sc = g.multiplication(q_second, cos);
        let q_second_rope = g.addition(fs, sc);

        // Concat back
        let _q_rope = g.concat(&[q_first_rope, q_second_rope], 1);

        let exec = g.compile(NSQualityOfService::Default);
        assert!(exec.is_ok(), "Fused RoPE compile failed: {:?}", exec.err());
        eprintln!("  Fused RoPE: COMPILES on ANE!");
    }
}
