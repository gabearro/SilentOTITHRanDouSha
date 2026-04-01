use ane::{Executable, Graph, NSQualityOfService, Shape};

use super::config::Gpt2Config;
use super::weights::LayerWeights;

pub struct PrefillLayer {
    pub attention: Executable,
    pub feed_forward: Executable,
}

pub struct DecodeLayer {
    pub attention: Executable,
    pub feed_forward: Executable,
}

pub struct CompiledExecutables {
    pub prefill: Box<[PrefillLayer]>,
    pub decode: Box<[DecodeLayer]>,
    pub lm_head: Executable,
}

pub const DECODE_SPATIAL_WIDTH: usize = 64;

fn scalar_shape() -> Shape {
    Shape {
        batch: 1,
        channels: 1,
        height: 1,
        width: 1,
    }
}

fn layer_norm(
    graph: &mut Graph,
    input: ane::Tensor,
    gamma: &[f32],
    beta: &[f32],
    embedding_dim: usize,
    epsilon: f64,
) -> ane::Tensor {
    let bs = scalar_shape();
    let inverse_dim = graph.constant_with_scalar(1.0 / embedding_dim as f32, bs);
    let epsilon_constant = graph.constant_with_scalar(epsilon as f32, bs);
    let neg_half = graph.constant_with_scalar(-0.5, bs);
    let neg_one = graph.constant_with_scalar(-1.0, bs);
    let gamma_c = graph.constant(
        gamma,
        Shape {
            batch: 1,
            channels: embedding_dim,
            height: 1,
            width: 1,
        },
    );
    let beta_c = graph.constant(
        beta,
        Shape {
            batch: 1,
            channels: embedding_dim,
            height: 1,
            width: 1,
        },
    );

    let channel_sum = graph.reduce_sum(input, 1);
    let mean = graph.multiplication(channel_sum, inverse_dim);
    let negative_mean = graph.multiplication(mean, neg_one);
    let centered = graph.addition(input, negative_mean);
    let squared = graph.multiplication(centered, centered);
    let variance_sum = graph.reduce_sum(squared, 1);
    let variance = graph.multiplication(variance_sum, inverse_dim);
    let variance_plus_eps = graph.addition(variance, epsilon_constant);
    let rstd = graph.power(variance_plus_eps, neg_half);
    let normalized = graph.multiplication(centered, rstd);
    let scaled = graph.multiplication(normalized, gamma_c);
    graph.addition(scaled, beta_c)
}

fn gelu(graph: &mut Graph, input: ane::Tensor) -> ane::Tensor {
    let bs = scalar_shape();
    let half_c = graph.constant_with_scalar(0.5, bs);
    let one_c = graph.constant_with_scalar(1.0, bs);
    let coeff = graph.constant_with_scalar(0.044715, bs);
    let sqrt_2_pi = graph.constant_with_scalar(0.7978845608028654, bs);

    let sq = graph.multiplication(input, input);
    let cube = graph.multiplication(sq, input);
    let scaled_cube = graph.multiplication(coeff, cube);
    let inner = graph.addition(input, scaled_cube);
    let tanh_arg = graph.multiplication(sqrt_2_pi, inner);
    let tanh_val = graph.tanh(tanh_arg);
    let one_plus = graph.addition(one_c, tanh_val);
    let half_in = graph.multiplication(half_c, input);
    graph.multiplication(half_in, one_plus)
}

fn causal_mask(sequence_length: usize) -> Box<[f32]> {
    (0..sequence_length * sequence_length)
        .map(|i| {
            if (i % sequence_length) <= (i / sequence_length) {
                0.0
            } else {
                -65504.0
            }
        })
        .collect()
}

fn attention_body(
    graph: &mut Graph,
    normalized: ane::Tensor,
    lw: &LayerWeights,
    config: &Gpt2Config,
    query_seq: usize,
    key_seq: usize,
    key_source: Option<ane::Tensor>,
    value_source: Option<ane::Tensor>,
    mask_tensor: Option<ane::Tensor>,
) -> (ane::Tensor, ane::Tensor, ane::Tensor) {
    let ed = config.n_embd;
    let nh = config.n_head;
    let hs = config.head_size();

    let qkv_w = graph.constant(&lw.qkv_weight, Shape::spatial(3 * ed, 1, 1));
    let qkv = graph.convolution_2d_1x1(normalized, qkv_w, None);
    let qkv_b = graph.constant(
        &lw.qkv_bias,
        Shape {
            batch: 1,
            channels: 3 * ed,
            height: 1,
            width: 1,
        },
    );
    let qkv = graph.addition(qkv, qkv_b);

    let q_flat = graph.slice(qkv, [0, 0, 0, 0], [1, ed, 1, query_seq]);
    let k_flat = graph.slice(qkv, [0, ed, 0, 0], [1, ed, 1, query_seq]);
    let v_flat = graph.slice(qkv, [0, 2 * ed, 0, 0], [1, ed, 1, query_seq]);

    let hw = [0, 1, 3, 2];
    let q = graph.reshape(
        q_flat,
        Shape {
            batch: 1,
            channels: nh,
            height: hs,
            width: query_seq,
        },
    );
    let q = graph.transpose(q, hw);

    let (k, v, attn_ks) = match (key_source, value_source) {
        (Some(kc), Some(vc)) => {
            let k = graph.reshape(
                kc,
                Shape {
                    batch: 1,
                    channels: nh,
                    height: hs,
                    width: key_seq,
                },
            );
            let k = graph.transpose(k, hw);
            let v = graph.reshape(
                vc,
                Shape {
                    batch: 1,
                    channels: nh,
                    height: hs,
                    width: key_seq,
                },
            );
            let v = graph.transpose(v, hw);
            (k, v, key_seq)
        }
        _ => {
            let k = graph.reshape(
                k_flat,
                Shape {
                    batch: 1,
                    channels: nh,
                    height: hs,
                    width: query_seq,
                },
            );
            let k = graph.transpose(k, hw);
            let v = graph.reshape(
                v_flat,
                Shape {
                    batch: 1,
                    channels: nh,
                    height: hs,
                    width: query_seq,
                },
            );
            let v = graph.transpose(v, hw);
            (k, v, query_seq)
        }
    };

    let scale = graph.constant_with_scalar(1.0 / (hs as f32).sqrt(), scalar_shape());
    let scores = graph.matrix_multiplication(q, k, false, true);
    let scores = graph.multiplication(scores, scale);

    let scores = match mask_tensor {
        Some(m) => graph.addition(scores, m),
        None => {
            let m = graph.constant(
                &causal_mask(query_seq),
                Shape {
                    batch: 1,
                    channels: 1,
                    height: query_seq,
                    width: attn_ks,
                },
            );
            graph.addition(scores, m)
        }
    };

    let probs = graph.soft_max(scores, -1);
    let attn = graph.matrix_multiplication(probs, v, false, false);
    let attn = graph.transpose(attn, hw);
    let attn = graph.reshape(attn, Shape::spatial(ed, 1, query_seq));

    let proj_w = graph.constant(&lw.attn_proj_weight, Shape::spatial(ed, 1, 1));
    let proj = graph.convolution_2d_1x1(attn, proj_w, None);
    let proj_b = graph.constant(
        &lw.attn_proj_bias,
        Shape {
            batch: 1,
            channels: ed,
            height: 1,
            width: 1,
        },
    );
    let o_proj = graph.addition(proj, proj_b);

    (o_proj, k_flat, v_flat)
}

fn ffn_body(
    graph: &mut Graph,
    input: ane::Tensor,
    lw: &LayerWeights,
    config: &Gpt2Config,
) -> ane::Tensor {
    let ed = config.n_embd;
    let norm = layer_norm(
        graph,
        input,
        &lw.ln2_weight,
        &lw.ln2_bias,
        ed,
        config.layer_norm_epsilon,
    );

    let fc_w = graph.constant(&lw.fc_weight, Shape::spatial(4 * ed, 1, 1));
    let h = graph.convolution_2d_1x1(norm, fc_w, None);
    let fc_b = graph.constant(
        &lw.fc_bias,
        Shape {
            batch: 1,
            channels: 4 * ed,
            height: 1,
            width: 1,
        },
    );
    let h = graph.addition(h, fc_b);
    let h = gelu(graph, h);

    let proj_w = graph.constant(&lw.fc_proj_weight, Shape::spatial(ed, 1, 1));
    let proj = graph.convolution_2d_1x1(h, proj_w, None);
    let proj_b = graph.constant(
        &lw.fc_proj_bias,
        Shape {
            batch: 1,
            channels: ed,
            height: 1,
            width: 1,
        },
    );
    let out = graph.addition(proj, proj_b);
    graph.addition(out, input)
}

pub fn build_prefill_attention(
    lw: &LayerWeights,
    config: &Gpt2Config,
    seq_len: usize,
) -> Result<Executable, ane::Error> {
    let ed = config.n_embd;
    let mut g = Graph::new();
    let input = g.placeholder(Shape::spatial(ed, 1, seq_len));
    let norm = layer_norm(
        &mut g,
        input,
        &lw.ln1_weight,
        &lw.ln1_bias,
        ed,
        config.layer_norm_epsilon,
    );
    let (o_proj, k, v) =
        attention_body(&mut g, norm, lw, config, seq_len, seq_len, None, None, None);
    let res = g.addition(o_proj, input);
    let _out = g.concat(&[res, k, v], 1);
    g.compile(NSQualityOfService::Default)
}

pub fn build_prefill_feed_forward(
    lw: &LayerWeights,
    config: &Gpt2Config,
    seq_len: usize,
) -> Result<Executable, ane::Error> {
    let mut g = Graph::new();
    let input = g.placeholder(Shape::spatial(config.n_embd, 1, seq_len));
    let _out = ffn_body(&mut g, input, lw, config);
    g.compile(NSQualityOfService::Default)
}

pub fn build_decode_attention(
    lw: &LayerWeights,
    config: &Gpt2Config,
    max_seq: usize,
) -> Result<Executable, ane::Error> {
    let ed = config.n_embd;
    let mut g = Graph::new();
    let x = g.placeholder(Shape::spatial(ed, 1, DECODE_SPATIAL_WIDTH));
    let kc = g.placeholder(Shape::spatial(ed, 1, max_seq));
    let vc = g.placeholder(Shape::spatial(ed, 1, max_seq));
    let mask = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: DECODE_SPATIAL_WIDTH,
        width: max_seq,
    });

    let norm = layer_norm(
        &mut g,
        x,
        &lw.ln1_weight,
        &lw.ln1_bias,
        ed,
        config.layer_norm_epsilon,
    );
    let (o_proj, k_new, v_new) = attention_body(
        &mut g,
        norm,
        lw,
        config,
        DECODE_SPATIAL_WIDTH,
        max_seq,
        Some(kc),
        Some(vc),
        Some(mask),
    );
    let res = g.addition(o_proj, x);
    let _out = g.concat(&[res, k_new, v_new], 1);
    g.compile(NSQualityOfService::Default)
}

pub fn build_decode_feed_forward(
    lw: &LayerWeights,
    config: &Gpt2Config,
) -> Result<Executable, ane::Error> {
    let mut g = Graph::new();
    let input = g.placeholder(Shape::spatial(config.n_embd, 1, DECODE_SPATIAL_WIDTH));
    let _out = ffn_body(&mut g, input, lw, config);
    g.compile(NSQualityOfService::Default)
}

/// QKV projection only: LayerNorm → 1×1 conv → output Q‖K‖V.
/// For private attention: ANE handles the linear projection, CPU handles the Beaver protocol.
pub fn build_decode_qkv(lw: &LayerWeights, config: &Gpt2Config) -> Result<Executable, ane::Error> {
    let ed = config.n_embd;
    let mut g = Graph::new();
    let input = g.placeholder(Shape::spatial(ed, 1, DECODE_SPATIAL_WIDTH));

    let norm = layer_norm(
        &mut g,
        input,
        &lw.ln1_weight,
        &lw.ln1_bias,
        ed,
        config.layer_norm_epsilon,
    );

    let qkv_w = g.constant(&lw.qkv_weight, Shape::spatial(3 * ed, 1, 1));
    let qkv = g.convolution_2d_1x1(norm, qkv_w, None);
    let qkv_b = g.constant(
        &lw.qkv_bias,
        Shape {
            batch: 1,
            channels: 3 * ed,
            height: 1,
            width: 1,
        },
    );
    let _out = g.addition(qkv, qkv_b);

    g.compile(NSQualityOfService::Default)
}

/// QKV projection linear-only kernel: 1x1 conv + bias (no LayerNorm).
///
/// Intended for private flow where LayerNorm is computed in MPC and only
/// linear projections are offloaded to ANE.
pub fn build_decode_qkv_linear(
    lw: &LayerWeights,
    config: &Gpt2Config,
) -> Result<Executable, ane::Error> {
    let ed = config.n_embd;
    let mut g = Graph::new();
    let input = g.placeholder(Shape::spatial(ed, 1, DECODE_SPATIAL_WIDTH));

    let qkv_w = g.constant(&lw.qkv_weight, Shape::spatial(3 * ed, 1, 1));
    let qkv = g.convolution_2d_1x1(input, qkv_w, None);
    let qkv_b = g.constant(
        &lw.qkv_bias,
        Shape {
            batch: 1,
            channels: 3 * ed,
            height: 1,
            width: 1,
        },
    );
    let _out = g.addition(qkv, qkv_b);

    g.compile(NSQualityOfService::Default)
}

/// Output projection only: 1×1 conv on attention output.
/// For private attention: takes reconstructed cleartext attention output.
pub fn build_decode_output_proj(
    lw: &LayerWeights,
    config: &Gpt2Config,
) -> Result<Executable, ane::Error> {
    let ed = config.n_embd;
    let mut g = Graph::new();
    let input = g.placeholder(Shape::spatial(ed, 1, DECODE_SPATIAL_WIDTH));

    let proj_w = g.constant(&lw.attn_proj_weight, Shape::spatial(ed, 1, 1));
    let proj = g.convolution_2d_1x1(input, proj_w, None);
    let proj_b = g.constant(
        &lw.attn_proj_bias,
        Shape {
            batch: 1,
            channels: ed,
            height: 1,
            width: 1,
        },
    );
    let _out = g.addition(proj, proj_b);

    g.compile(NSQualityOfService::Default)
}

/// FFN c_fc linear kernel: 1x1 conv + bias (no LayerNorm/GELU).
pub fn build_decode_fc_linear(
    lw: &LayerWeights,
    config: &Gpt2Config,
) -> Result<Executable, ane::Error> {
    let ed = config.n_embd;
    let mut g = Graph::new();
    let input = g.placeholder(Shape::spatial(ed, 1, DECODE_SPATIAL_WIDTH));

    let fc_w = g.constant(&lw.fc_weight, Shape::spatial(4 * ed, 1, 1));
    let h = g.convolution_2d_1x1(input, fc_w, None);
    let fc_b = g.constant(
        &lw.fc_bias,
        Shape {
            batch: 1,
            channels: 4 * ed,
            height: 1,
            width: 1,
        },
    );
    let _out = g.addition(h, fc_b);
    g.compile(NSQualityOfService::Default)
}

/// FFN c_proj linear kernel: 1x1 conv + bias.
pub fn build_decode_fc_proj_linear(
    lw: &LayerWeights,
    config: &Gpt2Config,
) -> Result<Executable, ane::Error> {
    let ed = config.n_embd;
    let mut g = Graph::new();
    let input = g.placeholder(Shape::spatial(4 * ed, 1, DECODE_SPATIAL_WIDTH));

    let proj_w = g.constant(&lw.fc_proj_weight, Shape::spatial(ed, 1, 1));
    let proj = g.convolution_2d_1x1(input, proj_w, None);
    let proj_b = g.constant(
        &lw.fc_proj_bias,
        Shape {
            batch: 1,
            channels: ed,
            height: 1,
            width: 1,
        },
    );
    let _out = g.addition(proj, proj_b);
    g.compile(NSQualityOfService::Default)
}

/// LM head linear-only kernel: token projection without final LayerNorm.
///
/// Intended for private flow where final LayerNorm is computed in MPC.
pub fn build_lm_head_linear(wte: &[f32], config: &Gpt2Config) -> Result<Executable, ane::Error> {
    let ed = config.n_embd;
    let vs = config.vocab_size;
    let mut g = Graph::new();
    let input = g.placeholder(Shape::spatial(ed, 1, DECODE_SPATIAL_WIDTH));
    let wte_w = g.constant(wte, Shape::spatial(vs, 1, 1));
    let _out = g.convolution_2d_1x1(input, wte_w, None);
    g.compile(NSQualityOfService::Default)
}

pub fn build_lm_head(
    ln_f_w: &[f32],
    ln_f_b: &[f32],
    wte: &[f32],
    config: &Gpt2Config,
) -> Result<Executable, ane::Error> {
    let ed = config.n_embd;
    let vs = config.vocab_size;
    let mut g = Graph::new();
    let input = g.placeholder(Shape::spatial(ed, 1, DECODE_SPATIAL_WIDTH));
    let norm = layer_norm(&mut g, input, ln_f_w, ln_f_b, ed, config.layer_norm_epsilon);
    let wte_w = g.constant(wte, Shape::spatial(vs, 1, 1));
    let _out = g.convolution_2d_1x1(norm, wte_w, None);
    g.compile(NSQualityOfService::Default)
}
