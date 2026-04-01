//! Private GPT-2 session: Q·K^T goes through the Beaver multiplication protocol.
//!
//! - ANE: linear projections only (QKV, attn output proj, FFN FC/FC_PROJ, LM head)
//! - CPU: nonlinear stages (layer norms, GELU), Beaver protocol for private Q·K^T,
//!   MPC softmax primitives backend, and secret-shared attn·V reconstruction
//! - GPU: Beaver triple generation in background (consumed by CPU Beaver protocol)
//!
//! Optimizations:
//! - Q shared once per head (not per key position)
//! - K cached in quantized+shared Fp32 form (not re-quantized every use)
//! - Inline sharing: for t=1, share = secret + r*point (no Vec alloc)
//! - Pre-allocated flat buffers reused across layers

use std::sync::mpsc::Receiver;

use ane::{Executable, Shape, TensorData};

use crate::beaver32::beaver_dot_product_batch_from_gpu_shared;
use crate::field32::Fp32;
use crate::field32_shamir::Shamir32;
use crate::gpu::BeaverTripleBatch32;
use crate::mpc_distributed32::{BeaverTripleShare32, InProcessBeaverExecutor32};
use crate::quantize;
use crate::secure_nonlinear::{
    encode_fixed, reconstruct_f32_batch, GELUCfg, LayerNormCfg, LocalCryptoProvider, SecureOps,
    SoftmaxCfg,
};
use rand::Rng;

use super::executables::DECODE_SPATIAL_WIDTH;
use super::model::CompiledModel;
use super::private_types::{PrivateExecutionPolicy, PrivateRuntimeStats};

struct PrivateLayerExecutables {
    qkv: Executable,
    output_proj: Executable,
    fc: Executable,
    fc_proj: Executable,
}

/// KV cache: both K and V in secret-shared Fp32 form.
struct KvCache {
    /// k_shared[layer][(pos * n_embd + dim) * n + party] — Fp32 shares, ready for Beaver
    k_shared: Vec<Vec<Fp32>>,
    /// v_shared[layer][(pos * n_embd + dim) * n + party] — Fp32 shares
    v_shared: Vec<Vec<Fp32>>,
    n: usize,
    n_embd: usize,
}

impl KvCache {
    fn new(n_layers: usize, n: usize, n_embd: usize, max_seq: usize) -> Self {
        KvCache {
            k_shared: (0..n_layers)
                .map(|_| vec![Fp32::ZERO; max_seq * n_embd * n])
                .collect(),
            v_shared: (0..n_layers)
                .map(|_| vec![Fp32::ZERO; max_seq * n_embd * n])
                .collect(),
            n,
            n_embd,
        }
    }

    /// Store K and V as pre-quantized+shared Fp32.
    fn write(
        &mut self,
        layer: usize,
        pos: usize,
        k_f32: &[f32],
        v_f32: &[f32],
        eval_points: &[Fp32],
        rng: &mut impl Rng,
    ) {
        let n = self.n;
        let ed = self.n_embd;
        let base = (pos * ed) * n;
        for d in 0..ed {
            let k_secret = quantize::quantize(k_f32[d]);
            let v_secret = quantize::quantize(v_f32[d]);
            let r_k = Fp32::random(rng);
            let r_v = Fp32::random(rng);
            for p in 0..n {
                let idx = base + d * n + p;
                self.k_shared[layer][idx] = k_secret + r_k * eval_points[p];
                self.v_shared[layer][idx] = v_secret + r_v * eval_points[p];
            }
        }
    }

    /// Get pre-shared K values for a position. Returns &[n_embd * n] Fp32 shares.
    #[inline]
    fn get_k_shared(&self, layer: usize, pos: usize) -> &[Fp32] {
        let base = (pos * self.n_embd) * self.n;
        &self.k_shared[layer][base..base + self.n_embd * self.n]
    }

    #[inline]
    fn get_v_shared(&self, layer: usize, pos: usize) -> &[Fp32] {
        let base = (pos * self.n_embd) * self.n;
        &self.v_shared[layer][base..base + self.n_embd * self.n]
    }
}

pub struct PrivateSession<'model> {
    model: &'model CompiledModel,
    layers: Vec<PrivateLayerExecutables>,
    kv_cache: KvCache,
    lag_coeffs: Vec<Fp32>,
    eval_points: Vec<Fp32>,
    n: usize,
    policy: PrivateExecutionPolicy,
    distributed_beaver: Option<InProcessBeaverExecutor32>,
    n_heads: usize,
    head_dim: usize,
    n_embd: usize,
    triple_rx: Receiver<BeaverTripleBatch32>,
    current_batch: Option<BeaverTripleBatch32>,
    triple_cursor: usize,
    position: usize,
    // Reusable ANE tensor buffers
    ane_hidden: TensorData,
    ane_qkv_in: TensorData,
    ane_qkv_out: TensorData,
    ane_proj_in: TensorData,
    ane_proj_out: TensorData,
    ane_fc_in: TensorData,
    ane_fc_out: TensorData,
    ane_fc_proj_out: TensorData,
    ane_lm_in: TensorData,
    ane_lm_out: TensorData,
    lm_head_linear: Executable,
    // Reusable flat buffers for Beaver protocol
    q_flat: Vec<Fp32>,
    k_flat: Vec<Fp32>,
    logits: Vec<f32>,
    pub triples_consumed: u64,
    pub runtime_stats: PrivateRuntimeStats,
}

impl<'model> PrivateSession<'model> {
    pub fn new(
        model: &'model CompiledModel,
        n: usize,
        t: usize,
        triple_rx: Receiver<BeaverTripleBatch32>,
    ) -> Self {
        assert_eq!(
            t, 1,
            "PrivateSession currently supports only t=1 sharing; got t={}",
            t
        );

        let ed = model.config.n_embd;
        let nh = model.config.n_head;
        let hd = model.config.head_size();
        let nl = model.config.n_layer;
        let max_seq = model.max_sequence_length;

        let layers: Vec<PrivateLayerExecutables> = model
            .weights
            .layers
            .iter()
            .enumerate()
            .map(|(i, lw)| {
                eprint!("\r  Compiling private layer {}/{}...", i + 1, nl);
                PrivateLayerExecutables {
                    qkv: super::executables::build_decode_qkv_linear(lw, &model.config).unwrap(),
                    output_proj: super::executables::build_decode_output_proj(lw, &model.config)
                        .unwrap(),
                    fc: super::executables::build_decode_fc_linear(lw, &model.config).unwrap(),
                    fc_proj: super::executables::build_decode_fc_proj_linear(lw, &model.config)
                        .unwrap(),
                }
            })
            .collect();
        eprintln!(" done");

        let shamir = Shamir32::new(n, t).unwrap();
        let lag_coeffs = shamir.lagrange_coefficients().to_vec();
        let eval_points: Vec<Fp32> = (1..=n as u32).map(Fp32::new).collect();

        let hidden_shape = Shape::spatial(ed, 1, DECODE_SPATIAL_WIDTH);
        let qkv_shape = Shape::spatial(3 * ed, 1, DECODE_SPATIAL_WIDTH);
        let fc_shape = Shape::spatial(4 * ed, 1, DECODE_SPATIAL_WIDTH);
        let lm_shape = Shape::spatial(model.config.vocab_size, 1, DECODE_SPATIAL_WIDTH);
        let lm_head_linear =
            super::executables::build_lm_head_linear(&model.weights.wte, &model.config)
                .unwrap_or_else(|e| panic!("failed to compile linear lm_head: {}", e));

        // Pre-allocate flat buffers for max capacity: nh * max_seq scores × hd × n
        let max_scores = nh * max_seq;
        let flat_size = max_scores * hd * n;
        let mut policy = PrivateExecutionPolicy::from_env();
        // Back-compat toggle: if explicitly disabled, force fast prototype policy.
        if std::env::var("PRIVATE_DISTRIBUTED_BEAVER")
            .map(|v| {
                let v = v.trim().to_ascii_lowercase();
                v == "0" || v == "false" || v == "off"
            })
            .unwrap_or(false)
        {
            policy = PrivateExecutionPolicy::PrototypeFast;
        }
        let distributed_beaver = if policy.use_distributed_beaver() {
            Some(InProcessBeaverExecutor32::new(n, t).unwrap_or_else(|e| {
                panic!("failed to initialize distributed beaver executor: {}", e)
            }))
        } else {
            None
        };

        PrivateSession {
            model,
            layers,
            kv_cache: KvCache::new(nl, n, ed, max_seq),
            lag_coeffs,
            eval_points,
            n,
            policy,
            distributed_beaver,
            n_heads: nh,
            head_dim: hd,
            n_embd: ed,
            triple_rx,
            current_batch: None,
            triple_cursor: 0,
            position: 0,
            ane_hidden: TensorData::new(hidden_shape),
            ane_qkv_in: TensorData::new(hidden_shape),
            ane_qkv_out: TensorData::new(qkv_shape),
            ane_proj_in: TensorData::new(hidden_shape),
            ane_proj_out: TensorData::new(hidden_shape),
            ane_fc_in: TensorData::new(hidden_shape),
            ane_fc_out: TensorData::new(fc_shape),
            ane_fc_proj_out: TensorData::new(hidden_shape),
            ane_lm_in: TensorData::new(hidden_shape),
            ane_lm_out: TensorData::new(lm_shape),
            lm_head_linear,
            q_flat: vec![Fp32::ZERO; flat_size],
            k_flat: vec![Fp32::ZERO; flat_size],
            logits: vec![0.0; model.config.vocab_size],
            triples_consumed: 0,
            runtime_stats: PrivateRuntimeStats::default(),
        }
    }

    fn ensure_triples(&mut self, needed: usize) {
        loop {
            if let Some(ref batch) = self.current_batch {
                if self.triple_cursor + needed <= batch.count {
                    return;
                }
            }
            self.current_batch = Some(self.triple_rx.recv().expect("triple supply exhausted"));
            self.triple_cursor = 0;
        }
    }

    pub fn decode_step_private(&mut self, token: u32, rng: &mut impl Rng) -> &[f32] {
        let ed = self.n_embd;
        let nh = self.n_heads;
        let hd = self.head_dim;
        let nl = self.layers.len();
        let n = self.n;
        let pos = self.position;
        let num_keys = pos + 1;
        let sw = DECODE_SPATIAL_WIDTH;
        let debug_numeric = std::env::var_os("PRIVATE_DEBUG_NUMERIC").is_some();
        let use_distributed_beaver = self.policy.use_distributed_beaver();

        // 1. Embedding lookup
        {
            let mut surf = self.ane_hidden.as_f32_slice_mut();
            let ti = token as usize;
            for d in 0..ed {
                surf[d * sw] =
                    self.model.weights.wte[ti * ed + d] + self.model.weights.wpe[pos * ed + d];
            }
        }

        for layer in 0..nl {
            // 2a. LN1 on CPU, then ANE linear QKV projection
            let hidden_before_attn = read_spatial_column0(&self.ane_hidden, ed, sw);
            let ln1_hidden = {
                let lw = &self.model.weights.layers[layer];
                mpc_layer_norm_from_clear(
                    &self.lag_coeffs,
                    &self.eval_points,
                    &hidden_before_attn,
                    &lw.ln1_weight,
                    &lw.ln1_bias,
                    self.model.config.layer_norm_epsilon as f32,
                    rng,
                )
            };
            write_spatial_column0(&mut self.ane_qkv_in, &ln1_hidden, sw);
            self.layers[layer]
                .qkv
                .run_cached(&[&self.ane_qkv_in], &[&self.ane_qkv_out])
                .unwrap_or_else(|e| panic!("qkv layer {layer}: {e}"));

            // 2b. Extract Q, K, V from ANE output
            let (q_f32, k_f32, v_f32) = {
                let qkv_slice = self.ane_qkv_out.as_f32_slice();
                let mut q = vec![0.0f32; ed];
                let mut k = vec![0.0f32; ed];
                let mut v = vec![0.0f32; ed];
                for d in 0..ed {
                    q[d] = qkv_slice[d * sw];
                    k[d] = qkv_slice[(ed + d) * sw];
                    v[d] = qkv_slice[(2 * ed + d) * sw];
                }
                (q, k, v)
            };

            // 2c. Store K and V in quantized+shared form
            self.kv_cache
                .write(layer, pos, &k_f32, &v_f32, &self.eval_points, rng);

            // 2d. Private Q·K^T via Beaver triples.
            // 2e. Secret-shared softmax via MPC primitives.
            // 2f. Secret-shared attn·V via Beaver triples.
            let triples_qk = nh * num_keys * hd;
            let triples_attn_v = nh * num_keys * hd;
            let triples_needed = triples_qk + triples_attn_v;
            self.ensure_triples(triples_needed);
            let num_scores = nh * num_keys;

            // Share Q once per head dimension (reused across all key positions)
            // Inline sharing: for t=1, share = secret + r*point
            for h in 0..nh {
                for d in 0..hd {
                    let q_val = quantize::quantize(q_f32[h * hd + d]);
                    let r = Fp32::random(rng);
                    // Compute shares into small stack buffer, then broadcast to all key positions
                    let mut q_shares = [Fp32::ZERO; 8]; // n <= 8
                    for p in 0..n {
                        q_shares[p] = q_val + r * self.eval_points[p];
                    }
                    for kp in 0..num_keys {
                        let score_idx = h * num_keys + kp;
                        let dst = (score_idx * hd + d) * n;
                        for p in 0..n {
                            self.q_flat[dst + p] = q_shares[p];
                        }
                    }
                }
            }

            // K: copy pre-shared values from cache (already quantized+shared)
            for h in 0..nh {
                for kp in 0..num_keys {
                    let score_idx = h * num_keys + kp;
                    let cached = self.kv_cache.get_k_shared(layer, kp);
                    for d in 0..hd {
                        let src_offset = (h * hd + d) * n;
                        let dst_offset = (score_idx * hd + d) * n;
                        self.k_flat[dst_offset..dst_offset + n]
                            .copy_from_slice(&cached[src_offset..src_offset + n]);
                    }
                }
            }

            let batch = self.current_batch.as_ref().unwrap();
            let dot_shares = if use_distributed_beaver {
                // k = num_scores * hd independent Beaver multiplications.
                let k = num_scores * hd;
                let distributed = self.distributed_beaver.as_ref().unwrap();
                let mul_shares = distributed_beaver_multiply_batch_from_gpu_shared(
                    distributed,
                    &self.q_flat[..num_scores * hd * n],
                    &self.k_flat[..num_scores * hd * n],
                    batch,
                    &mut self.triple_cursor,
                );
                self.runtime_stats.distributed_beaver_batches += 1;
                self.runtime_stats.distributed_open_values += (2 * k) as u64;
                let mut dots = vec![vec![Fp32::ZERO; n]; num_scores];
                for s in 0..num_scores {
                    let base = s * hd;
                    for d in 0..hd {
                        let prod = &mul_shares[base + d];
                        for p in 0..n {
                            dots[s][p] = dots[s][p] + prod[p];
                        }
                    }
                }
                dots
            } else {
                beaver_dot_product_batch_from_gpu_shared(
                    &self.lag_coeffs,
                    n,
                    hd,
                    num_scores,
                    &self.q_flat[..num_scores * hd * n],
                    &self.k_flat[..num_scores * hd * n],
                    batch,
                    &mut self.triple_cursor,
                )
            };

            // 2e. Secure softmax: rescale + CrypTEN-style exp/recip on shares.
            //     No plaintext scores are ever revealed (replaces open+reshare fallback).
            let weight_shares = {
                if self.policy == PrivateExecutionPolicy::StrictPrivate {
                    self.runtime_stats.strict_mode_blocks += 1;
                }
                secure_softmax_from_dot_shares(
                    &self.lag_coeffs,
                    &self.eval_points,
                    &dot_shares,
                    nh,
                    num_keys,
                    hd,
                    rng,
                )
            };

            // 2f. Private softmax(attn_scores) · V in secret-shared domain, then reconstruct.
            let attn_out = if use_distributed_beaver {
                // Pack k = ed * num_keys multiplications into reusable flat buffers.
                for out_d in 0..ed {
                    let h = out_d / hd;
                    for kp in 0..num_keys {
                        let score_idx = h * num_keys + kp;
                        let v_cached = self.kv_cache.get_v_shared(layer, kp);
                        let offset = out_d * n;
                        let idx = (out_d * num_keys + kp) * n;
                        for p in 0..n {
                            self.q_flat[idx + p] = weight_shares[score_idx][p];
                            self.k_flat[idx + p] = v_cached[offset + p];
                        }
                    }
                }
                let k = ed * num_keys;
                let distributed = self.distributed_beaver.as_ref().unwrap();
                let mul_shares = distributed_beaver_multiply_batch_from_gpu_shared(
                    distributed,
                    &self.q_flat[..ed * num_keys * n],
                    &self.k_flat[..ed * num_keys * n],
                    batch,
                    &mut self.triple_cursor,
                );
                self.runtime_stats.distributed_beaver_batches += 1;
                self.runtime_stats.distributed_open_values += (2 * k) as u64;
                let mut out = vec![0.0f32; ed];
                for out_d in 0..ed {
                    let mut acc_shares = vec![Fp32::ZERO; n];
                    for kp in 0..num_keys {
                        let idx = out_d * num_keys + kp;
                        for p in 0..n {
                            acc_shares[p] = acc_shares[p] + mul_shares[idx][p];
                        }
                    }
                    let acc_secret = reconstruct_share_values(&self.lag_coeffs, &acc_shares);
                    out[out_d] = quantize::dequantize_product(acc_secret);
                }
                out
            } else {
                let mut out = vec![0.0f32; ed];
                for h in 0..nh {
                    for d in 0..hd {
                        let out_d = h * hd + d;
                        let mut acc_shares = vec![Fp32::ZERO; n];
                        for kp in 0..num_keys {
                            let score_idx = h * num_keys + kp;
                            let v_cached = self.kv_cache.get_v_shared(layer, kp);
                            let offset = out_d * n;
                            let prod = beaver_multiply_shared_from_batch(
                                &self.lag_coeffs,
                                &weight_shares[score_idx],
                                &v_cached[offset..offset + n],
                                batch,
                                &mut self.triple_cursor,
                            );
                            for p in 0..n {
                                acc_shares[p] = acc_shares[p] + prod[p];
                            }
                        }
                        let acc_secret = reconstruct_share_values(&self.lag_coeffs, &acc_shares);
                        // weight is SCALE, V is SCALE -> product in SCALE² domain.
                        out[out_d] = quantize::dequantize_product(acc_secret);
                    }
                }
                out
            };
            if debug_numeric {
                let (finite, min_v, max_v, max_abs) = f32_stats(&attn_out);
                eprintln!(
                    "  [numdbg] pos={} layer={} attn_out finite={}/{} min={:.4} max={:.4} max_abs={:.4}",
                    pos,
                    layer,
                    finite,
                    attn_out.len(),
                    min_v,
                    max_v,
                    max_abs
                );
            }
            self.triples_consumed += triples_needed as u64;
            self.runtime_stats.triples_consumed += triples_needed as u64;

            // 2g. ANE: Output projection
            {
                let mut surf = self.ane_proj_in.as_f32_slice_mut();
                for d in 0..ed {
                    surf[d * sw] = attn_out[d];
                }
            }
            self.layers[layer]
                .output_proj
                .run_cached(&[&self.ane_proj_in], &[&self.ane_proj_out])
                .unwrap_or_else(|e| panic!("output_proj layer {layer}: {e}"));

            // 2h. Residual add
            {
                let proj_slice = self.ane_proj_out.as_f32_slice();
                let mut hidden_surf = self.ane_hidden.as_f32_slice_mut();
                for d in 0..ed {
                    hidden_surf[d * sw] += proj_slice[d * sw];
                }
            }
            if debug_numeric {
                let hidden_slice = self.ane_hidden.as_f32_slice();
                let mut compact = vec![0.0f32; ed];
                for d in 0..ed {
                    compact[d] = hidden_slice[d * sw];
                }
                let (finite, min_v, max_v, max_abs) = f32_stats(&compact);
                eprintln!(
                    "  [numdbg] pos={} layer={} post_residual finite={}/{} min={:.4} max={:.4} max_abs={:.4}",
                    pos, layer, finite, ed, min_v, max_v, max_abs
                );
            }

            // 2i. FFN split: LN2 + GELU on CPU, linear FC/FC_PROJ on ANE
            let hidden_before_ffn = read_spatial_column0(&self.ane_hidden, ed, sw);
            let ln2_hidden = {
                let lw = &self.model.weights.layers[layer];
                mpc_layer_norm_from_clear(
                    &self.lag_coeffs,
                    &self.eval_points,
                    &hidden_before_ffn,
                    &lw.ln2_weight,
                    &lw.ln2_bias,
                    self.model.config.layer_norm_epsilon as f32,
                    rng,
                )
            };
            write_spatial_column0(&mut self.ane_fc_in, &ln2_hidden, sw);
            self.layers[layer]
                .fc
                .run_cached(&[&self.ane_fc_in], &[&self.ane_fc_out])
                .unwrap_or_else(|e| panic!("fc layer {layer}: {e}"));

            let fc_hidden = mpc_gelu_from_clear(
                &self.lag_coeffs,
                &self.eval_points,
                &read_spatial_column0(&self.ane_fc_out, 4 * ed, sw),
                rng,
            );
            write_spatial_column0(&mut self.ane_fc_out, &fc_hidden, sw);

            self.layers[layer]
                .fc_proj
                .run_cached(&[&self.ane_fc_out], &[&self.ane_fc_proj_out])
                .unwrap_or_else(|e| panic!("fc_proj layer {layer}: {e}"));

            {
                let proj_slice = self.ane_fc_proj_out.as_f32_slice();
                let mut hidden_surf = self.ane_hidden.as_f32_slice_mut();
                for d in 0..ed {
                    hidden_surf[d * sw] += proj_slice[d * sw];
                }
            }
        }

        // 3. Final LN_F on CPU, then linear LM head on ANE
        let hidden_before_lm = read_spatial_column0(&self.ane_hidden, ed, sw);
        let ln_f_hidden = mpc_layer_norm_from_clear(
            &self.lag_coeffs,
            &self.eval_points,
            &hidden_before_lm,
            &self.model.weights.ln_f_weight,
            &self.model.weights.ln_f_bias,
            self.model.config.layer_norm_epsilon as f32,
            rng,
        );
        write_spatial_column0(&mut self.ane_lm_in, &ln_f_hidden, sw);
        self.lm_head_linear
            .run_cached(&[&self.ane_lm_in], &[&self.ane_lm_out])
            .unwrap_or_else(|e| panic!("lm_head_linear: {e}"));

        {
            let lm_slice = self.ane_lm_out.as_f32_slice();
            let vs = self.model.config.vocab_size;
            let mut finite = 0usize;
            let mut min_v = f32::INFINITY;
            let mut max_v = f32::NEG_INFINITY;
            for v in 0..vs {
                let val = lm_slice[v * DECODE_SPATIAL_WIDTH];
                self.logits[v] = val;
                if val.is_finite() {
                    finite += 1;
                    min_v = min_v.min(val);
                    max_v = max_v.max(val);
                }
            }
            if debug_numeric {
                eprintln!(
                    "  [numdbg] pos={} logits finite={}/{} min={:.4} max={:.4}",
                    pos, finite, vs, min_v, max_v
                );
            }
        }

        self.position += 1;
        &self.logits
    }
}

#[inline]
fn read_spatial_column0(tensor: &TensorData, channels: usize, spatial_width: usize) -> Vec<f32> {
    let surf = tensor.as_f32_slice();
    let mut out = vec![0.0f32; channels];
    for c in 0..channels {
        out[c] = surf[c * spatial_width];
    }
    out
}

#[inline]
fn write_spatial_column0(tensor: &mut TensorData, values: &[f32], spatial_width: usize) {
    let mut surf = tensor.as_f32_slice_mut();
    for c in 0..values.len() {
        surf[c * spatial_width] = values[c];
    }
}

/// Secure layer norm: shares input, applies CrypTEN-style LN on shares, reconstructs.
///
/// Only Beaver-masked values are revealed during computation — no intermediate
/// plaintext is ever exposed.
#[inline]
fn mpc_layer_norm_from_clear(
    lag_coeffs: &[Fp32],
    eval_points: &[Fp32],
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    epsilon: f32,
    rng: &mut impl Rng,
) -> Vec<f32> {
    const FB: u32 = 9; // SCALE = 512, matching quantize module
    let n = lag_coeffs.len();
    let mut prov = LocalCryptoProvider::new(rng, n, eval_points.to_vec());
    let shared: Vec<Vec<Fp32>> = input
        .iter()
        .map(|&v| prov.share_signed(encode_fixed(v as f64, FB)))
        .collect();
    let cfg = LayerNormCfg {
        frac_bits: FB,
        rsqrt_iters: 4,
    };
    let mut ops = SecureOps::new(&mut prov, lag_coeffs);
    let out = ops.layer_norm(&shared, 1, input.len(), Some(gamma), Some(beta), epsilon, &cfg);
    drop(ops);
    reconstruct_f32_batch(&out, lag_coeffs, FB)
}

/// Secure GELU: shares input, applies CrypTEN-style x·σ(1.702x) on shares, reconstructs.
#[inline]
fn mpc_gelu_from_clear(
    lag_coeffs: &[Fp32],
    eval_points: &[Fp32],
    input: &[f32],
    rng: &mut impl Rng,
) -> Vec<f32> {
    const FB: u32 = 9;
    let n = lag_coeffs.len();
    let mut prov = LocalCryptoProvider::new(rng, n, eval_points.to_vec());
    let shared: Vec<Vec<Fp32>> = input
        .iter()
        .map(|&v| prov.share_signed(encode_fixed(v as f64, FB)))
        .collect();
    let cfg = GELUCfg::default();
    let mut ops = SecureOps::new(&mut prov, lag_coeffs);
    let out = ops.gelu_batch(&shared, &cfg);
    drop(ops);
    reconstruct_f32_batch(&out, lag_coeffs, FB)
}

/// Secure softmax on dot-product shares (SCALE² domain).
///
/// Rescales from SCALE² to fixed-point, applies CrypTEN-style softmax on shares,
/// returns shares of integer weights summing to output_scale per head.
fn secure_softmax_from_dot_shares(
    lag_coeffs: &[Fp32],
    eval_points: &[Fp32],
    dot_shares: &[Vec<Fp32>],
    num_heads: usize,
    num_keys: usize,
    head_dim: usize,
    rng: &mut impl Rng,
) -> Vec<Vec<Fp32>> {
    let n = lag_coeffs.len();
    let mut prov = LocalCryptoProvider::new(rng, n, eval_points.to_vec());

    // Rescale from SCALE² to fb=9 fixed-point: divide by SCALE * sqrt(head_dim).
    // For head_dim=64: SCALE * sqrt(64) = 512 * 8 = 4096 = 2^12 → trunc by 12.
    // For general head_dim: use div_pub with the divisor.
    let sqrt_hd = (head_dim as f64).sqrt();
    let divisor = (quantize::scale_factor() as f64 * sqrt_hd).round() as u32;

    let mut ops = SecureOps::new(&mut prov, lag_coeffs);
    let rescaled: Vec<Vec<Fp32>> = if divisor.is_power_of_two() {
        let shift = divisor.trailing_zeros();
        dot_shares
            .iter()
            .map(|s| ops.trunc(s, shift))
            .collect()
    } else {
        dot_shares
            .iter()
            .map(|s| ops.div_pub(s, divisor))
            .collect()
    };

    let cfg = SoftmaxCfg::default();
    ops.softmax(&rescaled, num_heads, num_keys, &cfg)
}

#[inline]
fn f32_stats(data: &[f32]) -> (usize, f32, f32, f32) {
    let mut finite = 0usize;
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    let mut max_abs = 0.0f32;
    for &v in data {
        if v.is_finite() {
            finite += 1;
            min_v = min_v.min(v);
            max_v = max_v.max(v);
            max_abs = max_abs.max(v.abs());
        }
    }
    (finite, min_v, max_v, max_abs)
}

#[inline]
fn beaver_multiply_shared_from_batch(
    lag: &[Fp32],
    x_shares: &[Fp32],
    y_shares: &[Fp32],
    batch: &BeaverTripleBatch32,
    triple_cursor: &mut usize,
) -> Vec<Fp32> {
    debug_assert_eq!(x_shares.len(), y_shares.len());
    let n = x_shares.len();
    let tri_idx = *triple_cursor;
    *triple_cursor += 1;

    let mut d = Fp32::ZERO;
    let mut e = Fp32::ZERO;
    for p in 0..n {
        let (a_raw, b_raw, _) = batch.triple_values(tri_idx, p);
        let a_p = Fp32::from_reduced(a_raw);
        let b_p = Fp32::from_reduced(b_raw);
        d = d + (x_shares[p] - a_p) * lag[p];
        e = e + (y_shares[p] - b_p) * lag[p];
    }

    let de = d * e;
    let mut out = vec![Fp32::ZERO; n];
    for p in 0..n {
        let (_, _, c_raw) = batch.triple_values(tri_idx, p);
        let c_p = Fp32::from_reduced(c_raw);
        out[p] = c_p + e * x_shares[p] + d * y_shares[p] - de;
    }
    out
}

fn distributed_beaver_multiply_batch_from_gpu_shared(
    executor: &InProcessBeaverExecutor32,
    x_flat: &[Fp32], // [k][n]
    y_flat: &[Fp32], // [k][n]
    batch: &BeaverTripleBatch32,
    triple_cursor: &mut usize,
) -> Vec<Vec<Fp32>> {
    let n = executor.n();
    assert_eq!(
        x_flat.len(),
        y_flat.len(),
        "x/y flat buffers must have identical length"
    );
    assert_eq!(
        x_flat.len() % n,
        0,
        "flat buffers must be an integer multiple of n"
    );
    let k = x_flat.len() / n;

    let base = *triple_cursor;
    *triple_cursor += k;

    let mut tri_party = vec![Vec::<BeaverTripleShare32>::with_capacity(k); n];

    for i in 0..k {
        for p in 0..n {
            let (a_raw, b_raw, c_raw) = batch.triple_values(base + i, p);
            tri_party[p].push(BeaverTripleShare32 {
                a: Fp32::from_reduced(a_raw),
                b: Fp32::from_reduced(b_raw),
                c: Fp32::from_reduced(c_raw),
            });
        }
    }

    executor
        .run_batch(x_flat, y_flat, &tri_party)
        .unwrap_or_else(|e| panic!("distributed beaver batch failed: {}", e))
}

#[inline]
fn reconstruct_share_values(lag: &[Fp32], shares: &[Fp32]) -> Fp32 {
    let mut out = Fp32::ZERO;
    for i in 0..shares.len() {
        out = out + lag[i] * shares[i];
    }
    out
}


#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn test_kv_cache_v_is_secret_shared() {
        let n = 5;
        let n_embd = 4;
        let mut cache = KvCache::new(1, n, n_embd, 2);
        let eval_points: Vec<Fp32> = (1..=n as u32).map(Fp32::new).collect();
        let mut rng = ChaCha20Rng::seed_from_u64(42);

        let k = [0.25f32, -1.0, 2.0, 0.5];
        let v = [1.5f32, -0.75, 0.125, 3.0];
        cache.write(0, 0, &k, &v, &eval_points, &mut rng);

        let shamir = Shamir32::new(n, 1).unwrap();
        let lag = shamir.lagrange_coefficients();
        let v_shared = cache.get_v_shared(0, 0);

        for d in 0..n_embd {
            let mut rec = Fp32::ZERO;
            for p in 0..n {
                rec = rec + lag[p] * v_shared[d * n + p];
            }
            let got = quantize::dequantize(rec);
            assert!(
                (got - v[d]).abs() < 0.02,
                "dim {} expected {}, got {}",
                d,
                v[d],
                got
            );
        }
    }

    #[test]
    fn test_secure_softmax_from_dot_shares_weights_positive() {
        use crate::secure_nonlinear::{from_signed, to_signed};

        let n = 5usize;
        let shamir = Shamir32::new(n, 1).unwrap();
        let lag = shamir.lagrange_coefficients().to_vec();
        let eval_points: Vec<Fp32> = (1..=n as u32).map(Fp32::new).collect();
        let mut rng = ChaCha20Rng::seed_from_u64(1337);

        // Simulate SCALE² domain dot products (scores ~1.0 after rescaling)
        let dots = [262_144i64, 131_072, 0, 524_288]; // ≈ 1.0, 0.5, 0.0, 2.0 in real
        let dot_shares: Vec<Vec<Fp32>> = dots
            .iter()
            .map(|&v| {
                let secret = from_signed(v);
                let r = Fp32::random(&mut rng);
                (0..n)
                    .map(|p| secret + r * eval_points[p])
                    .collect::<Vec<_>>()
            })
            .collect();

        let out = secure_softmax_from_dot_shares(
            &lag,
            &eval_points,
            &dot_shares,
            1,
            dots.len(),
            64, // head_dim
            &mut rng,
        );

        // All weights should be non-negative
        for (i, w) in out.iter().enumerate() {
            let v = to_signed(reconstruct_share_values(&lag, w));
            assert!(
                v >= -5, // small tolerance for rounding
                "weight {} should be non-negative, got {}",
                i,
                v
            );
        }
    }
}
