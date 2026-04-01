//! Private attention: Q·K^T via Beaver multiplication protocol.
//!
//! Secret-shares the hidden state, computes QKV projections linearly on shares,
//! then performs the dot product Q·K^T using real Beaver triples from the GPU.
//! Attention scores are revealed (reconstructed) for softmax in cleartext.

use crate::beaver32::{beaver_dot_product_32, triples_from_gpu_batch, BeaverTriple32};
use crate::field32::Fp32;
use crate::field32_shamir::{Shamir32, Share32};
use crate::gpu::BeaverTripleBatch32;
use crate::quantize;
use rand::Rng;

pub struct PrivateAttentionConfig {
    pub n: usize,
    pub t: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub n_embd: usize,
    pub max_seq: usize,
}

/// KV cache storing secret-shared keys and values per layer per party.
/// Layout: data[layer][party * max_seq * n_embd + pos * n_embd + dim]
pub struct PrivateKvCache {
    pub keys: Vec<Vec<Fp32>>,   // [n_layers][n * max_seq * n_embd]
    pub values: Vec<Vec<Fp32>>, // [n_layers][n * max_seq * n_embd]
    pub n: usize,
    pub n_embd: usize,
    pub max_seq: usize,
}

impl PrivateKvCache {
    pub fn new(n_layers: usize, n: usize, n_embd: usize, max_seq: usize) -> Self {
        let size = n * max_seq * n_embd;
        PrivateKvCache {
            keys: (0..n_layers).map(|_| vec![Fp32::ZERO; size]).collect(),
            values: (0..n_layers).map(|_| vec![Fp32::ZERO; size]).collect(),
            n,
            n_embd,
            max_seq,
        }
    }

    #[inline]
    fn idx(&self, party: usize, pos: usize, dim: usize) -> usize {
        party * self.max_seq * self.n_embd + pos * self.n_embd + dim
    }

    pub fn write_kv(&mut self, layer: usize, party: usize, pos: usize, k: &[Fp32], v: &[Fp32]) {
        let base = party * self.max_seq * self.n_embd + pos * self.n_embd;
        for d in 0..self.n_embd {
            self.keys[layer][base + d] = k[d];
            self.values[layer][base + d] = v[d];
        }
    }

    pub fn get_k(
        &self,
        layer: usize,
        party: usize,
        pos: usize,
        head: usize,
        head_dim: usize,
    ) -> &[Fp32] {
        let base = self.idx(party, pos, head * head_dim);
        &self.keys[layer][base..base + head_dim]
    }

    pub fn get_v(&self, layer: usize, party: usize, pos: usize, dim: usize) -> Fp32 {
        self.values[layer][self.idx(party, pos, dim)]
    }
}

pub struct PrivateAttention {
    pub config: PrivateAttentionConfig,
    pub shamir: Shamir32,
    eval_points: Vec<Fp32>,
}

impl PrivateAttention {
    pub fn new(config: PrivateAttentionConfig) -> Self {
        let shamir = Shamir32::new(config.n, config.t).unwrap();
        let eval_points: Vec<Fp32> = (1..=config.n as u32).map(Fp32::new).collect();
        PrivateAttention {
            config,
            shamir,
            eval_points,
        }
    }

    /// Quantize and secret-share an f32 vector into n parties' Fp32 shares.
    /// Returns shares[party][dim].
    pub fn share_vec<R: Rng>(&self, data: &[f32], rng: &mut R) -> Vec<Vec<Fp32>> {
        let n = self.config.n;
        let dim = data.len();
        let mut shares = vec![vec![Fp32::ZERO; dim]; n];
        for d in 0..dim {
            let q = quantize::quantize(data[d]);
            let sh = self.shamir.share(q, rng);
            for p in 0..n {
                shares[p][d] = sh[p].value;
            }
        }
        shares
    }

    /// Reconstruct f32 vector from n parties' Fp32 shares.
    pub fn reconstruct_vec(&self, shares: &[Vec<Fp32>]) -> Vec<f32> {
        let n = self.config.n;
        let dim = shares[0].len();
        let lag = self.shamir.lagrange_coefficients();
        let mut result = vec![0.0f32; dim];
        for d in 0..dim {
            let mut val = Fp32::ZERO;
            for p in 0..n {
                val = val + shares[p][d] * lag[p];
            }
            result[d] = quantize::dequantize(val);
        }
        result
    }

    /// QKV projection on shares: Q_p = W_qkv · share_p (linear, no triples).
    /// `w_qkv` is [3*n_embd, n_embd] quantized weights, `bias` is [3*n_embd].
    /// Returns (q_shares, k_shares, v_shares) each [n][n_embd].
    pub fn qkv_project(
        &self,
        hidden_shares: &[Vec<Fp32>],
        w_qkv: &[Fp32],
        bias_qkv: &[Fp32],
    ) -> (Vec<Vec<Fp32>>, Vec<Vec<Fp32>>, Vec<Vec<Fp32>>) {
        let n = self.config.n;
        let ed = self.config.n_embd;
        let out_dim = 3 * ed;
        let mut q_shares = vec![vec![Fp32::ZERO; ed]; n];
        let mut k_shares = vec![vec![Fp32::ZERO; ed]; n];
        let mut v_shares = vec![vec![Fp32::ZERO; ed]; n];

        for p in 0..n {
            // matmul: out[o] = sum_i w[o*ed+i] * hidden[i] + bias[o]
            for o in 0..out_dim {
                let mut acc = Fp32::ZERO;
                for i in 0..ed {
                    acc = acc + w_qkv[o * ed + i] * hidden_shares[p][i];
                }
                // Bias is added only by party 0 (or split across parties, but for
                // simplicity: party 0 adds full bias, others add nothing)
                let bias_term = if p == 0 { bias_qkv[o] } else { Fp32::ZERO };
                let val = acc + bias_term;

                if o < ed {
                    q_shares[p][o] = val;
                } else if o < 2 * ed {
                    k_shares[p][o - ed] = val;
                } else {
                    v_shares[p][o - 2 * ed] = val;
                }
            }
        }
        (q_shares, k_shares, v_shares)
    }

    /// Private Q·K^T for one head using Beaver dot products.
    /// Returns cleartext attention scores for this head [num_keys].
    ///
    /// `q_shares[n][head_dim]` — party shares of the query vector for this head.
    /// `triple_cursor` is advanced by head_dim * num_keys triples.
    pub fn private_qk_dot_head(
        &self,
        q_shares: &[Vec<Fp32>], // [n][head_dim] for this head
        kv_cache: &PrivateKvCache,
        layer: usize,
        head: usize,
        num_keys: usize,
        triple_batch: &BeaverTripleBatch32,
        triple_cursor: &mut usize,
    ) -> Vec<f32> {
        let n = self.config.n;
        let hd = self.config.head_dim;
        let scale = 1.0 / (hd as f32).sqrt();
        let mut scores = Vec::with_capacity(num_keys);

        for key_pos in 0..num_keys {
            // Build x_shares[dim][n] and y_shares[dim][n] for dot product
            let mut x_shares = Vec::with_capacity(hd);
            let mut y_shares = Vec::with_capacity(hd);
            let mut triples_dim = Vec::with_capacity(hd);

            for d in 0..hd {
                // x = q[d], y = k[key_pos][d] for this head
                let mut x_sh = Vec::with_capacity(n);
                let mut y_sh = Vec::with_capacity(n);
                for p in 0..n {
                    x_sh.push(Share32 {
                        point: self.eval_points[p],
                        value: q_shares[p][d],
                    });
                    let k_val = kv_cache.get_k(layer, p, key_pos, head, hd)[d];
                    y_sh.push(Share32 {
                        point: self.eval_points[p],
                        value: k_val,
                    });
                }
                x_shares.push(x_sh);
                y_shares.push(y_sh);

                // Get triple from GPU batch
                let tri = triples_from_gpu_batch(triple_batch, *triple_cursor, &self.eval_points);
                triples_dim.push(tri);
                *triple_cursor += 1;
            }

            // Beaver dot product: sum_d q[d]*k[d] in MPC
            let dot_shares =
                beaver_dot_product_32(&self.shamir, &x_shares, &y_shares, &triples_dim).unwrap();

            // Reconstruct to get cleartext score
            let dot_val = self.shamir.reconstruct(&dot_shares).unwrap();

            // Dequantize: dot product is in SCALE² domain, then apply attention scaling
            scores.push(quantize::dequantize_product(dot_val) * scale);
        }

        scores
    }

    /// Full private Q·K^T across all heads for one layer.
    /// Returns cleartext scores [n_heads][num_keys].
    pub fn private_qk_all_heads(
        &self,
        q_shares: &[Vec<Fp32>], // [n][n_embd]
        kv_cache: &PrivateKvCache,
        layer: usize,
        num_keys: usize,
        triple_batch: &BeaverTripleBatch32,
        triple_cursor: &mut usize,
    ) -> Vec<Vec<f32>> {
        let nh = self.config.n_heads;
        let hd = self.config.head_dim;
        let n = self.config.n;

        let mut all_scores = Vec::with_capacity(nh);
        for h in 0..nh {
            // Extract this head's query shares: q_head[p][d] = q_shares[p][h*hd + d]
            let q_head: Vec<Vec<Fp32>> = (0..n)
                .map(|p| q_shares[p][h * hd..(h + 1) * hd].to_vec())
                .collect();

            let scores = self.private_qk_dot_head(
                &q_head,
                kv_cache,
                layer,
                h,
                num_keys,
                triple_batch,
                triple_cursor,
            );
            all_scores.push(scores);
        }
        all_scores
    }

    /// Compute attention output: attn_weights · V (linear on shares, no triples).
    /// `attn_weights[n_heads][num_keys]` is cleartext (after softmax).
    /// Returns output_shares[n][n_embd].
    pub fn attn_v_multiply(
        &self,
        attn_weights: &[Vec<f32>], // [n_heads][num_keys]
        kv_cache: &PrivateKvCache,
        layer: usize,
        num_keys: usize,
    ) -> Vec<Vec<Fp32>> {
        let n = self.config.n;
        let nh = self.config.n_heads;
        let hd = self.config.head_dim;
        let ed = self.config.n_embd;

        let mut output = vec![vec![Fp32::ZERO; ed]; n];

        for p in 0..n {
            for h in 0..nh {
                for d in 0..hd {
                    let out_d = h * hd + d;
                    let mut acc = Fp32::ZERO;
                    for k in 0..num_keys {
                        // Multiply cleartext weight by secret-shared V
                        let w = quantize::quantize(attn_weights[h][k]);
                        let v = kv_cache.get_v(layer, p, k, out_d);
                        acc = acc + w * v;
                    }
                    output[p][out_d] = acc;
                }
            }
        }
        output
    }

    /// Output projection on shares: out_p = W_out · attn_p + bias (linear).
    pub fn output_project(
        &self,
        attn_shares: &[Vec<Fp32>],
        w_out: &[Fp32],
        bias_out: &[Fp32],
    ) -> Vec<Vec<Fp32>> {
        let n = self.config.n;
        let ed = self.config.n_embd;
        let mut output = vec![vec![Fp32::ZERO; ed]; n];
        for p in 0..n {
            for o in 0..ed {
                let mut acc = Fp32::ZERO;
                for i in 0..ed {
                    acc = acc + w_out[o * ed + i] * attn_shares[p][i];
                }
                let bias_term = if p == 0 { bias_out[o] } else { Fp32::ZERO };
                output[p][o] = acc + bias_term;
            }
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::{ExpandedCorrelations32, GpuTripleGen32};
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn test_share_reconstruct_round_trip() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let pa = PrivateAttention::new(PrivateAttentionConfig {
            n: 5,
            t: 1,
            n_heads: 1,
            head_dim: 4,
            n_embd: 4,
            max_seq: 8,
        });

        let data = vec![1.5, -2.0, 0.25, 3.0];
        let shares = pa.share_vec(&data, &mut rng);
        let recovered = pa.reconstruct_vec(&shares);

        for i in 0..data.len() {
            assert!(
                (recovered[i] - data[i]).abs() < 0.01,
                "dim {}: expected {}, got {}",
                i,
                data[i],
                recovered[i]
            );
        }
    }

    #[test]
    fn test_private_dot_product_with_gpu_triples() {
        let mut rng = ChaCha20Rng::seed_from_u64(43);
        let n = 5;
        let t = 1;
        let head_dim = 64;
        let spr = n - 2 * t;
        let num_triples = head_dim; // 1 dot product = head_dim triples
        let num_rounds = (num_triples as usize).div_ceil(spr);

        let ot: Vec<ExpandedCorrelations32> = (0..n)
            .map(|i| ExpandedCorrelations32::from_random(i, num_rounds, &mut rng))
            .collect();
        let gpu = GpuTripleGen32::new(n, t).unwrap();
        let batch = gpu.generate(num_triples, &ot, &mut rng).unwrap();

        // Create random Q and K vectors
        let q: Vec<f32> = (0..head_dim)
            .map(|_| rng.gen::<f32>() * 2.0 - 1.0)
            .collect();
        let k: Vec<f32> = (0..head_dim)
            .map(|_| rng.gen::<f32>() * 2.0 - 1.0)
            .collect();

        // Cleartext dot product
        let expected: f32 = q.iter().zip(&k).map(|(a, b)| a * b).sum();

        // Private dot product
        let pa = PrivateAttention::new(PrivateAttentionConfig {
            n,
            t,
            n_heads: 1,
            head_dim,
            n_embd: head_dim,
            max_seq: 1,
        });
        let eval_points: Vec<Fp32> = (1..=n as u32).map(Fp32::new).collect();

        let q_q = quantize::quantize_vec(&q);
        let k_q = quantize::quantize_vec(&k);

        let q_shares: Vec<Vec<Share32>> =
            q_q.iter().map(|&v| pa.shamir.share(v, &mut rng)).collect();
        let k_shares: Vec<Vec<Share32>> =
            k_q.iter().map(|&v| pa.shamir.share(v, &mut rng)).collect();

        let triples: Vec<Vec<BeaverTriple32>> = (0..head_dim)
            .map(|i| triples_from_gpu_batch(&batch, i, &eval_points))
            .collect();

        let result_shares =
            beaver_dot_product_32(&pa.shamir, &q_shares, &k_shares, &triples).unwrap();
        let result_fp32 = pa.shamir.reconstruct(&result_shares).unwrap();
        let result = quantize::dequantize_product(result_fp32);

        let tolerance = expected.abs() * 0.05 + 0.1; // 5% relative + 0.1 absolute
        assert!(
            (result - expected).abs() < tolerance,
            "private dot product: expected {}, got {} (diff {})",
            expected,
            result,
            (result - expected).abs()
        );

        eprintln!(
            "  Private dot product: cleartext={:.4}, private={:.4}, diff={:.6}",
            expected,
            result,
            (result - expected).abs()
        );
    }
}
