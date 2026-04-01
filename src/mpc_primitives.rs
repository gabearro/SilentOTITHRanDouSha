//! MPC helper primitives and softmax building blocks over Fp32 shares.
//!
//! This module exposes the interfaces needed for private softmax:
//! fixed-point config, random bit-style primitives, comparison/truncation hooks,
//! and a softmax routine producing secret-shared attention weights.
//!
//! Current backend (`LocalMpcPrimitives`) is a local simulation backend that keeps
//! data in share form at the API boundaries while using local reconstruction
//! internally. For a party-local distributed runtime (networked openings + Beaver
//! multiplications without a central all-shares view), see `mpc_distributed32`.
//! Non-linear primitives (compare/trunc/exp/reciprocal) still need their full
//! distributed implementations (PRandBit/daBit/edaBit + secure compare/trunc).

use crate::field32::{Fp32, PRIME32};
use rand::Rng;

#[derive(Clone, Copy, Debug)]
pub struct FixedPointConfig {
    pub frac_bits: u32,
    pub int_bits: u32,
    pub total_bits: u32,
}

impl Default for FixedPointConfig {
    fn default() -> Self {
        FixedPointConfig {
            frac_bits: 16,
            int_bits: 15,
            total_bits: 31,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum TruncationMode {
    Deterministic,
    Probabilistic,
}

#[derive(Clone, Copy, Debug)]
pub struct SoftmaxMpcConfig {
    pub clip_bound: f32,
    pub exp_poly_degree: usize,
    pub recip_iters: usize,
    pub trunc_mode: TruncationMode,
    pub fp: FixedPointConfig,
    /// Output weight scale. 512 aligns with quantized activation scale.
    pub output_scale: u32,
}

impl Default for SoftmaxMpcConfig {
    fn default() -> Self {
        SoftmaxMpcConfig {
            clip_bound: 10.0,
            exp_poly_degree: 5,
            recip_iters: 3,
            trunc_mode: TruncationMode::Deterministic,
            fp: FixedPointConfig::default(),
            output_scale: 512,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DaBit {
    pub arithmetic_shares: Vec<Fp32>,
    pub bit_shares: Vec<Fp32>,
}

pub struct LocalMpcPrimitives<'a, R: Rng> {
    lag_coeffs: &'a [Fp32],
    eval_points: &'a [Fp32],
    rng: &'a mut R,
}

impl<'a, R: Rng> LocalMpcPrimitives<'a, R> {
    pub fn new(lag_coeffs: &'a [Fp32], eval_points: &'a [Fp32], rng: &'a mut R) -> Self {
        LocalMpcPrimitives {
            lag_coeffs,
            eval_points,
            rng,
        }
    }

    #[inline]
    fn n(&self) -> usize {
        self.lag_coeffs.len()
    }

    #[inline]
    fn reconstruct_signed_i64(&self, shares: &[Fp32]) -> i64 {
        debug_assert_eq!(shares.len(), self.n());
        let mut rec = Fp32::ZERO;
        for i in 0..shares.len() {
            rec = rec + shares[i] * self.lag_coeffs[i];
        }
        fp32_to_signed_i64(rec)
    }

    #[inline]
    fn share_signed_i64(&mut self, val: i64) -> Vec<Fp32> {
        let secret = fp32_from_signed_i64(val);
        let r = Fp32::random(self.rng);
        let mut out = vec![Fp32::ZERO; self.n()];
        for p in 0..self.n() {
            out[p] = secret + r * self.eval_points[p];
        }
        out
    }

    #[inline]
    fn fixed_scale(cfg: &FixedPointConfig) -> f64 {
        2f64.powi(cfg.frac_bits as i32)
    }

    #[inline]
    fn decode_fixed_signed_i64(v: i64, cfg: &FixedPointConfig) -> f64 {
        v as f64 / Self::fixed_scale(cfg)
    }

    #[inline]
    fn encode_fixed(v: f64, cfg: &FixedPointConfig) -> i64 {
        let scale = Self::fixed_scale(cfg);
        let max_int = ((1i128 << (cfg.total_bits - 1)) - 1) as f64;
        let min_int = (-(1i128 << (cfg.total_bits - 1))) as f64;
        (v * scale).round().clamp(min_int, max_int) as i64
    }

    /// Public helper for fixed-point encoding in the configured domain.
    pub fn encode_fixed_public(v: f64, cfg: &FixedPointConfig) -> i64 {
        Self::encode_fixed(v, cfg)
    }

    /// Public helper for fixed-point decoding from signed integer representation.
    pub fn decode_fixed_public(v: i64, cfg: &FixedPointConfig) -> f64 {
        Self::decode_fixed_signed_i64(v, cfg)
    }

    /// Share a cleartext float vector into fixed-point arithmetic shares.
    pub fn share_fixed_from_f32_batch(
        &mut self,
        values: &[f32],
        cfg: &FixedPointConfig,
    ) -> Vec<Vec<Fp32>> {
        values
            .iter()
            .map(|&v| self.share_signed_i64(Self::encode_fixed(v as f64, cfg)))
            .collect()
    }

    /// Reconstruct fixed-point arithmetic shares back to cleartext float values.
    pub fn reconstruct_fixed_to_f32_batch(
        &self,
        shares: &[Vec<Fp32>],
        cfg: &FixedPointConfig,
    ) -> Vec<f32> {
        shares
            .iter()
            .map(|s| Self::decode_fixed_signed_i64(self.reconstruct_signed_i64(s), cfg) as f32)
            .collect()
    }

    pub fn rand_bits(&mut self, count: usize) -> Vec<Vec<Fp32>> {
        (0..count)
            .map(|_| {
                let b = (self.rng.gen::<u8>() & 1) as i64;
                self.share_signed_i64(b)
            })
            .collect()
    }

    pub fn dabits(&mut self, count: usize) -> Vec<DaBit> {
        self.rand_bits(count)
            .into_iter()
            .map(|bit_shares| DaBit {
                arithmetic_shares: bit_shares.clone(),
                bit_shares,
            })
            .collect()
    }

    pub fn secure_trunc_batch(&mut self, values: &[Vec<Fp32>], frac_bits: u32) -> Vec<Vec<Fp32>> {
        values
            .iter()
            .map(|v| {
                let x = self.reconstruct_signed_i64(v);
                let trunc = if x >= 0 {
                    x >> frac_bits
                } else {
                    -(((-x) >> frac_bits) as i64)
                };
                self.share_signed_i64(trunc)
            })
            .collect()
    }

    pub fn secure_compare_batch(&mut self, lhs: &[Vec<Fp32>], rhs: &[Vec<Fp32>]) -> Vec<Vec<Fp32>> {
        debug_assert_eq!(lhs.len(), rhs.len());
        (0..lhs.len())
            .map(|i| {
                let a = self.reconstruct_signed_i64(&lhs[i]);
                let b = self.reconstruct_signed_i64(&rhs[i]);
                self.share_signed_i64((a < b) as i64)
            })
            .collect()
    }

    pub fn secure_max(&mut self, values: &[Vec<Fp32>]) -> Vec<Fp32> {
        let mut best = i64::MIN;
        for v in values {
            best = best.max(self.reconstruct_signed_i64(v));
        }
        self.share_signed_i64(best)
    }

    pub fn secure_clip_batch(
        &mut self,
        values: &[Vec<Fp32>],
        lo: f32,
        hi: f32,
        cfg: &FixedPointConfig,
    ) -> Vec<Vec<Fp32>> {
        values
            .iter()
            .map(|v| {
                let x = self.reconstruct_signed_i64(v);
                let x_f = Self::decode_fixed_signed_i64(x, cfg);
                let clipped = x_f.clamp(lo as f64, hi as f64);
                self.share_signed_i64(Self::encode_fixed(clipped, cfg))
            })
            .collect()
    }

    pub fn secure_exp_batch(
        &mut self,
        values: &[Vec<Fp32>],
        cfg: &FixedPointConfig,
    ) -> Vec<Vec<Fp32>> {
        values
            .iter()
            .map(|v| {
                let x = self.reconstruct_signed_i64(v);
                let x_f = Self::decode_fixed_signed_i64(x, cfg);
                let e = x_f.exp();
                self.share_signed_i64(Self::encode_fixed(e, cfg))
            })
            .collect()
    }

    pub fn secure_reciprocal(&mut self, value: &[Fp32], cfg: &FixedPointConfig) -> Vec<Fp32> {
        let x = self.reconstruct_signed_i64(value);
        let x_f = Self::decode_fixed_signed_i64(x, cfg);
        let inv = if x_f.abs() < 1e-9 { 0.0 } else { 1.0 / x_f };
        self.share_signed_i64(Self::encode_fixed(inv, cfg))
    }

    /// Batched reciprocal over fixed-point shares.
    pub fn secure_reciprocal_batch(
        &mut self,
        values: &[Vec<Fp32>],
        cfg: &FixedPointConfig,
    ) -> Vec<Vec<Fp32>> {
        values
            .iter()
            .map(|v| self.secure_reciprocal(v, cfg))
            .collect()
    }

    /// Batched inverse square root over fixed-point shares.
    pub fn secure_rsqrt_batch(
        &mut self,
        values: &[Vec<Fp32>],
        cfg: &FixedPointConfig,
    ) -> Vec<Vec<Fp32>> {
        values
            .iter()
            .map(|v| {
                let x = self.reconstruct_signed_i64(v);
                let x_f = Self::decode_fixed_signed_i64(x, cfg);
                let inv = if x_f <= 1e-9 { 0.0 } else { 1.0 / x_f.sqrt() };
                self.share_signed_i64(Self::encode_fixed(inv, cfg))
            })
            .collect()
    }

    /// Batched GELU non-linearity over fixed-point shares.
    pub fn secure_gelu_batch(
        &mut self,
        values: &[Vec<Fp32>],
        cfg: &FixedPointConfig,
    ) -> Vec<Vec<Fp32>> {
        values
            .iter()
            .map(|v| {
                let x = self.reconstruct_signed_i64(v);
                let x_f = Self::decode_fixed_signed_i64(x, cfg);
                let y = gelu_tanh_approx(x_f);
                self.share_signed_i64(Self::encode_fixed(y, cfg))
            })
            .collect()
    }

    /// Batched LayerNorm over fixed-point shares.
    ///
    /// Input layout is row-major flattened `[rows * cols][n]`.
    /// `gamma` and `beta` are public affine parameters applied per-column.
    pub fn secure_layer_norm_batch(
        &mut self,
        values: &[Vec<Fp32>],
        rows: usize,
        cols: usize,
        gamma: Option<&[f32]>,
        beta: Option<&[f32]>,
        epsilon: f32,
        cfg: &FixedPointConfig,
    ) -> Vec<Vec<Fp32>> {
        assert_eq!(
            values.len(),
            rows * cols,
            "values length must equal rows*cols"
        );
        if let Some(gamma) = gamma {
            assert_eq!(gamma.len(), cols, "gamma length mismatch");
        }
        if let Some(beta) = beta {
            assert_eq!(beta.len(), cols, "beta length mismatch");
        }

        let mut out = vec![vec![Fp32::ZERO; self.n()]; values.len()];
        for r in 0..rows {
            let base = r * cols;
            let mut row = vec![0.0f64; cols];
            let mut sum = 0.0f64;
            for c in 0..cols {
                let x = self.reconstruct_signed_i64(&values[base + c]);
                let xf = Self::decode_fixed_signed_i64(x, cfg);
                row[c] = xf;
                sum += xf;
            }

            let mean = sum / cols as f64;
            let mut var = 0.0f64;
            for c in 0..cols {
                let d = row[c] - mean;
                var += d * d;
            }
            var /= cols as f64;
            let inv_std = 1.0f64 / (var + epsilon as f64).sqrt();

            for c in 0..cols {
                let mut y = (row[c] - mean) * inv_std;
                if let Some(gamma) = gamma {
                    y *= gamma[c] as f64;
                }
                if let Some(beta) = beta {
                    y += beta[c] as f64;
                }
                out[base + c] = self.share_signed_i64(Self::encode_fixed(y, cfg));
            }
        }
        out
    }

    /// Convert QK score shares (SCALE² domain) into fixed-point shares using
    /// a public float scale factor.
    pub fn rescale_scores_to_fixed(
        &mut self,
        score_shares: &[Vec<Fp32>],
        public_scale: f64,
        cfg: &FixedPointConfig,
    ) -> Vec<Vec<Fp32>> {
        score_shares
            .iter()
            .map(|s| {
                let x = self.reconstruct_signed_i64(s) as f64;
                self.share_signed_i64(Self::encode_fixed(x * public_scale, cfg))
            })
            .collect()
    }

    /// Accurate softmax over secret-shared fixed-point scores.
    ///
    /// Returns shares with integer scale `cfg.output_scale`.
    pub fn softmax_scores(
        &mut self,
        score_shares: &[Vec<Fp32>], // [num_heads * num_keys][n]
        num_heads: usize,
        num_keys: usize,
        cfg: &SoftmaxMpcConfig,
    ) -> Vec<Vec<Fp32>> {
        assert_eq!(score_shares.len(), num_heads * num_keys);
        let mut out = vec![vec![Fp32::ZERO; self.n()]; score_shares.len()];

        for h in 0..num_heads {
            let base = h * num_keys;
            let mut scores = vec![0.0f64; num_keys];
            let mut max_v = f64::NEG_INFINITY;
            for kp in 0..num_keys {
                let x = self.reconstruct_signed_i64(&score_shares[base + kp]);
                let xf = Self::decode_fixed_signed_i64(x, &cfg.fp);
                scores[kp] = xf;
                if xf > max_v {
                    max_v = xf;
                }
            }

            if !max_v.is_finite() {
                let uniform = (cfg.output_scale as f64 / num_keys as f64).round() as i64;
                for kp in 0..num_keys {
                    out[base + kp] = self.share_signed_i64(uniform);
                }
                continue;
            }

            let mut sum = 0.0f64;
            for kp in 0..num_keys {
                let centered = (scores[kp] - max_v).clamp(-(cfg.clip_bound as f64), 0.0);
                let e = centered.exp();
                scores[kp] = e;
                sum += e;
            }

            if !sum.is_finite() || sum <= 0.0 {
                let uniform = (cfg.output_scale as f64 / num_keys as f64).round() as i64;
                for kp in 0..num_keys {
                    out[base + kp] = self.share_signed_i64(uniform);
                }
                continue;
            }

            for kp in 0..num_keys {
                let w = (scores[kp] / sum) * cfg.output_scale as f64;
                out[base + kp] = self.share_signed_i64(w.round() as i64);
            }
        }

        out
    }
}

#[inline]
fn fp32_from_signed_i64(v: i64) -> Fp32 {
    Fp32::from_reduced(v.rem_euclid(PRIME32 as i64) as u32)
}

#[inline]
fn fp32_to_signed_i64(v: Fp32) -> i64 {
    let raw = v.raw() as i64;
    if raw <= (PRIME32 as i64) / 2 {
        raw
    } else {
        raw - PRIME32 as i64
    }
}

#[inline]
fn gelu_tanh_approx(x: f64) -> f64 {
    let coeff = 0.044_715f64;
    let sqrt_2_pi = 0.797_884_560_802_865_4f64;
    0.5 * x * (1.0 + (sqrt_2_pi * (x + coeff * x * x * x)).tanh())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field32_shamir::Shamir32;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn test_rand_bits_reconstruct_to_binary() {
        let n = 5usize;
        let shamir = Shamir32::new(n, 1).unwrap();
        let lag: Vec<Fp32> = shamir.lagrange_coefficients().to_vec();
        let eval_points: Vec<Fp32> = (1..=n as u32).map(Fp32::new).collect();
        let mut rng = ChaCha20Rng::seed_from_u64(7);
        let mut mpc = LocalMpcPrimitives::new(&lag, &eval_points, &mut rng);
        let bits = mpc.rand_bits(64);
        for b in bits {
            let rec = mpc.reconstruct_signed_i64(&b);
            assert!(rec == 0 || rec == 1, "bit reconstructed to {}", rec);
        }
    }

    #[test]
    fn test_softmax_scores_sum_to_scale() {
        let n = 5usize;
        let shamir = Shamir32::new(n, 1).unwrap();
        let lag: Vec<Fp32> = shamir.lagrange_coefficients().to_vec();
        let eval_points: Vec<Fp32> = (1..=n as u32).map(Fp32::new).collect();
        let mut rng = ChaCha20Rng::seed_from_u64(9);
        let mut mpc = LocalMpcPrimitives::new(&lag, &eval_points, &mut rng);
        let cfg = SoftmaxMpcConfig::default();

        let score_vals = [1.5f64, 0.0, -1.0, 2.0];
        let shares: Vec<Vec<Fp32>> = score_vals
            .iter()
            .map(|&v| {
                let enc = LocalMpcPrimitives::<ChaCha20Rng>::encode_fixed(v, &cfg.fp);
                mpc.share_signed_i64(enc)
            })
            .collect();

        let out = mpc.softmax_scores(&shares, 1, 4, &cfg);
        let mut sum = 0i64;
        for w in out {
            sum += mpc.reconstruct_signed_i64(&w);
        }
        assert!(
            (sum - cfg.output_scale as i64).abs() <= 2,
            "softmax weights sum {}, expected ~{}",
            sum,
            cfg.output_scale
        );
    }

    #[test]
    fn test_secure_rsqrt_batch() {
        let n = 5usize;
        let shamir = Shamir32::new(n, 1).unwrap();
        let lag: Vec<Fp32> = shamir.lagrange_coefficients().to_vec();
        let eval_points: Vec<Fp32> = (1..=n as u32).map(Fp32::new).collect();
        let mut rng = ChaCha20Rng::seed_from_u64(11);
        let mut mpc = LocalMpcPrimitives::new(&lag, &eval_points, &mut rng);
        let cfg = FixedPointConfig::default();

        let vals = [0.25f32, 1.0, 4.0];
        let shared = mpc.share_fixed_from_f32_batch(&vals, &cfg);
        let out = mpc.secure_rsqrt_batch(&shared, &cfg);
        let dec = mpc.reconstruct_fixed_to_f32_batch(&out, &cfg);
        let expected = [2.0f32, 1.0, 0.5];
        for i in 0..vals.len() {
            assert!(
                (dec[i] - expected[i]).abs() < 0.02,
                "idx {} expected {}, got {}",
                i,
                expected[i],
                dec[i]
            );
        }
    }

    #[test]
    fn test_secure_gelu_batch() {
        let n = 5usize;
        let shamir = Shamir32::new(n, 1).unwrap();
        let lag: Vec<Fp32> = shamir.lagrange_coefficients().to_vec();
        let eval_points: Vec<Fp32> = (1..=n as u32).map(Fp32::new).collect();
        let mut rng = ChaCha20Rng::seed_from_u64(12);
        let mut mpc = LocalMpcPrimitives::new(&lag, &eval_points, &mut rng);
        let cfg = FixedPointConfig::default();

        let vals = [-3.0f32, -1.0, 0.0, 1.0, 3.0];
        let shared = mpc.share_fixed_from_f32_batch(&vals, &cfg);
        let out = mpc.secure_gelu_batch(&shared, &cfg);
        let dec = mpc.reconstruct_fixed_to_f32_batch(&out, &cfg);
        for i in 0..vals.len() {
            let expected = gelu_tanh_approx(vals[i] as f64) as f32;
            assert!(
                (dec[i] - expected).abs() < 0.03,
                "idx {} expected {}, got {}",
                i,
                expected,
                dec[i]
            );
        }
    }

    #[test]
    fn test_secure_layer_norm_batch() {
        let n = 5usize;
        let shamir = Shamir32::new(n, 1).unwrap();
        let lag: Vec<Fp32> = shamir.lagrange_coefficients().to_vec();
        let eval_points: Vec<Fp32> = (1..=n as u32).map(Fp32::new).collect();
        let mut rng = ChaCha20Rng::seed_from_u64(13);
        let mut mpc = LocalMpcPrimitives::new(&lag, &eval_points, &mut rng);
        let cfg = FixedPointConfig::default();

        let rows = 2usize;
        let cols = 4usize;
        let vals = [1.0f32, 2.0, 3.0, 4.0, -2.0, 0.0, 2.0, 4.0];
        let gamma = [1.0f32, 0.5, 1.5, 1.0];
        let beta = [0.0f32, 0.1, -0.1, 0.2];

        let shared = mpc.share_fixed_from_f32_batch(&vals, &cfg);
        let out =
            mpc.secure_layer_norm_batch(&shared, rows, cols, Some(&gamma), Some(&beta), 1e-5, &cfg);
        let dec = mpc.reconstruct_fixed_to_f32_batch(&out, &cfg);

        for r in 0..rows {
            let base = r * cols;
            let row = &vals[base..base + cols];
            let mean = row.iter().copied().sum::<f32>() / cols as f32;
            let var = row
                .iter()
                .map(|&x| {
                    let d = x - mean;
                    d * d
                })
                .sum::<f32>()
                / cols as f32;
            let inv = 1.0f32 / (var + 1e-5f32).sqrt();
            for c in 0..cols {
                let expected = (row[c] - mean) * inv * gamma[c] + beta[c];
                let got = dec[base + c];
                assert!(
                    (got - expected).abs() < 0.03,
                    "row {}, col {} expected {}, got {}",
                    r,
                    c,
                    expected,
                    got
                );
            }
        }
    }
}
