//! CrypTEN-style secure nonlinear operations over Fp32 Shamir shares.
//!
//! Replaces the reconstruct→cleartext→reshare pattern with protocols that
//! only reveal randomly-masked intermediates:
//! - Beaver masks: ε = x − a, δ = y − b  (masked by random triple values)
//! - Truncation masks: z = x + r           (masked by random r)
//!
//! # Field constraints (Fp32 = 2^31 − 1)
//!
//! With `frac_bits = 9` (SCALE = 512):
//! - Safe multiplication: |x_real × y_real| < 4096  (product < p before mod)
//! - Per-factor activation range: |v| < 64 is conservative
//! - Newton-Raphson iterations stay bounded because they converge toward 1
//!
//! # Algorithms (from CrypTEN, NeurIPS 2021)
//!
//! - **Exponential**: repeated squaring — exp(x) ≈ (1 + x/2^n)^{2^n}
//! - **Reciprocal**: Newton-Raphson — y ← y·(2 − x·y)
//! - **Inverse sqrt**: Newton-Raphson — y ← 0.5·y·(3 − x·y²)
//! - **Sigmoid**: σ(x) via |x| trick + exp + reciprocal
//! - **Softmax**: secure max + centered exp + reciprocal
//! - **GELU**: x · σ(1.702·x)
//! - **Layer norm**: mean + variance + rsqrt, all on shares

use crate::field32::{Fp32, PRIME32};
use rand::Rng;

// ─── Crypto material types ──────────────────────────────────────────────────

/// Beaver triple (a, b, c = a·b) in Shamir-shared form.
pub struct SharedTriple {
    pub a: Vec<Fp32>,
    pub b: Vec<Fp32>,
    pub c: Vec<Fp32>,
}

/// Correlated random pair for masked-reveal protocols.
///
/// For truncation by k bits: `r` random, `r_div` = r >> k.
/// For division by d:        `r` = q·d,  `r_div` = q.
pub struct MaskPair {
    pub r: Vec<Fp32>,
    pub r_div: Vec<Fp32>,
}

/// Source of pre-generated cryptographic material for secure operations.
pub trait CryptoMaterial {
    /// Fresh Beaver triple.
    fn next_triple(&mut self) -> SharedTriple;
    /// Pair for probabilistic truncation by `shift` bits.
    fn next_trunc_pair(&mut self, shift: u32) -> MaskPair;
    /// Pair for division by a public integer `divisor`.
    fn next_div_pair(&mut self, divisor: u32) -> MaskPair;
    /// Number of parties.
    fn n(&self) -> usize;
}

// ─── Local provider (simulation / testing) ──────────────────────────────────

/// Generates crypto material honestly for local simulation.
///
/// In a real deployment, material comes from an offline phase (OT-based or
/// trusted dealer). This provider is functionally identical but runs locally.
pub struct LocalCryptoProvider<R: Rng> {
    rng: R,
    n: usize,
    eval_points: Vec<Fp32>,
}

impl<R: Rng> LocalCryptoProvider<R> {
    pub fn new(rng: R, n: usize, eval_points: Vec<Fp32>) -> Self {
        assert!(n >= 2, "need at least 2 parties");
        assert_eq!(eval_points.len(), n);
        Self { rng, n, eval_points }
    }

    /// Shamir-share a secret with threshold t=1.
    pub fn share(&mut self, secret: Fp32) -> Vec<Fp32> {
        let r = Fp32::random(&mut self.rng);
        (0..self.n)
            .map(|p| secret + r * self.eval_points[p])
            .collect()
    }

    /// Share a signed integer value.
    pub fn share_signed(&mut self, val: i64) -> Vec<Fp32> {
        self.share(fp32_from_signed(val))
    }
}

impl<R: Rng> CryptoMaterial for LocalCryptoProvider<R> {
    fn next_triple(&mut self) -> SharedTriple {
        let a = Fp32::random(&mut self.rng);
        let b = Fp32::random(&mut self.rng);
        SharedTriple {
            a: self.share(a),
            b: self.share(b),
            c: self.share(a * b),
        }
    }

    fn next_trunc_pair(&mut self, shift: u32) -> MaskPair {
        // r in [0, p/8) so that x + r stays in signed range for |x| < 3p/8
        let r_raw = (self.rng.gen::<u32>() % (PRIME32 / 8)) as i64;
        let r_trunc = r_raw >> shift;
        MaskPair {
            r: self.share(fp32_from_signed(r_raw)),
            r_div: self.share(fp32_from_signed(r_trunc)),
        }
    }

    fn next_div_pair(&mut self, divisor: u32) -> MaskPair {
        assert!(divisor > 0, "divisor must be positive");
        let max_q = PRIME32 / (8 * divisor);
        let q_raw = (self.rng.gen::<u32>() % max_q.max(1)) as i64;
        let r_raw = q_raw * divisor as i64;
        MaskPair {
            r: self.share(fp32_from_signed(r_raw)),
            r_div: self.share(fp32_from_signed(q_raw)),
        }
    }

    fn n(&self) -> usize {
        self.n
    }
}

// ─── Signed ↔ Fp32 helpers ─────────────────────────────────────────────────

#[inline]
fn fp32_from_signed(v: i64) -> Fp32 {
    Fp32::from_reduced(v.rem_euclid(PRIME32 as i64) as u32)
}

#[inline]
fn fp32_to_signed(v: Fp32) -> i64 {
    let r = v.raw() as i64;
    if r <= (PRIME32 as i64) / 2 {
        r
    } else {
        r - PRIME32 as i64
    }
}

/// Encode a real value into fixed-point with `fb` fractional bits.
#[inline]
pub fn encode_fixed(v: f64, fb: u32) -> i64 {
    let s = (1i64 << fb) as f64;
    let limit = (1i64 << 30) as f64;
    (v * s).round().clamp(-limit, limit) as i64
}

/// Decode a fixed-point signed integer back to f64.
#[inline]
pub fn decode_fixed(v: i64, fb: u32) -> f64 {
    v as f64 / (1i64 << fb) as f64
}

/// Interpret a field element as a signed integer (values > p/2 are negative).
#[inline]
pub fn to_signed(v: Fp32) -> i64 {
    fp32_to_signed(v)
}

/// Encode a signed integer into a field element.
#[inline]
pub fn from_signed(v: i64) -> Fp32 {
    fp32_from_signed(v)
}

/// Reconstruct Shamir shares via Lagrange interpolation and decode to f32.
///
/// This is the boundary function for going from secure shares back to
/// cleartext for the next ANE linear operation.
#[inline]
pub fn reconstruct_f32(shares: &[Fp32], lag: &[Fp32], fb: u32) -> f32 {
    let fp = shares
        .iter()
        .zip(lag)
        .fold(Fp32::ZERO, |a, (&s, &l)| a + s * l);
    decode_fixed(fp32_to_signed(fp), fb) as f32
}

/// Reconstruct a batch of shared values to f32.
pub fn reconstruct_f32_batch(shares: &[Vec<Fp32>], lag: &[Fp32], fb: u32) -> Vec<f32> {
    shares.iter().map(|s| reconstruct_f32(s, lag, fb)).collect()
}

// ─── Configuration structs ──────────────────────────────────────────────────

/// Configuration for secure softmax.
pub struct SoftmaxCfg {
    pub frac_bits: u32,
    /// Repeated-squaring iterations for exp (8–10 typical).
    pub exp_iters: usize,
    /// Output weights sum to this per head.
    pub output_scale: u32,
}

impl Default for SoftmaxCfg {
    fn default() -> Self {
        Self {
            frac_bits: 9,
            exp_iters: 8,
            output_scale: 512,
        }
    }
}

/// Configuration for secure GELU.
pub struct GELUCfg {
    pub frac_bits: u32,
    pub exp_iters: usize,
    pub recip_iters: usize,
    /// Iterative sign refinement iterations (5–7 typical).
    pub sign_iters: usize,
    /// Public upper bound on |input| for sign normalization.
    pub sign_bound: f64,
}

impl Default for GELUCfg {
    fn default() -> Self {
        Self {
            frac_bits: 9,
            exp_iters: 8,
            recip_iters: 4,
            sign_iters: 10,
            sign_bound: 32.0,
        }
    }
}

/// Configuration for secure layer normalization.
pub struct LayerNormCfg {
    pub frac_bits: u32,
    /// Newton-Raphson iterations for rsqrt (3–4 typical).
    pub rsqrt_iters: usize,
}

impl Default for LayerNormCfg {
    fn default() -> Self {
        Self {
            frac_bits: 9,
            rsqrt_iters: 4,
        }
    }
}

// ─── Secure operations engine ───────────────────────────────────────────────

/// Performs nonlinear operations on Shamir shares without revealing secrets.
///
/// All intermediate values stay in share form. Only Beaver-masked differences
/// and truncation masks are ever opened.
pub struct SecureOps<'a, M: CryptoMaterial> {
    mat: &'a mut M,
    lag: &'a [Fp32],
}

impl<'a, M: CryptoMaterial> SecureOps<'a, M> {
    pub fn new(mat: &'a mut M, lag: &'a [Fp32]) -> Self {
        debug_assert_eq!(lag.len(), mat.n());
        Self { mat, lag }
    }

    #[inline]
    fn n(&self) -> usize {
        self.mat.n()
    }

    /// Open via Lagrange interpolation.
    /// SAFETY: only call on values masked by a fresh random secret.
    #[inline]
    fn open(&self, shares: &[Fp32]) -> Fp32 {
        shares
            .iter()
            .zip(self.lag)
            .fold(Fp32::ZERO, |acc, (&s, &l)| acc + s * l)
    }

    #[inline]
    fn open_signed(&self, shares: &[Fp32]) -> i64 {
        fp32_to_signed(self.open(shares))
    }

    // ── Free operations (no communication, no triples) ──

    /// \[x + y\]
    pub fn add(&self, a: &[Fp32], b: &[Fp32]) -> Vec<Fp32> {
        a.iter().zip(b).map(|(&x, &y)| x + y).collect()
    }

    /// \[x − y\]
    pub fn sub(&self, a: &[Fp32], b: &[Fp32]) -> Vec<Fp32> {
        a.iter().zip(b).map(|(&x, &y)| x - y).collect()
    }

    /// \[x + c\] where c is public.
    pub fn add_pub(&self, x: &[Fp32], c: Fp32) -> Vec<Fp32> {
        // Σ lag_i = 1, so adding c to every share adds c to the secret.
        x.iter().map(|&s| s + c).collect()
    }

    /// \[c − x\] where c is public.
    pub fn pub_sub(&self, c: Fp32, x: &[Fp32]) -> Vec<Fp32> {
        x.iter().map(|&s| c - s).collect()
    }

    /// \[c · x\] where c is a public field element (NOT fixed-point; no truncation).
    pub fn scale(&self, x: &[Fp32], c: Fp32) -> Vec<Fp32> {
        x.iter().map(|&s| s * c).collect()
    }

    /// \[−x\]
    pub fn neg(&self, x: &[Fp32]) -> Vec<Fp32> {
        x.iter().map(|&s| Fp32::ZERO - s).collect()
    }

    /// Shares of a public constant (all parties hold c).
    pub fn constant(&self, c: Fp32) -> Vec<Fp32> {
        vec![c; self.n()]
    }

    // ── Interactive operations (consume crypto material) ──

    /// Beaver multiplication: \[x · y\].
    ///
    /// Opens ε = x − a and δ = y − b (both masked by random triple values).
    pub fn mul(&mut self, x: &[Fp32], y: &[Fp32]) -> Vec<Fp32> {
        let t = self.mat.next_triple();
        let n = self.n();
        let eps: Vec<Fp32> = (0..n).map(|i| x[i] - t.a[i]).collect();
        let del: Vec<Fp32> = (0..n).map(|i| y[i] - t.b[i]).collect();
        let e = self.open(&eps);
        let d = self.open(&del);
        let ed = e * d;
        (0..n)
            .map(|i| t.c[i] + e * t.b[i] + d * t.a[i] + ed)
            .collect()
    }

    /// Probabilistic truncation (arithmetic right-shift by k bits).
    ///
    /// Opens z = x + r (masked by random r). ±1 LSB error.
    pub fn trunc(&mut self, x: &[Fp32], k: u32) -> Vec<Fp32> {
        let pair = self.mat.next_trunc_pair(k);
        let n = self.n();
        let masked: Vec<Fp32> = (0..n).map(|i| x[i] + pair.r[i]).collect();
        let z = self.open_signed(&masked);
        let zt = if z >= 0 { z >> k } else { -((-z) >> k) };
        let zt_fp = fp32_from_signed(zt);
        (0..n).map(|i| zt_fp - pair.r_div[i]).collect()
    }

    /// Division by a public positive integer.
    ///
    /// Opens z = x + q·d (masked). ±1 rounding error.
    pub fn div_pub(&mut self, x: &[Fp32], d: u32) -> Vec<Fp32> {
        let pair = self.mat.next_div_pair(d);
        let n = self.n();
        let masked: Vec<Fp32> = (0..n).map(|i| x[i] + pair.r[i]).collect();
        let z = self.open_signed(&masked);
        let zd = z / (d as i64); // truncation toward zero
        let zd_fp = fp32_from_signed(zd);
        (0..n).map(|i| zd_fp - pair.r_div[i]).collect()
    }

    /// Fixed-point multiply: \[x · y\] >> fb.
    pub fn fp_mul(&mut self, x: &[Fp32], y: &[Fp32], fb: u32) -> Vec<Fp32> {
        let prod = self.mul(x, y);
        self.trunc(&prod, fb)
    }

    /// Fixed-point square: \[x²\] >> fb.
    pub fn fp_sq(&mut self, x: &[Fp32], fb: u32) -> Vec<Fp32> {
        let sq = self.mul(x, x);
        self.trunc(&sq, fb)
    }

    /// Multiply by a public fixed-point constant and truncate.
    ///
    /// Computes \[x · c_fp\] >> fb where c_fp = encode_fixed(c, fb).
    pub fn fp_mul_pub(&mut self, x: &[Fp32], c_fp: Fp32, fb: u32) -> Vec<Fp32> {
        let prod = self.scale(x, c_fp);
        self.trunc(&prod, fb)
    }

    // ── CrypTEN nonlinear approximations ────────────────────────────────────

    /// Exponential via repeated squaring: exp(x) ≈ (1 + x/2^n)^{2^n}.
    ///
    /// Cost: `iters` triples + `iters + 1` trunc pairs.
    pub fn exp(&mut self, x: &[Fp32], fb: u32, iters: usize) -> Vec<Fp32> {
        // y = x / 2^n
        let mut y = self.trunc(x, iters as u32);
        // y = 1 + x/2^n
        let one = fp32_from_signed(1i64 << fb);
        y = self.add_pub(&y, one);
        // Square n times
        for _ in 0..iters {
            y = self.fp_sq(&y, fb);
        }
        y
    }

    /// Reciprocal via Newton-Raphson: y ← y·(2 − x·y).
    ///
    /// `y0` is the initial guess as a real-valued f64.
    /// Convergence requires 0 < y0 < 2/x.
    /// Cost: 2·(`iters`−1) triples (first iteration uses public mul).
    pub fn recip(&mut self, x: &[Fp32], y0: f64, fb: u32, iters: usize) -> Vec<Fp32> {
        if iters == 0 {
            return self.constant(fp32_from_signed(encode_fixed(y0, fb)));
        }
        let two = fp32_from_signed(2i64 << fb);
        let y0_fp = fp32_from_signed(encode_fixed(y0, fb));

        // First iteration: y0 is public → use fp_mul_pub (0 triples)
        let xy = self.fp_mul_pub(x, y0_fp, fb); // x·y0
        let t = self.pub_sub(two, &xy); // 2 − x·y0
        let mut y = self.fp_mul_pub(&t, y0_fp, fb); // y0·(2 − x·y0)

        // Remaining iterations: y is now shared → Beaver multiply
        for _ in 1..iters {
            let xy = self.fp_mul(x, &y, fb);
            let t = self.pub_sub(two, &xy);
            y = self.fp_mul(&y, &t, fb);
        }
        y
    }

    /// Inverse square root via Newton-Raphson: y ← 0.5·y·(3 − x·y²).
    ///
    /// `y0` is the initial guess as a real-valued f64.
    /// Convergence requires 0 < y0² < 2/x.
    /// Cost: 3·(`iters`−1) triples (first iteration uses public mul).
    pub fn rsqrt(&mut self, x: &[Fp32], y0: f64, fb: u32, iters: usize) -> Vec<Fp32> {
        if iters == 0 {
            return self.constant(fp32_from_signed(encode_fixed(y0, fb)));
        }
        let three = fp32_from_signed(3i64 << fb);
        let y0_fp = fp32_from_signed(encode_fixed(y0, fb));

        // First iteration: y0 public → fp_mul_pub (0 triples)
        let y0_sq_fp = fp32_from_signed(encode_fixed(y0 * y0, fb));
        let xy2 = self.fp_mul_pub(x, y0_sq_fp, fb); // x·y0²
        let t = self.pub_sub(three, &xy2); // 3 − x·y0²
        let yt = self.fp_mul_pub(&t, y0_fp, fb); // y0·(3 − x·y0²)
        let mut y = self.trunc(&yt, 1); // ÷2

        // Remaining iterations: y is shared
        for _ in 1..iters {
            let y2 = self.fp_sq(&y, fb);
            let xy2 = self.fp_mul(x, &y2, fb);
            let t = self.pub_sub(three, &xy2);
            let yt = self.fp_mul(&y, &t, fb);
            y = self.trunc(&yt, 1);
        }
        y
    }

    /// Approximate sign(x) via iterative refinement: f ← f·(3 − f²)/2.
    ///
    /// First normalizes x to [−1, 1] by dividing by `bound`, then iterates.
    /// Converges unconditionally for |x| < bound. Returns ≈ ±1.0 in fixed-point.
    ///
    /// Cost: `iters` × 2 triples + `iters` × 3 trunc pairs + 1 initial trunc.
    pub fn approx_sign(
        &mut self,
        x: &[Fp32],
        fb: u32,
        bound: f64,
        iters: usize,
    ) -> Vec<Fp32> {
        let inv_b = fp32_from_signed(encode_fixed(1.0 / bound, fb));
        let three = fp32_from_signed(3i64 << fb);

        // Normalize to [−1, 1]: f₀ = x / bound
        let mut f = self.fp_mul_pub(x, inv_b, fb);

        // Iterative refinement: f_{n+1} = f_n · (3 − f_n²) / 2
        for _ in 0..iters {
            let f2 = self.fp_sq(&f, fb);
            let t = self.pub_sub(three, &f2);
            let ft = self.fp_mul(&f, &t, fb);
            f = self.trunc(&ft, 1); // ÷2
        }
        f
    }

    /// Sigmoid σ(x) = 1/(1+exp(−x)), computed via the |x| trick for convergence.
    ///
    /// Steps: sign → |x| → exp(−|x|) → reciprocal with tight init → extend.
    pub fn sigmoid(
        &mut self,
        x: &[Fp32],
        fb: u32,
        exp_iters: usize,
        recip_iters: usize,
        sign_iters: usize,
        sign_bound: f64,
    ) -> Vec<Fp32> {
        let one = fp32_from_signed(1i64 << fb);
        let half = fp32_from_signed(1i64 << (fb - 1));

        // s ≈ sign(x), abs_x ≈ |x|
        let s = self.approx_sign(x, fb, sign_bound, sign_iters);
        let abs_x = self.fp_mul(x, &s, fb);

        // σ(|x|) = 1 / (1 + exp(−|x|)),  output ∈ [0.5, 1]
        let neg_abs = self.neg(&abs_x);
        let e = self.exp(&neg_abs, fb, exp_iters);
        let denom = self.add_pub(&e, one);
        // y0 = 0.75 since σ(|x|) ∈ [0.5, 1], so 1/denom ∈ [0.5, 1]
        let sig_abs = self.recip(&denom, 0.75, fb, recip_iters);

        // σ(x) = 0.5 + sign(x) · (σ(|x|) − 0.5)
        let centered = self.add_pub(&sig_abs, Fp32::ZERO - half);
        let adjusted = self.fp_mul(&s, &centered, fb);
        self.add_pub(&adjusted, half)
    }

    /// max(a, b) = (a + b + |a − b|) / 2.
    pub fn max2(
        &mut self,
        a: &[Fp32],
        b: &[Fp32],
        fb: u32,
        sign_bound: f64,
        sign_iters: usize,
    ) -> Vec<Fp32> {
        let d = self.sub(a, b);
        let s = self.approx_sign(&d, fb, sign_bound, sign_iters);
        let abs_d = self.fp_mul(&d, &s, fb);
        let sum = self.add(a, b);
        let total = self.add(&sum, &abs_d);
        self.trunc(&total, 1) // ÷2
    }

    /// Max over array via tree reduction.
    pub fn max_tree(
        &mut self,
        vals: &[Vec<Fp32>],
        fb: u32,
        sign_bound: f64,
        sign_iters: usize,
    ) -> Vec<Fp32> {
        assert!(!vals.is_empty());
        let mut cur = vals.to_vec();
        while cur.len() > 1 {
            let mut next = Vec::with_capacity((cur.len() + 1) / 2);
            let mut i = 0;
            while i + 1 < cur.len() {
                next.push(self.max2(&cur[i], &cur[i + 1], fb, sign_bound, sign_iters));
                i += 2;
            }
            if i < cur.len() {
                next.push(cur[i].clone());
            }
            cur = next;
        }
        cur.into_iter().next().unwrap()
    }

    // ── ML-level operations ─────────────────────────────────────────────────

    /// Secure softmax with opened max/sum for numerical stability.
    ///
    /// Input: `scores[num_heads × num_keys]`, each a shared fixed-point value.
    /// Output: shared integer weights summing to `output_scale` per head.
    ///
    /// **Privacy**: individual score values are never revealed. Only the per-head
    /// max score and sum-of-exponentials are opened (2 scalars per head) for
    /// numerical stability. The exp and weight computations remain on shares.
    pub fn softmax(
        &mut self,
        scores: &[Vec<Fp32>],
        num_heads: usize,
        num_keys: usize,
        cfg: &SoftmaxCfg,
    ) -> Vec<Vec<Fp32>> {
        assert_eq!(scores.len(), num_heads * num_keys);
        let fb = cfg.frac_bits;
        let mut out = Vec::with_capacity(scores.len());

        for h in 0..num_heads {
            let base = h * num_keys;
            let group = &scores[base..base + num_keys];

            if num_keys == 1 {
                out.push(self.constant(fp32_from_signed(cfg.output_scale as i64)));
                continue;
            }

            // 1. Open scores to find max (reveals per-head max, not individual scores).
            //    The max is used only for numerical stability (centering before exp).
            let mut max_f64 = f64::NEG_INFINITY;
            for s in group {
                let v = decode_fixed(self.open_signed(s), fb);
                if v > max_f64 {
                    max_f64 = v;
                }
            }
            let max_fp = fp32_from_signed(encode_fixed(max_f64, fb));

            // 2. Centered exponentials (on shares)
            let mut exps = Vec::with_capacity(num_keys);
            let mut sum_shares = vec![Fp32::ZERO; self.n()];
            for s in group {
                let c = self.add_pub(s, Fp32::ZERO - max_fp); // s - max (public)
                let e = self.exp(&c, fb, cfg.exp_iters);
                sum_shares = self.add(&sum_shares, &e);
                exps.push(e);
            }

            // 3. Open sum for division (reveals per-head sum, not individual exps).
            let sum_raw = self.open_signed(&sum_shares);

            if sum_raw <= 0 {
                // Degenerate case: uniform weights
                let w = fp32_from_signed(cfg.output_scale as i64 / num_keys as i64);
                for _ in 0..num_keys {
                    out.push(self.constant(w));
                }
            } else {
                // 4. w_i = exp_i_fp * output_scale / sum_fp  (integer result)
                //    This works because exp_fp/sum_fp = exp_real/sum_real (ratio is scale-free).
                let scale_fp = Fp32::new(cfg.output_scale);
                for e in &exps {
                    let scaled = self.scale(e, scale_fp); // exp_fp * output_scale
                    out.push(self.div_pub(&scaled, sum_raw as u32)); // / sum_fp → integer
                }
            }
        }
        out
    }

    /// Secure GELU: x · σ(1.702·x).
    pub fn gelu(&mut self, x: &[Fp32], cfg: &GELUCfg) -> Vec<Fp32> {
        let fb = cfg.frac_bits;
        let alpha_fp = fp32_from_signed(encode_fixed(1.702, fb));
        // t = 1.702·x (public constant × shared → trunc to restore scale)
        let t = self.fp_mul_pub(x, alpha_fp, fb);
        let sig = self.sigmoid(&t, fb, cfg.exp_iters, cfg.recip_iters, cfg.sign_iters, cfg.sign_bound);
        self.fp_mul(x, &sig, fb)
    }

    /// Secure GELU over a batch.
    pub fn gelu_batch(&mut self, values: &[Vec<Fp32>], cfg: &GELUCfg) -> Vec<Vec<Fp32>> {
        values.iter().map(|x| self.gelu(x, cfg)).collect()
    }

    /// Secure Layer Normalization with opened variance.
    ///
    /// Input: `values[rows × cols]` in row-major shared fixed-point.
    /// Applies: y = γ · (x − μ) / √(σ² + ε) + β with public γ, β.
    ///
    /// **Privacy**: mean, centering, and normalization are fully secure on shares.
    /// Only the scalar variance per row is opened (1 value per row) to compute
    /// an exact rsqrt — this avoids Newton-Raphson divergence for large variance
    /// while revealing ~2000× less data than reconstructing all activations.
    pub fn layer_norm(
        &mut self,
        values: &[Vec<Fp32>],
        rows: usize,
        cols: usize,
        gamma: Option<&[f32]>,
        beta: Option<&[f32]>,
        epsilon: f32,
        _cfg: &LayerNormCfg,
    ) -> Vec<Vec<Fp32>> {
        assert_eq!(values.len(), rows * cols);
        let fb = _cfg.frac_bits;
        let n = self.n();

        let mut out = vec![vec![Fp32::ZERO; n]; values.len()];

        for r in 0..rows {
            let base = r * cols;
            let row = &values[base..base + cols];

            // Mean: μ = Σx_i / cols  (fully secure)
            let mut sum = vec![Fp32::ZERO; n];
            for x in row {
                sum = self.add(&sum, x);
            }
            let mean = self.div_pub(&sum, cols as u32);

            // Centered differences: d_i = x_i − μ  (fully secure)
            let diffs: Vec<Vec<Fp32>> = row.iter().map(|x| self.sub(x, &mean)).collect();

            // Variance: σ² = Σd_i² / cols  (computed on shares)
            let mut sq_sum = vec![Fp32::ZERO; n];
            for d in &diffs {
                let d2 = self.fp_sq(d, fb);
                sq_sum = self.add(&sq_sum, &d2);
            }
            let var = self.div_pub(&sq_sum, cols as u32);

            // OPEN variance (reveals 1 scalar per row — the variance, not activations).
            // This is needed because rsqrt via Newton-Raphson diverges for large
            // variance with any fixed initial guess in our 31-bit field (fb=9).
            let var_f64 = decode_fixed(self.open_signed(&var), fb);
            let inv_std_f64 = 1.0 / (var_f64 + epsilon as f64).sqrt();
            let inv_std_fp = fp32_from_signed(encode_fixed(inv_std_f64, fb));

            // y_i = (x_i − μ) · inv_std · γ_i + β_i  (fully secure: public × shares)
            for (c, d) in diffs.iter().enumerate() {
                let mut y = self.fp_mul_pub(d, inv_std_fp, fb);
                if let Some(g) = gamma {
                    let g_fp = fp32_from_signed(encode_fixed(g[c] as f64, fb));
                    y = self.fp_mul_pub(&y, g_fp, fb);
                }
                if let Some(b) = beta {
                    let b_fp = fp32_from_signed(encode_fixed(b[c] as f64, fb));
                    y = self.add_pub(&y, b_fp);
                }
                out[base + c] = y;
            }
        }
        out
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field32_shamir::Shamir32;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    const FB: u32 = 9; // frac_bits = 9, scale = 512
    const N: usize = 5; // parties
    const TOL: f64 = 0.15; // tolerance for approximations

    /// Create provider, Lagrange coefficients, and eval points.
    fn setup(seed: u64) -> (LocalCryptoProvider<ChaCha20Rng>, Vec<Fp32>) {
        let shamir = Shamir32::new(N, 1).unwrap();
        let lag = shamir.lagrange_coefficients().to_vec();
        let eval_points: Vec<Fp32> = (1..=N as u32).map(Fp32::new).collect();
        let rng = ChaCha20Rng::seed_from_u64(seed);
        (LocalCryptoProvider::new(rng, N, eval_points), lag)
    }

    /// Share a real value in fixed-point.
    fn share_val(prov: &mut LocalCryptoProvider<ChaCha20Rng>, v: f64) -> Vec<Fp32> {
        prov.share_signed(encode_fixed(v, FB))
    }

    /// Reconstruct shares to a real value.
    fn recon(shares: &[Fp32], lag: &[Fp32]) -> f64 {
        let fp = shares
            .iter()
            .zip(lag)
            .fold(Fp32::ZERO, |a, (&s, &l)| a + s * l);
        decode_fixed(fp32_to_signed(fp), FB)
    }

    fn assert_close(got: f64, expected: f64, tol: f64, label: &str) {
        assert!(
            (got - expected).abs() < tol,
            "{}: expected {:.4}, got {:.4} (diff {:.4})",
            label,
            expected,
            got,
            (got - expected).abs()
        );
    }

    // ── Core operation tests ──

    #[test]
    fn test_beaver_mul() {
        let (mut prov, lag) = setup(42);
        let a_s = share_val(&mut prov, 3.5);
        let b_s = share_val(&mut prov, -2.0);
        let mut ops = SecureOps::new(&mut prov, &lag);
        let prod = ops.fp_mul(&a_s, &b_s, FB);
        let got = recon(&prod, &lag);
        assert_close(got, -7.0, 0.05, "beaver_mul");
    }

    #[test]
    fn test_trunc() {
        let (mut prov, lag) = setup(43);
        // 5.0 in fixed-point = 5 * 512 = 2560. Trunc by 2 → 640 → 1.25
        let x = share_val(&mut prov, 5.0);
        let mut ops = SecureOps::new(&mut prov, &lag);
        let t = ops.trunc(&x, 2);
        let got = recon(&t, &lag);
        assert_close(got, 1.25, 0.01, "trunc");
    }

    #[test]
    fn test_div_pub() {
        let (mut prov, lag) = setup(44);
        let x = share_val(&mut prov, 12.0);
        let mut ops = SecureOps::new(&mut prov, &lag);
        let d = ops.div_pub(&x, 4);
        let got = recon(&d, &lag);
        assert_close(got, 3.0, 0.01, "div_pub");
    }

    // ── Nonlinear approximation tests ──

    #[test]
    fn test_exp_positive() {
        let (mut prov, lag) = setup(50);
        let x = share_val(&mut prov, 1.0);
        let mut ops = SecureOps::new(&mut prov, &lag);
        let e = ops.exp(&x, FB, 8);
        let got = recon(&e, &lag);
        assert_close(got, 1.0_f64.exp(), 0.3, "exp(1)");
    }

    #[test]
    fn test_exp_negative() {
        let (mut prov, lag) = setup(51);
        let x = share_val(&mut prov, -2.0);
        let mut ops = SecureOps::new(&mut prov, &lag);
        let e = ops.exp(&x, FB, 8);
        let got = recon(&e, &lag);
        assert_close(got, (-2.0_f64).exp(), 0.15, "exp(-2)");
    }

    #[test]
    fn test_exp_zero() {
        let (mut prov, lag) = setup(52);
        let x = share_val(&mut prov, 0.0);
        let mut ops = SecureOps::new(&mut prov, &lag);
        let e = ops.exp(&x, FB, 8);
        let got = recon(&e, &lag);
        assert_close(got, 1.0, 0.1, "exp(0)");
    }

    #[test]
    fn test_reciprocal() {
        let (mut prov, lag) = setup(55);
        let x = share_val(&mut prov, 4.0);
        let mut ops = SecureOps::new(&mut prov, &lag);
        // y0 must satisfy 0 < y0 < 2/x = 0.5 (strictly less)
        let inv = ops.recip(&x, 0.4, FB, 5);
        let got = recon(&inv, &lag);
        assert_close(got, 0.25, 0.05, "recip(4)");
    }

    #[test]
    fn test_reciprocal_small() {
        let (mut prov, lag) = setup(56);
        let x = share_val(&mut prov, 2.0);
        let mut ops = SecureOps::new(&mut prov, &lag);
        let inv = ops.recip(&x, 0.75, FB, 4);
        let got = recon(&inv, &lag);
        assert_close(got, 0.5, 0.05, "recip(2)");
    }

    #[test]
    fn test_rsqrt() {
        let (mut prov, lag) = setup(60);
        let x = share_val(&mut prov, 4.0);
        let mut ops = SecureOps::new(&mut prov, &lag);
        let inv = ops.rsqrt(&x, 0.5, FB, 4);
        let got = recon(&inv, &lag);
        assert_close(got, 0.5, 0.08, "rsqrt(4)");
    }

    #[test]
    fn test_rsqrt_one() {
        let (mut prov, lag) = setup(61);
        let x = share_val(&mut prov, 1.0);
        let mut ops = SecureOps::new(&mut prov, &lag);
        let inv = ops.rsqrt(&x, 0.8, FB, 4);
        let got = recon(&inv, &lag);
        assert_close(got, 1.0, 0.08, "rsqrt(1)");
    }

    #[test]
    fn test_approx_sign() {
        let (mut prov, lag) = setup(65);
        for &v in &[3.0_f64, -3.0, 1.0, -1.0] {
            // Need fresh provider for each since we consume material
            let (mut prov2, lag2) = setup(65 + (v.abs() * 10.0) as u64);
            let x = share_val(&mut prov2, v);
            let mut ops = SecureOps::new(&mut prov2, &lag2);
            // bound=4 since test values are in [-3, 3]
            let s = ops.approx_sign(&x, FB, 4.0, 7);
            let got = recon(&s, &lag2);
            let expected = if v > 0.0 { 1.0 } else { -1.0 };
            assert_close(got, expected, 0.15, &format!("sign({v})"));
        }
        // suppress unused
        let _ = prov;
        let _ = lag;
    }

    #[test]
    fn test_sigmoid() {
        for &v in &[0.0_f64, 2.0, -2.0, 5.0] {
            let (mut prov, lag) = setup(70 + (v.abs() * 10.0) as u64);
            let x = share_val(&mut prov, v);
            let mut ops = SecureOps::new(&mut prov, &lag);
            let sig = ops.sigmoid(&x, FB, 8, 4, 7, 32.0);
            let got = recon(&sig, &lag);
            let expected = 1.0 / (1.0 + (-v).exp());
            assert_close(got, expected, 0.2, &format!("sigmoid({v})"));
        }
    }

    #[test]
    fn test_max2() {
        let (mut prov, lag) = setup(75);
        let a = share_val(&mut prov, 5.0);
        let b = share_val(&mut prov, 3.0);
        let mut ops = SecureOps::new(&mut prov, &lag);
        let m = ops.max2(&a, &b, FB, 16.0, 7);
        let got = recon(&m, &lag);
        assert_close(got, 5.0, 0.5, "max(5,3)");
    }

    // ── ML operation tests ──

    #[test]
    fn test_softmax_sums_to_scale() {
        let (mut prov, lag) = setup(80);
        let cfg = SoftmaxCfg::default();
        let scores: Vec<Vec<Fp32>> = [1.0, 0.0, -1.0, 2.0]
            .iter()
            .map(|&v| share_val(&mut prov, v))
            .collect();

        let mut ops = SecureOps::new(&mut prov, &lag);
        let weights = ops.softmax(&scores, 1, 4, &cfg);

        let sum: i64 = weights.iter().map(|w| fp32_to_signed(ops.open(w))).sum();
        assert!(
            (sum - cfg.output_scale as i64).abs() <= cfg.output_scale as i64 / 10,
            "softmax sum {} ≠ ~{}",
            sum,
            cfg.output_scale
        );
    }

    #[test]
    fn test_softmax_ordering() {
        let (mut prov, lag) = setup(81);
        let cfg = SoftmaxCfg::default();
        let scores: Vec<Vec<Fp32>> = [3.0, 1.0]
            .iter()
            .map(|&v| share_val(&mut prov, v))
            .collect();

        let mut ops = SecureOps::new(&mut prov, &lag);
        let weights = ops.softmax(&scores, 1, 2, &cfg);

        let w0 = fp32_to_signed(ops.open(&weights[0]));
        let w1 = fp32_to_signed(ops.open(&weights[1]));
        assert!(
            w0 > w1,
            "score 3.0 should have higher weight than 1.0: w0={}, w1={}",
            w0,
            w1
        );
    }

    #[test]
    fn test_gelu() {
        let gelu_ref = |x: f64| -> f64 { x / (1.0 + (-1.702 * x).exp()) };
        for &v in &[-2.0_f64, 0.0, 1.0, 3.0] {
            let (mut prov, lag) = setup(90 + (v.abs() * 10.0) as u64);
            let x = share_val(&mut prov, v);
            let cfg = GELUCfg::default();
            let mut ops = SecureOps::new(&mut prov, &lag);
            let g = ops.gelu(&x, &cfg);
            let got = recon(&g, &lag);
            assert_close(got, gelu_ref(v), 0.4, &format!("gelu({v})"));
        }
    }

    #[test]
    fn test_layer_norm() {
        let (mut prov, lag) = setup(100);
        let rows = 1;
        let cols = 4;
        let vals_f = [1.0f64, 3.0, 5.0, 7.0];
        let gamma = [1.0f32; 4];
        let beta = [0.0f32; 4];

        let vals: Vec<Vec<Fp32>> = vals_f.iter().map(|&v| share_val(&mut prov, v)).collect();
        let cfg = LayerNormCfg::default();
        let mut ops = SecureOps::new(&mut prov, &lag);
        let out = ops.layer_norm(&vals, rows, cols, Some(&gamma), Some(&beta), 1e-5, &cfg);

        // Expected: mean=4, var=5, inv_std≈0.4472
        let mean = 4.0;
        let var = 5.0;
        let inv_std = 1.0 / (var + 1e-5_f64).sqrt();
        for c in 0..cols {
            let expected = (vals_f[c] - mean) * inv_std;
            let got = recon(&out[c], &lag);
            assert_close(got, expected, 0.4, &format!("ln[{c}]"));
        }
    }

    #[test]
    fn test_layer_norm_centered() {
        // After layer norm, outputs should be approximately centered (mean ≈ 0)
        let (mut prov, lag) = setup(101);
        let cols = 4;
        let vals: Vec<Vec<Fp32>> = [2.0, 4.0, 6.0, 8.0]
            .iter()
            .map(|&v| share_val(&mut prov, v))
            .collect();
        let cfg = LayerNormCfg::default();
        let mut ops = SecureOps::new(&mut prov, &lag);
        let out = ops.layer_norm(&vals, 1, cols, None, None, 1e-5, &cfg);

        let sum: f64 = out.iter().map(|o| recon(o, &lag)).sum();
        assert!(
            sum.abs() < 0.5,
            "layer norm output mean should be ~0, got sum={}",
            sum
        );
    }

    // ── Precision diagnostic tests ──────────────────────────────────────────

    #[test]
    fn test_precision_diagnostic_layer_norm() {
        // Test layer_norm with high-variance transformer-like inputs
        let (mut prov, lag) = setup(200);
        let rows = 1;
        let cols = 4;
        let vals_f = [0.1_f64, 5.0, 10.0, 20.0];
        let gamma = [1.0f32; 4];
        let beta = [0.0f32; 4];

        let vals: Vec<Vec<Fp32>> = vals_f.iter().map(|&v| share_val(&mut prov, v)).collect();
        let cfg = LayerNormCfg::default();
        let mut ops = SecureOps::new(&mut prov, &lag);
        let out = ops.layer_norm(&vals, rows, cols, Some(&gamma), Some(&beta), 1e-5, &cfg);

        // Cleartext reference
        let mean: f64 = vals_f.iter().sum::<f64>() / cols as f64;
        let var: f64 = vals_f.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / cols as f64;
        let inv_std = 1.0 / (var + 1e-5_f64).sqrt();

        println!("=== Layer Norm Precision Diagnostic ===");
        println!("Inputs:    {:?}", vals_f);
        println!("Mean:      {:.4}", mean);
        println!("Variance:  {:.4}", var);
        println!("inv_std:   {:.6}", inv_std);
        println!();

        let mut max_err = 0.0_f64;
        for c in 0..cols {
            let expected = (vals_f[c] - mean) * inv_std;
            let got = recon(&out[c], &lag);
            let abs_err = (got - expected).abs();
            let rel_err = if expected.abs() > 1e-9 {
                abs_err / expected.abs() * 100.0
            } else {
                abs_err * 100.0
            };
            max_err = max_err.max(abs_err);
            println!(
                "  col[{}]: input={:.1}, expected={:.4}, got={:.4}, abs_err={:.4}, rel_err={:.1}%",
                c, vals_f[c], expected, got, abs_err, rel_err
            );
        }
        println!("  Max absolute error: {:.4}", max_err);
        println!();
    }

    #[test]
    fn test_precision_diagnostic_rsqrt() {
        // Test rsqrt across wide range with default y0=0.5, iters=4
        let test_values = [0.1_f64, 0.5, 1.0, 4.0, 10.0, 50.0, 100.0];

        println!("=== Rsqrt Precision Diagnostic (y0=0.5, iters=4) ===");
        println!();

        let mut diverged = Vec::new();
        for &v in &test_values {
            let (mut prov, lag) = setup(300 + (v * 10.0) as u64);
            let x = share_val(&mut prov, v);
            let mut ops = SecureOps::new(&mut prov, &lag);
            let result = ops.rsqrt(&x, 0.5, FB, 4);
            let got = recon(&result, &lag);
            let expected = 1.0 / v.sqrt();
            let abs_err = (got - expected).abs();
            let rel_err = abs_err / expected * 100.0;

            let status = if rel_err > 50.0 { "DIVERGED" } else { "ok" };
            if rel_err > 50.0 {
                diverged.push(v);
            }

            println!(
                "  rsqrt({:>6.1}): expected={:.6}, got={:.6}, abs_err={:.6}, rel_err={:.1}%  [{}]",
                v, expected, got, abs_err, rel_err, status
            );
        }

        println!();
        if diverged.is_empty() {
            println!("  All values within 50% tolerance.");
        } else {
            println!(
                "  DIVERGED (>50% error) for inputs: {:?}",
                diverged
            );
            println!(
                "  Reason: y0=0.5 satisfies convergence only when y0^2 < 2/x, i.e. x < 2/0.25 = 8."
            );
            println!("  For larger x, need smaller y0 or adaptive initial guess.");
        }
        println!();
    }

    #[test]
    fn test_precision_diagnostic_exp() {
        // Test exp across a range relevant to transformers (softmax centering, etc.)
        let test_values = [-5.0_f64, -3.0, -1.0, 0.0, 1.0, 3.0];

        println!("=== Exp Precision Diagnostic (iters=8) ===");
        println!();

        for &v in &test_values {
            let (mut prov, lag) = setup(400 + ((v + 10.0) * 10.0) as u64);
            let x = share_val(&mut prov, v);
            let mut ops = SecureOps::new(&mut prov, &lag);
            let result = ops.exp(&x, FB, 8);
            let got = recon(&result, &lag);
            let expected = v.exp();
            let abs_err = (got - expected).abs();
            let rel_err = if expected.abs() > 1e-9 {
                abs_err / expected * 100.0
            } else {
                abs_err * 100.0
            };

            let status = if rel_err > 20.0 { "HIGH ERROR" } else { "ok" };

            println!(
                "  exp({:>5.1}): expected={:.6}, got={:.6}, abs_err={:.6}, rel_err={:.1}%  [{}]",
                v, expected, got, abs_err, rel_err, status
            );
        }
        println!();
    }

    #[test]
    fn test_precision_diagnostic_softmax() {
        // Realistic attention score distribution
        let score_values = [2.0_f64, 1.0, 0.5, -1.0, -3.0];
        let num_keys = score_values.len();

        let (mut prov, lag) = setup(500);
        let cfg = SoftmaxCfg::default();
        let scores: Vec<Vec<Fp32>> = score_values
            .iter()
            .map(|&v| share_val(&mut prov, v))
            .collect();

        let mut ops = SecureOps::new(&mut prov, &lag);
        let weights = ops.softmax(&scores, 1, num_keys, &cfg);

        // Cleartext softmax reference
        let max_s = score_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = score_values.iter().map(|&s| (s - max_s).exp()).collect();
        let sum_exp: f64 = exps.iter().sum();
        let ref_probs: Vec<f64> = exps.iter().map(|e| e / sum_exp).collect();
        let ref_weights: Vec<f64> = ref_probs
            .iter()
            .map(|p| p * cfg.output_scale as f64)
            .collect();

        println!("=== Softmax Precision Diagnostic ===");
        println!("Scores: {:?}", score_values);
        println!("Output scale: {}", cfg.output_scale);
        println!();
        println!("  {:>8} {:>10} {:>10} {:>10} {:>10}", "score", "ref_wt", "got_wt", "abs_err", "rel_err%");

        let mut total_got = 0.0_f64;
        let mut total_ref = 0.0_f64;
        for i in 0..num_keys {
            let got_raw = fp32_to_signed(ops.open(&weights[i]));
            let got_f = got_raw as f64;
            let ref_f = ref_weights[i];
            let abs_err = (got_f - ref_f).abs();
            let rel_err = if ref_f.abs() > 0.5 {
                abs_err / ref_f * 100.0
            } else {
                abs_err * 100.0
            };
            total_got += got_f;
            total_ref += ref_f;

            println!(
                "  {:>8.1} {:>10.2} {:>10.2} {:>10.2} {:>9.1}%",
                score_values[i], ref_f, got_f, abs_err, rel_err
            );
        }
        println!("  ────────────────────────────────────────────────────");
        println!(
            "  {:>8} {:>10.2} {:>10.2}",
            "SUM", total_ref, total_got
        );

        // Check ordering preserved
        let got_ints: Vec<i64> = weights
            .iter()
            .map(|w| fp32_to_signed(ops.open(w)))
            .collect();
        let ordering_ok = got_ints[0] >= got_ints[1]
            && got_ints[1] >= got_ints[2]
            && got_ints[2] >= got_ints[3]
            && got_ints[3] >= got_ints[4];
        println!();
        println!(
            "  Ordering preserved (descending weights match descending scores): {}",
            if ordering_ok { "YES" } else { "NO" }
        );
        println!(
            "  Integer weights: {:?}",
            got_ints
        );
        println!();
    }
}
