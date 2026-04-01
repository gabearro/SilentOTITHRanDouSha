//! TurboQuant primitives for GPT-OSS KV-cache quantization.
//!
//! Implements the paper:
//! "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
//! (arXiv:2504.19874)
//!
//! - Algorithm 1 (TurboQuant_mse): random rotation + Lloyd-Max scalar codebook
//! - Algorithm 2 (TurboQuant_prod): (b-1)-bit MSE stage + 1-bit QJL residual

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[inline]
fn l2_norm(x: &[f32]) -> f32 {
    x.iter().map(|v| v * v).sum::<f32>().sqrt()
}

#[inline]
fn sample_standard_normal(rng: &mut ChaCha8Rng) -> f32 {
    // Box-Muller transform
    let u1 = rng
        .gen::<f32>()
        .clamp(f32::MIN_POSITIVE, 1.0 - f32::EPSILON);
    let u2 = rng.gen::<f32>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

fn mat_vec(mat: &[f32], x: &[f32], out: &mut [f32], dim: usize) {
    for i in 0..dim {
        let row = &mat[i * dim..(i + 1) * dim];
        let mut acc = 0.0f32;
        for j in 0..dim {
            acc += row[j] * x[j];
        }
        out[i] = acc;
    }
}

fn mat_t_vec(mat: &[f32], x: &[f32], out: &mut [f32], dim: usize) {
    out.fill(0.0);
    for i in 0..dim {
        let xi = x[i];
        let row = &mat[i * dim..(i + 1) * dim];
        for j in 0..dim {
            out[j] += row[j] * xi;
        }
    }
}

fn random_orthogonal_matrix(dim: usize, rng: &mut ChaCha8Rng) -> Vec<f32> {
    // Random Gaussian matrix followed by modified Gram-Schmidt on rows.
    let mut q = vec![0.0f32; dim * dim];
    for v in &mut q {
        *v = sample_standard_normal(rng);
    }

    for i in 0..dim {
        for j in 0..i {
            let mut dot = 0.0f32;
            for k in 0..dim {
                dot += q[i * dim + k] * q[j * dim + k];
            }
            for k in 0..dim {
                q[i * dim + k] -= dot * q[j * dim + k];
            }
        }

        let mut norm = 0.0f32;
        for k in 0..dim {
            let v = q[i * dim + k];
            norm += v * v;
        }
        norm = norm.sqrt();
        if norm < 1e-8 {
            for k in 0..dim {
                q[i * dim + k] = if k == i { 1.0 } else { 0.0 };
            }
            continue;
        }
        let inv = 1.0 / norm;
        for k in 0..dim {
            q[i * dim + k] *= inv;
        }
    }

    q
}

fn nearest_centroid_index(centroids: &[f32], x: f32) -> usize {
    let mut best_i = 0usize;
    let mut best_d = (x - centroids[0]).abs();
    for (i, &c) in centroids.iter().enumerate().skip(1) {
        let d = (x - c).abs();
        if d < best_d {
            best_d = d;
            best_i = i;
        }
    }
    best_i
}

fn build_gaussian_lloyd_max_codebook(dim: usize, bits: u8, seed: u64) -> Vec<f32> {
    let levels = 1usize << bits;
    let sigma = 1.0f32 / (dim as f32).sqrt();
    let samples_n = 24_576usize;
    let iters = 24usize;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut samples: Vec<f32> = (0..samples_n)
        .map(|_| sample_standard_normal(&mut rng) * sigma)
        .collect();
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut centroids = vec![0.0f32; levels];
    for (i, c) in centroids.iter_mut().enumerate() {
        let qi = (i as f32 + 0.5) / levels as f32;
        let idx = (qi * (samples_n as f32 - 1.0)).round() as usize;
        *c = samples[idx];
    }

    let mut sums = vec![0.0f32; levels];
    let mut counts = vec![0usize; levels];
    for _ in 0..iters {
        sums.fill(0.0);
        counts.fill(0);

        for &s in &samples {
            let idx = nearest_centroid_index(&centroids, s);
            sums[idx] += s;
            counts[idx] += 1;
        }

        for i in 0..levels {
            if counts[i] > 0 {
                centroids[i] = sums[i] / counts[i] as f32;
            }
        }
        centroids.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    }

    // Symmetrize around zero to reduce sampling noise.
    let half = levels / 2;
    for i in 0..half {
        let mag = 0.5 * (centroids[levels - 1 - i] - centroids[i]).abs();
        centroids[i] = -mag;
        centroids[levels - 1 - i] = mag;
    }
    if levels % 2 == 1 {
        centroids[half] = 0.0;
    }

    centroids
}

#[inline]
pub fn packed_bits_len(dim: usize) -> usize {
    dim.div_ceil(8)
}

#[inline]
fn set_packed_sign(bits: &mut [u8], idx: usize, positive: bool) {
    let b = idx / 8;
    let off = idx % 8;
    if positive {
        bits[b] |= 1u8 << off;
    } else {
        bits[b] &= !(1u8 << off);
    }
}

#[inline]
fn get_packed_sign(bits: &[u8], idx: usize) -> f32 {
    let b = idx / 8;
    let off = idx % 8;
    if ((bits[b] >> off) & 1u8) != 0 {
        1.0
    } else {
        -1.0
    }
}

pub struct MseScratch {
    pub unit: Vec<f32>,
    pub rotated: Vec<f32>,
    pub recon: Vec<f32>,
}

impl MseScratch {
    pub fn new(dim: usize) -> Self {
        MseScratch {
            unit: vec![0.0; dim],
            rotated: vec![0.0; dim],
            recon: vec![0.0; dim],
        }
    }
}

pub struct ProdScratch {
    pub mse: MseScratch,
    pub residual: Vec<f32>,
    pub proj: Vec<f32>,
}

impl ProdScratch {
    pub fn new(dim: usize) -> Self {
        ProdScratch {
            mse: MseScratch::new(dim),
            residual: vec![0.0; dim],
            proj: vec![0.0; dim],
        }
    }
}

/// TurboQuant_mse (Algorithm 1): random rotation + scalar Lloyd-Max.
pub struct TurboQuantMse {
    dim: usize,
    bits: u8,
    centroids: Vec<f32>,
    rotation: Vec<f32>, // row-major orthonormal matrix Pi
    zero_index: u8,
}

impl TurboQuantMse {
    pub fn new(dim: usize, bits: u8, seed: u64) -> Self {
        assert!(dim > 0, "dim must be > 0");
        assert!(bits > 0 && bits <= 8, "bits must be in [1, 8]");

        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xA5A5_1234_9E37_79B9);
        let rotation = random_orthogonal_matrix(dim, &mut rng);
        let centroids = build_gaussian_lloyd_max_codebook(dim, bits, seed ^ 0x7F4A_7C15_DA3E_39CB);
        let zero_index = nearest_centroid_index(&centroids, 0.0) as u8;

        TurboQuantMse {
            dim,
            bits,
            centroids,
            rotation,
            zero_index,
        }
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    #[inline]
    pub fn bits(&self) -> u8 {
        self.bits
    }

    #[inline]
    pub fn rotate(&self, x: &[f32], out: &mut [f32]) {
        mat_vec(&self.rotation, x, out, self.dim);
    }

    #[inline]
    pub fn inverse_rotate(&self, x: &[f32], out: &mut [f32]) {
        mat_t_vec(&self.rotation, x, out, self.dim);
    }

    pub fn quantize_unit_into(
        &self,
        x_unit: &[f32],
        idx_out: &mut [u8],
        rotated_scratch: &mut [f32],
    ) {
        assert_eq!(x_unit.len(), self.dim);
        assert_eq!(idx_out.len(), self.dim);
        assert_eq!(rotated_scratch.len(), self.dim);

        self.rotate(x_unit, rotated_scratch);
        for i in 0..self.dim {
            idx_out[i] = nearest_centroid_index(&self.centroids, rotated_scratch[i]) as u8;
        }
    }

    pub fn quantize_into(
        &self,
        x: &[f32],
        idx_out: &mut [u8],
        norm_out: &mut f32,
        scratch: &mut MseScratch,
    ) {
        assert_eq!(x.len(), self.dim);
        assert_eq!(idx_out.len(), self.dim);
        assert_eq!(scratch.unit.len(), self.dim);
        assert_eq!(scratch.rotated.len(), self.dim);

        let norm = l2_norm(x);
        *norm_out = norm;

        if norm <= 1e-12 {
            idx_out.fill(self.zero_index);
            return;
        }

        let inv = 1.0 / norm;
        for i in 0..self.dim {
            scratch.unit[i] = x[i] * inv;
        }
        self.quantize_unit_into(&scratch.unit, idx_out, &mut scratch.rotated);
    }

    pub fn dequantize_unit_from_indices_into(
        &self,
        idx: &[u8],
        out: &mut [f32],
        rotated_scratch: &mut [f32],
    ) {
        assert_eq!(idx.len(), self.dim);
        assert_eq!(out.len(), self.dim);
        assert_eq!(rotated_scratch.len(), self.dim);

        for i in 0..self.dim {
            rotated_scratch[i] = self.centroids[idx[i] as usize];
        }
        self.inverse_rotate(rotated_scratch, out);
    }

    pub fn dequantize_into(
        &self,
        idx: &[u8],
        norm: f32,
        out: &mut [f32],
        rotated_scratch: &mut [f32],
    ) {
        self.dequantize_unit_from_indices_into(idx, out, rotated_scratch);
        for v in out {
            *v *= norm;
        }
    }

    /// Accumulate `coeff * (rotated centroid vector)` into `acc_rot`.
    ///
    /// This corresponds to linear accumulation in rotated domain:
    /// sum_i coeff_i * c[idx_i], and a single inverse rotation at the end.
    pub fn accumulate_rotated_from_indices(&self, idx: &[u8], coeff: f32, acc_rot: &mut [f32]) {
        assert_eq!(idx.len(), self.dim);
        assert_eq!(acc_rot.len(), self.dim);
        for i in 0..self.dim {
            acc_rot[i] += coeff * self.centroids[idx[i] as usize];
        }
    }

    pub fn finalize_rotated(&self, acc_rot: &[f32], out: &mut [f32]) {
        assert_eq!(acc_rot.len(), self.dim);
        assert_eq!(out.len(), self.dim);
        self.inverse_rotate(acc_rot, out);
    }
}

/// TurboQuant_prod (Algorithm 2): (b-1)-bit TurboQuant_mse + 1-bit QJL residual.
pub struct TurboQuantProd {
    dim: usize,
    bits: u8,
    mse: TurboQuantMse,
    projection: Vec<f32>, // row-major Gaussian S
    qjl_scale: f32,       // sqrt(pi/2) / d
}

impl TurboQuantProd {
    pub fn new(dim: usize, bits: u8, seed: u64) -> Self {
        assert!(
            bits >= 2 && bits <= 8,
            "TurboQuantProd total bits must be in [2, 8]"
        );

        let mse = TurboQuantMse::new(dim, bits - 1, seed ^ 0xD1B5_4A32_0F77_EE01);
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xBEEF_0123_7788_99AA);
        let mut projection = vec![0.0f32; dim * dim];
        for v in &mut projection {
            *v = sample_standard_normal(&mut rng);
        }

        TurboQuantProd {
            dim,
            bits,
            mse,
            projection,
            qjl_scale: (std::f32::consts::FRAC_PI_2).sqrt() / dim as f32,
        }
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    #[inline]
    pub fn bits(&self) -> u8 {
        self.bits
    }

    #[inline]
    pub fn mse(&self) -> &TurboQuantMse {
        &self.mse
    }

    pub fn prepare_query(&self, q: &[f32], q_rot: &mut [f32], q_proj: &mut [f32]) {
        assert_eq!(q.len(), self.dim);
        assert_eq!(q_rot.len(), self.dim);
        assert_eq!(q_proj.len(), self.dim);
        self.mse.rotate(q, q_rot);
        mat_vec(&self.projection, q, q_proj, self.dim);
    }

    pub fn quantize_into(
        &self,
        x: &[f32],
        idx_out: &mut [u8],
        qjl_out: &mut [u8],
        norm_out: &mut f32,
        gamma_out: &mut f32,
        scratch: &mut ProdScratch,
    ) {
        assert_eq!(x.len(), self.dim);
        assert_eq!(idx_out.len(), self.dim);
        assert_eq!(qjl_out.len(), packed_bits_len(self.dim));
        assert_eq!(scratch.mse.unit.len(), self.dim);
        assert_eq!(scratch.mse.rotated.len(), self.dim);
        assert_eq!(scratch.mse.recon.len(), self.dim);
        assert_eq!(scratch.residual.len(), self.dim);
        assert_eq!(scratch.proj.len(), self.dim);

        let norm = l2_norm(x);
        *norm_out = norm;
        qjl_out.fill(0);

        if norm <= 1e-12 {
            idx_out.fill(self.mse.zero_index);
            *gamma_out = 0.0;
            // Store all +1 signs as a neutral default.
            for i in 0..self.dim {
                set_packed_sign(qjl_out, i, true);
            }
            return;
        }

        let inv_norm = 1.0 / norm;
        for i in 0..self.dim {
            scratch.mse.unit[i] = x[i] * inv_norm;
        }

        self.mse
            .quantize_unit_into(&scratch.mse.unit, idx_out, &mut scratch.mse.rotated);
        self.mse.dequantize_unit_from_indices_into(
            idx_out,
            &mut scratch.mse.recon,
            &mut scratch.mse.rotated,
        );

        for i in 0..self.dim {
            scratch.residual[i] = scratch.mse.unit[i] - scratch.mse.recon[i];
        }

        let gamma = l2_norm(&scratch.residual);
        *gamma_out = gamma;
        if gamma <= 1e-12 {
            for i in 0..self.dim {
                set_packed_sign(qjl_out, i, true);
            }
            return;
        }

        mat_vec(
            &self.projection,
            &scratch.residual,
            &mut scratch.proj,
            self.dim,
        );
        for i in 0..self.dim {
            set_packed_sign(qjl_out, i, scratch.proj[i] >= 0.0);
        }
    }

    pub fn dequantize_into(
        &self,
        idx: &[u8],
        qjl: &[u8],
        norm: f32,
        gamma: f32,
        out: &mut [f32],
        scratch: &mut ProdScratch,
    ) {
        assert_eq!(idx.len(), self.dim);
        assert_eq!(qjl.len(), packed_bits_len(self.dim));
        assert_eq!(out.len(), self.dim);
        assert_eq!(scratch.residual.len(), self.dim);
        assert_eq!(scratch.proj.len(), self.dim);

        self.mse
            .dequantize_unit_from_indices_into(idx, out, &mut scratch.mse.rotated);

        if gamma > 1e-12 {
            for i in 0..self.dim {
                scratch.residual[i] = get_packed_sign(qjl, i);
            }
            mat_t_vec(
                &self.projection,
                &scratch.residual,
                &mut scratch.proj,
                self.dim,
            );
            let qjl_coeff = self.qjl_scale * gamma;
            for i in 0..self.dim {
                out[i] += qjl_coeff * scratch.proj[i];
            }
        }

        for i in 0..self.dim {
            out[i] *= norm;
        }
    }

    /// Dot product with a quantized key using precomputed query transforms.
    ///
    /// If `q_rot = Pi * q` and `q_proj = S * q`, this computes:
    ///   <q, x_tilde> = norm * ( <q_rot, c[idx]> + (sqrt(pi/2)/d) * gamma * <q_proj, qjl> )
    pub fn dot_from_quantized_prepared(
        &self,
        q_rot: &[f32],
        q_proj: &[f32],
        idx: &[u8],
        qjl: &[u8],
        norm: f32,
        gamma: f32,
    ) -> f32 {
        assert_eq!(q_rot.len(), self.dim);
        assert_eq!(q_proj.len(), self.dim);
        assert_eq!(idx.len(), self.dim);
        assert_eq!(qjl.len(), packed_bits_len(self.dim));

        let mut mse_term = 0.0f32;
        for i in 0..self.dim {
            mse_term += q_rot[i] * self.mse.centroids[idx[i] as usize];
        }

        let mut qjl_term = 0.0f32;
        for i in 0..self.dim {
            qjl_term += q_proj[i] * get_packed_sign(qjl, i);
        }

        norm * (mse_term + self.qjl_scale * gamma * qjl_term)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packed_bits_roundtrip() {
        let dim = 23usize;
        let mut bits = vec![0u8; packed_bits_len(dim)];
        for i in 0..dim {
            set_packed_sign(&mut bits, i, i % 3 != 0);
        }
        for i in 0..dim {
            let want = if i % 3 != 0 { 1.0 } else { -1.0 };
            assert_eq!(get_packed_sign(&bits, i), want);
        }
    }

    #[test]
    fn test_prod_dot_matches_explicit_dequant() {
        let dim = 64usize;
        let q = TurboQuantProd::new(dim, 3, 1234);

        let mut rng = ChaCha8Rng::seed_from_u64(4321);
        let x: Vec<f32> = (0..dim).map(|_| sample_standard_normal(&mut rng)).collect();
        let y: Vec<f32> = (0..dim).map(|_| sample_standard_normal(&mut rng)).collect();

        let mut idx = vec![0u8; dim];
        let mut qjl = vec![0u8; packed_bits_len(dim)];
        let mut norm = 0.0;
        let mut gamma = 0.0;
        let mut scratch = ProdScratch::new(dim);

        q.quantize_into(&x, &mut idx, &mut qjl, &mut norm, &mut gamma, &mut scratch);

        let mut x_hat = vec![0.0f32; dim];
        q.dequantize_into(&idx, &qjl, norm, gamma, &mut x_hat, &mut scratch);
        let explicit_dot: f32 = x_hat.iter().zip(&y).map(|(a, b)| a * b).sum();

        let mut y_rot = vec![0.0f32; dim];
        let mut y_proj = vec![0.0f32; dim];
        q.prepare_query(&y, &mut y_rot, &mut y_proj);
        let fast_dot = q.dot_from_quantized_prepared(&y_rot, &y_proj, &idx, &qjl, norm, gamma);

        let diff = (explicit_dot - fast_dot).abs();
        assert!(
            diff < 1e-3,
            "dot mismatch: explicit={explicit_dot}, fast={fast_dot}, diff={diff}"
        );
    }

    #[test]
    fn test_mse_accumulate_rotated_matches_explicit() {
        let dim = 64usize;
        let q = TurboQuantMse::new(dim, 3, 99);
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let x: Vec<f32> = (0..dim).map(|_| sample_standard_normal(&mut rng)).collect();

        let mut idx = vec![0u8; dim];
        let mut norm = 0.0f32;
        let mut scratch = MseScratch::new(dim);
        q.quantize_into(&x, &mut idx, &mut norm, &mut scratch);

        let mut explicit = vec![0.0f32; dim];
        q.dequantize_into(&idx, norm, &mut explicit, &mut scratch.rotated);

        let mut acc_rot = vec![0.0f32; dim];
        q.accumulate_rotated_from_indices(&idx, norm, &mut acc_rot);
        let mut via_rot = vec![0.0f32; dim];
        q.finalize_rotated(&acc_rot, &mut via_rot);

        let max_diff = explicit
            .iter()
            .zip(&via_rot)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3, "max diff={max_diff}");
    }
}
