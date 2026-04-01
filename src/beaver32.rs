//! Beaver multiplication protocol over Fp32 (p = 2^31-1).
//!
//! Consumes triples from the GPU `BeaverTripleBatch32` for private attention.

use crate::error::{ProtocolError, Result};
use crate::field32::Fp32;
use crate::field32_shamir::{Shamir32, Share32};

#[derive(Clone, Copy, Debug)]
pub struct BeaverTriple32 {
    pub a: Share32,
    pub b: Share32,
    pub c: Share32,
}

/// Open masked differences d = x - a, e = y - b using Lagrange interpolation.
pub fn beaver_multiply_open_32(
    shamir_t: &Shamir32,
    x_shares: &[Share32],
    y_shares: &[Share32],
    triples: &[BeaverTriple32],
) -> Result<(Fp32, Fp32)> {
    let n = shamir_t.n;
    assert_eq!(x_shares.len(), n);
    assert_eq!(y_shares.len(), n);
    assert_eq!(triples.len(), n);

    // d_p = x_p - a_p, e_p = y_p - b_p
    let d_shares: Vec<Share32> = (0..n)
        .map(|p| Share32 {
            point: x_shares[p].point,
            value: x_shares[p].value - triples[p].a.value,
        })
        .collect();

    let e_shares: Vec<Share32> = (0..n)
        .map(|p| Share32 {
            point: y_shares[p].point,
            value: y_shares[p].value - triples[p].b.value,
        })
        .collect();

    let d = shamir_t.reconstruct(&d_shares)?;
    let e = shamir_t.reconstruct(&e_shares)?;
    Ok((d, e))
}

/// Compute output shares: z_p = c_p + e·x_p + d·y_p - d·e
pub fn beaver_multiply_finish_32(
    n: usize,
    d: Fp32,
    e: Fp32,
    x_shares: &[Share32],
    y_shares: &[Share32],
    triples: &[BeaverTriple32],
) -> Vec<Share32> {
    let de = d * e;
    (0..n)
        .map(|p| Share32 {
            point: x_shares[p].point,
            value: triples[p].c.value + e * x_shares[p].value + d * y_shares[p].value - de,
        })
        .collect()
}

/// Complete Beaver multiplication (local simulation, no network).
pub fn beaver_multiply_local_32(
    n: usize,
    t: usize,
    x_shares: &[Share32],
    y_shares: &[Share32],
    triples: &[BeaverTriple32],
) -> Result<Vec<Share32>> {
    let shamir_t = Shamir32::new(n, t)?;
    let (d, e) = beaver_multiply_open_32(&shamir_t, x_shares, y_shares, triples)?;
    Ok(beaver_multiply_finish_32(
        n, d, e, x_shares, y_shares, triples,
    ))
}

/// Compute shares of dot_product(x, y) = sum_i x[i]*y[i] using Beaver triples.
///
/// Fuses all element-wise multiplications: opens all (d_i, e_i) in one batch,
/// then sums z_p values. This is 1 communication round for the entire dot product.
///
/// `x_shares[dim][n]`, `y_shares[dim][n]`, `triples[dim][n]`.
/// Returns `n` shares of the dot product.
pub fn beaver_dot_product_32(
    shamir_t: &Shamir32,
    x_shares: &[Vec<Share32>],
    y_shares: &[Vec<Share32>],
    triples: &[Vec<BeaverTriple32>],
) -> Result<Vec<Share32>> {
    let dim = x_shares.len();
    let n = shamir_t.n;
    assert_eq!(y_shares.len(), dim);
    assert_eq!(triples.len(), dim);

    // Open all (d_i, e_i) pairs
    let mut d_vals = Vec::with_capacity(dim);
    let mut e_vals = Vec::with_capacity(dim);
    for i in 0..dim {
        let (d, e) = beaver_multiply_open_32(shamir_t, &x_shares[i], &y_shares[i], &triples[i])?;
        d_vals.push(d);
        e_vals.push(e);
    }

    // Sum z_p = sum_i (c_p_i + e_i*x_p_i + d_i*y_p_i - d_i*e_i) across all dims
    let mut result = Vec::with_capacity(n);
    for p in 0..n {
        let mut sum = Fp32::ZERO;
        for i in 0..dim {
            let de = d_vals[i] * e_vals[i];
            sum = sum
                + triples[i][p].c.value
                + e_vals[i] * x_shares[i][p].value
                + d_vals[i] * y_shares[i][p].value
                - de;
        }
        result.push(Share32 {
            point: x_shares[0][p].point,
            value: sum,
        });
    }

    Ok(result)
}

/// Convert GPU triple batch into BeaverTriple32 structs for the protocol.
///
/// `batch` is the GPU output, `offset` is the starting triple index.
/// Returns `n` triples (one per party) for a single multiplication.
#[cfg(target_os = "macos")]
pub fn triples_from_gpu_batch(
    batch: &crate::gpu::BeaverTripleBatch32,
    triple_index: usize,
    eval_points: &[Fp32],
) -> Vec<BeaverTriple32> {
    let n = batch.n;
    (0..n)
        .map(|p| {
            let (a, b, c) = batch.triple_values(triple_index, p);
            BeaverTriple32 {
                a: Share32 {
                    point: eval_points[p],
                    value: Fp32::from_reduced(a),
                },
                b: Share32 {
                    point: eval_points[p],
                    value: Fp32::from_reduced(b),
                },
                c: Share32 {
                    point: eval_points[p],
                    value: Fp32::from_reduced(c),
                },
            }
        })
        .collect()
}

/// Batched private dot product directly from GPU triple batch.
///
/// Computes `num_scores` dot products of dimension `head_dim` in one pass.
/// `q_flat[score_idx * head_dim * n + dim * n + party]` — interleaved Q values.
/// `k_flat` — same layout for K values.
///
/// Returns per-score per-party shares in SCALE² domain:
/// `result[score_idx][party]`.
/// Advances `cursor` by `num_scores * head_dim`.
#[cfg(target_os = "macos")]
pub fn beaver_dot_product_batch_from_gpu_shared(
    lag_coeffs: &[Fp32],
    n: usize,
    head_dim: usize,
    num_scores: usize,
    q_flat: &[Fp32], // [num_scores][head_dim][n]
    k_flat: &[Fp32], // [num_scores][head_dim][n]
    batch: &crate::gpu::BeaverTripleBatch32,
    cursor: &mut usize,
) -> Vec<Vec<Fp32>> {
    let mut results = Vec::with_capacity(num_scores);

    for s in 0..num_scores {
        let base = s * head_dim * n;
        let mut dot_per_party = vec![Fp32::ZERO; n];

        for d in 0..head_dim {
            let tri_idx = *cursor;
            *cursor += 1;

            // Read triple (a, b, c) for all parties directly from GPU batch
            // Compute d_p = q_p - a_p, e_p = k_p - b_p for each party
            let mut d_val = Fp32::ZERO; // reconstructed d
            let mut e_val = Fp32::ZERO; // reconstructed e

            for p in 0..n {
                let qi = q_flat[base + d * n + p];
                let ki = k_flat[base + d * n + p];
                let (a_raw, b_raw, _) = batch.triple_values(tri_idx, p);
                let a_p = Fp32::from_reduced(a_raw);
                let b_p = Fp32::from_reduced(b_raw);
                let dp = qi - a_p;
                let ep = ki - b_p;
                d_val = d_val + dp * lag_coeffs[p];
                e_val = e_val + ep * lag_coeffs[p];
            }

            // Finish: z_p = c_p + e*q_p + d*k_p - d*e
            let de = d_val * e_val;
            for p in 0..n {
                let qi = q_flat[base + d * n + p];
                let ki = k_flat[base + d * n + p];
                let (_, _, c_raw) = batch.triple_values(tri_idx, p);
                let c_p = Fp32::from_reduced(c_raw);
                dot_per_party[p] = dot_per_party[p] + c_p + e_val * qi + d_val * ki - de;
            }
        }

        results.push(dot_per_party);
    }

    results
}

/// Batched private dot product directly from GPU triple batch.
///
/// Computes `num_scores` dot products of dimension `head_dim` in one pass.
/// `q_flat[score_idx * head_dim * n + dim * n + party]` — interleaved Q values.
/// `k_flat` — same layout for K values.
///
/// Returns `num_scores` reconstructed Fp32 dot products (still in SCALE² domain).
/// Advances `cursor` by `num_scores * head_dim`.
#[cfg(target_os = "macos")]
pub fn beaver_dot_product_batch_from_gpu(
    lag_coeffs: &[Fp32],
    n: usize,
    head_dim: usize,
    num_scores: usize,
    q_flat: &[Fp32], // [num_scores][head_dim][n]
    k_flat: &[Fp32], // [num_scores][head_dim][n]
    batch: &crate::gpu::BeaverTripleBatch32,
    cursor: &mut usize,
) -> Vec<Fp32> {
    let shared = beaver_dot_product_batch_from_gpu_shared(
        lag_coeffs, n, head_dim, num_scores, q_flat, k_flat, batch, cursor,
    );
    shared
        .into_iter()
        .map(|dot_per_party| {
            let mut result = Fp32::ZERO;
            for p in 0..n {
                result = result + dot_per_party[p] * lag_coeffs[p];
            }
            result
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn test_beaver32_multiply_correctness() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let n = 5;
        let t = 1;
        let shamir = Shamir32::new(n, t).unwrap();

        for _ in 0..100 {
            let x = Fp32::random(&mut rng);
            let y = Fp32::random(&mut rng);
            let a = Fp32::random(&mut rng);
            let b = Fp32::random(&mut rng);
            let c = a * b;

            let x_shares = shamir.share(x, &mut rng);
            let y_shares = shamir.share(y, &mut rng);
            let a_shares = shamir.share(a, &mut rng);
            let b_shares = shamir.share(b, &mut rng);
            let c_shares = shamir.share(c, &mut rng);

            let triples: Vec<BeaverTriple32> = (0..n)
                .map(|p| BeaverTriple32 {
                    a: a_shares[p],
                    b: b_shares[p],
                    c: c_shares[p],
                })
                .collect();

            let z_shares = beaver_multiply_local_32(n, t, &x_shares, &y_shares, &triples).unwrap();
            let z = shamir.reconstruct(&z_shares).unwrap();
            assert_eq!(
                z,
                x * y,
                "beaver multiply failed: {} * {} = {}, got {}",
                x,
                y,
                x * y,
                z
            );
        }
    }

    #[test]
    fn test_beaver32_dot_product() {
        let mut rng = ChaCha20Rng::seed_from_u64(43);
        let n = 5;
        let t = 1;
        let shamir = Shamir32::new(n, t).unwrap();
        let dim = 64; // head_dim

        let x: Vec<Fp32> = (0..dim).map(|_| Fp32::random(&mut rng)).collect();
        let y: Vec<Fp32> = (0..dim).map(|_| Fp32::random(&mut rng)).collect();

        // Expected dot product
        let mut expected = Fp32::ZERO;
        for i in 0..dim {
            expected = expected + x[i] * y[i];
        }

        // Share x, y and create triples
        let x_shares: Vec<Vec<Share32>> = x.iter().map(|&v| shamir.share(v, &mut rng)).collect();
        let y_shares: Vec<Vec<Share32>> = y.iter().map(|&v| shamir.share(v, &mut rng)).collect();

        let triples: Vec<Vec<BeaverTriple32>> = (0..dim)
            .map(|_| {
                let a = Fp32::random(&mut rng);
                let b = Fp32::random(&mut rng);
                let c = a * b;
                let a_sh = shamir.share(a, &mut rng);
                let b_sh = shamir.share(b, &mut rng);
                let c_sh = shamir.share(c, &mut rng);
                (0..n)
                    .map(|p| BeaverTriple32 {
                        a: a_sh[p],
                        b: b_sh[p],
                        c: c_sh[p],
                    })
                    .collect()
            })
            .collect();

        let result_shares = beaver_dot_product_32(&shamir, &x_shares, &y_shares, &triples).unwrap();
        let result = shamir.reconstruct(&result_shares).unwrap();
        assert_eq!(result, expected, "dot product failed");
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_beaver32_with_gpu_triples() {
        use crate::gpu::{ExpandedCorrelations32, GpuTripleGen32};
        use rand::Rng;

        let mut rng = ChaCha20Rng::seed_from_u64(44);
        let n = 5;
        let t = 1;
        let count = 100;
        let spr = n - 2 * t;
        let num_rounds = (count as usize).div_ceil(spr);
        let shamir = Shamir32::new(n, t).unwrap();
        let eval_points: Vec<Fp32> = (1..=n as u32).map(Fp32::new).collect();

        let ot: Vec<ExpandedCorrelations32> = (0..n)
            .map(|i| ExpandedCorrelations32::from_random(i, num_rounds, &mut rng))
            .collect();

        let gpu = GpuTripleGen32::new(n, t).unwrap();
        let batch = gpu.generate(count, &ot, &mut rng).unwrap();

        // Verify triples: c = a*b
        for k in 0..count.min(50) {
            let triples = triples_from_gpu_batch(&batch, k, &eval_points);
            let a_vals: Vec<Fp32> = triples.iter().map(|t| t.a.value).collect();
            let b_vals: Vec<Fp32> = triples.iter().map(|t| t.b.value).collect();
            let c_vals: Vec<Fp32> = triples.iter().map(|t| t.c.value).collect();

            let a = shamir.reconstruct_raw(&a_vals);
            let b = shamir.reconstruct_raw(&b_vals);
            let c = shamir.reconstruct_raw(&c_vals);
            assert_eq!(
                c,
                a * b,
                "GPU triple {} failed: c={} != a*b={}*{}",
                k,
                c,
                a,
                b
            );
        }

        // Use GPU triples in a Beaver multiplication
        let x = Fp32::random(&mut rng);
        let y = Fp32::random(&mut rng);
        let x_shares = shamir.share(x, &mut rng);
        let y_shares = shamir.share(y, &mut rng);
        let triples = triples_from_gpu_batch(&batch, 0, &eval_points);

        let z_shares = beaver_multiply_local_32(n, t, &x_shares, &y_shares, &triples).unwrap();
        let z = shamir.reconstruct(&z_shares).unwrap();
        assert_eq!(z, x * y, "Beaver multiply with GPU triples failed");
    }
}
