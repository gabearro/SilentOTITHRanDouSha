use crate::error::{ProtocolError, Result};
use crate::field::Fp;
use crate::randousha::{DoubleShare, HyperInvertibleMatrix};
use crate::shamir::{Shamir, Share};
use crate::silent_ot::ExpandedCorrelations;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// A Beaver triple: degree-t shares of random a, b, and c = a*b.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct BeaverTriple {
    pub a: Share,
    pub b: Share,
    pub c: Share,
}

/// Build an n×cols Vandermonde matrix V[i][j] = α_{i+1}^j.
///
/// Uses points α = 1, 2, ..., n (matching Shamir evaluation points).
/// In a distributed batch-reconstruction protocol, this matrix encodes
/// t+1 degree-2t shared values into n linear combinations that can be
/// opened in a single all-to-all round.
pub fn make_vandermonde(n: usize, cols: usize) -> Vec<Vec<Fp>> {
    (0..n)
        .map(|i| {
            let alpha = Fp::new((i + 1) as u64);
            let mut row = Vec::with_capacity(cols);
            let mut power = Fp::ONE;
            for _ in 0..cols {
                row.push(power);
                power *= alpha;
            }
            row
        })
        .collect()
}

/// Reconstruct multiple degree-2t shared secrets in batch.
///
/// `shares_per_value[k]` contains n shares of the k-th degree-2t value.
/// Returns the reconstructed secrets.
///
/// Uses precomputed Lagrange coefficients from `shamir_2t` so the cost is
/// a single dot-product per value (amortised over all values in the batch).
/// In a distributed protocol this would additionally use Vandermonde encoding
/// to batch t+1 openings per all-to-all communication round.
pub fn batch_reconstruct(
    shares_per_value: &[Vec<Share>],
    shamir_2t: &Shamir,
) -> Result<Vec<Fp>> {
    if shares_per_value.is_empty() {
        return Ok(Vec::new());
    }
    let n = shamir_2t.n;
    let lagrange = shamir_2t.lagrange_coefficients();

    for (k, shares) in shares_per_value.iter().enumerate() {
        if shares.len() != n {
            return Err(ProtocolError::InvalidParams(format!(
                "value {} has {} shares, expected {}",
                k,
                shares.len(),
                n
            )));
        }
    }

    Ok(shares_per_value
        .iter()
        .map(|shares| {
            shares
                .iter()
                .zip(lagrange)
                .map(|(s, &c)| s.value * c)
                .sum()
        })
        .collect())
}

/// Generate Beaver triples from double shares.
///
/// `double_shares[k][p]` is party p's double share for triple k.
/// Returns `triples[k][p]` = party p's BeaverTriple for triple k.
///
/// Per triple:
/// 1. Generate random a, b; share at degree t (inline polynomial evaluation)
/// 2. Fused: compute masked products and reconstruct δ via Lagrange dot product
/// 3. Each party: c_p = r_t_p + δ  →  c(0) = r + (a·b − r) = a·b
///
/// Uses rayon parallelism over chunks of triples with per-chunk RNGs.
pub fn generate_triples<R: Rng>(
    n: usize,
    t: usize,
    double_shares: &[Vec<DoubleShare>],
    rng: &mut R,
) -> Result<Vec<Vec<BeaverTriple>>> {
    if n <= 2 * t {
        return Err(ProtocolError::InvalidParams(format!(
            "need n > 2t, got n={}, t={}",
            n, t
        )));
    }
    if double_shares.is_empty() {
        return Err(ProtocolError::InvalidParams(
            "empty double_shares".into(),
        ));
    }
    let count = double_shares.len();
    for (k, ds) in double_shares.iter().enumerate() {
        if ds.len() != n {
            return Err(ProtocolError::InvalidParams(format!(
                "double_shares[{}] has {} entries, expected {}",
                k,
                ds.len(),
                n
            )));
        }
    }

    let shamir_t = Shamir::new(n, t)?;
    let shamir_2t = Shamir::new(n, 2 * t)?;
    let eval_points = &shamir_t.eval_points;
    let lagrange = shamir_2t.lagrange_coefficients();

    const CHUNK: usize = 4096;
    let num_chunks = count.div_ceil(CHUNK);
    let seeds: Vec<u64> = (0..num_chunks).map(|_| rng.gen()).collect();

    let chunk_results: Vec<Vec<Vec<BeaverTriple>>> = (0..num_chunks)
        .into_par_iter()
        .map(|ci| {
            let start = ci * CHUNK;
            let end = (start + CHUNK).min(count);
            let mut local_rng = ChaCha20Rng::seed_from_u64(seeds[ci]);
            let mut out = Vec::with_capacity(end - start);

            for k in start..end {
                out.push(gen_triple_core(
                    n,
                    t,
                    eval_points,
                    lagrange,
                    |p| double_shares[k][p].share_2t.value,
                    |p| double_shares[k][p].share_t.point,
                    |p| double_shares[k][p].share_t.value,
                    &mut local_rng,
                ));
            }
            out
        })
        .collect();

    let mut result = Vec::with_capacity(count);
    for chunk in chunk_results {
        result.extend(chunk);
    }
    Ok(result)
}

/// Core triple generation for a single triple. Inlines polynomial evaluation
/// and fuses masked-product computation with Lagrange reconstruction to avoid
/// intermediate allocations.
#[inline]
#[allow(clippy::too_many_arguments)]
fn gen_triple_core(
    n: usize,
    t: usize,
    eval_points: &[Fp],
    lagrange: &[Fp],
    ds_2t_value: impl Fn(usize) -> Fp,
    ds_t_point: impl Fn(usize) -> Fp,
    ds_t_value: impl Fn(usize) -> Fp,
    rng: &mut ChaCha20Rng,
) -> Vec<BeaverTriple> {
    let a_secret = Fp::random(rng);
    let b_secret = Fp::random(rng);

    // Random polynomial coefficients (degree-t: t coefficients beyond the secret)
    let mut a_coeffs = [Fp::ZERO; 8];
    let mut b_coeffs = [Fp::ZERO; 8];
    for i in 0..t {
        a_coeffs[i] = Fp::random(rng);
        b_coeffs[i] = Fp::random(rng);
    }

    // Evaluate polynomials at each party's point, compute masked product,
    // and accumulate Lagrange reconstruction of δ — all in one pass.
    let mut a_vals = [Fp::ZERO; 16];
    let mut b_vals = [Fp::ZERO; 16];
    let mut delta = Fp::ZERO;

    for p in 0..n {
        let x = eval_points[p];

        // poly_a(x) = a_secret + a_coeffs[0]*x + a_coeffs[1]*x² + ...
        let mut a_val = a_secret;
        let mut x_pow = x;
        for i in 0..t {
            a_val += a_coeffs[i] * x_pow;
            x_pow *= x;
        }

        let mut b_val = b_secret;
        x_pow = x;
        for i in 0..t {
            b_val += b_coeffs[i] * x_pow;
            x_pow *= x;
        }

        a_vals[p] = a_val;
        b_vals[p] = b_val;

        // masked = a_p * b_p − r_2t_p;  delta += lagrange[p] * masked
        delta += lagrange[p] * (a_val * b_val - ds_2t_value(p));
    }

    // Build triple: c_p = r_t_p + δ
    (0..n)
        .map(|p| BeaverTriple {
            a: Share { point: eval_points[p], value: a_vals[p] },
            b: Share { point: eval_points[p], value: b_vals[p] },
            c: Share {
                point: ds_t_point(p),
                value: ds_t_value(p) + delta,
            },
        })
        .collect()
}

/// Generate Beaver triples from party-indexed double shares `[party][index]`.
///
/// Works directly with the party-indexed layout (no transpose), using rayon
/// parallelism over chunks of triples with per-chunk RNGs.
pub fn generate_triples_from_party_indexed<R: Rng>(
    n: usize,
    t: usize,
    party_double_shares: &[Vec<DoubleShare>],
    rng: &mut R,
) -> Result<Vec<Vec<BeaverTriple>>> {
    if n == 0 || party_double_shares.len() != n {
        return Err(ProtocolError::InvalidParams(format!(
            "expected {} > 0 party vectors, got {}",
            n,
            party_double_shares.len()
        )));
    }
    if party_double_shares[0].is_empty() {
        return Err(ProtocolError::InvalidParams(
            "empty double shares".into(),
        ));
    }
    let count = party_double_shares[0].len();
    for (p, ds) in party_double_shares.iter().enumerate() {
        if ds.len() != count {
            return Err(ProtocolError::InvalidParams(format!(
                "party {} has {} double shares, expected {}",
                p,
                ds.len(),
                count
            )));
        }
    }

    if n <= 2 * t {
        return Err(ProtocolError::InvalidParams(format!(
            "need n > 2t, got n={}, t={}",
            n, t
        )));
    }

    let shamir_t = Shamir::new(n, t)?;
    let shamir_2t = Shamir::new(n, 2 * t)?;
    let eval_points = &shamir_t.eval_points;
    let lagrange = shamir_2t.lagrange_coefficients();

    const CHUNK: usize = 4096;
    let num_chunks = count.div_ceil(CHUNK);
    let seeds: Vec<u64> = (0..num_chunks).map(|_| rng.gen()).collect();

    let chunk_results: Vec<Vec<Vec<BeaverTriple>>> = (0..num_chunks)
        .into_par_iter()
        .map(|ci| {
            let start = ci * CHUNK;
            let end = (start + CHUNK).min(count);
            let mut local_rng = ChaCha20Rng::seed_from_u64(seeds[ci]);
            let mut out = Vec::with_capacity(end - start);

            for k in start..end {
                out.push(gen_triple_core(
                    n,
                    t,
                    eval_points,
                    lagrange,
                    |p| party_double_shares[p][k].share_2t.value,
                    |p| party_double_shares[p][k].share_t.point,
                    |p| party_double_shares[p][k].share_t.value,
                    &mut local_rng,
                ));
            }
            out
        })
        .collect();

    let mut result = Vec::with_capacity(count);
    for chunk in chunk_results {
        result.extend(chunk);
    }
    Ok(result)
}

/// Generate Beaver triples directly from OT correlations, fusing HIM mixing
/// and triple generation in a single parallel pass.
///
/// This eliminates the intermediate double share storage (~320MB for 2M triples),
/// combining HIM mixing + polynomial evaluation + Lagrange reconstruction in one loop.
///
/// Returns `triples[k][p]` = party p's BeaverTriple for triple k.
pub fn generate_triples_from_ot<R: Rng>(
    n: usize,
    t: usize,
    count: usize,
    ot_correlations: &[ExpandedCorrelations],
    rng: &mut R,
) -> Result<Vec<Vec<BeaverTriple>>> {
    if n <= 2 * t {
        return Err(ProtocolError::InvalidParams(format!(
            "need n > 2t, got n={}, t={}",
            n, t
        )));
    }
    if count == 0 {
        return Err(ProtocolError::InvalidParams("count must be > 0".into()));
    }
    if ot_correlations.len() != n {
        return Err(ProtocolError::InvalidParams(format!(
            "expected {} OT correlations, got {}",
            n,
            ot_correlations.len()
        )));
    }

    let sharings_per_round = n - 2 * t;
    let num_rounds = count.div_ceil(sharings_per_round);

    let shamir_t = Shamir::new(n, t)?;
    let shamir_2t = Shamir::new(n, 2 * t)?;
    let him = HyperInvertibleMatrix::new(n);
    let eval_points = &shamir_t.eval_points;
    let eval_sq: Vec<Fp> = eval_points.iter().map(|&x| x * x).collect();
    let lagrange_2t = shamir_2t.lagrange_coefficients();

    // Pre-compute sums for factoring out constant terms from delta accumulation.
    // delta = sum_p lag[p] * (a_val*b_val - a_mixed - x*b1 - x^2*b2)
    //       = sum_p lag[p]*a_val*b_val - a_mixed*lag_sum - b1*lag_x_sum - b2*lag_xsq_sum
    let lag_sum: Fp = lagrange_2t.iter().copied().sum();
    let lag_x_sum: Fp = (0..n).map(|p| lagrange_2t[p] * eval_points[p]).sum();
    let lag_xsq_sum: Fp = (0..n).map(|p| lagrange_2t[p] * eval_sq[p]).sum();

    // Pre-compute HIM rows for the output sharings (first sharings_per_round rows)
    let him_rows: Vec<Vec<Fp>> = (0..sharings_per_round)
        .map(|j| (0..n).map(|i| him.get(j, i)).collect())
        .collect();

    const CHUNK: usize = 4096; // rounds per chunk
    let num_chunks = num_rounds.div_ceil(CHUNK);
    let seeds: Vec<u64> = (0..num_chunks).map(|_| rng.gen()).collect();

    let chunk_results: Vec<Vec<Vec<BeaverTriple>>> = (0..num_chunks)
        .into_par_iter()
        .map(|ci| {
            let round_start = ci * CHUNK;
            let round_end = (round_start + CHUNK).min(num_rounds);
            let mut local_rng = ChaCha20Rng::seed_from_u64(seeds[ci]);
            let capacity = (round_end - round_start) * sharings_per_round;
            let mut out: Vec<Vec<BeaverTriple>> = Vec::with_capacity(capacity);

            for round in round_start..round_end {
                // Get OT secrets for this round
                let mut secrets = [Fp::ZERO; 16];
                for i in 0..n {
                    secrets[i] = ot_correlations[i].get_random(round);
                }

                // Random coefficients for Shamir sharing (HIM input)
                let mut r_t = [Fp::ZERO; 16];
                let mut r1_2t = [Fp::ZERO; 16];
                let mut r2_2t = [Fp::ZERO; 16];
                for i in 0..n {
                    r_t[i] = Fp::random(&mut local_rng);
                    r1_2t[i] = Fp::random(&mut local_rng);
                    r2_2t[i] = Fp::random(&mut local_rng);
                }

                for j in 0..sharings_per_round {
                    // HIM mix: compute mixed secret and random coefficients
                    let mut a_mixed = Fp::ZERO;
                    let mut bt = Fp::ZERO;
                    let mut b1 = Fp::ZERO;
                    let mut b2 = Fp::ZERO;
                    for i in 0..n {
                        let m = him_rows[j][i];
                        a_mixed += m * secrets[i];
                        bt += m * r_t[i];
                        b1 += m * r1_2t[i];
                        b2 += m * r2_2t[i];
                    }

                    // Generate Beaver triple a, b polynomials
                    let a0 = Fp::random(&mut local_rng);
                    let b0 = Fp::random(&mut local_rng);
                    let mut a_coeffs = [Fp::ZERO; 8];
                    let mut b_coeffs = [Fp::ZERO; 8];
                    for i in 0..t {
                        a_coeffs[i] = Fp::random(&mut local_rng);
                        b_coeffs[i] = Fp::random(&mut local_rng);
                    }

                    // Fused: poly eval + inline double share + Lagrange reconstruction.
                    // Delta factored: delta = sum_p lag[p]*a*b - a_mixed*lag_sum - b1*lag_x_sum - b2*lag_xsq_sum
                    // This reduces 3 muls per p (ds_2t terms) to 3 muls total (outside the loop).
                    let mut a_vals = [Fp::ZERO; 16];
                    let mut b_vals = [Fp::ZERO; 16];
                    let mut ds_t_vals = [Fp::ZERO; 16];
                    let mut delta_ab = Fp::ZERO;

                    for p in 0..n {
                        let x = eval_points[p];

                        // Horner evaluation: a0 + a_coeffs[0]*x + ...
                        let mut a_val = a0;
                        let mut x_pow = x;
                        for i in 0..t {
                            a_val += a_coeffs[i] * x_pow;
                            x_pow *= x;
                        }

                        let mut b_val = b0;
                        x_pow = x;
                        for i in 0..t {
                            b_val += b_coeffs[i] * x_pow;
                            x_pow *= x;
                        }

                        a_vals[p] = a_val;
                        b_vals[p] = b_val;
                        ds_t_vals[p] = a_mixed + x * bt;

                        // Only the a*b product contribution varies per p
                        delta_ab += lagrange_2t[p] * a_val * b_val;
                    }

                    // Subtract the constant ds_2t terms (factored out of the p-loop)
                    let delta = delta_ab - a_mixed * lag_sum - b1 * lag_x_sum - b2 * lag_xsq_sum;

                    // Build triples: c_p = r_t_p + delta
                    out.push(
                        (0..n)
                            .map(|p| BeaverTriple {
                                a: Share { point: eval_points[p], value: a_vals[p] },
                                b: Share { point: eval_points[p], value: b_vals[p] },
                                c: Share { point: eval_points[p], value: ds_t_vals[p] + delta },
                            })
                            .collect(),
                    );
                }
            }
            out
        })
        .collect();

    let mut result: Vec<Vec<BeaverTriple>> = chunk_results.into_iter().flatten().collect();
    result.truncate(count);
    Ok(result)
}

/// Beaver multiplication: compute degree-t shares of x·y.
///
/// Each party holds x_p, y_p, and a triple (a_p, b_p, c_p).
/// 1. Open d = x − a, e = y − b via degree-t reconstruction
/// 2. z_p = c_p + e·x_p + d·y_p − d·e
///
/// Correctness: z(0) = ab + (y−b)·x + (x−a)·y − (x−a)(y−b) = x·y
pub fn beaver_multiply_local(
    n: usize,
    t: usize,
    x_shares: &[Share],
    y_shares: &[Share],
    triples: &[BeaverTriple],
) -> Result<Vec<Share>> {
    if x_shares.len() != n || y_shares.len() != n || triples.len() != n {
        return Err(ProtocolError::InvalidParams(format!(
            "expected {} shares, got x={}, y={}, triples={}",
            n,
            x_shares.len(),
            y_shares.len(),
            triples.len()
        )));
    }

    let shamir_t = Shamir::new(n, t)?;

    // d_p = x_p − a_p, e_p = y_p − b_p
    let d_shares: Vec<Share> = (0..n)
        .map(|p| Share {
            point: x_shares[p].point,
            value: x_shares[p].value - triples[p].a.value,
        })
        .collect();
    let e_shares: Vec<Share> = (0..n)
        .map(|p| Share {
            point: y_shares[p].point,
            value: y_shares[p].value - triples[p].b.value,
        })
        .collect();

    let d = shamir_t.reconstruct(&d_shares)?;
    let e = shamir_t.reconstruct(&e_shares)?;
    let de = d * e;

    Ok((0..n)
        .map(|p| Share {
            point: x_shares[p].point,
            value: triples[p].c.value + e * x_shares[p].value + d * y_shares[p].value - de,
        })
        .collect())
}

/// Chain multiplication using Beaver triples.
///
/// Computes shares of value_shares[0] * value_shares[1] * ... * value_shares[m-1].
/// `triples[k][p]` is party p's triple for the k-th multiplication.
pub fn beaver_multiply_chain(
    n: usize,
    t: usize,
    value_shares: &[Vec<Share>],
    triples: &[Vec<BeaverTriple>],
) -> Result<Vec<Share>> {
    let m = value_shares.len();
    if m < 2 {
        return Err(ProtocolError::InvalidParams(
            "need at least 2 values to multiply".into(),
        ));
    }
    if triples.len() != m - 1 {
        return Err(ProtocolError::InvalidParams(format!(
            "need {} triples for {} multiplications, got {}",
            m - 1,
            m - 1,
            triples.len()
        )));
    }
    for (idx, vs) in value_shares.iter().enumerate() {
        if vs.len() != n {
            return Err(ProtocolError::InvalidParams(format!(
                "value_shares[{}] has {} shares, expected {}",
                idx,
                vs.len(),
                n
            )));
        }
    }
    for (idx, ts) in triples.iter().enumerate() {
        if ts.len() != n {
            return Err(ProtocolError::InvalidParams(format!(
                "triples[{}] has {} entries, expected {}",
                idx,
                ts.len(),
                n
            )));
        }
    }

    let mut current = value_shares[0].clone();
    for i in 1..m {
        current =
            beaver_multiply_local(n, t, &current, &value_shares[i], &triples[i - 1])?;
    }
    Ok(current)
}

/// Struct-of-arrays layout for Beaver triples, storing raw u64 values.
///
/// Layout is party-major: `a_values[party * count + k]` is party `party`'s
/// a-share for triple `k`. Eval points are implicit: party p uses point p+1.
///
/// This eliminates per-triple heap allocation (vs `Vec<Vec<BeaverTriple>>`)
/// and allows the hot path to work entirely with raw u64 arithmetic.
#[derive(Clone, Debug)]
pub struct BeaverTripleBatch {
    pub n: usize,
    pub count: usize,
    pub a_values: Vec<u64>,
    pub b_values: Vec<u64>,
    pub c_values: Vec<u64>,
}

impl BeaverTripleBatch {
    /// Retrieve a single `BeaverTriple` for compatibility with the AoS API.
    #[inline]
    pub fn triple(&self, k: usize, party: usize) -> BeaverTriple {
        debug_assert!(k < self.count && party < self.n);
        let idx = party * self.count + k;
        let point = Fp::from_reduced((party + 1) as u64);
        BeaverTriple {
            a: Share { point, value: Fp::from_reduced(self.a_values[idx]) },
            b: Share { point, value: Fp::from_reduced(self.b_values[idx]) },
            c: Share { point, value: Fp::from_reduced(self.c_values[idx]) },
        }
    }

    /// Collect all n parties' shares for triple `k` as a `Vec<BeaverTriple>`.
    pub fn triple_shares(&self, k: usize) -> Vec<BeaverTriple> {
        (0..self.n).map(|p| self.triple(k, p)).collect()
    }
}

/// Generate Beaver triples directly from OT correlations using SoA layout.
///
/// Same algorithm as `generate_triples_from_ot` but:
/// - Output is a flat `BeaverTripleBatch` (no per-triple Vec allocation)
/// - Inner loops use raw u64 arithmetic (avoids Fp wrapper overhead)
/// - Delta accumulation uses u128 lazy reduction (saves 4 reductions per triple)
/// - Output buffer is pre-allocated; rayon chunks write directly to slices
pub fn generate_triples_from_ot_batch<R: Rng>(
    n: usize,
    t: usize,
    count: usize,
    ot_correlations: &[ExpandedCorrelations],
    rng: &mut R,
) -> Result<BeaverTripleBatch> {
    if n <= 2 * t {
        return Err(ProtocolError::InvalidParams(format!(
            "need n > 2t, got n={}, t={}",
            n, t
        )));
    }
    if count == 0 {
        return Err(ProtocolError::InvalidParams("count must be > 0".into()));
    }
    if ot_correlations.len() != n {
        return Err(ProtocolError::InvalidParams(format!(
            "expected {} OT correlations, got {}",
            n,
            ot_correlations.len()
        )));
    }

    let sharings_per_round = n - 2 * t;
    let num_rounds = count.div_ceil(sharings_per_round);

    let shamir_t = Shamir::new(n, t)?;
    let shamir_2t = Shamir::new(n, 2 * t)?;
    let him = HyperInvertibleMatrix::new(n);

    // Pre-compute eval points as raw u64
    let eval_raw: Vec<u64> = shamir_t.eval_points.iter().map(|fp| fp.raw()).collect();
    let eval_sq_raw: Vec<u64> = eval_raw.iter().map(|&x| Fp::mul_raw(x, x)).collect();

    // Pre-compute Lagrange coefficients as raw u64
    let lagrange_2t = shamir_2t.lagrange_coefficients();
    let lag_raw: Vec<u64> = lagrange_2t.iter().map(|fp| fp.raw()).collect();

    // Pre-compute factored sums as raw u64
    let mut lag_sum: u64 = 0;
    let mut lag_x_sum: u64 = 0;
    let mut lag_xsq_sum: u64 = 0;
    for p in 0..n {
        lag_sum = Fp::add_raw(lag_sum, lag_raw[p]);
        lag_x_sum = Fp::add_raw(lag_x_sum, Fp::mul_raw(lag_raw[p], eval_raw[p]));
        lag_xsq_sum = Fp::add_raw(lag_xsq_sum, Fp::mul_raw(lag_raw[p], eval_sq_raw[p]));
    }

    // Pre-compute HIM rows as raw u64
    let him_rows_raw: Vec<Vec<u64>> = (0..sharings_per_round)
        .map(|j| (0..n).map(|i| him.get(j, i).raw()).collect())
        .collect();

    // Pre-allocate output buffers (party-major layout: a_values[p * count + k])
    let total = n * count;
    let mut a_values = vec![0u64; total];
    let mut b_values = vec![0u64; total];
    let mut c_values = vec![0u64; total];

    // Dynamic chunk size: ensure at least 8 chunks for parallelism,
    // with a maximum of 16384 rounds per chunk to amortize scheduling overhead.
    const MAX_CHUNK: usize = 16384;
    const MIN_CHUNKS: usize = 8;
    let chunk = (num_rounds.div_ceil(MIN_CHUNKS)).min(MAX_CHUNK).max(1);
    let num_chunks = num_rounds.div_ceil(chunk);
    let seeds: Vec<u64> = (0..num_chunks).map(|_| rng.gen()).collect();

    // Number of random u64 values needed per round:
    // 3*n (r_t, r1_2t, r2_2t) + spr * (2 + 2*t) (a0, b0, a_coeffs, b_coeffs per sharing)
    let rands_per_round = 3 * n + sharings_per_round * (2 + 2 * t);

    // Raw pointers for direct partitioned writes from rayon chunks.
    // SAFETY: each chunk writes to disjoint triple indices [chunk_start_triple..chunk_end_triple)
    // within each party's band [p*count..(p+1)*count). No two chunks share any index.
    // We store as usize for Send+Sync; reconstructed as *mut u64 inside each chunk.
    let a_base = a_values.as_mut_ptr() as usize;
    let b_base = b_values.as_mut_ptr() as usize;
    let c_base = c_values.as_mut_ptr() as usize;

    (0..num_chunks)
        .into_par_iter()
        .for_each(|ci| {
            let round_start = ci * chunk;
            let round_end = (round_start + chunk).min(num_rounds);
            let mut local_rng = ChaCha20Rng::seed_from_u64(seeds[ci]);
            let mut triple_idx = round_start * sharings_per_round;

            // Pre-allocate batch random buffer (reused across rounds)
            let mut rand_buf = vec![0u64; rands_per_round];

            for round in round_start..round_end {
                // Load OT secrets as raw u64
                let mut secrets = [0u64; 16];
                for i in 0..n {
                    secrets[i] = ot_correlations[i].get_random_raw(round);
                }

                // Batch random generation for this round
                Fp::random_batch_raw(&mut local_rng, &mut rand_buf[..rands_per_round]);
                let mut ri = 0usize;

                let mut r_t = [0u64; 16];
                let mut r1_2t = [0u64; 16];
                let mut r2_2t = [0u64; 16];
                for i in 0..n {
                    r_t[i] = rand_buf[ri]; ri += 1;
                    r1_2t[i] = rand_buf[ri]; ri += 1;
                    r2_2t[i] = rand_buf[ri]; ri += 1;
                }

                for j in 0..sharings_per_round {
                    if triple_idx >= count {
                        return;
                    }

                    // HIM mix (raw u64 dot products)
                    let mut a_mixed: u64 = 0;
                    let mut bt: u64 = 0;
                    let mut b1: u64 = 0;
                    let mut b2: u64 = 0;
                    for i in 0..n {
                        let m = him_rows_raw[j][i];
                        a_mixed = Fp::add_raw(a_mixed, Fp::mul_raw(m, secrets[i]));
                        bt = Fp::add_raw(bt, Fp::mul_raw(m, r_t[i]));
                        b1 = Fp::add_raw(b1, Fp::mul_raw(m, r1_2t[i]));
                        b2 = Fp::add_raw(b2, Fp::mul_raw(m, r2_2t[i]));
                    }

                    // Random polynomial coefficients from batch buffer
                    let a0 = rand_buf[ri]; ri += 1;
                    let b0 = rand_buf[ri]; ri += 1;
                    let mut a_coeffs = [0u64; 8];
                    let mut b_coeffs = [0u64; 8];
                    for i in 0..t {
                        a_coeffs[i] = rand_buf[ri]; ri += 1;
                        b_coeffs[i] = rand_buf[ri]; ri += 1;
                    }

                    // Evaluate polynomials and compute delta with lazy u128 accumulation
                    let mut a_vals = [0u64; 16];
                    let mut b_vals = [0u64; 16];
                    let mut ds_t_vals = [0u64; 16];
                    let mut delta_ab_wide: u128 = 0;

                    for p in 0..n {
                        let x = eval_raw[p];

                        // Horner eval for a(x)
                        let mut a_val = a0;
                        let mut x_pow = x;
                        for i in 0..t {
                            a_val = Fp::add_raw(a_val, Fp::mul_raw(a_coeffs[i], x_pow));
                            x_pow = Fp::mul_raw(x_pow, x);
                        }

                        // Horner eval for b(x)
                        let mut b_val = b0;
                        x_pow = x;
                        for i in 0..t {
                            b_val = Fp::add_raw(b_val, Fp::mul_raw(b_coeffs[i], x_pow));
                            x_pow = Fp::mul_raw(x_pow, x);
                        }

                        a_vals[p] = a_val;
                        b_vals[p] = b_val;
                        ds_t_vals[p] = Fp::add_raw(a_mixed, Fp::mul_raw(x, bt));

                        delta_ab_wide += (lag_raw[p] as u128) * (Fp::mul_raw(a_val, b_val) as u128);
                    }

                    let delta_ab = Fp::reduce_wide(delta_ab_wide);

                    let delta = Fp::sub_raw(
                        delta_ab,
                        Fp::add_raw(
                            Fp::mul_raw(a_mixed, lag_sum),
                            Fp::add_raw(
                                Fp::mul_raw(b1, lag_x_sum),
                                Fp::mul_raw(b2, lag_xsq_sum),
                            ),
                        ),
                    );

                    // Write directly to party-major output via raw pointers.
                    // SAFETY: triple_idx is unique per chunk (disjoint ranges),
                    // and p*count+triple_idx is within bounds for each party p.
                    for p in 0..n {
                        let dst = p * count + triple_idx;
                        unsafe {
                            *(a_base as *mut u64).add(dst) = a_vals[p];
                            *(b_base as *mut u64).add(dst) = b_vals[p];
                            *(c_base as *mut u64).add(dst) = Fp::add_raw(ds_t_vals[p], delta);
                        }
                    }
                    triple_idx += 1;
                }
            }
        });

    Ok(BeaverTripleBatch {
        n,
        count,
        a_values,
        b_values,
        c_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multiply::DnMultiply;
    use crate::randousha::{RanDouShaParams, RanDouShaProtocol};
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    /// Helper: generate `count` double shares in [index][party] layout.
    fn gen_double_shares(
        n: usize,
        t: usize,
        count: usize,
        rng: &mut ChaCha20Rng,
    ) -> Vec<Vec<DoubleShare>> {
        let params = RanDouShaParams::new(n, t, count).unwrap();
        let party_ds = RanDouShaProtocol::new(params)
            .generate_local(rng)
            .unwrap();
        // Transpose [party][index] → [index][party]
        (0..count)
            .map(|k| (0..n).map(|p| party_ds[p][k].clone()).collect())
            .collect()
    }

    /// Helper: generate `count` double shares in [party][index] layout.
    fn gen_party_indexed_double_shares(
        n: usize,
        t: usize,
        count: usize,
        rng: &mut ChaCha20Rng,
    ) -> Vec<Vec<DoubleShare>> {
        let params = RanDouShaParams::new(n, t, count).unwrap();
        RanDouShaProtocol::new(params)
            .generate_local(rng)
            .unwrap()
    }

    // ── Batch reconstruction ──────────────────────────────────────────

    #[test]
    fn test_batch_reconstruct_basic() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let n = 5;
        let t = 1;
        let shamir_t = Shamir::new(n, t).unwrap();
        // degree-2t Shamir for reconstruction (degree-t poly is valid degree-2t)
        let shamir_2t = Shamir::new(n, 2 * t).unwrap();

        let secrets = [Fp::new(10), Fp::new(20), Fp::new(30)];
        let shares_per_value: Vec<Vec<Share>> =
            secrets.iter().map(|&s| shamir_t.share(s, &mut rng)).collect();

        let recovered = batch_reconstruct(&shares_per_value, &shamir_2t).unwrap();
        assert_eq!(recovered.len(), 3);
        for (i, &s) in secrets.iter().enumerate() {
            assert_eq!(recovered[i], s, "mismatch at index {}", i);
        }
    }

    #[test]
    fn test_batch_reconstruct_degree_2t() {
        let mut rng = ChaCha20Rng::seed_from_u64(99);
        let n = 5;
        let t = 1;
        let shamir_2t = Shamir::new(n, 2 * t).unwrap();

        let secrets = [Fp::new(111), Fp::new(222)];
        let shares_per_value: Vec<Vec<Share>> = secrets
            .iter()
            .map(|&s| shamir_2t.share(s, &mut rng))
            .collect();

        let recovered = batch_reconstruct(&shares_per_value, &shamir_2t).unwrap();
        for (i, &s) in secrets.iter().enumerate() {
            assert_eq!(recovered[i], s, "mismatch at index {}", i);
        }
    }

    #[test]
    fn test_batch_reconstruct_empty() {
        let shamir_2t = Shamir::new(5, 2).unwrap();
        let recovered = batch_reconstruct(&[], &shamir_2t).unwrap();
        assert!(recovered.is_empty());
    }

    #[test]
    fn test_batch_reconstruct_wrong_count() {
        let shamir_2t = Shamir::new(5, 2).unwrap();
        // 3 shares instead of 5
        let bad = vec![vec![
            Share { point: Fp::new(1), value: Fp::new(1) },
            Share { point: Fp::new(2), value: Fp::new(2) },
            Share { point: Fp::new(3), value: Fp::new(3) },
        ]];
        let result = batch_reconstruct(&bad, &shamir_2t);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("expected"));
    }

    // ── Triple generation ─────────────────────────────────────────────

    #[test]
    fn test_generate_triples_basic() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let n = 5;
        let t = 1;
        let shamir_t = Shamir::new(n, t).unwrap();
        let ds = gen_double_shares(n, t, 3, &mut rng);

        let triples = generate_triples(n, t, &ds, &mut rng).unwrap();
        assert_eq!(triples.len(), 3);

        for k in 0..3 {
            assert_eq!(triples[k].len(), n);
            let a_shares: Vec<Share> = triples[k].iter().map(|tr| tr.a).collect();
            let b_shares: Vec<Share> = triples[k].iter().map(|tr| tr.b).collect();
            let c_shares: Vec<Share> = triples[k].iter().map(|tr| tr.c).collect();

            let a = shamir_t.reconstruct(&a_shares).unwrap();
            let b = shamir_t.reconstruct(&b_shares).unwrap();
            let c = shamir_t.reconstruct(&c_shares).unwrap();

            assert_eq!(c, a * b, "c != a*b for triple {}", k);
        }
    }

    #[test]
    fn test_generate_triples_c_is_degree_t() {
        let mut rng = ChaCha20Rng::seed_from_u64(77);
        let n = 5;
        let t = 1;
        let ds = gen_double_shares(n, t, 1, &mut rng);
        let triples = generate_triples(n, t, &ds, &mut rng).unwrap();

        let c_shares: Vec<Share> = triples[0].iter().map(|tr| tr.c).collect();

        // Any t+1 = 2 shares should reconstruct the same value
        let shamir_t = Shamir::new(n, t).unwrap();
        let c_full = shamir_t.reconstruct(&c_shares).unwrap();

        for start in 0..n - t {
            let subset = &c_shares[start..start + t + 1];
            let c_sub = shamir_t.reconstruct(subset).unwrap();
            assert_eq!(c_sub, c_full, "subset starting at {} disagrees", start);
        }
    }

    #[test]
    fn test_generate_triples_from_party_indexed() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let n = 5;
        let t = 1;
        let shamir_t = Shamir::new(n, t).unwrap();
        let party_ds = gen_party_indexed_double_shares(n, t, 3, &mut rng);

        let triples =
            generate_triples_from_party_indexed(n, t, &party_ds, &mut rng).unwrap();
        assert_eq!(triples.len(), 3);

        for k in 0..3 {
            let a = shamir_t
                .reconstruct(&triples[k].iter().map(|tr| tr.a).collect::<Vec<_>>())
                .unwrap();
            let b = shamir_t
                .reconstruct(&triples[k].iter().map(|tr| tr.b).collect::<Vec<_>>())
                .unwrap();
            let c = shamir_t
                .reconstruct(&triples[k].iter().map(|tr| tr.c).collect::<Vec<_>>())
                .unwrap();
            assert_eq!(c, a * b, "triple {} failed", k);
        }
    }

    #[test]
    fn test_generate_triples_from_ot_basic() {
        use crate::silent_ot::{DistributedSilentOt, SilentOtParams};

        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let n = 5;
        let t = 1;
        let count: usize = 10;
        let sharings_per_round = n - 2 * t;
        let num_rounds = count.div_ceil(sharings_per_round);

        let ot_params = SilentOtParams::new(n, t, num_rounds).unwrap();
        let ot_protocol = DistributedSilentOt::new(ot_params);
        let mut ot_states: Vec<_> =
            (0..n).map(|i| ot_protocol.init_party(i, &mut rng)).collect();

        // Run all 4 rounds
        let mut r0 = vec![Vec::new(); n];
        for s in ot_states.iter() {
            for (to, c) in DistributedSilentOt::round0_commitments(s) {
                r0[to].push((s.party_id, c));
            }
        }
        for (i, s) in ot_states.iter_mut().enumerate() {
            DistributedSilentOt::process_round0(s, &r0[i]).unwrap();
        }
        let mut r1 = vec![Vec::new(); n];
        for s in ot_states.iter() {
            for (to, idx) in DistributedSilentOt::round1_puncture_choices(s) {
                r1[to].push((s.party_id, idx));
            }
        }
        for (i, s) in ot_states.iter_mut().enumerate() {
            DistributedSilentOt::process_round1(s, &r1[i]).unwrap();
        }
        let mut r2 = vec![Vec::new(); n];
        for s in ot_states.iter() {
            for (to, path) in DistributedSilentOt::round2_sibling_paths(s).unwrap() {
                r2[to].push((s.party_id, path));
            }
        }
        for (i, s) in ot_states.iter_mut().enumerate() {
            DistributedSilentOt::process_round2(s, &r2[i]).unwrap();
        }
        let mut r3 = vec![Vec::new(); n];
        for s in ot_states.iter() {
            for (to, seed) in DistributedSilentOt::round3_seed_reveals(s) {
                r3[to].push((s.party_id, seed));
            }
        }
        for (i, s) in ot_states.iter_mut().enumerate() {
            DistributedSilentOt::process_round3(s, &r3[i]).unwrap();
        }

        let ot_correlations: Vec<_> = ot_states
            .iter()
            .map(|s| DistributedSilentOt::expand(s).unwrap())
            .collect();

        let triples =
            generate_triples_from_ot(n, t, count, &ot_correlations, &mut rng).unwrap();
        assert_eq!(triples.len(), count);

        let shamir_t = Shamir::new(n, t).unwrap();
        for k in 0..count {
            assert_eq!(triples[k].len(), n);
            let a_shares: Vec<Share> = triples[k].iter().map(|tr| tr.a).collect();
            let b_shares: Vec<Share> = triples[k].iter().map(|tr| tr.b).collect();
            let c_shares: Vec<Share> = triples[k].iter().map(|tr| tr.c).collect();

            let a = shamir_t.reconstruct(&a_shares).unwrap();
            let b = shamir_t.reconstruct(&b_shares).unwrap();
            let c = shamir_t.reconstruct(&c_shares).unwrap();
            assert_eq!(c, a * b, "triple {} failed: c != a*b", k);
        }
    }

    #[test]
    fn test_generate_triples_from_ot_batch_basic() {
        use crate::silent_ot::{DistributedSilentOt, SilentOtParams};

        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let n = 5;
        let t = 1;
        let count: usize = 10;
        let sharings_per_round = n - 2 * t;
        let num_rounds = count.div_ceil(sharings_per_round);

        let ot_params = SilentOtParams::new(n, t, num_rounds).unwrap();
        let ot_protocol = DistributedSilentOt::new(ot_params);
        let mut ot_states: Vec<_> =
            (0..n).map(|i| ot_protocol.init_party(i, &mut rng)).collect();

        // Run all 4 rounds
        let mut r0 = vec![Vec::new(); n];
        for s in ot_states.iter() {
            for (to, c) in DistributedSilentOt::round0_commitments(s) {
                r0[to].push((s.party_id, c));
            }
        }
        for (i, s) in ot_states.iter_mut().enumerate() {
            DistributedSilentOt::process_round0(s, &r0[i]).unwrap();
        }
        let mut r1 = vec![Vec::new(); n];
        for s in ot_states.iter() {
            for (to, idx) in DistributedSilentOt::round1_puncture_choices(s) {
                r1[to].push((s.party_id, idx));
            }
        }
        for (i, s) in ot_states.iter_mut().enumerate() {
            DistributedSilentOt::process_round1(s, &r1[i]).unwrap();
        }
        let mut r2 = vec![Vec::new(); n];
        for s in ot_states.iter() {
            for (to, path) in DistributedSilentOt::round2_sibling_paths(s).unwrap() {
                r2[to].push((s.party_id, path));
            }
        }
        for (i, s) in ot_states.iter_mut().enumerate() {
            DistributedSilentOt::process_round2(s, &r2[i]).unwrap();
        }
        let mut r3 = vec![Vec::new(); n];
        for s in ot_states.iter() {
            for (to, seed) in DistributedSilentOt::round3_seed_reveals(s) {
                r3[to].push((s.party_id, seed));
            }
        }
        for (i, s) in ot_states.iter_mut().enumerate() {
            DistributedSilentOt::process_round3(s, &r3[i]).unwrap();
        }

        let ot_correlations: Vec<_> = ot_states
            .iter()
            .map(|s| DistributedSilentOt::expand(s).unwrap())
            .collect();

        // Generate using both APIs with same seed
        let triples_vec =
            generate_triples_from_ot(n, t, count, &ot_correlations, &mut ChaCha20Rng::seed_from_u64(99)).unwrap();
        let triples_batch =
            generate_triples_from_ot_batch(n, t, count, &ot_correlations, &mut ChaCha20Rng::seed_from_u64(99)).unwrap();

        assert_eq!(triples_batch.count, count);
        assert_eq!(triples_batch.n, n);

        // Verify batch produces correct triples (c = a*b)
        let shamir_t = Shamir::new(n, t).unwrap();
        for k in 0..count {
            let shares = triples_batch.triple_shares(k);
            let a_shares: Vec<Share> = shares.iter().map(|tr| tr.a).collect();
            let b_shares: Vec<Share> = shares.iter().map(|tr| tr.b).collect();
            let c_shares: Vec<Share> = shares.iter().map(|tr| tr.c).collect();

            let a = shamir_t.reconstruct(&a_shares).unwrap();
            let b = shamir_t.reconstruct(&b_shares).unwrap();
            let c = shamir_t.reconstruct(&c_shares).unwrap();
            assert_eq!(c, a * b, "batch triple {} failed: c != a*b", k);
        }

        // Verify old API also produces correct triples (independent check)
        for k in 0..count {
            let a_shares: Vec<Share> = triples_vec[k].iter().map(|tr| tr.a).collect();
            let b_shares: Vec<Share> = triples_vec[k].iter().map(|tr| tr.b).collect();
            let c_shares: Vec<Share> = triples_vec[k].iter().map(|tr| tr.c).collect();

            let a = shamir_t.reconstruct(&a_shares).unwrap();
            let b = shamir_t.reconstruct(&b_shares).unwrap();
            let c = shamir_t.reconstruct(&c_shares).unwrap();
            assert_eq!(c, a * b, "vec triple {} failed: c != a*b", k);
        }
    }

    #[test]
    fn test_generate_triples_invalid_params() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);

        // n <= 2t
        let ds = vec![vec![]; 4];
        assert!(generate_triples(4, 2, &ds, &mut rng).is_err());

        // empty double_shares
        assert!(generate_triples(5, 1, &[], &mut rng).is_err());

        // wrong inner length
        let bad_ds = vec![vec![
            DoubleShare {
                share_t: Share { point: Fp::new(1), value: Fp::ZERO },
                share_2t: Share { point: Fp::new(1), value: Fp::ZERO },
            };
            3 // 3 instead of 5
        ]];
        assert!(generate_triples(5, 1, &bad_ds, &mut rng).is_err());
    }

    // ── Beaver multiply ───────────────────────────────────────────────

    #[test]
    fn test_beaver_multiply_local_basic() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let n = 5;
        let t = 1;
        let shamir_t = Shamir::new(n, t).unwrap();

        let x = Fp::new(7);
        let y = Fp::new(6);
        let x_shares = shamir_t.share(x, &mut rng);
        let y_shares = shamir_t.share(y, &mut rng);

        let ds = gen_double_shares(n, t, 1, &mut rng);
        let triples = generate_triples(n, t, &ds, &mut rng).unwrap();

        let z_shares =
            beaver_multiply_local(n, t, &x_shares, &y_shares, &triples[0]).unwrap();
        let z = shamir_t.reconstruct(&z_shares).unwrap();
        assert_eq!(z, Fp::new(42));
    }

    #[test]
    fn test_beaver_multiply_with_zero() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let n = 5;
        let t = 1;
        let shamir_t = Shamir::new(n, t).unwrap();

        let x_shares = shamir_t.share(Fp::new(42), &mut rng);
        let y_shares = shamir_t.share(Fp::ZERO, &mut rng);

        let ds = gen_double_shares(n, t, 1, &mut rng);
        let triples = generate_triples(n, t, &ds, &mut rng).unwrap();

        let z_shares =
            beaver_multiply_local(n, t, &x_shares, &y_shares, &triples[0]).unwrap();
        let z = shamir_t.reconstruct(&z_shares).unwrap();
        assert_eq!(z, Fp::ZERO);
    }

    #[test]
    fn test_beaver_multiply_large_values() {
        use crate::field::PRIME;

        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let n = 5;
        let t = 1;
        let shamir_t = Shamir::new(n, t).unwrap();

        let x = Fp::new(PRIME - 1);
        let y = Fp::new(PRIME - 2);
        let expected = x * y;

        let x_shares = shamir_t.share(x, &mut rng);
        let y_shares = shamir_t.share(y, &mut rng);

        let ds = gen_double_shares(n, t, 1, &mut rng);
        let triples = generate_triples(n, t, &ds, &mut rng).unwrap();

        let z_shares =
            beaver_multiply_local(n, t, &x_shares, &y_shares, &triples[0]).unwrap();
        let z = shamir_t.reconstruct(&z_shares).unwrap();
        assert_eq!(z, expected);
    }

    #[test]
    fn test_beaver_multiply_agrees_with_dn() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let n = 5;
        let t = 1;
        let shamir_t = Shamir::new(n, t).unwrap();

        let x = Fp::new(13);
        let y = Fp::new(17);
        let x_shares = shamir_t.share(x, &mut rng);
        let y_shares = shamir_t.share(y, &mut rng);

        // DN multiply
        let params = RanDouShaParams::new(n, t, 1).unwrap();
        let party_ds_dn = RanDouShaProtocol::new(params)
            .generate_local(&mut rng)
            .unwrap();
        let dn_ds: Vec<DoubleShare> =
            (0..n).map(|p| party_ds_dn[p][0].clone()).collect();
        let dn = DnMultiply::new(n, t, 0).unwrap();
        let dn_result = shamir_t
            .reconstruct(
                &dn.multiply_local(&x_shares, &y_shares, &dn_ds)
                    .unwrap(),
            )
            .unwrap();

        // Beaver multiply
        let ds = gen_double_shares(n, t, 1, &mut rng);
        let triples = generate_triples(n, t, &ds, &mut rng).unwrap();
        let beaver_result = shamir_t
            .reconstruct(
                &beaver_multiply_local(n, t, &x_shares, &y_shares, &triples[0])
                    .unwrap(),
            )
            .unwrap();

        assert_eq!(dn_result, Fp::new(13 * 17));
        assert_eq!(beaver_result, Fp::new(13 * 17));
        assert_eq!(dn_result, beaver_result);
    }

    #[test]
    fn test_beaver_multiply_local_invalid_params() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let n = 5;
        let t = 1;
        let shamir_t = Shamir::new(n, t).unwrap();

        let x_shares = shamir_t.share(Fp::new(1), &mut rng);
        let y_shares = shamir_t.share(Fp::new(2), &mut rng);

        let ds = gen_double_shares(n, t, 1, &mut rng);
        let triples = generate_triples(n, t, &ds, &mut rng).unwrap();

        // Wrong number of x shares
        assert!(
            beaver_multiply_local(n, t, &x_shares[..3], &y_shares, &triples[0])
                .is_err()
        );
        // Wrong number of y shares
        assert!(
            beaver_multiply_local(n, t, &x_shares, &y_shares[..3], &triples[0])
                .is_err()
        );
        // Wrong number of triples
        assert!(
            beaver_multiply_local(n, t, &x_shares, &y_shares, &triples[0][..3])
                .is_err()
        );
    }

    #[test]
    fn test_beaver_minimum_params_n3_t1() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let n = 3;
        let t = 1;
        let shamir_t = Shamir::new(n, t).unwrap();

        let x = Fp::new(5);
        let y = Fp::new(9);
        let x_shares = shamir_t.share(x, &mut rng);
        let y_shares = shamir_t.share(y, &mut rng);

        let ds = gen_double_shares(n, t, 1, &mut rng);
        let triples = generate_triples(n, t, &ds, &mut rng).unwrap();

        let z_shares =
            beaver_multiply_local(n, t, &x_shares, &y_shares, &triples[0]).unwrap();
        let z = shamir_t.reconstruct(&z_shares).unwrap();
        assert_eq!(z, Fp::new(45));
    }

    // ── Chain multiply ────────────────────────────────────────────────

    #[test]
    fn test_beaver_chain_basic() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let n = 5;
        let t = 1;
        let shamir_t = Shamir::new(n, t).unwrap();

        let values = [Fp::new(2), Fp::new(3), Fp::new(4), Fp::new(5)];
        let value_shares: Vec<Vec<Share>> =
            values.iter().map(|&v| shamir_t.share(v, &mut rng)).collect();

        let ds = gen_double_shares(n, t, 3, &mut rng);
        let triples = generate_triples(n, t, &ds, &mut rng).unwrap();

        let result_shares =
            beaver_multiply_chain(n, t, &value_shares, &triples).unwrap();
        let result = shamir_t.reconstruct(&result_shares).unwrap();
        assert_eq!(result, Fp::new(120));
    }

    #[test]
    fn test_beaver_chain_long() {
        let mut rng = ChaCha20Rng::seed_from_u64(777);
        let n = 5;
        let t = 1;
        let shamir_t = Shamir::new(n, t).unwrap();

        let num_values = 50;
        let values: Vec<Fp> = (1..=num_values as u64).map(Fp::new).collect();
        let expected: Fp = values.iter().copied().reduce(|a, b| a * b).unwrap();

        let value_shares: Vec<Vec<Share>> =
            values.iter().map(|&v| shamir_t.share(v, &mut rng)).collect();

        let ds = gen_double_shares(n, t, num_values - 1, &mut rng);
        let triples = generate_triples(n, t, &ds, &mut rng).unwrap();

        let result_shares =
            beaver_multiply_chain(n, t, &value_shares, &triples).unwrap();
        let result = shamir_t.reconstruct(&result_shares).unwrap();
        assert_eq!(result, expected, "50! mod p mismatch");
    }

    #[test]
    fn test_beaver_chain_invalid_params() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let n = 5;
        let t = 1;
        let shamir_t = Shamir::new(n, t).unwrap();

        let v = shamir_t.share(Fp::new(1), &mut rng);

        // Fewer than 2 values
        assert!(beaver_multiply_chain(n, t, &[v.clone()], &[]).is_err());

        // Wrong number of triples
        let vs = vec![v.clone(), v.clone(), v];
        assert!(beaver_multiply_chain(n, t, &vs, &[]).is_err());

        // Wrong inner share count
        let bad_triple = vec![BeaverTriple {
            a: Share { point: Fp::new(1), value: Fp::ZERO },
            b: Share { point: Fp::new(1), value: Fp::ZERO },
            c: Share { point: Fp::new(1), value: Fp::ZERO },
        }; 3]; // 3 instead of 5
        assert!(
            beaver_multiply_chain(n, t, &vs[..2], &[bad_triple]).is_err()
        );
    }

    // ── Vandermonde ───────────────────────────────────────────────────

    #[test]
    fn test_make_vandermonde() {
        let v = make_vandermonde(3, 2);
        assert_eq!(v.len(), 3);
        // Row i: [1, α_{i+1}]
        // Row 0 (α=1): [1, 1]
        assert_eq!(v[0], vec![Fp::ONE, Fp::new(1)]);
        // Row 1 (α=2): [1, 2]
        assert_eq!(v[1], vec![Fp::ONE, Fp::new(2)]);
        // Row 2 (α=3): [1, 3]
        assert_eq!(v[2], vec![Fp::ONE, Fp::new(3)]);
    }

    // ── Stress test ───────────────────────────────────────────────────

    #[test]
    #[ignore]
    fn test_2m_beaver_triples_silent_ot() {
        use crate::silent_ot::{DistributedSilentOt, SilentOtParams};
        use rayon::prelude::*;

        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let count: usize = 2_000_000;
        let n: usize = 5;
        let t: usize = 1;
        let sharings_per_round = n - 2 * t;
        let num_rounds = count.div_ceil(sharings_per_round);

        eprintln!(
            "=== 2M Beaver triples: {} triples (n={}, t={}, {} HIM rounds) ===",
            count, n, t, num_rounds
        );

        // ── Phase 1: Silent OT setup ──────────────────────────────────
        let ot_start = std::time::Instant::now();
        let ot_params = SilentOtParams::new(n, t, num_rounds).unwrap();
        let ot_protocol = DistributedSilentOt::new(ot_params);
        let mut ot_states: Vec<_> =
            (0..n).map(|i| ot_protocol.init_party(i, &mut rng)).collect();

        let mut r0 = vec![Vec::new(); n];
        for s in ot_states.iter() {
            for (to, c) in DistributedSilentOt::round0_commitments(s) {
                r0[to].push((s.party_id, c));
            }
        }
        for (i, s) in ot_states.iter_mut().enumerate() {
            DistributedSilentOt::process_round0(s, &r0[i]).unwrap();
        }

        let mut r1 = vec![Vec::new(); n];
        for s in ot_states.iter() {
            for (to, idx) in DistributedSilentOt::round1_puncture_choices(s) {
                r1[to].push((s.party_id, idx));
            }
        }
        for (i, s) in ot_states.iter_mut().enumerate() {
            DistributedSilentOt::process_round1(s, &r1[i]).unwrap();
        }

        let mut r2 = vec![Vec::new(); n];
        for s in ot_states.iter() {
            for (to, path) in DistributedSilentOt::round2_sibling_paths(s).unwrap() {
                r2[to].push((s.party_id, path));
            }
        }
        for (i, s) in ot_states.iter_mut().enumerate() {
            DistributedSilentOt::process_round2(s, &r2[i]).unwrap();
        }

        let mut r3 = vec![Vec::new(); n];
        for s in ot_states.iter() {
            for (to, seed) in DistributedSilentOt::round3_seed_reveals(s) {
                r3[to].push((s.party_id, seed));
            }
        }
        for (i, s) in ot_states.iter_mut().enumerate() {
            DistributedSilentOt::process_round3(s, &r3[i]).unwrap();
        }
        let rounds_elapsed = ot_start.elapsed();
        eprintln!("  OT rounds (init+4 rounds): {:.2?}", rounds_elapsed);

        let expand_start = std::time::Instant::now();
        let ot_correlations: Vec<_> = ot_states
            .par_iter()
            .map(|s| DistributedSilentOt::expand(s).unwrap())
            .collect();
        let expand_elapsed = expand_start.elapsed();
        let ot_elapsed = ot_start.elapsed();
        eprintln!("  OT expand (5 parties): {:.2?}", expand_elapsed);
        eprintln!("silent OT total: {:.2?}", ot_elapsed);

        // ── Phase 2: Fused HIM + triple generation (batch API) ─────────
        let fused_start = std::time::Instant::now();
        let batch =
            generate_triples_from_ot_batch(n, t, count, &ot_correlations, &mut rng).unwrap();
        let fused_elapsed = fused_start.elapsed();
        eprintln!(
            "fused HIM+triple gen batch ({} triples): {:.2?}",
            batch.count,
            fused_elapsed
        );

        assert_eq!(batch.count, count);
        assert_eq!(batch.n, n);

        // ── Phase 3: Sample verification ──────────────────────────────
        let verify_start = std::time::Instant::now();
        let shamir_t = Shamir::new(n, t).unwrap();
        let sample_size = 1000;
        for k in 0..sample_size {
            let shares = batch.triple_shares(k);
            let a_shares: Vec<Share> = shares.iter().map(|tr| tr.a).collect();
            let b_shares: Vec<Share> = shares.iter().map(|tr| tr.b).collect();
            let c_shares: Vec<Share> = shares.iter().map(|tr| tr.c).collect();

            let a = shamir_t.reconstruct(&a_shares).unwrap();
            let b = shamir_t.reconstruct(&b_shares).unwrap();
            let c = shamir_t.reconstruct(&c_shares).unwrap();
            assert_eq!(c, a * b, "triple {} failed: c != a*b", k);
        }
        eprintln!(
            "verified {} sample triples in {:.2?}",
            sample_size,
            verify_start.elapsed()
        );

        // ── Phase 4: Functional test with chain multiply ──────────────
        let chain_start = std::time::Instant::now();
        let num_mults = 1000;
        let values: Vec<Fp> = (2..2 + num_mults as u64).map(Fp::new).collect();
        let expected: Fp = values.iter().copied().reduce(|a, b| a * b).unwrap();

        let value_shares: Vec<Vec<Share>> =
            values.iter().map(|&v| shamir_t.share(v, &mut rng)).collect();

        // Extract triples from batch for chain multiply
        let chain_triples: Vec<Vec<BeaverTriple>> = (0..num_mults - 1)
            .map(|k| batch.triple_shares(k))
            .collect();
        let result_shares =
            beaver_multiply_chain(n, t, &value_shares, &chain_triples).unwrap();
        let result = shamir_t.reconstruct(&result_shares).unwrap();
        assert_eq!(result, expected, "chain multiply mismatch");
        eprintln!(
            "chain multiply ({} values): {:.2?}",
            num_mults,
            chain_start.elapsed()
        );

        eprintln!(
            "grand total: {:.2?}",
            ot_elapsed + fused_elapsed
        );
        eprintln!("=== PASSED ===");
    }
}
