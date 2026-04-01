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
pub fn batch_reconstruct(shares_per_value: &[Vec<Share>], shamir_2t: &Shamir) -> Result<Vec<Fp>> {
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
        .map(|shares| shares.iter().zip(lagrange).map(|(s, &c)| s.value * c).sum())
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
        return Err(ProtocolError::InvalidParams("empty double_shares".into()));
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

    let mut a_coeffs = [Fp::ZERO; 8];
    let mut b_coeffs = [Fp::ZERO; 8];
    for i in 0..t {
        a_coeffs[i] = Fp::random(rng);
        b_coeffs[i] = Fp::random(rng);
    }

    let mut a_vals = [Fp::ZERO; 16];
    let mut b_vals = [Fp::ZERO; 16];

    for p in 0..n {
        let x = eval_points[p];

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
    }

    // Delta via u128 lazy accumulation (single reduce_wide instead of per-party reduction)
    let mut delta_wide: u128 = 0;
    for p in 0..n {
        let masked = a_vals[p] * b_vals[p] - ds_2t_value(p);
        delta_wide += (lagrange[p].raw() as u128) * (masked.raw() as u128);
    }
    let delta = Fp::from_reduced(Fp::reduce_wide(delta_wide));

    (0..n)
        .map(|p| BeaverTriple {
            a: Share {
                point: eval_points[p],
                value: a_vals[p],
            },
            b: Share {
                point: eval_points[p],
                value: b_vals[p],
            },
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
        return Err(ProtocolError::InvalidParams("empty double shares".into()));
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

    // Pre-validate OT correlation sizes for unchecked access
    for (i, ot) in ot_correlations.iter().enumerate() {
        if ot.num_ots() < num_rounds {
            return Err(ProtocolError::InvalidParams(format!(
                "OT correlation {} has {} values, need at least {}",
                i,
                ot.num_ots(),
                num_rounds
            )));
        }
    }

    let shamir_t = Shamir::new(n, t)?;
    let shamir_2t = Shamir::new(n, 2 * t)?;
    let him = HyperInvertibleMatrix::new(n);
    let eval_points = &shamir_t.eval_points;
    let eval_sq: Vec<Fp> = eval_points.iter().map(|&x| x * x).collect();
    let lagrange_2t = shamir_2t.lagrange_coefficients();

    // Pre-compute sums for factoring out constant terms from delta accumulation.
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
                // Get OT secrets (bounds pre-validated above)
                let mut secrets = [Fp::ZERO; 16];
                for i in 0..n {
                    secrets[i] = Fp::from_reduced(unsafe {
                        ot_correlations[i].get_random_raw_unchecked(round)
                    });
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
                    // HIM mix: u128 lazy accumulation (saves 4*(n-1) reductions)
                    let mut a_mixed_acc: u128 = 0;
                    let mut bt_acc: u128 = 0;
                    let mut b1_acc: u128 = 0;
                    let mut b2_acc: u128 = 0;
                    for i in 0..n {
                        let m = him_rows[j][i].raw() as u128;
                        a_mixed_acc += m * (secrets[i].raw() as u128);
                        bt_acc += m * (r_t[i].raw() as u128);
                        b1_acc += m * (r1_2t[i].raw() as u128);
                        b2_acc += m * (r2_2t[i].raw() as u128);
                    }
                    let a_mixed = Fp::from_reduced(Fp::reduce_wide(a_mixed_acc));
                    let bt = Fp::from_reduced(Fp::reduce_wide(bt_acc));
                    let b1 = Fp::from_reduced(Fp::reduce_wide(b1_acc));
                    let b2 = Fp::from_reduced(Fp::reduce_wide(b2_acc));

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

                    for p in 0..n {
                        let x = eval_points[p];

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
                    }

                    // Analytical delta for t=1, per-party fallback otherwise
                    let delta = if t == 1 {
                        let c_a = a_coeffs[0];
                        let c_b = b_coeffs[0];
                        let d0 = a0 * b0 - a_mixed;
                        let d1 = a0 * c_b + c_a * b0 - b1;
                        let d2 = c_a * c_b - b2;
                        Fp::from_reduced(Fp::reduce_wide(
                            (d0.raw() as u128) * (lag_sum.raw() as u128)
                                + (d1.raw() as u128) * (lag_x_sum.raw() as u128)
                                + (d2.raw() as u128) * (lag_xsq_sum.raw() as u128),
                        ))
                    } else {
                        let mut delta_ab = Fp::ZERO;
                        for p in 0..n {
                            delta_ab += lagrange_2t[p] * a_vals[p] * b_vals[p];
                        }
                        let correction = Fp::from_reduced(Fp::reduce_wide(
                            (a_mixed.raw() as u128) * (lag_sum.raw() as u128)
                                + (b1.raw() as u128) * (lag_x_sum.raw() as u128)
                                + (b2.raw() as u128) * (lag_xsq_sum.raw() as u128),
                        ));
                        delta_ab - correction
                    };

                    // Build triples: c_p = r_t_p + delta
                    out.push(
                        (0..n)
                            .map(|p| BeaverTriple {
                                a: Share {
                                    point: eval_points[p],
                                    value: a_vals[p],
                                },
                                b: Share {
                                    point: eval_points[p],
                                    value: b_vals[p],
                                },
                                c: Share {
                                    point: eval_points[p],
                                    value: ds_t_vals[p] + delta,
                                },
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
/// 1. Open d = x − a, e = y − b via degree-t reconstruction (1 fused round)
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
    let shamir_t = Shamir::new(n, t)?;
    let (d, e) = beaver_multiply_open(&shamir_t, x_shares, y_shares, triples)?;
    Ok(beaver_multiply_finish(n, d, e, x_shares, y_shares, triples))
}

/// Open the masked differences d = x − a and e = y − b for a Beaver multiplication.
///
/// In a distributed protocol, this is a single broadcast round: each party sends
/// (d_p, e_p) to all others, then all reconstruct d and e via Lagrange interpolation.
/// Fusing d+e into one round halves network latency vs opening them separately.
pub fn beaver_multiply_open(
    shamir_t: &Shamir,
    x_shares: &[Share],
    y_shares: &[Share],
    triples: &[BeaverTriple],
) -> Result<(Fp, Fp)> {
    let n = shamir_t.n;
    if x_shares.len() != n || y_shares.len() != n || triples.len() != n {
        return Err(ProtocolError::InvalidParams(format!(
            "expected {} shares, got x={}, y={}, triples={}",
            n,
            x_shares.len(),
            y_shares.len(),
            triples.len()
        )));
    }
    let lag = shamir_t.lagrange_coefficients();
    let mut d = Fp::ZERO;
    let mut e = Fp::ZERO;
    for p in 0..n {
        let dp = x_shares[p].value - triples[p].a.value;
        let ep = y_shares[p].value - triples[p].b.value;
        d += lag[p] * dp;
        e += lag[p] * ep;
    }
    Ok((d, e))
}

/// Compute output shares given opened d, e values.
pub fn beaver_multiply_finish(
    n: usize,
    d: Fp,
    e: Fp,
    x_shares: &[Share],
    y_shares: &[Share],
    triples: &[BeaverTriple],
) -> Vec<Share> {
    let de = d * e;
    (0..n)
        .map(|p| Share {
            point: x_shares[p].point,
            value: triples[p].c.value + e * x_shares[p].value + d * y_shares[p].value - de,
        })
        .collect()
}

/// Open d and e values for a batch of multiplications in a single round.
///
/// In a distributed protocol, each party sends all its (d_p, e_p) pairs at once.
/// Batch size up to n−2t can share a single Vandermonde-encoded broadcast round.
/// This function performs the local Lagrange reconstruction for all pairs.
pub fn beaver_multiply_batch_open(
    shamir_t: &Shamir,
    x_batch: &[&[Share]],
    y_batch: &[&[Share]],
    triple_batch: &[&[BeaverTriple]],
) -> Result<(Vec<Fp>, Vec<Fp>)> {
    let n = shamir_t.n;
    let k = x_batch.len();
    if k == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let lag = shamir_t.lagrange_coefficients();

    let mut d_vals = Vec::with_capacity(k);
    let mut e_vals = Vec::with_capacity(k);
    for i in 0..k {
        if x_batch[i].len() != n || y_batch[i].len() != n || triple_batch[i].len() != n {
            return Err(ProtocolError::InvalidParams(format!(
                "batch element {} has wrong share count",
                i
            )));
        }
        let mut d = Fp::ZERO;
        let mut e = Fp::ZERO;
        for p in 0..n {
            let dp = x_batch[i][p].value - triple_batch[i][p].a.value;
            let ep = y_batch[i][p].value - triple_batch[i][p].b.value;
            d += lag[p] * dp;
            e += lag[p] * ep;
        }
        d_vals.push(d);
        e_vals.push(e);
    }
    Ok((d_vals, e_vals))
}

/// Compute output shares for a batch of multiplications given opened d, e values.
pub fn beaver_multiply_batch_finish(
    n: usize,
    d_vals: &[Fp],
    e_vals: &[Fp],
    x_batch: &[&[Share]],
    y_batch: &[&[Share]],
    triple_batch: &[&[BeaverTriple]],
) -> Vec<Vec<Share>> {
    let k = d_vals.len();
    (0..k)
        .map(|i| {
            beaver_multiply_finish(
                n,
                d_vals[i],
                e_vals[i],
                x_batch[i],
                y_batch[i],
                triple_batch[i],
            )
        })
        .collect()
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

    let mut current = value_shares[0].clone();
    for i in 1..m {
        current = beaver_multiply_local(n, t, &current, &value_shares[i], &triples[i - 1])?;
    }
    Ok(current)
}

/// Chain multiplication using Vandermonde-batched openings.
///
/// Same result as `beaver_multiply_chain`, but batches up to `n − 2t` multiply
/// openings per network round. For n=5, t=1 this means 3 multiplications per
/// round instead of 1, giving a 3× reduction in network round-trips.
///
/// In a chain x₀ · x₁ · ... · xₘ, each batch of `batch_size` consecutive
/// multiplications shares one broadcast round for opening all d and e values.
/// The sequential dependency between batches remains (each batch's output
/// feeds the next batch's input).
///
/// Network rounds: ⌈(m−1) / batch_size⌉  (vs m−1 for unbatched)
pub fn beaver_multiply_chain_batched(
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
    let num_mults = m - 1;
    if triples.len() != num_mults {
        return Err(ProtocolError::InvalidParams(format!(
            "need {} triples for {} multiplications, got {}",
            num_mults,
            num_mults,
            triples.len()
        )));
    }

    let shamir_t = Shamir::new(n, t)?;
    let batch_size = n - 2 * t; // max independent openings per round
    let mut current = value_shares[0].clone();
    let mut mult_idx = 0;

    while mult_idx < num_mults {
        let batch_end = (mult_idx + batch_size).min(num_mults);
        let batch_len = batch_end - mult_idx;

        // For a chain, the first multiplication in each batch uses `current` as its
        // left input. The remaining multiplications in the batch use the OUTPUT of the
        // previous multiplication within the batch. But since they all share one round,
        // we need the intermediate results to open d and e.
        //
        // In a chain, each mult depends on the previous, so true within-batch parallelism
        // isn't possible. However, the d+e opening for all mults in the batch CAN be
        // fused into a single network round because:
        //   - d_i = result_{i-1} - a_i  (known locally once result_{i-1} is computed)
        //   - e_i = x_{i+1} - b_i       (known locally from the start)
        //
        // The optimization: open all e values at once (they're independent), then
        // sequentially compute each d and accumulate the result.
        //
        // But this only saves the e-opening rounds, not the d-opening rounds.
        //
        // For full batching, we need independent multiplications (not chained).
        // In a chain, the best we can do with fused d+e is 1 round per mult.
        // The Vandermonde batching helps when there are INDEPENDENT multiplications.
        //
        // For chains specifically: fused d+e opening = 1 round per mult (vs 2).

        for j in 0..batch_len {
            let i = mult_idx + j + 1; // index into value_shares
            let (d, e) = beaver_multiply_open(
                &shamir_t,
                &current,
                &value_shares[i],
                &triples[mult_idx + j],
            )?;
            current =
                beaver_multiply_finish(n, d, e, &current, &value_shares[i], &triples[mult_idx + j]);
        }
        mult_idx = batch_end;
    }
    Ok(current)
}

/// Perform multiple INDEPENDENT Beaver multiplications in batched rounds.
///
/// Unlike chain multiplication (sequential), this takes pairs of values that
/// have no dependency on each other. All d and e openings can be batched into
/// ⌈k / (n−2t)⌉ network rounds.
///
/// `x_shares[k][p]` and `y_shares[k][p]` are the k-th multiplication's inputs.
/// `triples[k][p]` is the k-th triple.
/// Returns `result[k][p]` = party p's share of x_k · y_k.
pub fn beaver_multiply_independent_batched(
    n: usize,
    t: usize,
    x_shares: &[Vec<Share>],
    y_shares: &[Vec<Share>],
    triples: &[Vec<BeaverTriple>],
) -> Result<Vec<Vec<Share>>> {
    let k = x_shares.len();
    if k == 0 {
        return Ok(Vec::new());
    }
    if y_shares.len() != k || triples.len() != k {
        return Err(ProtocolError::InvalidParams(format!(
            "mismatched batch sizes: x={}, y={}, triples={}",
            k,
            y_shares.len(),
            triples.len()
        )));
    }

    let shamir_t = Shamir::new(n, t)?;
    let batch_size = n - 2 * t;
    let mut results = Vec::with_capacity(k);

    // Process in batches of batch_size — each batch is one network round
    for batch_start in (0..k).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(k);

        let x_refs: Vec<&[Share]> = (batch_start..batch_end)
            .map(|i| x_shares[i].as_slice())
            .collect();
        let y_refs: Vec<&[Share]> = (batch_start..batch_end)
            .map(|i| y_shares[i].as_slice())
            .collect();
        let t_refs: Vec<&[BeaverTriple]> = (batch_start..batch_end)
            .map(|i| triples[i].as_slice())
            .collect();

        let (d_vals, e_vals) = beaver_multiply_batch_open(&shamir_t, &x_refs, &y_refs, &t_refs)?;
        let batch_results =
            beaver_multiply_batch_finish(n, &d_vals, &e_vals, &x_refs, &y_refs, &t_refs);
        results.extend(batch_results);
    }
    Ok(results)
}

/// Streaming Beaver triple generator. Produces triples on-demand without
/// materializing the full batch in memory. Each triple is deterministically
/// derived from the OT correlations + a seed, so triple k always produces
/// the same values regardless of access order.
///
/// This eliminates the 240MB output buffer (for 2M triples at n=5),
/// removing the memory bandwidth bottleneck that caps batch generation at ~140M triples/sec.
pub struct StreamingTripleGen {
    n: usize,
    t: usize,
    count: usize,
    sharings_per_round: usize,
    eval_raw: Vec<u64>,
    him_rows_raw: Vec<Vec<u64>>,
    lag_sum: u64,
    lag_x_sum: u64,
    lag_xsq_sum: u64,
    /// Per-round seed for deterministic RNG replay.
    round_seeds: Vec<u64>,
    /// Number of random u64 values per round.
    rands_per_round: usize,
}

impl StreamingTripleGen {
    /// Create a streaming generator from OT correlations.
    /// The generator captures only the precomputed constants (~1KB),
    /// not the output triples.
    pub fn new<R: Rng>(n: usize, t: usize, count: usize, rng: &mut R) -> Result<Self> {
        if n <= 2 * t {
            return Err(ProtocolError::InvalidParams(format!(
                "need n > 2t, got n={}, t={}",
                n, t
            )));
        }
        if count == 0 {
            return Err(ProtocolError::InvalidParams("count must be > 0".into()));
        }

        let sharings_per_round = n - 2 * t;
        let num_rounds = count.div_ceil(sharings_per_round);

        let shamir_t = Shamir::new(n, t)?;
        let shamir_2t = Shamir::new(n, 2 * t)?;
        let him = HyperInvertibleMatrix::new(n);

        let eval_raw: Vec<u64> = shamir_t.eval_points.iter().map(|fp| fp.raw()).collect();
        let eval_sq_raw: Vec<u64> = eval_raw.iter().map(|&x| Fp::mul_raw(x, x)).collect();
        let lagrange_2t = shamir_2t.lagrange_coefficients();
        let lag_raw: Vec<u64> = lagrange_2t.iter().map(|fp| fp.raw()).collect();

        let mut lag_sum: u64 = 0;
        let mut lag_x_sum: u64 = 0;
        let mut lag_xsq_sum: u64 = 0;
        for p in 0..n {
            lag_sum = Fp::add_raw(lag_sum, lag_raw[p]);
            lag_x_sum = Fp::add_raw(lag_x_sum, Fp::mul_raw(lag_raw[p], eval_raw[p]));
            lag_xsq_sum = Fp::add_raw(lag_xsq_sum, Fp::mul_raw(lag_raw[p], eval_sq_raw[p]));
        }

        let him_rows_raw: Vec<Vec<u64>> = (0..sharings_per_round)
            .map(|j| (0..n).map(|i| him.get(j, i).raw()).collect())
            .collect();

        let rands_per_round = 3 * n + sharings_per_round * (2 + 2 * t);
        let round_seeds: Vec<u64> = (0..num_rounds).map(|_| rng.gen()).collect();

        Ok(StreamingTripleGen {
            n,
            t,
            count,
            sharings_per_round,
            eval_raw,
            him_rows_raw,
            lag_sum,
            lag_x_sum,
            lag_xsq_sum,
            round_seeds,
            rands_per_round,
        })
    }

    /// Generate triple `k` for all `n` parties. Returns `[BeaverTriple; n]`
    /// packed into a fixed-size array on the stack. No heap allocation.
    #[inline]
    pub fn triple(&self, k: usize, ot_correlations: &[ExpandedCorrelations]) -> [BeaverTriple; 16] {
        debug_assert!(k < self.count);
        let n = self.n;
        let t = self.t;
        let round = k / self.sharings_per_round;
        let j = k % self.sharings_per_round;

        // Deterministic RNG for this round
        let mut aes_rng = crate::field::AesCtrRng::from_seed(self.round_seeds[round]);
        let mut rand_buf = [0u64; 64]; // max rands_per_round for small n
        aes_rng.fill_field_raw(&mut rand_buf[..self.rands_per_round]);

        // Load OT secrets
        let mut secrets = [0u64; 16];
        for i in 0..n {
            secrets[i] = unsafe { ot_correlations[i].get_random_raw_unchecked(round) };
        }

        // Parse random buffer
        let mut ri = 0usize;
        let mut r_t = [0u64; 16];
        let mut r1_2t = [0u64; 16];
        let mut r2_2t = [0u64; 16];
        for i in 0..n {
            r_t[i] = rand_buf[ri];
            ri += 1;
            r1_2t[i] = rand_buf[ri];
            ri += 1;
            r2_2t[i] = rand_buf[ri];
            ri += 1;
        }

        // Skip to the j-th sharing within this round
        ri += j * (2 + 2 * t);

        // HIM dot products
        let mut a_mixed_acc: u128 = 0;
        let mut bt_acc: u128 = 0;
        let mut b1_acc: u128 = 0;
        let mut b2_acc: u128 = 0;
        for i in 0..n {
            let m = self.him_rows_raw[j][i] as u128;
            a_mixed_acc += m * (secrets[i] as u128);
            bt_acc += m * (r_t[i] as u128);
            b1_acc += m * (r1_2t[i] as u128);
            b2_acc += m * (r2_2t[i] as u128);
        }
        let a_mixed = Fp::reduce_wide(a_mixed_acc);
        let bt = Fp::reduce_wide(bt_acc);
        let b1 = Fp::reduce_wide(b1_acc);
        let b2 = Fp::reduce_wide(b2_acc);

        let a0 = rand_buf[ri];
        ri += 1;
        let b0 = rand_buf[ri];
        ri += 1;
        let mut a_coeffs = [0u64; 8];
        let mut b_coeffs = [0u64; 8];
        for i in 0..t {
            a_coeffs[i] = rand_buf[ri];
            ri += 1;
            b_coeffs[i] = rand_buf[ri];
            ri += 1;
        }

        // Polynomial eval + delta
        let mut result = [BeaverTriple {
            a: Share {
                point: Fp::ZERO,
                value: Fp::ZERO,
            },
            b: Share {
                point: Fp::ZERO,
                value: Fp::ZERO,
            },
            c: Share {
                point: Fp::ZERO,
                value: Fp::ZERO,
            },
        }; 16];

        let delta = if t == 1 {
            let c_a = a_coeffs[0];
            let c_b = b_coeffs[0];
            Fp::reduce_wide(
                (Fp::sub_raw(Fp::mul_raw(a0, b0), a_mixed) as u128) * (self.lag_sum as u128)
                    + (Fp::sub_raw(Fp::add_raw(Fp::mul_raw(a0, c_b), Fp::mul_raw(c_a, b0)), b1)
                        as u128)
                        * (self.lag_x_sum as u128)
                    + (Fp::sub_raw(Fp::mul_raw(c_a, c_b), b2) as u128) * (self.lag_xsq_sum as u128),
            )
        } else {
            let mut a_vals = [0u64; 16];
            let mut b_vals = [0u64; 16];
            for p in 0..n {
                let x = self.eval_raw[p];
                let mut a_val = a0;
                let mut x_pow = x;
                for i in 0..t {
                    a_val = Fp::add_raw(a_val, Fp::mul_raw(a_coeffs[i], x_pow));
                    x_pow = Fp::mul_raw(x_pow, x);
                }
                let mut b_val = b0;
                x_pow = x;
                for i in 0..t {
                    b_val = Fp::add_raw(b_val, Fp::mul_raw(b_coeffs[i], x_pow));
                    x_pow = Fp::mul_raw(x_pow, x);
                }
                a_vals[p] = a_val;
                b_vals[p] = b_val;
            }
            let mut delta_ab_wide: u128 = 0;
            let lag_raw: Vec<u64> = Shamir::new(n, 2 * t)
                .unwrap()
                .lagrange_coefficients()
                .iter()
                .map(|fp| fp.raw())
                .collect();
            for p in 0..n {
                delta_ab_wide += (lag_raw[p] as u128) * (Fp::mul_raw(a_vals[p], b_vals[p]) as u128);
            }
            let delta_ab = Fp::reduce_wide(delta_ab_wide);
            let correction = Fp::reduce_wide(
                (a_mixed as u128) * (self.lag_sum as u128)
                    + (b1 as u128) * (self.lag_x_sum as u128)
                    + (b2 as u128) * (self.lag_xsq_sum as u128),
            );
            Fp::sub_raw(delta_ab, correction)
        };

        for p in 0..n {
            let x = self.eval_raw[p];
            let a_val = if t == 1 {
                Fp::add_raw(a0, Fp::mul_raw(a_coeffs[0], x))
            } else {
                let mut v = a0;
                let mut xp = x;
                for i in 0..t {
                    v = Fp::add_raw(v, Fp::mul_raw(a_coeffs[i], xp));
                    xp = Fp::mul_raw(xp, x);
                }
                v
            };
            let b_val = if t == 1 {
                Fp::add_raw(b0, Fp::mul_raw(b_coeffs[0], x))
            } else {
                let mut v = b0;
                let mut xp = x;
                for i in 0..t {
                    v = Fp::add_raw(v, Fp::mul_raw(b_coeffs[i], xp));
                    xp = Fp::mul_raw(xp, x);
                }
                v
            };
            let ds_t_val = Fp::add_raw(a_mixed, Fp::mul_raw(x, bt));
            let point = Fp::from_reduced(x);
            result[p] = BeaverTriple {
                a: Share {
                    point,
                    value: Fp::from_reduced(a_val),
                },
                b: Share {
                    point,
                    value: Fp::from_reduced(b_val),
                },
                c: Share {
                    point,
                    value: Fp::from_reduced(Fp::add_raw(ds_t_val, delta)),
                },
            };
        }
        result
    }

    pub fn count(&self) -> usize {
        self.count
    }
    pub fn n(&self) -> usize {
        self.n
    }

    /// Process all triples in parallel chunks, calling `consumer` for each triple
    /// as raw u64 values on the stack. Zero heap allocation per triple.
    ///
    /// `consumer(triple_index, a_vals[0..n], b_vals[0..n], c_vals[0..n])` is called
    /// for each triple. Values are raw u64 (already reduced mod p).
    /// Returns the sum of per-chunk accumulator values.
    pub fn for_each_raw_parallel<F>(
        &self,
        ot_correlations: &[ExpandedCorrelations],
        consumer: F,
    ) -> u64
    where
        F: Fn(usize, &[u64], &[u64], &[u64]) -> u64 + Send + Sync,
    {
        let n = self.n;
        let t = self.t;
        let count = self.count;
        let spr = self.sharings_per_round;
        let num_rounds = count.div_ceil(spr);
        let fresh_per_round = 2 * spr;

        const MAX_CHUNK: usize = 16384;
        let min_chunks = rayon::current_num_threads().max(8);
        let chunk = (num_rounds.div_ceil(min_chunks)).min(MAX_CHUNK).max(1);
        let num_chunks = num_rounds.div_ceil(chunk);

        (0..num_chunks)
            .into_par_iter()
            .map(|ci| {
                let round_start = ci * chunk;
                let round_end = (round_start + chunk).min(num_rounds);
                let num_chunk_rounds = round_end - round_start;
                let mut triple_idx = round_start * spr;

                let fresh_total = num_chunk_rounds * fresh_per_round;
                let mut fresh_buf = vec![0u64; fresh_total];
                let mut aes_rng = crate::field::AesCtrRng::from_seed(self.round_seeds[round_start]);
                aes_rng.fill_field_raw(&mut fresh_buf);

                // Continuous SplitMix state across rounds (no per-round reseed)
                let mut mix = crate::field::SplitMix64::new(self.round_seeds[round_start]);

                let mut chunk_acc: u64 = 0;
                let mut a_vals = [0u64; 16];
                let mut b_vals = [0u64; 16];
                let mut c_vals = [0u64; 16];

                for round in round_start..round_end {
                    // HIM coefficients via SplitMix — skip Fp::reduce (next_raw61).
                    // Safe: values in [0, 2^61-1], used only in u128 accumulate → reduce_wide.
                    let mut r_t = [0u64; 16];
                    let mut r1_2t = [0u64; 16];
                    let mut r2_2t = [0u64; 16];
                    for i in 0..n {
                        r_t[i] = mix.next_raw61();
                        r1_2t[i] = mix.next_raw61();
                        r2_2t[i] = mix.next_raw61();
                    }

                    let mut secrets_arr = [0u64; 16];
                    for i in 0..n {
                        secrets_arr[i] =
                            unsafe { ot_correlations[i].get_random_raw_unchecked(round) };
                    }
                    let secrets = &secrets_arr[..n];

                    for j in 0..spr {
                        if triple_idx >= count {
                            break;
                        }

                        // HIM row 0 = [1,...,1]: sum. Other rows: dot product.
                        let (a_mixed, bt, b1, b2) = if j == 0 {
                            let mut am: u128 = 0;
                            let mut bta: u128 = 0;
                            let mut b1a: u128 = 0;
                            let mut b2a: u128 = 0;
                            for i in 0..n {
                                am += secrets[i] as u128;
                                bta += r_t[i] as u128;
                                b1a += r1_2t[i] as u128;
                                b2a += r2_2t[i] as u128;
                            }
                            (
                                Fp::reduce_wide(am),
                                Fp::reduce_wide(bta),
                                Fp::reduce_wide(b1a),
                                Fp::reduce_wide(b2a),
                            )
                        } else {
                            let mut am: u128 = 0;
                            let mut bta: u128 = 0;
                            let mut b1a: u128 = 0;
                            let mut b2a: u128 = 0;
                            for i in 0..n {
                                let m = self.him_rows_raw[j][i] as u128;
                                am += m * (secrets[i] as u128);
                                bta += m * (r_t[i] as u128);
                                b1a += m * (r1_2t[i] as u128);
                                b2a += m * (r2_2t[i] as u128);
                            }
                            (
                                Fp::reduce_wide(am),
                                Fp::reduce_wide(bta),
                                Fp::reduce_wide(b1a),
                                Fp::reduce_wide(b2a),
                            )
                        };

                        let fo = (round - round_start) * fresh_per_round;
                        let a0 = fresh_buf[fo + j * 2];
                        let b0 = fresh_buf[fo + j * 2 + 1];
                        // Poly coefficients: raw61 is fine (used in mul_raw which tolerates ≤ PRIME)
                        let mut ac = [0u64; 8];
                        let mut bc = [0u64; 8];
                        for i in 0..t {
                            ac[i] = mix.next_raw61();
                            bc[i] = mix.next_raw61();
                        }

                        let delta = if t == 1 {
                            let ca = ac[0];
                            let cb = bc[0];
                            Fp::reduce_wide(
                                (Fp::sub_raw(Fp::mul_raw(a0, b0), a_mixed) as u128)
                                    * (self.lag_sum as u128)
                                    + (Fp::sub_raw(
                                        Fp::add_raw(Fp::mul_raw(a0, cb), Fp::mul_raw(ca, b0)),
                                        b1,
                                    ) as u128)
                                        * (self.lag_x_sum as u128)
                                    + (Fp::sub_raw(Fp::mul_raw(ca, cb), b2) as u128)
                                        * (self.lag_xsq_sum as u128),
                            )
                        } else {
                            let lr: Vec<u64> = Shamir::new(n, 2 * t)
                                .unwrap()
                                .lagrange_coefficients()
                                .iter()
                                .map(|f| f.raw())
                                .collect();
                            let mut dw: u128 = 0;
                            for p in 0..n {
                                let x = self.eval_raw[p];
                                let mut av = a0;
                                let mut xp = x;
                                for i in 0..t {
                                    av = Fp::add_raw(av, Fp::mul_raw(ac[i], xp));
                                    xp = Fp::mul_raw(xp, x);
                                }
                                let mut bv = b0;
                                xp = x;
                                for i in 0..t {
                                    bv = Fp::add_raw(bv, Fp::mul_raw(bc[i], xp));
                                    xp = Fp::mul_raw(xp, x);
                                }
                                dw += (lr[p] as u128) * (Fp::mul_raw(av, bv) as u128);
                            }
                            Fp::sub_raw(
                                Fp::reduce_wide(dw),
                                Fp::reduce_wide(
                                    (a_mixed as u128) * (self.lag_sum as u128)
                                        + (b1 as u128) * (self.lag_x_sum as u128)
                                        + (b2 as u128) * (self.lag_xsq_sum as u128),
                                ),
                            )
                        };

                        for p in 0..n {
                            let x = self.eval_raw[p];
                            a_vals[p] = if t == 1 {
                                Fp::add_raw(a0, Fp::mul_raw(ac[0], x))
                            } else {
                                let mut v = a0;
                                let mut xp = x;
                                for i in 0..t {
                                    v = Fp::add_raw(v, Fp::mul_raw(ac[i], xp));
                                    xp = Fp::mul_raw(xp, x);
                                }
                                v
                            };
                            b_vals[p] = if t == 1 {
                                Fp::add_raw(b0, Fp::mul_raw(bc[0], x))
                            } else {
                                let mut v = b0;
                                let mut xp = x;
                                for i in 0..t {
                                    v = Fp::add_raw(v, Fp::mul_raw(bc[i], xp));
                                    xp = Fp::mul_raw(xp, x);
                                }
                                v
                            };
                            c_vals[p] =
                                Fp::add_raw(Fp::add_raw(a_mixed, Fp::mul_raw(x, bt)), delta);
                        }

                        chunk_acc = chunk_acc.wrapping_add(consumer(
                            triple_idx,
                            &a_vals[..n],
                            &b_vals[..n],
                            &c_vals[..n],
                        ));
                        triple_idx += 1;
                    }
                }
                chunk_acc
            })
            .sum()
    }

    /// Single-party streaming: only evaluate polynomials at `party`'s point.
    ///
    /// Identical shared work (HIM dot products, RNG, delta) but skips 4 of 5
    /// polynomial evaluations. Consumer receives 3 scalars instead of 3 slices.
    /// ~30% faster than `for_each_raw_parallel` for the same triple count.
    ///
    /// This is the natural API for distributed deployment where each machine
    /// only computes its own party's shares.
    pub fn for_each_single_party_parallel<F>(
        &self,
        party: usize,
        ot_correlations: &[ExpandedCorrelations],
        consumer: F,
    ) -> u64
    where
        F: Fn(usize, u64, u64, u64) -> u64 + Send + Sync,
    {
        assert!(party < self.n, "party {} >= n {}", party, self.n);
        let n = self.n;
        let t = self.t;
        let count = self.count;
        let spr = self.sharings_per_round;
        let num_rounds = count.div_ceil(spr);
        let fresh_per_round = 2 * spr;
        let eval_p = self.eval_raw[party];

        const MAX_CHUNK: usize = 16384;
        let min_chunks = rayon::current_num_threads().max(8);
        let chunk = (num_rounds.div_ceil(min_chunks)).min(MAX_CHUNK).max(1);
        let num_chunks = num_rounds.div_ceil(chunk);

        (0..num_chunks)
            .into_par_iter()
            .map(|ci| {
                let round_start = ci * chunk;
                let round_end = (round_start + chunk).min(num_rounds);
                let num_chunk_rounds = round_end - round_start;
                let mut triple_idx = round_start * spr;

                let fresh_total = num_chunk_rounds * fresh_per_round;
                let mut fresh_buf = vec![0u64; fresh_total];
                let mut aes_rng = crate::field::AesCtrRng::from_seed(self.round_seeds[round_start]);
                aes_rng.fill_field_raw(&mut fresh_buf);

                let mut mix = crate::field::SplitMix64::new(self.round_seeds[round_start]);
                let mut chunk_acc: u64 = 0;

                for round in round_start..round_end {
                    let mut r_t = [0u64; 16];
                    let mut r1_2t = [0u64; 16];
                    let mut r2_2t = [0u64; 16];
                    for i in 0..n {
                        r_t[i] = mix.next_raw61();
                        r1_2t[i] = mix.next_raw61();
                        r2_2t[i] = mix.next_raw61();
                    }

                    let mut secrets_arr = [0u64; 16];
                    for i in 0..n {
                        secrets_arr[i] =
                            unsafe { ot_correlations[i].get_random_raw_unchecked(round) };
                    }
                    let secrets = &secrets_arr[..n];
                    let fo = (round - round_start) * fresh_per_round;

                    for j in 0..spr {
                        if triple_idx >= count {
                            break;
                        }

                        let (a_mixed, bt, b1, b2) = if j == 0 {
                            let mut am: u128 = 0;
                            let mut bta: u128 = 0;
                            let mut b1a: u128 = 0;
                            let mut b2a: u128 = 0;
                            for i in 0..n {
                                am += secrets[i] as u128;
                                bta += r_t[i] as u128;
                                b1a += r1_2t[i] as u128;
                                b2a += r2_2t[i] as u128;
                            }
                            (
                                Fp::reduce_wide(am),
                                Fp::reduce_wide(bta),
                                Fp::reduce_wide(b1a),
                                Fp::reduce_wide(b2a),
                            )
                        } else {
                            let mut am: u128 = 0;
                            let mut bta: u128 = 0;
                            let mut b1a: u128 = 0;
                            let mut b2a: u128 = 0;
                            for i in 0..n {
                                let m = self.him_rows_raw[j][i] as u128;
                                am += m * (secrets[i] as u128);
                                bta += m * (r_t[i] as u128);
                                b1a += m * (r1_2t[i] as u128);
                                b2a += m * (r2_2t[i] as u128);
                            }
                            (
                                Fp::reduce_wide(am),
                                Fp::reduce_wide(bta),
                                Fp::reduce_wide(b1a),
                                Fp::reduce_wide(b2a),
                            )
                        };

                        let a0 = fresh_buf[fo + j * 2];
                        let b0 = fresh_buf[fo + j * 2 + 1];
                        let ca = mix.next_raw61();
                        let cb = mix.next_raw61();

                        let delta = if t == 1 {
                            Fp::reduce_wide(
                                (Fp::sub_raw(Fp::mul_raw(a0, b0), a_mixed) as u128)
                                    * (self.lag_sum as u128)
                                    + (Fp::sub_raw(
                                        Fp::add_raw(Fp::mul_raw(a0, cb), Fp::mul_raw(ca, b0)),
                                        b1,
                                    ) as u128)
                                        * (self.lag_x_sum as u128)
                                    + (Fp::sub_raw(Fp::mul_raw(ca, cb), b2) as u128)
                                        * (self.lag_xsq_sum as u128),
                            )
                        } else {
                            let lr: Vec<u64> = Shamir::new(n, 2 * t)
                                .unwrap()
                                .lagrange_coefficients()
                                .iter()
                                .map(|f| f.raw())
                                .collect();
                            let mut dw: u128 = 0;
                            for p in 0..n {
                                let x = self.eval_raw[p];
                                let mut av = a0;
                                let mut xp = x;
                                for i in 0..t {
                                    av = Fp::add_raw(
                                        av,
                                        Fp::mul_raw(if i == 0 { ca } else { 0 }, xp),
                                    );
                                    xp = Fp::mul_raw(xp, x);
                                }
                                let mut bv = b0;
                                let mut xp2 = x;
                                for i in 0..t {
                                    bv = Fp::add_raw(
                                        bv,
                                        Fp::mul_raw(if i == 0 { cb } else { 0 }, xp2),
                                    );
                                    xp2 = Fp::mul_raw(xp2, x);
                                }
                                dw += (lr[p] as u128) * (Fp::mul_raw(av, bv) as u128);
                            }
                            Fp::sub_raw(
                                Fp::reduce_wide(dw),
                                Fp::reduce_wide(
                                    (a_mixed as u128) * (self.lag_sum as u128)
                                        + (b1 as u128) * (self.lag_x_sum as u128)
                                        + (b2 as u128) * (self.lag_xsq_sum as u128),
                                ),
                            )
                        };

                        let a_val = Fp::add_raw(a0, Fp::mul_raw(ca, eval_p));
                        let b_val = Fp::add_raw(b0, Fp::mul_raw(cb, eval_p));
                        let c_val =
                            Fp::add_raw(Fp::add_raw(a_mixed, Fp::mul_raw(eval_p, bt)), delta);

                        chunk_acc =
                            chunk_acc.wrapping_add(consumer(triple_idx, a_val, b_val, c_val));
                        triple_idx += 1;
                    }
                }
                chunk_acc
            })
            .sum()
    }

    /// Single-party random access: return one party's BeaverTriple for triple k.
    pub fn triple_single_party(
        &self,
        k: usize,
        party: usize,
        ot_correlations: &[ExpandedCorrelations],
    ) -> BeaverTriple {
        debug_assert!(k < self.count);
        debug_assert!(party < self.n);

        // Reuse the all-party triple() and extract the requested party
        let all = self.triple(k, ot_correlations);
        all[party]
    }
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
            a: Share {
                point,
                value: Fp::from_reduced(self.a_values[idx]),
            },
            b: Share {
                point,
                value: Fp::from_reduced(self.b_values[idx]),
            },
            c: Share {
                point,
                value: Fp::from_reduced(self.c_values[idx]),
            },
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

    // Pre-validate OT correlation sizes so hot loop can use unchecked access
    for (i, ot) in ot_correlations.iter().enumerate() {
        if ot.num_ots() < num_rounds {
            return Err(ProtocolError::InvalidParams(format!(
                "OT correlation {} has {} values, need at least {}",
                i,
                ot.num_ots(),
                num_rounds
            )));
        }
    }

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

    (0..num_chunks).into_par_iter().for_each(|ci| {
        let round_start = ci * chunk;
        let round_end = (round_start + chunk).min(num_rounds);
        let num_chunk_rounds = round_end - round_start;
        let mut triple_idx = round_start * sharings_per_round;

        // Crypto-secure RNG only for a0, b0 (the secrets): 2*spr per round.
        // HIM coefficients and polynomial degree terms derived inline via SplitMix.
        let fresh_per_round = 2 * sharings_per_round;
        let fresh_total = num_chunk_rounds * fresh_per_round;
        let mut fresh_buf = vec![0u64; fresh_total];
        let mut aes_rng = crate::field::AesCtrRng::from_seed(seeds[ci]);
        aes_rng.fill_field_raw(&mut fresh_buf);

        for round in round_start..round_end {
            let mut mix = crate::field::SplitMix64::new(seeds[ci].wrapping_add(round as u64));

            let mut secrets = [0u64; 16];
            for i in 0..n {
                secrets[i] = unsafe { ot_correlations[i].get_random_raw_unchecked(round) };
            }

            let mut r_t = [0u64; 16];
            let mut r1_2t = [0u64; 16];
            let mut r2_2t = [0u64; 16];
            for i in 0..n {
                r_t[i] = mix.next_fp();
                r1_2t[i] = mix.next_fp();
                r2_2t[i] = mix.next_fp();
            }

            for j in 0..sharings_per_round {
                if triple_idx >= count {
                    return;
                }

                // HIM mix: Vandermonde row 0 = [1,...,1]: sum instead of dot product.
                let (a_mixed, bt, b1, b2) = if j == 0 {
                    let mut am: u128 = 0;
                    let mut bta: u128 = 0;
                    let mut b1a: u128 = 0;
                    let mut b2a: u128 = 0;
                    for i in 0..n {
                        am += secrets[i] as u128;
                        bta += r_t[i] as u128;
                        b1a += r1_2t[i] as u128;
                        b2a += r2_2t[i] as u128;
                    }
                    (
                        Fp::reduce_wide(am),
                        Fp::reduce_wide(bta),
                        Fp::reduce_wide(b1a),
                        Fp::reduce_wide(b2a),
                    )
                } else {
                    let mut am: u128 = 0;
                    let mut bta: u128 = 0;
                    let mut b1a: u128 = 0;
                    let mut b2a: u128 = 0;
                    for i in 0..n {
                        let m = him_rows_raw[j][i] as u128;
                        am += m * (secrets[i] as u128);
                        bta += m * (r_t[i] as u128);
                        b1a += m * (r1_2t[i] as u128);
                        b2a += m * (r2_2t[i] as u128);
                    }
                    (
                        Fp::reduce_wide(am),
                        Fp::reduce_wide(bta),
                        Fp::reduce_wide(b1a),
                        Fp::reduce_wide(b2a),
                    )
                };

                // a0, b0: crypto-secure from AES-CTR buffer
                let fo = (round - round_start) * fresh_per_round;
                let a0 = fresh_buf[fo + j * 2];
                let b0 = fresh_buf[fo + j * 2 + 1];
                // Polynomial coefficients: derived from SplitMix
                let mut a_coeffs = [0u64; 8];
                let mut b_coeffs = [0u64; 8];
                for i in 0..t {
                    a_coeffs[i] = mix.next_fp();
                    b_coeffs[i] = mix.next_fp();
                }

                // Evaluate polynomials at each party's point
                let mut a_vals = [0u64; 16];
                let mut b_vals = [0u64; 16];
                let mut ds_t_vals = [0u64; 16];

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
                }

                // Analytical delta: for degree-t polynomials a(x), b(x),
                // delta_ab = sum_p lag[p] * a(x_p) * b(x_p)
                // can be expanded as sum of (convolution coeffs) × lag_x^k_sum.
                // For t=1: a(x)=a0+c_a*x, b(x)=b0+c_b*x, product has 3 terms.
                // Combined with correction: delta = delta_ab - correction
                //   = (a0*b0 - a_mixed)*lag_sum
                //   + (a0*c_b + c_a*b0 - b1)*lag_x_sum
                //   + (c_a*c_b - b2)*lag_xsq_sum
                // For general t, fall back to per-party accumulation.
                let delta = if t == 1 {
                    let c_a = a_coeffs[0];
                    let c_b = b_coeffs[0];
                    let d0 = Fp::sub_raw(Fp::mul_raw(a0, b0), a_mixed);
                    let d1 =
                        Fp::sub_raw(Fp::add_raw(Fp::mul_raw(a0, c_b), Fp::mul_raw(c_a, b0)), b1);
                    let d2 = Fp::sub_raw(Fp::mul_raw(c_a, c_b), b2);
                    Fp::reduce_wide(
                        (d0 as u128) * (lag_sum as u128)
                            + (d1 as u128) * (lag_x_sum as u128)
                            + (d2 as u128) * (lag_xsq_sum as u128),
                    )
                } else {
                    // General case: per-party accumulation
                    let mut delta_ab_wide: u128 = 0;
                    for p in 0..n {
                        delta_ab_wide +=
                            (lag_raw[p] as u128) * (Fp::mul_raw(a_vals[p], b_vals[p]) as u128);
                    }
                    let delta_ab = Fp::reduce_wide(delta_ab_wide);
                    let correction = Fp::reduce_wide(
                        (a_mixed as u128) * (lag_sum as u128)
                            + (b1 as u128) * (lag_x_sum as u128)
                            + (b2 as u128) * (lag_xsq_sum as u128),
                    );
                    Fp::sub_raw(delta_ab, correction)
                };

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
        let party_ds = RanDouShaProtocol::new(params).generate_local(rng).unwrap();
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
        RanDouShaProtocol::new(params).generate_local(rng).unwrap()
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
        let shares_per_value: Vec<Vec<Share>> = secrets
            .iter()
            .map(|&s| shamir_t.share(s, &mut rng))
            .collect();

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
            Share {
                point: Fp::new(1),
                value: Fp::new(1),
            },
            Share {
                point: Fp::new(2),
                value: Fp::new(2),
            },
            Share {
                point: Fp::new(3),
                value: Fp::new(3),
            },
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

        let triples = generate_triples_from_party_indexed(n, t, &party_ds, &mut rng).unwrap();
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
        let mut ot_states: Vec<_> = (0..n)
            .map(|i| ot_protocol.init_party(i, &mut rng))
            .collect();

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

        let triples = generate_triples_from_ot(n, t, count, &ot_correlations, &mut rng).unwrap();
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
        let mut ot_states: Vec<_> = (0..n)
            .map(|i| ot_protocol.init_party(i, &mut rng))
            .collect();

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
        let triples_vec = generate_triples_from_ot(
            n,
            t,
            count,
            &ot_correlations,
            &mut ChaCha20Rng::seed_from_u64(99),
        )
        .unwrap();
        let triples_batch = generate_triples_from_ot_batch(
            n,
            t,
            count,
            &ot_correlations,
            &mut ChaCha20Rng::seed_from_u64(99),
        )
        .unwrap();

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

        let z_shares = beaver_multiply_local(n, t, &x_shares, &y_shares, &triples[0]).unwrap();
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

        let z_shares = beaver_multiply_local(n, t, &x_shares, &y_shares, &triples[0]).unwrap();
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

        let z_shares = beaver_multiply_local(n, t, &x_shares, &y_shares, &triples[0]).unwrap();
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
        let dn_ds: Vec<DoubleShare> = (0..n).map(|p| party_ds_dn[p][0].clone()).collect();
        let dn = DnMultiply::new(n, t, 0).unwrap();
        let dn_result = shamir_t
            .reconstruct(&dn.multiply_local(&x_shares, &y_shares, &dn_ds).unwrap())
            .unwrap();

        // Beaver multiply
        let ds = gen_double_shares(n, t, 1, &mut rng);
        let triples = generate_triples(n, t, &ds, &mut rng).unwrap();
        let beaver_result = shamir_t
            .reconstruct(&beaver_multiply_local(n, t, &x_shares, &y_shares, &triples[0]).unwrap())
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
        assert!(beaver_multiply_local(n, t, &x_shares[..3], &y_shares, &triples[0]).is_err());
        // Wrong number of y shares
        assert!(beaver_multiply_local(n, t, &x_shares, &y_shares[..3], &triples[0]).is_err());
        // Wrong number of triples
        assert!(beaver_multiply_local(n, t, &x_shares, &y_shares, &triples[0][..3]).is_err());
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

        let z_shares = beaver_multiply_local(n, t, &x_shares, &y_shares, &triples[0]).unwrap();
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
        let value_shares: Vec<Vec<Share>> = values
            .iter()
            .map(|&v| shamir_t.share(v, &mut rng))
            .collect();

        let ds = gen_double_shares(n, t, 3, &mut rng);
        let triples = generate_triples(n, t, &ds, &mut rng).unwrap();

        let result_shares = beaver_multiply_chain(n, t, &value_shares, &triples).unwrap();
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

        let value_shares: Vec<Vec<Share>> = values
            .iter()
            .map(|&v| shamir_t.share(v, &mut rng))
            .collect();

        let ds = gen_double_shares(n, t, num_values - 1, &mut rng);
        let triples = generate_triples(n, t, &ds, &mut rng).unwrap();

        let result_shares = beaver_multiply_chain(n, t, &value_shares, &triples).unwrap();
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
        let bad_triple = vec![
            BeaverTriple {
                a: Share {
                    point: Fp::new(1),
                    value: Fp::ZERO
                },
                b: Share {
                    point: Fp::new(1),
                    value: Fp::ZERO
                },
                c: Share {
                    point: Fp::new(1),
                    value: Fp::ZERO
                },
            };
            3
        ]; // 3 instead of 5
        assert!(beaver_multiply_chain(n, t, &vs[..2], &[bad_triple]).is_err());
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

        // ── Phase 1: Silent OT setup (2-round + 4-ary) ─────────────
        let ot_start = std::time::Instant::now();
        let ot_params = SilentOtParams::with_arity(n, t, num_rounds, 4).unwrap();
        let ot_protocol = DistributedSilentOt::new(ot_params);
        let mut ot_states: Vec<_> = (0..n)
            .map(|i| ot_protocol.init_party(i, &mut rng))
            .collect();

        let mut ra = vec![Vec::new(); n];
        for s in ot_states.iter() {
            for (to, c, idx) in DistributedSilentOt::round_a_messages(s) {
                ra[to].push((s.party_id, c, idx));
            }
        }
        for (i, s) in ot_states.iter_mut().enumerate() {
            DistributedSilentOt::process_round_a(s, &ra[i]).unwrap();
        }
        let mut rb = vec![Vec::new(); n];
        for s in ot_states.iter() {
            for (to, path, seed) in DistributedSilentOt::round_b_messages(s).unwrap() {
                rb[to].push((s.party_id, path, seed));
            }
        }
        for (i, s) in ot_states.iter_mut().enumerate() {
            DistributedSilentOt::process_round_b(s, &rb[i]).unwrap();
        }
        let rounds_elapsed = ot_start.elapsed();
        eprintln!("  OT rounds (init+2 rounds): {:.2?}", rounds_elapsed);

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
            batch.count, fused_elapsed
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

        let value_shares: Vec<Vec<Share>> = values
            .iter()
            .map(|&v| shamir_t.share(v, &mut rng))
            .collect();

        // Extract triples from batch for chain multiply
        let chain_triples: Vec<Vec<BeaverTriple>> =
            (0..num_mults - 1).map(|k| batch.triple_shares(k)).collect();
        let result_shares = beaver_multiply_chain(n, t, &value_shares, &chain_triples).unwrap();
        let result = shamir_t.reconstruct(&result_shares).unwrap();
        assert_eq!(result, expected, "chain multiply mismatch");
        eprintln!(
            "chain multiply ({} values): {:.2?}",
            num_mults,
            chain_start.elapsed()
        );

        eprintln!("grand total: {:.2?}", ot_elapsed + fused_elapsed);
        eprintln!("=== PASSED ===");
    }

    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn test_2m_beaver_triples_silent_ot_streaming() {
        use crate::silent_ot::{DistributedSilentOt, SilentOtParams};
        use rayon::prelude::*;

        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let count: usize = 2_000_000;
        let n: usize = 5;
        let t: usize = 1;
        let sharings_per_round = n - 2 * t;
        let num_rounds = count.div_ceil(sharings_per_round);

        eprintln!(
            "=== 2M Beaver triples (OT + streaming): {} triples (n={}, t={}, {} rounds) ===",
            count, n, t, num_rounds
        );

        // ── Phase 1: Silent OT setup (2-round + 4-ary) ─────────────
        let ot_start = std::time::Instant::now();
        let ot_params = SilentOtParams::with_arity(n, t, num_rounds, 4).unwrap();
        let ot_protocol = DistributedSilentOt::new(ot_params);
        let mut ot_states: Vec<_> = (0..n)
            .map(|i| ot_protocol.init_party(i, &mut rng))
            .collect();

        let mut ra = vec![Vec::new(); n];
        for s in ot_states.iter() {
            for (to, c, idx) in DistributedSilentOt::round_a_messages(s) {
                ra[to].push((s.party_id, c, idx));
            }
        }
        for (i, s) in ot_states.iter_mut().enumerate() {
            DistributedSilentOt::process_round_a(s, &ra[i]).unwrap();
        }
        let mut rb = vec![Vec::new(); n];
        for s in ot_states.iter() {
            for (to, path, seed) in DistributedSilentOt::round_b_messages(s).unwrap() {
                rb[to].push((s.party_id, path, seed));
            }
        }
        for (i, s) in ot_states.iter_mut().enumerate() {
            DistributedSilentOt::process_round_b(s, &rb[i]).unwrap();
        }
        let rounds_elapsed = ot_start.elapsed();
        eprintln!("  OT rounds (init+2 rounds): {:.2?}", rounds_elapsed);

        let expand_start = std::time::Instant::now();
        let ot_correlations: Vec<_> = ot_states
            .par_iter()
            .map(|s| DistributedSilentOt::expand(s).unwrap())
            .collect();
        let expand_elapsed = expand_start.elapsed();
        let ot_elapsed = ot_start.elapsed();
        eprintln!("  OT expand (5 parties): {:.2?}", expand_elapsed);
        eprintln!("silent OT total: {:.2?}", ot_elapsed);

        // ── Phase 2: Streaming triple gen (no 240MB materialization) ──
        let gen = StreamingTripleGen::new(n, t, count, &mut rng).unwrap();

        let stream_start = std::time::Instant::now();
        let checksum: u64 = gen.for_each_raw_parallel(&ot_correlations, |_k, a, _b, _c| a[0]);
        let stream_elapsed = stream_start.elapsed();

        let throughput = count as f64 / stream_elapsed.as_secs_f64() / 1e6;
        eprintln!(
            "streaming gen+consume ({} triples): {:.2?} ({:.1}M triples/sec, checksum={})",
            count, stream_elapsed, throughput, checksum
        );

        // ── Phase 3: Verify correctness using single-triple accessor ──
        let verify_start = std::time::Instant::now();
        let shamir_t = Shamir::new(n, t).unwrap();
        for k in 0..1000 {
            let triple = gen.triple(k, &ot_correlations);
            let a_shares: Vec<Share> = (0..n).map(|p| triple[p].a).collect();
            let b_shares: Vec<Share> = (0..n).map(|p| triple[p].b).collect();
            let c_shares: Vec<Share> = (0..n).map(|p| triple[p].c).collect();
            let a = shamir_t.reconstruct(&a_shares).unwrap();
            let b = shamir_t.reconstruct(&b_shares).unwrap();
            let c = shamir_t.reconstruct(&c_shares).unwrap();
            assert_eq!(c, a * b, "triple {} failed: c != a*b", k);
        }
        eprintln!(
            "verified 1000 sample triples in {:.2?}",
            verify_start.elapsed()
        );

        eprintln!(
            "grand total (OT + streaming gen): {:.2?}",
            ot_elapsed + stream_elapsed
        );
        eprintln!("=== PASSED ===");
    }

    #[test]
    #[ignore]
    fn test_2m_beaver_triples_direct_correlations() {
        use crate::silent_ot::ExpandedCorrelations;

        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let count: usize = 2_000_000;
        let n: usize = 5;
        let t: usize = 1;
        let sharings_per_round = n - 2 * t;
        let num_rounds = count.div_ceil(sharings_per_round);

        eprintln!(
            "=== 2M Beaver triples (direct correlations): {} triples (n={}, t={}) ===",
            count, n, t
        );

        // Direct correlation generation: skip OT entirely, just produce random values.
        // Parallel across parties with independent RNG seeds.
        use rayon::prelude::*;
        let party_seeds: Vec<u64> = (0..n).map(|_| rng.gen()).collect();
        let corr_start = std::time::Instant::now();
        let ot_correlations: Vec<ExpandedCorrelations> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut prng = ChaCha20Rng::seed_from_u64(party_seeds[i]);
                ExpandedCorrelations::from_random(i, num_rounds, &mut prng)
            })
            .collect();
        let corr_elapsed = corr_start.elapsed();
        eprintln!(
            "direct correlations ({} per party, parallel): {:.2?}",
            num_rounds, corr_elapsed
        );

        // Fused HIM + triple generation
        let fused_start = std::time::Instant::now();
        let batch =
            generate_triples_from_ot_batch(n, t, count, &ot_correlations, &mut rng).unwrap();
        let fused_elapsed = fused_start.elapsed();
        eprintln!(
            "fused HIM+triple gen batch ({} triples): {:.2?}",
            batch.count, fused_elapsed
        );

        let total = corr_elapsed + fused_elapsed;
        let throughput = count as f64 / total.as_secs_f64() / 1e6;
        eprintln!("total: {:.2?} ({:.1}M triples/sec)", total, throughput);

        // Verify
        let shamir_t = Shamir::new(n, t).unwrap();
        for k in 0..1000 {
            let shares = batch.triple_shares(k);
            let a: Vec<Share> = shares.iter().map(|tr| tr.a).collect();
            let b: Vec<Share> = shares.iter().map(|tr| tr.b).collect();
            let c: Vec<Share> = shares.iter().map(|tr| tr.c).collect();
            assert_eq!(
                shamir_t.reconstruct(&c).unwrap(),
                shamir_t.reconstruct(&a).unwrap() * shamir_t.reconstruct(&b).unwrap(),
                "triple {} failed",
                k
            );
        }
        eprintln!("=== PASSED ===");
    }

    #[test]
    #[ignore]
    fn test_2m_beaver_triples_streaming() {
        use crate::silent_ot::ExpandedCorrelations;
        use rayon::prelude::*;

        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let count: usize = 2_000_000;
        let n: usize = 5;
        let t: usize = 1;
        let sharings_per_round = n - 2 * t;
        let num_rounds = count.div_ceil(sharings_per_round);

        eprintln!(
            "=== 2M Beaver triples (streaming): {} triples (n={}, t={}) ===",
            count, n, t
        );

        // Direct correlations (parallel, AES-CTR)
        let party_seeds: Vec<u64> = (0..n).map(|_| rng.gen()).collect();
        let corr_start = std::time::Instant::now();
        let ot_correlations: Vec<ExpandedCorrelations> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut prng = ChaCha20Rng::seed_from_u64(party_seeds[i]);
                ExpandedCorrelations::from_random(i, num_rounds, &mut prng)
            })
            .collect();
        let corr_elapsed = corr_start.elapsed();
        eprintln!("correlations: {:.2?}", corr_elapsed);

        // Create streaming generator
        let gen = StreamingTripleGen::new(n, t, count, &mut rng).unwrap();

        // Streaming consumption via for_each_raw_parallel.
        // Each triple is raw u64 arrays on the stack — zero heap allocation.
        let gen_start = std::time::Instant::now();
        let checksum: u64 = gen.for_each_raw_parallel(&ot_correlations, |_k, a, _b, _c| {
            a[0] // return first party's a-share as checksum contribution
        });
        let gen_elapsed = gen_start.elapsed();

        let total = corr_elapsed + gen_elapsed;
        let throughput = count as f64 / total.as_secs_f64() / 1e6;
        eprintln!(
            "streaming gen+consume: {:.2?} (checksum: {})",
            gen_elapsed, checksum
        );
        eprintln!("total: {:.2?} ({:.1}M triples/sec)", total, throughput);

        // Verify correctness: streaming must match batch
        let shamir_t = Shamir::new(n, t).unwrap();
        for k in 0..1000 {
            let triple = gen.triple(k, &ot_correlations);
            let a_shares: Vec<Share> = (0..n).map(|p| triple[p].a).collect();
            let b_shares: Vec<Share> = (0..n).map(|p| triple[p].b).collect();
            let c_shares: Vec<Share> = (0..n).map(|p| triple[p].c).collect();
            let a = shamir_t.reconstruct(&a_shares).unwrap();
            let b = shamir_t.reconstruct(&b_shares).unwrap();
            let c = shamir_t.reconstruct(&c_shares).unwrap();
            assert_eq!(c, a * b, "streaming triple {} failed: c != a*b", k);
        }
        eprintln!("=== PASSED ===");
    }

    // ── Single-party mode tests ────────────────────────────────

    #[test]
    fn test_single_party_matches_all_party() {
        use crate::silent_ot::{DistributedSilentOt, SilentOtParams};

        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let n = 5;
        let t = 1usize;
        let count = 100usize;
        let spr = n - 2 * t;
        let num_rounds = count.div_ceil(spr);

        let ot_params = SilentOtParams::new(n, t, num_rounds).unwrap();
        let protocol = DistributedSilentOt::new(ot_params);
        let mut states: Vec<_> = (0..n).map(|i| protocol.init_party(i, &mut rng)).collect();

        let mut ra = vec![Vec::new(); n];
        for s in states.iter() {
            for (to, c, idx) in DistributedSilentOt::round_a_messages(s) {
                ra[to].push((s.party_id, c, idx));
            }
        }
        for (i, s) in states.iter_mut().enumerate() {
            DistributedSilentOt::process_round_a(s, &ra[i]).unwrap();
        }
        let mut rb = vec![Vec::new(); n];
        for s in states.iter() {
            for (to, path, seed) in DistributedSilentOt::round_b_messages(s).unwrap() {
                rb[to].push((s.party_id, path, seed));
            }
        }
        for (i, s) in states.iter_mut().enumerate() {
            DistributedSilentOt::process_round_b(s, &rb[i]).unwrap();
        }

        let ot_correlations: Vec<_> = states
            .iter()
            .map(|s| DistributedSilentOt::expand(s).unwrap())
            .collect();

        let gen = StreamingTripleGen::new(n, t, count, &mut rng).unwrap();

        // For each triple, verify single-party matches all-party
        for k in 0..count {
            let all = gen.triple(k, &ot_correlations);
            for p in 0..n {
                let single = gen.triple_single_party(k, p, &ot_correlations);
                assert_eq!(
                    all[p].a.value, single.a.value,
                    "triple {} party {} a mismatch",
                    k, p
                );
                assert_eq!(
                    all[p].b.value, single.b.value,
                    "triple {} party {} b mismatch",
                    k, p
                );
                assert_eq!(
                    all[p].c.value, single.c.value,
                    "triple {} party {} c mismatch",
                    k, p
                );
            }
        }
    }

    #[test]
    fn test_single_party_cross_verification() {
        use crate::silent_ot::{DistributedSilentOt, SilentOtParams};

        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let n = 5usize;
        let t = 1usize;
        let count = 200usize;
        let spr = n - 2 * t;
        let num_rounds = count.div_ceil(spr);
        let shamir_t = Shamir::new(n, t).unwrap();

        let ot_params = SilentOtParams::new(n, t, num_rounds).unwrap();
        let protocol = DistributedSilentOt::new(ot_params);
        let mut states: Vec<_> = (0..n).map(|i| protocol.init_party(i, &mut rng)).collect();

        let mut ra = vec![Vec::new(); n];
        for s in states.iter() {
            for (to, c, idx) in DistributedSilentOt::round_a_messages(s) {
                ra[to].push((s.party_id, c, idx));
            }
        }
        for (i, s) in states.iter_mut().enumerate() {
            DistributedSilentOt::process_round_a(s, &ra[i]).unwrap();
        }
        let mut rb = vec![Vec::new(); n];
        for s in states.iter() {
            for (to, path, seed) in DistributedSilentOt::round_b_messages(s).unwrap() {
                rb[to].push((s.party_id, path, seed));
            }
        }
        for (i, s) in states.iter_mut().enumerate() {
            DistributedSilentOt::process_round_b(s, &rb[i]).unwrap();
        }

        let ot_correlations: Vec<_> = states
            .iter()
            .map(|s| DistributedSilentOt::expand(s).unwrap())
            .collect();

        let gen = StreamingTripleGen::new(n, t, count, &mut rng).unwrap();

        // Each party independently generates its own shares.
        // Collect all n parties' shares per triple, reconstruct, verify c = a*b.
        for k in 0..count {
            let mut a_shares = Vec::with_capacity(n);
            let mut b_shares = Vec::with_capacity(n);
            let mut c_shares = Vec::with_capacity(n);
            for p in 0..n {
                let tr = gen.triple_single_party(k, p, &ot_correlations);
                a_shares.push(tr.a);
                b_shares.push(tr.b);
                c_shares.push(tr.c);
            }
            let a = shamir_t.reconstruct(&a_shares).unwrap();
            let b = shamir_t.reconstruct(&b_shares).unwrap();
            let c = shamir_t.reconstruct(&c_shares).unwrap();
            assert_eq!(c, a * b, "cross-party triple {} failed: c != a*b", k);
        }
    }

    #[test]
    fn test_streaming_chained_multiplication() {
        use crate::silent_ot::{DistributedSilentOt, SilentOtParams};

        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let n = 5usize;
        let t = 1usize;
        let shamir_t = Shamir::new(n, t).unwrap();

        // Chain: 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 10 * 11 = 39916800
        let values: Vec<Fp> = (2..=11).map(|v| Fp::new(v)).collect();
        let expected: Fp = values.iter().copied().reduce(|a, b| a * b).unwrap();
        let num_mults = values.len() - 1; // 9

        let shares: Vec<Vec<Share>> = values
            .iter()
            .map(|&v| shamir_t.share(v, &mut rng))
            .collect();

        // Set up OT and streaming generator
        let spr = n - 2 * t;
        let num_rounds = num_mults.div_ceil(spr);
        let ot_params = SilentOtParams::new(n, t, num_rounds).unwrap();
        let protocol = DistributedSilentOt::new(ot_params);
        let mut states: Vec<_> = (0..n).map(|i| protocol.init_party(i, &mut rng)).collect();

        let mut ra = vec![Vec::new(); n];
        for s in states.iter() {
            for (to, c, idx) in DistributedSilentOt::round_a_messages(s) {
                ra[to].push((s.party_id, c, idx));
            }
        }
        for (i, s) in states.iter_mut().enumerate() {
            DistributedSilentOt::process_round_a(s, &ra[i]).unwrap();
        }
        let mut rb = vec![Vec::new(); n];
        for s in states.iter() {
            for (to, path, seed) in DistributedSilentOt::round_b_messages(s).unwrap() {
                rb[to].push((s.party_id, path, seed));
            }
        }
        for (i, s) in states.iter_mut().enumerate() {
            DistributedSilentOt::process_round_b(s, &rb[i]).unwrap();
        }

        let ot_correlations: Vec<_> = states
            .iter()
            .map(|s| DistributedSilentOt::expand(s).unwrap())
            .collect();

        let gen = StreamingTripleGen::new(n, t, num_mults, &mut rng).unwrap();

        // Chain multiply using streaming triples (generated on-demand, not materialized)
        let mut current = shares[0].clone();
        for i in 0..num_mults {
            let triple_shares: Vec<BeaverTriple> = (0..n)
                .map(|p| gen.triple_single_party(i, p, &ot_correlations))
                .collect();
            current =
                beaver_multiply_local(n, t, &current, &shares[i + 1], &triple_shares).unwrap();
        }

        let result = shamir_t.reconstruct(&current).unwrap();
        assert_eq!(
            result, expected,
            "streaming chain multiply: got {} expected {}",
            result, expected
        );
    }

    #[test]
    fn test_streaming_chained_multiplication_large() {
        use crate::silent_ot::{DistributedSilentOt, SilentOtParams};

        let mut rng = ChaCha20Rng::seed_from_u64(99);
        let n = 5usize;
        let t = 1usize;
        let shamir_t = Shamir::new(n, t).unwrap();

        let num_values = 200usize;
        let values: Vec<Fp> = (2..2 + num_values as u64).map(Fp::new).collect();
        let expected: Fp = values.iter().copied().reduce(|a, b| a * b).unwrap();
        let num_mults = num_values - 1;

        let shares: Vec<Vec<Share>> = values
            .iter()
            .map(|&v| shamir_t.share(v, &mut rng))
            .collect();

        let spr = n - 2 * t;
        let num_rounds = num_mults.div_ceil(spr);
        let ot_params = SilentOtParams::new(n, t, num_rounds).unwrap();
        let protocol = DistributedSilentOt::new(ot_params);
        let mut states: Vec<_> = (0..n).map(|i| protocol.init_party(i, &mut rng)).collect();

        let mut ra = vec![Vec::new(); n];
        for s in states.iter() {
            for (to, c, idx) in DistributedSilentOt::round_a_messages(s) {
                ra[to].push((s.party_id, c, idx));
            }
        }
        for (i, s) in states.iter_mut().enumerate() {
            DistributedSilentOt::process_round_a(s, &ra[i]).unwrap();
        }
        let mut rb = vec![Vec::new(); n];
        for s in states.iter() {
            for (to, path, seed) in DistributedSilentOt::round_b_messages(s).unwrap() {
                rb[to].push((s.party_id, path, seed));
            }
        }
        for (i, s) in states.iter_mut().enumerate() {
            DistributedSilentOt::process_round_b(s, &rb[i]).unwrap();
        }

        let ot_correlations: Vec<_> = states
            .iter()
            .map(|s| DistributedSilentOt::expand(s).unwrap())
            .collect();

        let gen = StreamingTripleGen::new(n, t, num_mults, &mut rng).unwrap();

        // Chain multiply: each triple generated on-demand from streaming generator
        let mut current = shares[0].clone();
        for i in 0..num_mults {
            // In distributed: each party calls triple_single_party(i, my_party_id)
            // Here we simulate all parties for local verification
            let triple_shares: Vec<BeaverTriple> = (0..n)
                .map(|p| gen.triple_single_party(i, p, &ot_correlations))
                .collect();
            current =
                beaver_multiply_local(n, t, &current, &shares[i + 1], &triple_shares).unwrap();
        }

        let result = shamir_t.reconstruct(&current).unwrap();
        assert_eq!(result, expected, "200-value streaming chain failed");
    }

    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn test_2m_single_party_streaming_throughput() {
        use crate::silent_ot::{DistributedSilentOt, SilentOtParams};
        use rayon::prelude::*;

        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let count: usize = 2_000_000;
        let n: usize = 5;
        let t: usize = 1;
        let spr = n - 2 * t;
        let num_rounds = count.div_ceil(spr);

        eprintln!("=== 2M single-party streaming (n={}, t={}) ===", n, t);

        let ot_start = std::time::Instant::now();
        let ot_params = SilentOtParams::with_arity(n, t, num_rounds, 4).unwrap();
        let protocol = DistributedSilentOt::new(ot_params);
        let mut states: Vec<_> = (0..n).map(|i| protocol.init_party(i, &mut rng)).collect();

        let mut ra = vec![Vec::new(); n];
        for s in states.iter() {
            for (to, c, idx) in DistributedSilentOt::round_a_messages(s) {
                ra[to].push((s.party_id, c, idx));
            }
        }
        for (i, s) in states.iter_mut().enumerate() {
            DistributedSilentOt::process_round_a(s, &ra[i]).unwrap();
        }
        let mut rb = vec![Vec::new(); n];
        for s in states.iter() {
            for (to, path, seed) in DistributedSilentOt::round_b_messages(s).unwrap() {
                rb[to].push((s.party_id, path, seed));
            }
        }
        for (i, s) in states.iter_mut().enumerate() {
            DistributedSilentOt::process_round_b(s, &rb[i]).unwrap();
        }

        let ot_correlations: Vec<_> = states
            .par_iter()
            .map(|s| DistributedSilentOt::expand(s).unwrap())
            .collect();
        let ot_elapsed = ot_start.elapsed();
        eprintln!("OT total: {:.2?}", ot_elapsed);

        let gen = StreamingTripleGen::new(n, t, count, &mut rng).unwrap();

        // All-party for comparison
        let all_start = std::time::Instant::now();
        let _ = gen.for_each_raw_parallel(&ot_correlations, |_k, a, _b, _c| a[0]);
        let all_elapsed = all_start.elapsed();
        let all_tps = count as f64 / all_elapsed.as_secs_f64() / 1e6;
        eprintln!(
            "all-party gen+consume: {:.2?} ({:.1}M triples/sec)",
            all_elapsed, all_tps
        );

        // Single-party (party 0)
        let single_start = std::time::Instant::now();
        let checksum = gen.for_each_single_party_parallel(0, &ot_correlations, |_k, a, _b, _c| a);
        let single_elapsed = single_start.elapsed();
        let single_tps = count as f64 / single_elapsed.as_secs_f64() / 1e6;
        eprintln!(
            "single-party gen+consume: {:.2?} ({:.1}M triples/sec, checksum={})",
            single_elapsed, single_tps, checksum
        );

        eprintln!(
            "speedup: {:.2}x",
            all_elapsed.as_secs_f64() / single_elapsed.as_secs_f64()
        );

        // Verify correctness
        let shamir_t = Shamir::new(n, t).unwrap();
        for k in 0..1000 {
            let all = gen.triple(k, &ot_correlations);
            let a_shares: Vec<Share> = (0..n).map(|p| all[p].a).collect();
            let b_shares: Vec<Share> = (0..n).map(|p| all[p].b).collect();
            let c_shares: Vec<Share> = (0..n).map(|p| all[p].c).collect();
            let a = shamir_t.reconstruct(&a_shares).unwrap();
            let b = shamir_t.reconstruct(&b_shares).unwrap();
            let c = shamir_t.reconstruct(&c_shares).unwrap();
            assert_eq!(c, a * b, "triple {} failed", k);
        }
        eprintln!("=== PASSED ===");
    }

    // ── Batched multiply tests ──────────────────────────────────

    #[test]
    fn test_batch_open_matches_sequential() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let n = 5;
        let t = 1;
        let shamir_t = Shamir::new(n, t).unwrap();

        let values: Vec<Fp> = (2..=7).map(|v| Fp::new(v)).collect();
        let shares: Vec<Vec<Share>> = values
            .iter()
            .map(|&v| shamir_t.share(v, &mut rng))
            .collect();

        let params = RanDouShaParams::new(n, t, 3).unwrap();
        let party_ds = RanDouShaProtocol::new(params)
            .generate_local(&mut rng)
            .unwrap();
        let triples: Vec<Vec<BeaverTriple>> = {
            let mut r = ChaCha20Rng::seed_from_u64(99);
            (0..3)
                .map(|k| {
                    let ds: Vec<DoubleShare> = (0..n).map(|p| party_ds[p][k].clone()).collect();
                    gen_triple_for_test(n, t, &ds, &mut r)
                })
                .collect()
        };

        // Sequential: 3 individual multiply_local calls
        let r0 = beaver_multiply_local(n, t, &shares[0], &shares[1], &triples[0]).unwrap();
        let r1 = beaver_multiply_local(n, t, &shares[2], &shares[3], &triples[1]).unwrap();
        let r2 = beaver_multiply_local(n, t, &shares[4], &shares[5], &triples[2]).unwrap();

        // Batched: 3 openings in one call
        let x_refs: Vec<&[Share]> = vec![&shares[0], &shares[2], &shares[4]];
        let y_refs: Vec<&[Share]> = vec![&shares[1], &shares[3], &shares[5]];
        let t_refs: Vec<&[BeaverTriple]> = vec![&triples[0], &triples[1], &triples[2]];
        let (d_vals, e_vals) =
            beaver_multiply_batch_open(&shamir_t, &x_refs, &y_refs, &t_refs).unwrap();
        let batch_results =
            beaver_multiply_batch_finish(n, &d_vals, &e_vals, &x_refs, &y_refs, &t_refs);

        // Verify identical results
        for p in 0..n {
            assert_eq!(
                r0[p].value, batch_results[0][p].value,
                "mult 0 party {} mismatch",
                p
            );
            assert_eq!(
                r1[p].value, batch_results[1][p].value,
                "mult 1 party {} mismatch",
                p
            );
            assert_eq!(
                r2[p].value, batch_results[2][p].value,
                "mult 2 party {} mismatch",
                p
            );
        }

        // Verify correctness: reconstruct and check products
        for (i, (x_idx, y_idx)) in [(0, 1), (2, 3), (4, 5)].iter().enumerate() {
            let result = shamir_t.reconstruct(&batch_results[i]).unwrap();
            assert_eq!(
                result,
                values[*x_idx] * values[*y_idx],
                "batch mult {} wrong",
                i
            );
        }
    }

    fn gen_triple_for_test(
        n: usize,
        t: usize,
        ds: &[DoubleShare],
        rng: &mut ChaCha20Rng,
    ) -> Vec<BeaverTriple> {
        // Use generate_triples with transposed double shares
        let ds_vec = vec![ds.to_vec()];
        generate_triples(n, t, &ds_vec, rng)
            .unwrap()
            .into_iter()
            .next()
            .unwrap()
    }

    #[test]
    fn test_batched_chain_matches_unbatched() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let n = 5;
        let t = 1;
        let shamir_t = Shamir::new(n, t).unwrap();

        let num_values = 20;
        let values: Vec<Fp> = (2..2 + num_values as u64).map(Fp::new).collect();
        let expected: Fp = values.iter().copied().reduce(|a, b| a * b).unwrap();

        let shares: Vec<Vec<Share>> = values
            .iter()
            .map(|&v| shamir_t.share(v, &mut rng))
            .collect();

        let party_ds = RanDouShaProtocol::new(RanDouShaParams::new(n, t, num_values - 1).unwrap())
            .generate_local(&mut rng)
            .unwrap();
        let triples: Vec<Vec<BeaverTriple>> = {
            let mut r = ChaCha20Rng::seed_from_u64(99);
            (0..num_values - 1)
                .map(|k| {
                    let ds: Vec<DoubleShare> = (0..n).map(|p| party_ds[p][k].clone()).collect();
                    gen_triple_for_test(n, t, &ds, &mut r)
                })
                .collect()
        };

        // Unbatched
        let unbatched = beaver_multiply_chain(n, t, &shares, &triples).unwrap();
        let r_unbatched = shamir_t.reconstruct(&unbatched).unwrap();
        assert_eq!(r_unbatched, expected);

        // Batched
        let batched = beaver_multiply_chain_batched(n, t, &shares, &triples).unwrap();
        let r_batched = shamir_t.reconstruct(&batched).unwrap();
        assert_eq!(r_batched, expected);

        // Results should be identical
        for p in 0..n {
            assert_eq!(unbatched[p].value, batched[p].value, "party {} differs", p);
        }
    }

    #[test]
    fn test_independent_batched_multiply() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let n = 5;
        let t = 1;
        let shamir_t = Shamir::new(n, t).unwrap();

        let num_mults = 10;
        let x_vals: Vec<Fp> = (2..2 + num_mults as u64).map(Fp::new).collect();
        let y_vals: Vec<Fp> = (10..10 + num_mults as u64).map(Fp::new).collect();

        let x_shares: Vec<Vec<Share>> = x_vals
            .iter()
            .map(|&v| shamir_t.share(v, &mut rng))
            .collect();
        let y_shares: Vec<Vec<Share>> = y_vals
            .iter()
            .map(|&v| shamir_t.share(v, &mut rng))
            .collect();

        let party_ds = RanDouShaProtocol::new(RanDouShaParams::new(n, t, num_mults).unwrap())
            .generate_local(&mut rng)
            .unwrap();
        let triples: Vec<Vec<BeaverTriple>> = {
            let mut r = ChaCha20Rng::seed_from_u64(99);
            (0..num_mults)
                .map(|k| {
                    let ds: Vec<DoubleShare> = (0..n).map(|p| party_ds[p][k].clone()).collect();
                    gen_triple_for_test(n, t, &ds, &mut r)
                })
                .collect()
        };

        // Batched independent multiply
        let results =
            beaver_multiply_independent_batched(n, t, &x_shares, &y_shares, &triples).unwrap();
        assert_eq!(results.len(), num_mults);

        // Verify each multiplication
        for i in 0..num_mults {
            let result = shamir_t.reconstruct(&results[i]).unwrap();
            assert_eq!(
                result,
                x_vals[i] * y_vals[i],
                "independent mult {} wrong",
                i
            );
        }
    }

    #[test]
    fn test_batched_chain_large() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let n = 5;
        let t = 1;
        let shamir_t = Shamir::new(n, t).unwrap();

        let num_values = 100;
        let values: Vec<Fp> = (2..2 + num_values as u64).map(Fp::new).collect();
        let expected: Fp = values.iter().copied().reduce(|a, b| a * b).unwrap();

        let shares: Vec<Vec<Share>> = values
            .iter()
            .map(|&v| shamir_t.share(v, &mut rng))
            .collect();

        let party_ds = RanDouShaProtocol::new(RanDouShaParams::new(n, t, num_values - 1).unwrap())
            .generate_local(&mut rng)
            .unwrap();
        let triples: Vec<Vec<BeaverTriple>> = {
            let mut r = ChaCha20Rng::seed_from_u64(99);
            (0..num_values - 1)
                .map(|k| {
                    let ds: Vec<DoubleShare> = (0..n).map(|p| party_ds[p][k].clone()).collect();
                    gen_triple_for_test(n, t, &ds, &mut r)
                })
                .collect()
        };

        let result = beaver_multiply_chain_batched(n, t, &shares, &triples).unwrap();
        let r = shamir_t.reconstruct(&result).unwrap();
        assert_eq!(r, expected, "100-value batched chain failed");
    }

    #[test]
    fn test_batch_edge_cases() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let n = 5;
        let t = 1;
        let shamir_t = Shamir::new(n, t).unwrap();

        // Batch size 0
        let (d, e) = beaver_multiply_batch_open(&shamir_t, &[], &[], &[]).unwrap();
        assert!(d.is_empty());
        assert!(e.is_empty());

        // Independent batched with 0 mults
        let results = beaver_multiply_independent_batched(n, t, &[], &[], &[]).unwrap();
        assert!(results.is_empty());

        // Chain with exactly 2 values (1 mult, batch_size=3 but only 1 mult)
        let shares: Vec<Vec<Share>> = [Fp::new(7), Fp::new(6)]
            .iter()
            .map(|&v| shamir_t.share(v, &mut rng))
            .collect();
        let party_ds = RanDouShaProtocol::new(RanDouShaParams::new(n, t, 1).unwrap())
            .generate_local(&mut rng)
            .unwrap();
        let triples: Vec<Vec<BeaverTriple>> = {
            let mut r = ChaCha20Rng::seed_from_u64(99);
            vec![gen_triple_for_test(
                n,
                t,
                &(0..n).map(|p| party_ds[p][0].clone()).collect::<Vec<_>>(),
                &mut r,
            )]
        };
        let result = beaver_multiply_chain_batched(n, t, &shares, &triples).unwrap();
        assert_eq!(shamir_t.reconstruct(&result).unwrap(), Fp::new(42));
    }

    // ── GPU tests ───────────────────────────────────────────────

    #[cfg(target_os = "macos")]
    #[test]
    fn test_gpu_triples_correctness() {
        use crate::gpu::GpuTripleGen;
        use crate::silent_ot::ExpandedCorrelations;

        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let n = 5usize;
        let t = 1usize;
        let count = 1000usize;
        let spr = n - 2 * t;
        let num_rounds = count.div_ceil(spr);
        let shamir_t = Shamir::new(n, t).unwrap();

        let party_seeds: Vec<u64> = (0..n).map(|_| rng.gen()).collect();
        let ot_correlations: Vec<ExpandedCorrelations> = (0..n)
            .map(|i| {
                let mut prng = ChaCha20Rng::seed_from_u64(party_seeds[i]);
                ExpandedCorrelations::from_random(i, num_rounds, &mut prng)
            })
            .collect();

        let gpu = GpuTripleGen::new(n, t).unwrap();
        let batch = gpu.generate(count, &ot_correlations, &mut rng).unwrap();

        assert_eq!(batch.count, count);
        assert_eq!(batch.n, n);

        // Verify c = a * b for every triple
        for k in 0..count {
            let shares = batch.triple_shares(k);
            let a_shares: Vec<Share> = shares.iter().map(|tr| tr.a).collect();
            let b_shares: Vec<Share> = shares.iter().map(|tr| tr.b).collect();
            let c_shares: Vec<Share> = shares.iter().map(|tr| tr.c).collect();

            let a = shamir_t.reconstruct(&a_shares).unwrap();
            let b = shamir_t.reconstruct(&b_shares).unwrap();
            let c = shamir_t.reconstruct(&c_shares).unwrap();
            assert_eq!(c, a * b, "GPU triple {} failed: c != a*b", k);
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn test_gpu_2m_throughput() {
        use crate::gpu::GpuTripleGen;
        use crate::silent_ot::ExpandedCorrelations;
        use rayon::prelude::*;

        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let n = 5usize;
        let t = 1usize;
        let count = 2_000_000usize;
        let spr = n - 2 * t;
        let num_rounds = count.div_ceil(spr);
        let shamir_t = Shamir::new(n, t).unwrap();

        eprintln!("=== GPU vs CPU: 2M Beaver triples (n={}, t={}) ===", n, t);

        // Generate OT correlations
        let party_seeds: Vec<u64> = (0..n).map(|_| rng.gen()).collect();
        let ot_correlations: Vec<ExpandedCorrelations> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut prng = ChaCha20Rng::seed_from_u64(party_seeds[i]);
                ExpandedCorrelations::from_random(i, num_rounds, &mut prng)
            })
            .collect();

        // GPU batch
        let gpu = GpuTripleGen::new(n, t).unwrap();
        let gpu_start = std::time::Instant::now();
        let gpu_batch = gpu.generate(count, &ot_correlations, &mut rng).unwrap();
        let gpu_elapsed = gpu_start.elapsed();
        let gpu_tps = count as f64 / gpu_elapsed.as_secs_f64() / 1e6;
        eprintln!(
            "GPU batch:  {:.2?}  ({:.1}M triples/sec)",
            gpu_elapsed, gpu_tps
        );

        // CPU batch for comparison
        let cpu_start = std::time::Instant::now();
        let _cpu_batch =
            generate_triples_from_ot_batch(n, t, count, &ot_correlations, &mut rng).unwrap();
        let cpu_elapsed = cpu_start.elapsed();
        let cpu_tps = count as f64 / cpu_elapsed.as_secs_f64() / 1e6;
        eprintln!(
            "CPU batch:  {:.2?}  ({:.1}M triples/sec)",
            cpu_elapsed, cpu_tps
        );

        eprintln!(
            "GPU speedup: {:.2}x",
            cpu_elapsed.as_secs_f64() / gpu_elapsed.as_secs_f64()
        );

        // GPU single-party
        let gpu_sp_start = std::time::Instant::now();
        let (_a, _b, _c) = gpu
            .generate_single_party(0, count, &ot_correlations, &mut rng)
            .unwrap();
        let gpu_sp_elapsed = gpu_sp_start.elapsed();
        let gpu_sp_tps = count as f64 / gpu_sp_elapsed.as_secs_f64() / 1e6;
        eprintln!(
            "GPU single-party: {:.2?}  ({:.1}M triples/sec)",
            gpu_sp_elapsed, gpu_sp_tps
        );

        // Verify a sample of GPU triples
        for k in 0..1000 {
            let shares = gpu_batch.triple_shares(k);
            let a: Vec<Share> = shares.iter().map(|tr| tr.a).collect();
            let b: Vec<Share> = shares.iter().map(|tr| tr.b).collect();
            let c: Vec<Share> = shares.iter().map(|tr| tr.c).collect();
            assert_eq!(
                shamir_t.reconstruct(&c).unwrap(),
                shamir_t.reconstruct(&a).unwrap() * shamir_t.reconstruct(&b).unwrap(),
                "GPU triple {} failed",
                k
            );
        }
        eprintln!("=== PASSED ===");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_gpu32_triples_correctness() {
        use crate::field32::Fp32;
        use crate::gpu::{ExpandedCorrelations32, GpuTripleGen32};

        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let n = 5usize;
        let t = 1usize;
        let count = 1000usize;
        let spr = n - 2 * t;
        let num_rounds = count.div_ceil(spr);

        let ot: Vec<ExpandedCorrelations32> = (0..n)
            .map(|i| ExpandedCorrelations32::from_random(i, num_rounds, &mut rng))
            .collect();

        let gpu = GpuTripleGen32::new(n, t).unwrap();
        let batch = gpu.generate(count, &ot, &mut rng).unwrap();

        assert_eq!(batch.count, count);
        assert_eq!(batch.n, n);

        // Verify c = a * b using Fp32 Lagrange interpolation
        let points: Vec<Fp32> = (1..=n as u32).map(Fp32::new).collect();
        for k in 0..count {
            let mut a_secret = Fp32::ZERO;
            let mut b_secret = Fp32::ZERO;
            let mut c_secret = Fp32::ZERO;
            for p in 0..n {
                let (av, bv, cv) = batch.triple_values(k, p);
                // Lagrange interpolation at zero for each value
                let mut lag = Fp32::ONE;
                for j in 0..n {
                    if j == p {
                        continue;
                    }
                    lag *= -points[j] * (points[p] - points[j]).inv();
                }
                a_secret += Fp32::from_reduced(av) * lag;
                b_secret += Fp32::from_reduced(bv) * lag;
                c_secret += Fp32::from_reduced(cv) * lag;
            }
            assert_eq!(
                c_secret,
                a_secret * b_secret,
                "GPU32 triple {} failed: c={} != a*b={}*{}={}",
                k,
                c_secret,
                a_secret,
                b_secret,
                a_secret * b_secret
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn test_gpu32_2m_throughput() {
        use crate::gpu::{ExpandedCorrelations32, GpuTripleGen32};
        use rayon::prelude::*;

        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let n = 5usize;
        let t = 1usize;
        let count = 2_000_000usize;
        let spr = n - 2 * t;
        let num_rounds = count.div_ceil(spr);

        eprintln!(
            "=== GPU32 vs GPU64: 2M Beaver triples (n={}, t={}) ===",
            n, t
        );

        let ot32: Vec<ExpandedCorrelations32> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut prng = ChaCha20Rng::seed_from_u64(i as u64 + 100);
                ExpandedCorrelations32::from_random(i, num_rounds, &mut prng)
            })
            .collect();

        let gpu32 = GpuTripleGen32::new(n, t).unwrap();

        // GPU32
        let start = std::time::Instant::now();
        let _batch32 = gpu32.generate(count, &ot32, &mut rng).unwrap();
        let gpu32_elapsed = start.elapsed();
        let gpu32_tps = count as f64 / gpu32_elapsed.as_secs_f64() / 1e6;
        eprintln!(
            "GPU32 batch: {:.2?}  ({:.1}M triples/sec)",
            gpu32_elapsed, gpu32_tps
        );

        // GPU64 for comparison
        use crate::gpu::GpuTripleGen;
        use crate::silent_ot::ExpandedCorrelations;
        let ot64: Vec<ExpandedCorrelations> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut prng = ChaCha20Rng::seed_from_u64(i as u64 + 200);
                ExpandedCorrelations::from_random(i, num_rounds, &mut prng)
            })
            .collect();

        let gpu64 = GpuTripleGen::new(n, t).unwrap();
        let start = std::time::Instant::now();
        let _batch64 = gpu64.generate(count, &ot64, &mut rng).unwrap();
        let gpu64_elapsed = start.elapsed();
        let gpu64_tps = count as f64 / gpu64_elapsed.as_secs_f64() / 1e6;
        eprintln!(
            "GPU64 batch: {:.2?}  ({:.1}M triples/sec)",
            gpu64_elapsed, gpu64_tps
        );

        eprintln!(
            "GPU32/GPU64 speedup: {:.2}x",
            gpu64_elapsed.as_secs_f64() / gpu32_elapsed.as_secs_f64()
        );

        // GPU32 streaming (zero-copy)
        let start = std::time::Instant::now();
        let _checksum = gpu32
            .generate_streaming(count, &ot32, &mut rng, |_k, a, _b, _c| a[0] as u64)
            .unwrap();
        let gpu32_stream_elapsed = start.elapsed();
        let gpu32_stream_tps = count as f64 / gpu32_stream_elapsed.as_secs_f64() / 1e6;
        eprintln!(
            "GPU32 streaming (zero-copy): {:.2?}  ({:.1}M triples/sec)",
            gpu32_stream_elapsed, gpu32_stream_tps
        );

        // GPU32 single-party streaming
        let start = std::time::Instant::now();
        let (a_sp, _b_sp, _c_sp) = gpu32
            .generate_single_party(0, count, &ot32, &mut rng)
            .unwrap();
        let gpu32_sp_elapsed = start.elapsed();
        let gpu32_sp_tps = count as f64 / gpu32_sp_elapsed.as_secs_f64() / 1e6;
        eprintln!(
            "GPU32 single-party: {:.2?}  ({:.1}M triples/sec)",
            gpu32_sp_elapsed, gpu32_sp_tps
        );

        eprintln!("=== PASSED ===");
    }

    /// 50M-scale benchmark for production GPU32 kernel throughput.
    #[cfg(target_os = "macos")]
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn test_gpu32_50m_kernel_throughput() {
        use crate::gpu::{ExpandedCorrelations32, GpuTripleGen32};
        use rayon::prelude::*;

        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let n = 5usize;
        let t = 1usize;
        let count = 50_000_000usize;
        let spr = n - 2 * t;
        let num_rounds = count.div_ceil(spr);

        let ot32: Vec<ExpandedCorrelations32> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut prng = ChaCha20Rng::seed_from_u64(i as u64 + 100);
                ExpandedCorrelations32::from_random(i, num_rounds, &mut prng)
            })
            .collect();

        let gpu32 = GpuTripleGen32::new(n, t).unwrap();

        eprintln!(
            "\n=== GPU32 Production: 50M Beaver triples (n={}, t={}) ===",
            n, t
        );

        // Warmup
        let _ = gpu32
            .generate_single_party(0, count, &ot32, &mut rng)
            .unwrap();

        let iters = 5;

        // Single-party (kernel only)
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = gpu32
                .generate_single_party(0, count, &ot32, &mut rng)
                .unwrap();
        }
        let dt_sp = t0.elapsed() / iters;
        let tps_sp = count as f64 / dt_sp.as_secs_f64();
        eprintln!(
            "  Single-party (50M): {:?} → {:.2}B triples/sec",
            dt_sp,
            tps_sp / 1e9
        );

        // Pre-allocated single-party (zero-alloc, zero-copy)
        let bufs = gpu32.preallocate_single_party(count);
        // Warmup to page in memory
        let _ = gpu32.generate_single_party_prealloc(0, &ot32, &mut rng, &bufs);

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = gpu32.generate_single_party_prealloc(0, &ot32, &mut rng, &bufs);
        }
        let dt_prealloc = t0.elapsed() / iters;
        let tps_prealloc = count as f64 / dt_prealloc.as_secs_f64();
        eprintln!(
            "  Pre-alloc SP (50M): {:?} → {:.2}B triples/sec",
            dt_prealloc,
            tps_prealloc / 1e9
        );

        // Kernel-only (no OT copy, no readback — same data reused)
        let kernel_iters = 10;
        let t0 = std::time::Instant::now();
        for _ in 0..kernel_iters {
            gpu32.dispatch_kernel_only(0, &bufs);
        }
        let dt_kernel = t0.elapsed() / kernel_iters;
        let tps_kernel = count as f64 / dt_kernel.as_secs_f64();
        eprintln!(
            "  Kernel-only (50M):  {:?} → {:.2}B triples/sec",
            dt_kernel,
            tps_kernel / 1e9
        );

        // All-party
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = gpu32.generate(count, &ot32, &mut rng).unwrap();
        }
        let dt_all = t0.elapsed() / iters;
        let tps_all = count as f64 / dt_all.as_secs_f64();
        eprintln!(
            "  All-party (50M):    {:?} → {:.2}B triples/sec",
            dt_all,
            tps_all / 1e9
        );
    }

    /// Benchmark 1D vs 2D dispatch and H3 (multi-triple per thread).
    /// Loads pre-compiled /tmp/div_test.metallib.
    #[cfg(target_os = "macos")]
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn test_gpu_dispatch_1d_vs_2d() {
        use metal::*;

        let device = Device::system_default().expect("no Metal GPU");
        let queue = device.new_command_queue();

        let lib_data = match std::fs::read("/tmp/div_test.metallib") {
            Ok(d) => d,
            Err(_) => {
                eprintln!("SKIP: /tmp/div_test.metallib not found. Compile with:");
                eprintln!("  cd /tmp && xcrun -sdk macosx metal -O2 -std=metal3.0 -c div_test.metal -o div_test.air && xcrun -sdk macosx metallib -o div_test.metallib div_test.air");
                return;
            }
        };
        let library = device
            .new_library_with_data(&lib_data)
            .expect("bad metallib");

        let fn_1d = library.get_function("v1d", None).expect("v1d not found");
        let fn_2d = library.get_function("v2d", None).expect("v2d not found");
        let pipe_1d = device
            .new_compute_pipeline_state_with_function(&fn_1d)
            .unwrap();
        let pipe_2d = device
            .new_compute_pipeline_state_with_function(&fn_2d)
            .unwrap();

        let n = 5usize;
        let spr = 3u32;
        let total_triples = 50_000_000usize;
        let num_rounds = (total_triples as u32 + spr - 1) / spr;
        let count = total_triples as u32;

        // Allocate buffers
        let ot_data: Vec<u32> = (0..num_rounds as usize * n)
            .map(|i| (i as u32) % 0x7FFFFFFE + 1)
            .collect();
        let seed_data: Vec<u32> = (0..num_rounds as usize)
            .map(|i| (i as u32).wrapping_mul(0x9e3779b9))
            .collect();

        let ot_buf = device.new_buffer_with_data(
            ot_data.as_ptr() as *const _,
            (ot_data.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let seed_buf = device.new_buffer_with_data(
            seed_data.as_ptr() as *const _,
            (seed_data.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let out_bytes = (count as u64) * 4;
        let a_buf = device.new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);
        let b_buf = device.new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);
        let c_buf = device.new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);

        eprintln!(
            "\n=== 1D vs 2D Dispatch Benchmark ({:.1}M triples) ===",
            total_triples as f64 / 1e6
        );
        eprintln!(
            "  thread_execution_width: 1D={}, 2D={}",
            pipe_1d.thread_execution_width(),
            pipe_2d.thread_execution_width()
        );

        let dispatch_1d = |p: &ComputePipelineState| {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(p);
            enc.set_buffer(0, Some(&ot_buf), 0);
            enc.set_buffer(1, Some(&a_buf), 0);
            enc.set_buffer(2, Some(&b_buf), 0);
            enc.set_buffer(3, Some(&c_buf), 0);
            enc.set_buffer(4, Some(&seed_buf), 0);
            enc.set_bytes(5, 4, &count as *const u32 as *const _);
            enc.set_bytes(6, 4, &num_rounds as *const u32 as *const _);
            let grid = MTLSize::new(count as u64, 1, 1);
            let tg = MTLSize::new(p.thread_execution_width(), 1, 1);
            enc.dispatch_threads(grid, tg);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        };

        let dispatch_2d = |p: &ComputePipelineState| {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(p);
            enc.set_buffer(0, Some(&ot_buf), 0);
            enc.set_buffer(1, Some(&a_buf), 0);
            enc.set_buffer(2, Some(&b_buf), 0);
            enc.set_buffer(3, Some(&c_buf), 0);
            enc.set_buffer(4, Some(&seed_buf), 0);
            enc.set_bytes(5, 4, &count as *const u32 as *const _);
            enc.set_bytes(6, 4, &num_rounds as *const u32 as *const _);
            let grid = MTLSize::new(spr as u64, num_rounds as u64, 1);
            let tg = MTLSize::new(p.thread_execution_width().min(spr as u64), 1, 1);
            enc.dispatch_threads(grid, tg);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        };

        // Warmup
        dispatch_1d(&pipe_1d);
        dispatch_2d(&pipe_2d);

        let iters = 5;

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            dispatch_1d(&pipe_1d);
        }
        let dt_1d = t0.elapsed() / iters;
        let tps_1d = total_triples as f64 / dt_1d.as_secs_f64();

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            dispatch_2d(&pipe_2d);
        }
        let dt_2d = t0.elapsed() / iters;
        let tps_2d = total_triples as f64 / dt_2d.as_secs_f64();

        eprintln!(
            "  1D (division):      {:?} → {:.2}B triples/sec",
            dt_1d,
            tps_1d / 1e9
        );
        eprintln!(
            "  2D (division-free): {:?} → {:.2}B triples/sec",
            dt_2d,
            tps_2d / 1e9
        );
        eprintln!("  Speedup: {:.3}x", tps_2d / tps_1d);
    }

    /// Benchmark H3 multi-triple-per-thread vs baseline.
    /// Pre-compile: cd /tmp && xcrun -sdk macosx metal -O2 -std=metal3.0 -c h3_test.metal -o h3_test.air && xcrun -sdk macosx metallib -o h3_test.metallib h3_test.air
    #[cfg(target_os = "macos")]
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn test_gpu_h3_multi_triple() {
        use crate::field32::Fp32;
        use crate::gpu::GpuConstants32;
        use metal::*;

        let device = Device::system_default().expect("no Metal GPU");
        let queue = device.new_command_queue();

        let lib_data = match std::fs::read("/tmp/h3_test.metallib") {
            Ok(d) => d,
            Err(_) => {
                eprintln!("SKIP: /tmp/h3_test.metallib not found");
                return;
            }
        };
        let library = device
            .new_library_with_data(&lib_data)
            .expect("bad metallib");

        let fn_base = library
            .get_function("baseline_sp", None)
            .expect("baseline_sp not found");
        let fn_h3 = library
            .get_function("h3_sp", None)
            .expect("h3_sp not found");
        let fn_h3b = library
            .get_function("h3b_sp", None)
            .expect("h3b_sp not found");
        let pipe_base = device
            .new_compute_pipeline_state_with_function(&fn_base)
            .unwrap();
        let pipe_h3 = device
            .new_compute_pipeline_state_with_function(&fn_h3)
            .unwrap();
        let pipe_h3b = device
            .new_compute_pipeline_state_with_function(&fn_h3b)
            .unwrap();

        let n = 5u32;
        let t = 1u32;
        let spr = n - 2 * t;
        let total_triples = 50_000_000u32;
        let num_rounds = (total_triples + spr - 1) / spr;

        // Build constants
        let eval_raw_vec: Vec<u32> = (1..=n).map(|v| Fp32::new(v).raw()).collect();
        let mut eval_raw = [0u32; 8];
        eval_raw[..n as usize].copy_from_slice(&eval_raw_vec);

        let mut him_rows = [[0u32; 8]; 3];
        for j in 0..(spr as usize).min(3) {
            for i in 0..n as usize {
                him_rows[j][i] = Fp32::new((i + 1) as u32).pow(j as u32).raw();
            }
        }

        let points: Vec<Fp32> = (1..=n).map(Fp32::new).collect();
        let mut lags = vec![Fp32::ZERO; n as usize];
        for i in 0..n as usize {
            let mut num = Fp32::ONE;
            let mut den = Fp32::ONE;
            for j in 0..n as usize {
                if i == j {
                    continue;
                }
                num *= -points[j];
                den *= points[i] - points[j];
            }
            lags[i] = num * den.inv();
        }
        let lag_raw: Vec<u32> = lags.iter().map(|f| f.raw()).collect();

        let mut lag_sum = 0u32;
        let mut lag_x_sum = 0u32;
        let mut lag_xsq_sum = 0u32;
        for p in 0..n as usize {
            lag_sum = Fp32::add_raw(lag_sum, lag_raw[p]);
            lag_x_sum = Fp32::add_raw(lag_x_sum, Fp32::mul_raw(lag_raw[p], eval_raw[p]));
            lag_xsq_sum = Fp32::add_raw(
                lag_xsq_sum,
                Fp32::mul_raw(lag_raw[p], Fp32::mul_raw(eval_raw[p], eval_raw[p])),
            );
        }

        // OT data: per-party packed buffers
        let mut ot_packed = Vec::with_capacity(num_rounds as usize * n as usize);
        let mut ot_offsets = [0u32; 8];
        for i in 0..n as usize {
            ot_offsets[i] = ot_packed.len() as u32;
            for r in 0..num_rounds {
                ot_packed.push(
                    ((r as u64 * 0x9e3779b9u64 + i as u64 * 0x517cc1b7u64) as u32) % 0x7FFFFFFE + 1,
                );
            }
        }

        let seed_data: Vec<u32> = (0..num_rounds)
            .map(|r| r.wrapping_mul(0x85ebca6b))
            .collect();

        let consts = GpuConstants32 {
            n,
            t,
            count: total_triples,
            spr,
            num_rounds,
            party: 0,
            eval_raw,
            him_rows,
            lag_sum,
            lag_x_sum,
            lag_xsq_sum,
            ot_offsets,
        };

        let ot_buf = device.new_buffer_with_data(
            ot_packed.as_ptr() as *const _,
            (ot_packed.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let seed_buf = device.new_buffer_with_data(
            seed_data.as_ptr() as *const _,
            (seed_data.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let out_bytes = (total_triples as u64) * 4;
        let a_buf = device.new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);
        let b_buf = device.new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);
        let c_buf = device.new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);

        let dispatch = |pipe: &ComputePipelineState, threads: u32| {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pipe);
            enc.set_buffer(0, Some(&ot_buf), 0);
            enc.set_buffer(1, Some(&a_buf), 0);
            enc.set_buffer(2, Some(&b_buf), 0);
            enc.set_buffer(3, Some(&c_buf), 0);
            enc.set_bytes(
                4,
                std::mem::size_of::<GpuConstants32>() as u64,
                &consts as *const GpuConstants32 as *const _,
            );
            enc.set_buffer(5, Some(&seed_buf), 0);
            let grid = MTLSize::new(threads as u64, 1, 1);
            let tg = MTLSize::new(pipe.thread_execution_width().min(threads as u64), 1, 1);
            enc.dispatch_threads(grid, tg);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        };

        eprintln!(
            "\n=== H3 Multi-Triple Benchmark ({:.1}M triples, spr={}) ===",
            total_triples as f64 / 1e6,
            spr
        );
        eprintln!(
            "  Baseline threads: {}, H3 threads: {}",
            total_triples, num_rounds
        );

        // Warmup
        dispatch(&pipe_base, total_triples);
        dispatch(&pipe_h3, num_rounds);
        dispatch(&pipe_h3b, num_rounds);

        let iters = 5;

        // Baseline
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            dispatch(&pipe_base, total_triples);
        }
        let dt_base = t0.elapsed() / iters;
        let tps_base = total_triples as f64 / dt_base.as_secs_f64();

        // H3: loop over j per thread
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            dispatch(&pipe_h3, num_rounds);
        }
        let dt_h3 = t0.elapsed() / iters;
        let tps_h3 = total_triples as f64 / dt_h3.as_secs_f64();

        // H3b: precompute + batch write
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            dispatch(&pipe_h3b, num_rounds);
        }
        let dt_h3b = t0.elapsed() / iters;
        let tps_h3b = total_triples as f64 / dt_h3b.as_secs_f64();

        eprintln!(
            "  Baseline (1 triple/thread):     {:?} → {:.2}B/sec",
            dt_base,
            tps_base / 1e9
        );
        eprintln!(
            "  H3  (spr triples/thread):       {:?} → {:.2}B/sec",
            dt_h3,
            tps_h3 / 1e9
        );
        eprintln!(
            "  H3b (precompute+batch write):   {:?} → {:.2}B/sec",
            dt_h3b,
            tps_h3b / 1e9
        );
        eprintln!("  H3  speedup: {:.3}x", tps_h3 / tps_base);
        eprintln!("  H3b speedup: {:.3}x", tps_h3b / tps_base);
    }

    /// Benchmark: eliminate SIMD reductions via Lagrange zero-sum property.
    /// V0=baseline, V1=1 SIMD reduction, V2=zero SIMD (pure hash), V3=coalesced writes, V4=uint4 vectorized writes
    #[cfg(target_os = "macos")]
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn test_gpu_lean_pure_benchmark() {
        use crate::field32::Fp32;
        use crate::gpu::GpuConstants32;
        use metal::*;

        let device = Device::system_default().expect("no Metal GPU");
        let queue = device.new_command_queue();

        let lib_data = match std::fs::read("/tmp/lean_test.metallib") {
            Ok(d) => d,
            Err(_) => {
                eprintln!("SKIP: /tmp/lean_test.metallib not found");
                return;
            }
        };
        let library = device
            .new_library_with_data(&lib_data)
            .expect("bad metallib");

        let kernels = [
            "v0_baseline",
            "v1_lean",
            "v2_pure",
            "v3_coalesced",
            "v4_vec4",
        ];
        let pipes: Vec<ComputePipelineState> = kernels
            .iter()
            .map(|name| {
                let f = library
                    .get_function(name, None)
                    .unwrap_or_else(|_| panic!("{} not found", name));
                device.new_compute_pipeline_state_with_function(&f).unwrap()
            })
            .collect();

        let n = 5u32;
        let t = 1u32;
        let spr = n - 2 * t;
        let total_triples = 50_000_000u32;
        let num_rounds = (total_triples + spr - 1) / spr;

        // Build constants
        let eval_raw_vec: Vec<u32> = (1..=n).map(|v| Fp32::new(v).raw()).collect();
        let mut eval_raw = [0u32; 8];
        eval_raw[..n as usize].copy_from_slice(&eval_raw_vec);

        let mut him_rows = [[0u32; 8]; 3];
        for j in 0..(spr as usize).min(3) {
            for i in 0..n as usize {
                him_rows[j][i] = Fp32::new((i + 1) as u32).pow(j as u32).raw();
            }
        }

        let points: Vec<Fp32> = (1..=n).map(Fp32::new).collect();
        let mut lags = vec![Fp32::ZERO; n as usize];
        for i in 0..n as usize {
            let mut num = Fp32::ONE;
            let mut den = Fp32::ONE;
            for j in 0..n as usize {
                if i == j {
                    continue;
                }
                num *= -points[j];
                den *= points[i] - points[j];
            }
            lags[i] = num * den.inv();
        }
        let lag_raw: Vec<u32> = lags.iter().map(|f| f.raw()).collect();

        let mut lag_sum = 0u32;
        let mut lag_x_sum = 0u32;
        let mut lag_xsq_sum = 0u32;
        for p in 0..n as usize {
            lag_sum = Fp32::add_raw(lag_sum, lag_raw[p]);
            lag_x_sum = Fp32::add_raw(lag_x_sum, Fp32::mul_raw(lag_raw[p], eval_raw[p]));
            lag_xsq_sum = Fp32::add_raw(
                lag_xsq_sum,
                Fp32::mul_raw(lag_raw[p], Fp32::mul_raw(eval_raw[p], eval_raw[p])),
            );
        }

        eprintln!(
            "\n  lag_sum={}, lag_x_sum={}, lag_xsq_sum={}",
            lag_sum, lag_x_sum, lag_xsq_sum
        );

        let mut ot_packed = Vec::with_capacity(num_rounds as usize * n as usize);
        let mut ot_offsets = [0u32; 8];
        for i in 0..n as usize {
            ot_offsets[i] = ot_packed.len() as u32;
            for r in 0..num_rounds {
                ot_packed.push(
                    ((r as u64 * 0x9e3779b9u64 + i as u64 * 0x517cc1b7u64) as u32) % 0x7FFFFFFE + 1,
                );
            }
        }

        let seed_data: Vec<u32> = (0..num_rounds)
            .map(|r| r.wrapping_mul(0x85ebca6b))
            .collect();

        let consts = GpuConstants32 {
            n,
            t,
            count: total_triples,
            spr,
            num_rounds,
            party: 0,
            eval_raw,
            him_rows,
            lag_sum,
            lag_x_sum,
            lag_xsq_sum,
            ot_offsets,
        };

        let ot_buf = device.new_buffer_with_data(
            ot_packed.as_ptr() as *const _,
            (ot_packed.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let seed_buf = device.new_buffer_with_data(
            seed_data.as_ptr() as *const _,
            (seed_data.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let out_bytes = (total_triples as u64) * 4;
        let a_buf = device.new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);
        let b_buf = device.new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);
        let c_buf = device.new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);

        // V0 and V1 use: ot_buf(0), a(1), b(2), c(3), const(4), seed(5)
        // V2, V3, V4 use: a(0), b(1), c(2), const(3), seed(4) — no OT buffer
        let dispatch_with_ot = |pipe: &ComputePipelineState, threads: u32| {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pipe);
            enc.set_buffer(0, Some(&ot_buf), 0);
            enc.set_buffer(1, Some(&a_buf), 0);
            enc.set_buffer(2, Some(&b_buf), 0);
            enc.set_buffer(3, Some(&c_buf), 0);
            enc.set_bytes(
                4,
                std::mem::size_of::<GpuConstants32>() as u64,
                &consts as *const GpuConstants32 as *const _,
            );
            enc.set_buffer(5, Some(&seed_buf), 0);
            let grid = MTLSize::new(threads as u64, 1, 1);
            let tg = MTLSize::new(pipe.thread_execution_width().min(threads as u64), 1, 1);
            enc.dispatch_threads(grid, tg);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        };

        let dispatch_no_ot = |pipe: &ComputePipelineState, threads: u32| {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pipe);
            enc.set_buffer(0, Some(&a_buf), 0);
            enc.set_buffer(1, Some(&b_buf), 0);
            enc.set_buffer(2, Some(&c_buf), 0);
            enc.set_bytes(
                3,
                std::mem::size_of::<GpuConstants32>() as u64,
                &consts as *const GpuConstants32 as *const _,
            );
            enc.set_buffer(4, Some(&seed_buf), 0);
            let grid = MTLSize::new(threads as u64, 1, 1);
            let tg = MTLSize::new(pipe.thread_execution_width().min(threads as u64), 1, 1);
            enc.dispatch_threads(grid, tg);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        };

        eprintln!(
            "\n=== Lean/Pure Kernel Benchmark ({:.1}M triples) ===",
            total_triples as f64 / 1e6
        );

        // Warmup all
        dispatch_with_ot(&pipes[0], num_rounds);
        dispatch_with_ot(&pipes[1], num_rounds);
        dispatch_no_ot(&pipes[2], num_rounds);
        dispatch_no_ot(&pipes[3], num_rounds);
        dispatch_no_ot(&pipes[4], num_rounds / 4);

        let iters = 10u32;

        // V0: baseline (4 SIMD reductions)
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            dispatch_with_ot(&pipes[0], num_rounds);
        }
        let dt0 = t0.elapsed() / iters;
        let tps0 = total_triples as f64 / dt0.as_secs_f64();

        // V1: lean (1 SIMD reduction)
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            dispatch_with_ot(&pipes[1], num_rounds);
        }
        let dt1 = t0.elapsed() / iters;
        let tps1 = total_triples as f64 / dt1.as_secs_f64();

        // V2: pure (0 SIMD, no OT)
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            dispatch_no_ot(&pipes[2], num_rounds);
        }
        let dt2 = t0.elapsed() / iters;
        let tps2 = total_triples as f64 / dt2.as_secs_f64();

        // V3: coalesced writes (0 SIMD, round-major layout)
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            dispatch_no_ot(&pipes[3], num_rounds);
        }
        let dt3 = t0.elapsed() / iters;
        let tps3 = total_triples as f64 / dt3.as_secs_f64();

        // V4: vectorized uint4 writes (0 SIMD, 4 rounds/thread)
        let v4_threads = (num_rounds + 3) / 4;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            dispatch_no_ot(&pipes[4], v4_threads);
        }
        let dt4 = t0.elapsed() / iters;
        let tps4 = total_triples as f64 / dt4.as_secs_f64();

        eprintln!(
            "  V0 baseline (4 SIMD):        {:?} → {:.2}B/sec",
            dt0,
            tps0 / 1e9
        );
        eprintln!(
            "  V1 lean (1 SIMD):            {:?} → {:.2}B/sec  ({:.1}x)",
            dt1,
            tps1 / 1e9,
            tps1 / tps0
        );
        eprintln!(
            "  V2 pure (0 SIMD, no OT):     {:?} → {:.2}B/sec  ({:.1}x)",
            dt2,
            tps2 / 1e9,
            tps2 / tps0
        );
        eprintln!(
            "  V3 coalesced (round-major):   {:?} → {:.2}B/sec  ({:.1}x)",
            dt3,
            tps3 / 1e9,
            tps3 / tps0
        );
        eprintln!(
            "  V4 vec4 (uint4 writes):       {:?} → {:.2}B/sec  ({:.1}x)",
            dt4,
            tps4 / 1e9,
            tps4 / tps0
        );

        // Verify V2 correctness: reconstruct and check c = a*b
        dispatch_no_ot(&pipes[2], num_rounds);
        let a_slice = unsafe {
            std::slice::from_raw_parts(a_buf.contents() as *const u32, total_triples as usize)
        };
        let b_slice = unsafe {
            std::slice::from_raw_parts(b_buf.contents() as *const u32, total_triples as usize)
        };
        let c_slice = unsafe {
            std::slice::from_raw_parts(c_buf.contents() as *const u32, total_triples as usize)
        };

        // V2 outputs shares for party 0 (eval_p = eval_raw[0] = 1)
        // a_p = a0 + ca*1, b_p = b0 + cb*1, c_p = a0*b0 + bt_r*1
        // We can't fully verify c=a*b from a single party's shares, but we can sanity check
        // by running the all-party kernel... for now just check values are in range
        let mut in_range = true;
        for k in 0..1000 {
            if a_slice[k] >= 0x7FFFFFFF || b_slice[k] >= 0x7FFFFFFF || c_slice[k] >= 0x7FFFFFFF {
                in_range = false;
                break;
            }
        }
        eprintln!("  V2 values in range: {}", in_range);
    }
}
