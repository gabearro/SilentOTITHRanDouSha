use crate::error::{ProtocolError, Result};
use crate::field::Fp;
use crate::shamir::{Shamir, Share};
use crate::silent_ot::{DistributedSilentOt, ExpandedCorrelations, SilentOtParams};
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DoubleShare {
    pub share_t: Share,
    pub share_2t: Share,
}

#[derive(Clone, Copy, Debug)]
pub struct RanDouShaParams {
    pub n: usize,
    pub t: usize,
    pub count: usize,
}

impl RanDouShaParams {
    pub fn new(n: usize, t: usize, count: usize) -> Result<Self> {
        if n <= 2 * t {
            return Err(ProtocolError::InvalidParams(format!(
                "RanDouSha requires n > 2t, got n={}, t={}",
                n, t
            )));
        }
        if count == 0 {
            return Err(ProtocolError::InvalidParams(
                "count must be > 0".into(),
            ));
        }
        Ok(RanDouShaParams { n, t, count })
    }
}

pub struct HyperInvertibleMatrix {
    entries: Vec<Vec<Fp>>,
    n: usize,
}

impl HyperInvertibleMatrix {
    pub fn new(n: usize) -> Self {
        let alphas: Vec<Fp> = (1..=n as u64).map(Fp::new).collect();
        let mut entries = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = Vec::with_capacity(n);
            for j in 0..n {
                row.push(alphas[j].pow(i as u64));
            }
            entries.push(row);
        }
        HyperInvertibleMatrix { entries, n }
    }

    pub fn get(&self, row: usize, col: usize) -> Fp {
        self.entries[row][col]
    }

    pub fn mul_vec(&self, v: &[Fp]) -> Vec<Fp> {
        assert_eq!(v.len(), self.n);
        (0..self.n)
            .map(|i| {
                (0..self.n)
                    .map(|j| self.entries[i][j] * v[j])
                    .sum()
            })
            .collect()
    }
}

pub struct RanDouShaProtocol {
    pub params: RanDouShaParams,
}

impl RanDouShaProtocol {
    pub fn new(params: RanDouShaParams) -> Self {
        RanDouShaProtocol { params }
    }

    pub fn generate_local<R: Rng>(&self, rng: &mut R) -> Result<Vec<Vec<DoubleShare>>> {
        let n = self.params.n;
        let t = self.params.t;
        let count = self.params.count;
        let shamir_t = Shamir::new(n, t)?;
        let shamir_2t = Shamir::new(n, 2 * t)?;

        let ot_params = SilentOtParams::new(n, t, std::cmp::max(count * 2, 16))?;
        let protocol = DistributedSilentOt::new(ot_params);

        let mut ot_states: Vec<_> = (0..n).map(|i| protocol.init_party(i, rng)).collect();

        let r0_msgs: Vec<_> = ot_states
            .iter()
            .flat_map(DistributedSilentOt::round0_commitments)
            .collect();
        for s in &mut ot_states {
            DistributedSilentOt::process_round0(s, &r0_msgs)?;
        }

        let r1_msgs: Vec<_> = ot_states
            .iter()
            .flat_map(DistributedSilentOt::round1_puncture_choices)
            .collect();
        for s in &mut ot_states {
            DistributedSilentOt::process_round1(s, &r1_msgs)?;
        }

        let mut r2_msgs = Vec::new();
        for s in &ot_states {
            r2_msgs.extend(DistributedSilentOt::round2_sibling_paths(s)?);
        }
        for s in &mut ot_states {
            DistributedSilentOt::process_round2(s, &r2_msgs)?;
        }

        let r3_msgs: Vec<_> = ot_states
            .iter()
            .flat_map(DistributedSilentOt::round3_seed_reveals)
            .collect();
        for s in &mut ot_states {
            DistributedSilentOt::process_round3(s, &r3_msgs)?;
        }

        let ot_correlations: Vec<ExpandedCorrelations> = ot_states
            .iter()
            .map(DistributedSilentOt::expand)
            .collect::<Result<_>>()?;

        let him = HyperInvertibleMatrix::new(n);
        let sharings_per_round = n - 2 * t;
        let num_rounds = count.div_ceil(sharings_per_round);

        let mut all_party_shares: Vec<Vec<DoubleShare>> = vec![Vec::new(); n];

        for round in 0..num_rounds {
            let secrets: Vec<Fp> = (0..n)
                .map(|i| ot_correlations[i].get_random(round))
                .collect();

            let mut all_shares_t: Vec<Vec<Share>> = Vec::with_capacity(n);
            let mut all_shares_2t: Vec<Vec<Share>> = Vec::with_capacity(n);

            for i in 0..n {
                all_shares_t.push(shamir_t.share(secrets[i], rng));
                all_shares_2t.push(shamir_2t.share(secrets[i], rng));
            }

            for j in 0..sharings_per_round {
                if all_party_shares[0].len() >= count {
                    break;
                }

                let shares_t_out: Vec<Share> = (0..n)
                    .map(|p| {
                        let point = shamir_t.eval_points[p];
                        let val: Fp = (0..n)
                            .map(|i| him.get(j, i) * all_shares_t[i][p].value)
                            .sum();
                        Share { point, value: val }
                    })
                    .collect();

                let shares_2t_out: Vec<Share> = (0..n)
                    .map(|p| {
                        let point = shamir_2t.eval_points[p];
                        let val: Fp = (0..n)
                            .map(|i| him.get(j, i) * all_shares_2t[i][p].value)
                            .sum();
                        Share { point, value: val }
                    })
                    .collect();

                for p in 0..n {
                    all_party_shares[p].push(DoubleShare {
                        share_t: shares_t_out[p],
                        share_2t: shares_2t_out[p],
                    });
                }
            }

            for check_row in sharings_per_round..n {
                let shares_t_check: Vec<Share> = (0..n)
                    .map(|p| {
                        let point = shamir_t.eval_points[p];
                        let val: Fp = (0..n)
                            .map(|i| him.get(check_row, i) * all_shares_t[i][p].value)
                            .sum();
                        Share { point, value: val }
                    })
                    .collect();

                let shares_2t_check: Vec<Share> = (0..n)
                    .map(|p| {
                        let point = shamir_2t.eval_points[p];
                        let val: Fp = (0..n)
                            .map(|i| him.get(check_row, i) * all_shares_2t[i][p].value)
                            .sum();
                        Share { point, value: val }
                    })
                    .collect();

                let secret_t = shamir_t.reconstruct(&shares_t_check)?;
                let secret_2t = shamir_2t.reconstruct(&shares_2t_check)?;

                if secret_t != secret_2t {
                    return Err(ProtocolError::MaliciousParty(format!(
                        "HIM check row {} in round {} failed: degree-t secret={} != degree-2t secret={}",
                        check_row, round, secret_t, secret_2t
                    )));
                }
            }
        }

        for shares in &mut all_party_shares {
            shares.truncate(count);
        }

        Ok(all_party_shares)
    }

    pub fn verify(party_shares: &[Vec<DoubleShare>], n: usize, t: usize) -> Result<bool> {
        let shamir_t = Shamir::new(n, t)?;
        let shamir_2t = Shamir::new(n, 2 * t)?;

        if party_shares.is_empty() {
            return Ok(true);
        }
        let count = party_shares[0].len();

        for k in 0..count {
            let shares_t: Vec<Share> = (0..n).map(|p| party_shares[p][k].share_t).collect();
            let shares_2t: Vec<Share> = (0..n).map(|p| party_shares[p][k].share_2t).collect();

            let secret_t = shamir_t.reconstruct(&shares_t)?;
            let secret_2t = shamir_2t.reconstruct(&shares_2t)?;

            if secret_t != secret_2t {
                return Err(ProtocolError::VerificationFailed(format!(
                    "double share {} inconsistent: degree-t secret={} vs degree-2t secret={}",
                    k, secret_t, secret_2t
                )));
            }
        }

        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn test_hyper_invertible_matrix() {
        let m = HyperInvertibleMatrix::new(5);
        let v: Vec<Fp> = (1..=5u64).map(Fp::new).collect();
        let result = m.mul_vec(&v);
        assert_eq!(result.len(), 5);

        let expected_first = Fp::new(1) + Fp::new(2) + Fp::new(3) + Fp::new(4) + Fp::new(5);
        assert_eq!(result[0], expected_first);
    }

    #[test]
    fn test_randousha_basic() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let params = RanDouShaParams::new(5, 1, 3).unwrap();
        let protocol = RanDouShaProtocol::new(params);
        let party_shares = protocol.generate_local(&mut rng).unwrap();

        assert_eq!(party_shares.len(), 5);
        for shares in &party_shares {
            assert_eq!(shares.len(), 3);
        }

        assert!(RanDouShaProtocol::verify(&party_shares, 5, 1).unwrap());
    }

    #[test]
    fn test_randousha_larger_batch() {
        let mut rng = ChaCha20Rng::seed_from_u64(99);
        let params = RanDouShaParams::new(5, 1, 10).unwrap();
        let protocol = RanDouShaProtocol::new(params);
        let party_shares = protocol.generate_local(&mut rng).unwrap();

        assert_eq!(party_shares.len(), 5);
        for shares in &party_shares {
            assert_eq!(shares.len(), 10);
        }
        assert!(RanDouShaProtocol::verify(&party_shares, 5, 1).unwrap());
    }

    #[test]
    fn test_randousha_secrets_are_random() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let params = RanDouShaParams::new(5, 1, 5).unwrap();
        let protocol = RanDouShaProtocol::new(params);
        let party_shares = protocol.generate_local(&mut rng).unwrap();
        let shamir_t = Shamir::new(5, 1).unwrap();

        let secrets: Vec<Fp> = (0..5)
            .map(|k| {
                let shares: Vec<Share> = (0..5).map(|p| party_shares[p][k].share_t).collect();
                shamir_t.reconstruct(&shares).unwrap()
            })
            .collect();

        let all_same = secrets.iter().all(|s| *s == secrets[0]);
        assert!(!all_same, "all secrets identical - randomness broken");
    }

    #[test]
    fn test_double_share_degree() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let params = RanDouShaParams::new(5, 1, 3).unwrap();
        let protocol = RanDouShaProtocol::new(params);
        let party_shares = protocol.generate_local(&mut rng).unwrap();

        let shamir_t = Shamir::new(5, 1).unwrap();
        let shamir_2t = Shamir::new(5, 2).unwrap();

        for k in 0..3 {
            let shares_t: Vec<Share> = (0..5).map(|p| party_shares[p][k].share_t).collect();
            let shares_2t: Vec<Share> = (0..5).map(|p| party_shares[p][k].share_2t).collect();

            let s1 = shamir_t.reconstruct(&shares_t[0..2]).unwrap();
            let s2 = shamir_t.reconstruct(&shares_t[1..3]).unwrap();
            let s3 = shamir_t.reconstruct(&shares_t[2..4]).unwrap();
            assert_eq!(s1, s2);
            assert_eq!(s2, s3);

            let s1 = shamir_2t.reconstruct(&shares_2t[0..3]).unwrap();
            let s2 = shamir_2t.reconstruct(&shares_2t[1..4]).unwrap();
            let s3 = shamir_2t.reconstruct(&shares_2t[2..5]).unwrap();
            assert_eq!(s1, s2);
            assert_eq!(s2, s3);
        }
    }

    #[test]
    fn test_him_verification_passes_honest() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let params = RanDouShaParams::new(5, 1, 5).unwrap();
        let protocol = RanDouShaProtocol::new(params);
        let party_shares = protocol.generate_local(&mut rng).unwrap();
        assert!(RanDouShaProtocol::verify(&party_shares, 5, 1).unwrap());
    }

    #[test]
    fn test_him_verification_catches_tampered_shares() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let params = RanDouShaParams::new(5, 1, 3).unwrap();
        let protocol = RanDouShaProtocol::new(params);
        let mut party_shares = protocol.generate_local(&mut rng).unwrap();

        party_shares[2][0].share_t.value += Fp::new(1);

        let result = RanDouShaProtocol::verify(&party_shares, 5, 1);
        assert!(result.is_err(), "tampered shares should be detected");
        let err = result.unwrap_err();
        assert!(
            format!("{}", err).contains("inconsistent"),
            "error should mention inconsistency: {}",
            err
        );
    }

    #[test]
    fn test_invalid_params() {
        assert!(RanDouShaParams::new(2, 1, 5).is_err());
        assert!(RanDouShaParams::new(5, 0, 0).is_err());
        assert!(RanDouShaParams::new(4, 2, 5).is_err());
    }

    #[test]
    #[ignore]
    fn test_randousha_2m_silent_ot() {
        use crate::silent_ot::{DistributedSilentOt, SilentOtParams};

        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let count: usize = 2_000_000;
        let n: usize = 5;
        let t: usize = 1;
        let sharings_per_round = n - 2 * t;
        let num_rounds = count.div_ceil(sharings_per_round);

        eprintln!(
            "generating {} double shares (n={}, t={}, {} HIM rounds)",
            count, n, t, num_rounds
        );

        let ot_start = std::time::Instant::now();
        let ot_params = SilentOtParams::new(n, t, num_rounds).unwrap();
        let ot_protocol = DistributedSilentOt::new(ot_params);
        let mut ot_states: Vec<_> =
            (0..n).map(|i| ot_protocol.init_party(i, &mut rng)).collect();

        let r0_msgs: Vec<_> = ot_states
            .iter()
            .flat_map(DistributedSilentOt::round0_commitments)
            .collect();
        for s in &mut ot_states {
            DistributedSilentOt::process_round0(s, &r0_msgs).unwrap();
        }

        let r1_msgs: Vec<_> = ot_states
            .iter()
            .flat_map(DistributedSilentOt::round1_puncture_choices)
            .collect();
        for s in &mut ot_states {
            DistributedSilentOt::process_round1(s, &r1_msgs).unwrap();
        }

        let mut r2_msgs = Vec::new();
        for s in &ot_states {
            r2_msgs.extend(DistributedSilentOt::round2_sibling_paths(s).unwrap());
        }
        for s in &mut ot_states {
            DistributedSilentOt::process_round2(s, &r2_msgs).unwrap();
        }

        let r3_msgs: Vec<_> = ot_states
            .iter()
            .flat_map(DistributedSilentOt::round3_seed_reveals)
            .collect();
        for s in &mut ot_states {
            DistributedSilentOt::process_round3(s, &r3_msgs).unwrap();
        }

        let ot_correlations: Vec<_> = ot_states
            .iter()
            .map(|s| DistributedSilentOt::expand(s).unwrap())
            .collect();
        let ot_elapsed = ot_start.elapsed();
        eprintln!("silent OT setup: {:.2?}", ot_elapsed);

        let him_start = std::time::Instant::now();
        let shamir_t = Shamir::new(n, t).unwrap();
        let shamir_2t = Shamir::new(n, 2 * t).unwrap();
        let him = HyperInvertibleMatrix::new(n);
        let mut all_party_shares: Vec<Vec<DoubleShare>> = vec![Vec::new(); n];

        for round in 0..num_rounds {
            let secrets: Vec<Fp> = (0..n)
                .map(|i| ot_correlations[i].get_random(round))
                .collect();

            let mut all_shares_t: Vec<Vec<Share>> = Vec::with_capacity(n);
            let mut all_shares_2t: Vec<Vec<Share>> = Vec::with_capacity(n);
            for i in 0..n {
                all_shares_t.push(shamir_t.share(secrets[i], &mut rng));
                all_shares_2t.push(shamir_2t.share(secrets[i], &mut rng));
            }

            for j in 0..sharings_per_round {
                if all_party_shares[0].len() >= count {
                    break;
                }

                let shares_t_out: Vec<Share> = (0..n)
                    .map(|p| {
                        let point = shamir_t.eval_points[p];
                        let val: Fp = (0..n)
                            .map(|i| him.get(j, i) * all_shares_t[i][p].value)
                            .sum();
                        Share { point, value: val }
                    })
                    .collect();

                let shares_2t_out: Vec<Share> = (0..n)
                    .map(|p| {
                        let point = shamir_2t.eval_points[p];
                        let val: Fp = (0..n)
                            .map(|i| him.get(j, i) * all_shares_2t[i][p].value)
                            .sum();
                        Share { point, value: val }
                    })
                    .collect();

                for p in 0..n {
                    all_party_shares[p].push(DoubleShare {
                        share_t: shares_t_out[p],
                        share_2t: shares_2t_out[p],
                    });
                }
            }
        }

        for shares in &mut all_party_shares {
            shares.truncate(count);
        }
        let him_elapsed = him_start.elapsed();
        eprintln!("HIM generation: {:.2?}", him_elapsed);
        eprintln!(
            "total: {:.2?}",
            ot_elapsed + him_elapsed
        );

        assert_eq!(all_party_shares.len(), n);
        for shares in &all_party_shares {
            assert_eq!(shares.len(), count);
        }

        let verify_start = std::time::Instant::now();
        assert!(RanDouShaProtocol::verify(&all_party_shares, n, t).unwrap());
        eprintln!("verified {} double shares in {:.2?}", count, verify_start.elapsed());
    }

    #[test]
    #[ignore]
    fn test_2m_double_shares_chain_multiply() {
        use crate::multiply::multiply_sequence;
        use crate::silent_ot::{DistributedSilentOt, SilentOtParams};

        let mut rng = ChaCha20Rng::seed_from_u64(12345);
        let n: usize = 5;
        let t: usize = 1;
        let num_mults: usize = 2_000_000;
        let num_values = num_mults + 1;
        let sharings_per_round = n - 2 * t;
        let num_rounds = num_mults.div_ceil(sharings_per_round);

        eprintln!(
            "=== 2M chain multiply: {} multiplications (n={}, t={}, {} HIM rounds) ===",
            num_mults, n, t, num_rounds
        );

        let ot_start = std::time::Instant::now();
        let ot_params = SilentOtParams::new(n, t, num_rounds).unwrap();
        let ot_protocol = DistributedSilentOt::new(ot_params);
        let mut ot_states: Vec<_> =
            (0..n).map(|i| ot_protocol.init_party(i, &mut rng)).collect();

        let r0_msgs: Vec<_> = ot_states
            .iter()
            .flat_map(DistributedSilentOt::round0_commitments)
            .collect();
        for s in &mut ot_states {
            DistributedSilentOt::process_round0(s, &r0_msgs).unwrap();
        }

        let r1_msgs: Vec<_> = ot_states
            .iter()
            .flat_map(DistributedSilentOt::round1_puncture_choices)
            .collect();
        for s in &mut ot_states {
            DistributedSilentOt::process_round1(s, &r1_msgs).unwrap();
        }

        let mut r2_msgs = Vec::new();
        for s in &ot_states {
            r2_msgs.extend(DistributedSilentOt::round2_sibling_paths(s).unwrap());
        }
        for s in &mut ot_states {
            DistributedSilentOt::process_round2(s, &r2_msgs).unwrap();
        }

        let r3_msgs: Vec<_> = ot_states
            .iter()
            .flat_map(DistributedSilentOt::round3_seed_reveals)
            .collect();
        for s in &mut ot_states {
            DistributedSilentOt::process_round3(s, &r3_msgs).unwrap();
        }

        let ot_correlations: Vec<_> = ot_states
            .iter()
            .map(|s| DistributedSilentOt::expand(s).unwrap())
            .collect();
        let ot_elapsed = ot_start.elapsed();
        eprintln!("silent OT setup + expand: {:.2?}", ot_elapsed);

        let him_start = std::time::Instant::now();
        let shamir_t = Shamir::new(n, t).unwrap();
        let shamir_2t = Shamir::new(n, 2 * t).unwrap();
        let him = HyperInvertibleMatrix::new(n);
        let mut all_party_shares: Vec<Vec<DoubleShare>> = vec![Vec::new(); n];

        for round in 0..num_rounds {
            let secrets: Vec<Fp> = (0..n)
                .map(|i| ot_correlations[i].get_random(round))
                .collect();

            let mut all_shares_t: Vec<Vec<Share>> = Vec::with_capacity(n);
            let mut all_shares_2t: Vec<Vec<Share>> = Vec::with_capacity(n);
            for i in 0..n {
                all_shares_t.push(shamir_t.share(secrets[i], &mut rng));
                all_shares_2t.push(shamir_2t.share(secrets[i], &mut rng));
            }

            for j in 0..sharings_per_round {
                if all_party_shares[0].len() >= num_mults {
                    break;
                }

                let shares_t_out: Vec<Share> = (0..n)
                    .map(|p| {
                        let point = shamir_t.eval_points[p];
                        let val: Fp = (0..n)
                            .map(|i| him.get(j, i) * all_shares_t[i][p].value)
                            .sum();
                        Share { point, value: val }
                    })
                    .collect();

                let shares_2t_out: Vec<Share> = (0..n)
                    .map(|p| {
                        let point = shamir_2t.eval_points[p];
                        let val: Fp = (0..n)
                            .map(|i| him.get(j, i) * all_shares_2t[i][p].value)
                            .sum();
                        Share { point, value: val }
                    })
                    .collect();

                for p in 0..n {
                    all_party_shares[p].push(DoubleShare {
                        share_t: shares_t_out[p],
                        share_2t: shares_2t_out[p],
                    });
                }
            }
        }

        for shares in &mut all_party_shares {
            shares.truncate(num_mults);
        }
        let him_elapsed = him_start.elapsed();
        eprintln!("HIM generation: {:.2?}", him_elapsed);
        eprintln!(
            "offline total (OT + HIM): {:.2?}",
            ot_elapsed + him_elapsed
        );

        let verify_start = std::time::Instant::now();
        let sample_size = 1000;
        let sample_shares: Vec<Vec<DoubleShare>> = (0..n)
            .map(|p| all_party_shares[p][..sample_size].to_vec())
            .collect();
        assert!(RanDouShaProtocol::verify(&sample_shares, n, t).unwrap());
        eprintln!(
            "verified {} sample double shares in {:.2?}",
            sample_size,
            verify_start.elapsed()
        );

        let values: Vec<Fp> = (0..num_values)
            .map(|i| Fp::new((i % 7 + 2) as u64))
            .collect();

        let expected: Fp = values.iter().copied().reduce(|a, b| a * b).unwrap();
        eprintln!("expected product (mod p): {}", expected);

        let share_start = std::time::Instant::now();
        let value_shares: Vec<Vec<Share>> = values
            .iter()
            .map(|v| shamir_t.share(*v, &mut rng))
            .collect();

        let double_shares: Vec<Vec<DoubleShare>> = (0..num_mults)
            .map(|k| (0..n).map(|p| all_party_shares[p][k].clone()).collect())
            .collect();
        eprintln!("sharing + reshape: {:.2?}", share_start.elapsed());

        let online_start = std::time::Instant::now();
        let result_shares =
            multiply_sequence(n, t, 0, &value_shares, &double_shares).unwrap();
        let online_elapsed = online_start.elapsed();
        eprintln!(
            "online phase ({} multiplications): {:.2?}",
            num_mults, online_elapsed
        );

        let result = shamir_t.reconstruct(&result_shares).unwrap();
        assert_eq!(
            result, expected,
            "chain multiply result mismatch: got {} expected {}",
            result, expected
        );
        eprintln!("revealed result: {} (correct!)", result);
        eprintln!(
            "grand total: {:.2?}",
            ot_elapsed + him_elapsed + online_elapsed
        );
        eprintln!("=== PASSED ===");
    }
}
