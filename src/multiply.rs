use crate::error::{ProtocolError, Result};
use crate::field::Fp;
use crate::randousha::DoubleShare;
use crate::shamir::{Shamir, Share};

pub struct DnMultiply {
    pub n: usize,
    pub t: usize,
    pub king: usize,
}

impl DnMultiply {
    pub fn new(n: usize, t: usize, king: usize) -> Result<Self> {
        if king >= n {
            return Err(ProtocolError::InvalidParams(format!(
                "king {} >= n {}",
                king, n
            )));
        }
        if n <= 2 * t {
            return Err(ProtocolError::InvalidParams(format!(
                "need n > 2t, got n={}, t={}",
                n, t
            )));
        }
        Ok(DnMultiply { n, t, king })
    }

    pub fn compute_masked_share(
        x_share: &Share,
        y_share: &Share,
        double_share: &DoubleShare,
    ) -> Share {
        let masked = x_share.value * y_share.value + double_share.share_2t.value;
        Share {
            point: x_share.point,
            value: masked,
        }
    }

    pub fn king_reconstruct(&self, masked_shares: &[Share]) -> Result<Fp> {
        if masked_shares.len() < 2 * self.t + 1 {
            return Err(ProtocolError::MultiplyError(format!(
                "need >= {} masked shares, got {}",
                2 * self.t + 1,
                masked_shares.len()
            )));
        }
        let shamir_2t = Shamir::new(self.n, 2 * self.t)?;
        shamir_2t.reconstruct(masked_shares)
    }

    pub fn compute_output_share(opened_value: Fp, double_share: &DoubleShare) -> Share {
        Share {
            point: double_share.share_t.point,
            value: opened_value - double_share.share_t.value,
        }
    }

    pub fn verify_king_broadcast(
        &self,
        masked_shares: &[Share],
        claimed_value: Fp,
    ) -> Result<()> {
        if masked_shares.len() < 2 * self.t + 1 {
            return Err(ProtocolError::MultiplyError(format!(
                "need >= {} masked shares for verification, got {}",
                2 * self.t + 1,
                masked_shares.len()
            )));
        }

        let shamir_2t = Shamir::new(self.n, 2 * self.t)?;
        let reconstructed = shamir_2t.reconstruct(masked_shares)?;
        if reconstructed != claimed_value {
            return Err(ProtocolError::MaliciousParty(format!(
                "king broadcast verification failed: reconstructed {} but king claimed {}",
                reconstructed, claimed_value
            )));
        }
        Ok(())
    }

    pub fn multiply_local(
        &self,
        x_shares: &[Share],
        y_shares: &[Share],
        double_shares: &[DoubleShare],
    ) -> Result<Vec<Share>> {
        if x_shares.len() != self.n || y_shares.len() != self.n || double_shares.len() != self.n {
            return Err(ProtocolError::InvalidParams(format!(
                "expected {} shares, got x={}, y={}, ds={}",
                self.n,
                x_shares.len(),
                y_shares.len(),
                double_shares.len()
            )));
        }

        let masked_shares: Vec<Share> = (0..self.n)
            .map(|i| Self::compute_masked_share(&x_shares[i], &y_shares[i], &double_shares[i]))
            .collect();

        let opened = self.king_reconstruct(&masked_shares)?;

        Ok((0..self.n)
            .map(|i| Self::compute_output_share(opened, &double_shares[i]))
            .collect())
    }
}

pub fn multiply_sequence(
    n: usize,
    t: usize,
    _king: usize,
    value_shares: &[Vec<Share>],
    double_shares: &[Vec<DoubleShare>],
) -> Result<Vec<Share>> {
    let m = value_shares.len();
    if m < 2 {
        return Err(ProtocolError::InvalidParams(
            "need at least 2 values to multiply".into(),
        ));
    }
    if double_shares.len() != m - 1 {
        return Err(ProtocolError::InvalidParams(format!(
            "need {} double shares for {} multiplications, got {}",
            m - 1,
            m - 1,
            double_shares.len()
        )));
    }

    let shamir_2t = Shamir::new(n, 2 * t)?;
    let lagrange = shamir_2t.lagrange_coefficients();
    let mut current: Vec<Share> = value_shares[0].clone();

    for i in 1..m {
        let mut opened = Fp::ZERO;
        for p in 0..n {
            let masked = current[p].value * value_shares[i][p].value
                + double_shares[i - 1][p].share_2t.value;
            opened += lagrange[p] * masked;
        }

        for p in 0..n {
            current[p] = Share {
                point: double_shares[i - 1][p].share_t.point,
                value: opened - double_shares[i - 1][p].share_t.value,
            };
        }
    }

    Ok(current)
}

pub fn multiply_sequence_party_indexed(
    n: usize,
    t: usize,
    value_shares: &[Vec<Share>],
    party_double_shares: &[Vec<DoubleShare>],
) -> Result<Vec<Share>> {
    let m = value_shares.len();
    if m < 2 {
        return Err(ProtocolError::InvalidParams(
            "need at least 2 values to multiply".into(),
        ));
    }
    let num_mults = m - 1;
    if party_double_shares.len() != n {
        return Err(ProtocolError::InvalidParams(format!(
            "need {} party share vectors, got {}",
            n,
            party_double_shares.len()
        )));
    }
    for p in 0..n {
        if party_double_shares[p].len() < num_mults {
            return Err(ProtocolError::InvalidParams(format!(
                "party {} has {} double shares, need {}",
                p,
                party_double_shares[p].len(),
                num_mults
            )));
        }
    }

    let shamir_2t = Shamir::new(n, 2 * t)?;
    let lagrange = shamir_2t.lagrange_coefficients();
    let mut current: Vec<Share> = value_shares[0].clone();

    for i in 1..m {
        let k = i - 1;
        let mut opened = Fp::ZERO;
        for p in 0..n {
            let masked = current[p].value * value_shares[i][p].value
                + party_double_shares[p][k].share_2t.value;
            opened += lagrange[p] * masked;
        }

        for p in 0..n {
            current[p] = Share {
                point: party_double_shares[p][k].share_t.point,
                value: opened - party_double_shares[p][k].share_t.value,
            };
        }
    }

    Ok(current)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::randousha::{RanDouShaParams, RanDouShaProtocol};
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn test_dn_multiply_basic() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let n = 5;
        let t = 1;
        let shamir_t = Shamir::new(n, t).unwrap();

        let x = Fp::new(7);
        let y = Fp::new(6);
        let expected = x * y;

        let x_shares = shamir_t.share(x, &mut rng);
        let y_shares = shamir_t.share(y, &mut rng);

        let params = RanDouShaParams::new(n, t, 1).unwrap();
        let protocol = RanDouShaProtocol::new(params);
        let party_ds = protocol.generate_local(&mut rng).unwrap();
        let double_shares: Vec<DoubleShare> = (0..n).map(|p| party_ds[p][0].clone()).collect();

        let dn = DnMultiply::new(n, t, 0).unwrap();
        let result_shares = dn.multiply_local(&x_shares, &y_shares, &double_shares).unwrap();

        let result = shamir_t.reconstruct(&result_shares).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_dn_multiply_with_zero() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let n = 5;
        let t = 1;
        let shamir_t = Shamir::new(n, t).unwrap();

        let x_shares = shamir_t.share(Fp::new(42), &mut rng);
        let y_shares = shamir_t.share(Fp::ZERO, &mut rng);

        let params = RanDouShaParams::new(n, t, 1).unwrap();
        let party_ds = RanDouShaProtocol::new(params).generate_local(&mut rng).unwrap();
        let ds: Vec<DoubleShare> = (0..n).map(|p| party_ds[p][0].clone()).collect();

        let dn = DnMultiply::new(n, t, 0).unwrap();
        let result_shares = dn.multiply_local(&x_shares, &y_shares, &ds).unwrap();
        let result = shamir_t.reconstruct(&result_shares).unwrap();
        assert_eq!(result, Fp::ZERO);
    }

    #[test]
    fn test_multiply_sequence() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let n = 5;
        let t = 1;
        let shamir_t = Shamir::new(n, t).unwrap();

        let values = [Fp::new(2), Fp::new(3), Fp::new(4), Fp::new(5)];
        let expected: Fp = values.iter().copied().reduce(|a, b| a * b).unwrap();
        assert_eq!(expected, Fp::new(120));

        let value_shares: Vec<Vec<Share>> =
            values.iter().map(|v| shamir_t.share(*v, &mut rng)).collect();

        let params = RanDouShaParams::new(n, t, values.len() - 1).unwrap();
        let party_ds = RanDouShaProtocol::new(params).generate_local(&mut rng).unwrap();
        let double_shares: Vec<Vec<DoubleShare>> = (0..(values.len() - 1))
            .map(|k| (0..n).map(|p| party_ds[p][k].clone()).collect())
            .collect();

        let result_shares = multiply_sequence(n, t, 0, &value_shares, &double_shares).unwrap();
        let result = shamir_t.reconstruct(&result_shares).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_multiply_large_values() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let n = 5;
        let t = 1;
        let shamir_t = Shamir::new(n, t).unwrap();

        let x = Fp::new(1_000_000);
        let y = Fp::new(2_000_000);
        let expected = x * y;

        let x_shares = shamir_t.share(x, &mut rng);
        let y_shares = shamir_t.share(y, &mut rng);

        let params = RanDouShaParams::new(n, t, 1).unwrap();
        let party_ds = RanDouShaProtocol::new(params).generate_local(&mut rng).unwrap();
        let ds: Vec<DoubleShare> = (0..n).map(|p| party_ds[p][0].clone()).collect();

        let dn = DnMultiply::new(n, t, 0).unwrap();
        let result_shares = dn.multiply_local(&x_shares, &y_shares, &ds).unwrap();
        let result = shamir_t.reconstruct(&result_shares).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_chained_multiplication_local() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let n = 5;
        let t = 1;
        let shamir_t = Shamir::new(n, t).unwrap();

        let num_values = 20;
        let values: Vec<Fp> = (2..2 + num_values as u64).map(Fp::new).collect();
        let expected: Fp = values.iter().copied().reduce(|a, b| a * b).unwrap();

        let value_shares: Vec<Vec<Share>> =
            values.iter().map(|v| shamir_t.share(*v, &mut rng)).collect();

        let params = RanDouShaParams::new(n, t, num_values - 1).unwrap();
        let party_ds = RanDouShaProtocol::new(params).generate_local(&mut rng).unwrap();
        let double_shares: Vec<Vec<DoubleShare>> = (0..(num_values - 1))
            .map(|k| (0..n).map(|p| party_ds[p][k].clone()).collect())
            .collect();

        let result_shares = multiply_sequence(n, t, 0, &value_shares, &double_shares).unwrap();
        let result = shamir_t.reconstruct(&result_shares).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_mass_chained_multiplication_50() {
        let mut rng = ChaCha20Rng::seed_from_u64(777);
        let n = 5;
        let t = 1;
        let shamir_t = Shamir::new(n, t).unwrap();

        let num_values = 50;
        let values: Vec<Fp> = (1..=num_values as u64).map(Fp::new).collect();
        let expected: Fp = values.iter().copied().reduce(|a, b| a * b).unwrap();

        let value_shares: Vec<Vec<Share>> =
            values.iter().map(|v| shamir_t.share(*v, &mut rng)).collect();

        let params = RanDouShaParams::new(n, t, num_values - 1).unwrap();
        let party_ds = RanDouShaProtocol::new(params).generate_local(&mut rng).unwrap();
        let double_shares: Vec<Vec<DoubleShare>> = (0..(num_values - 1))
            .map(|k| (0..n).map(|p| party_ds[p][k].clone()).collect())
            .collect();

        let result_shares = multiply_sequence(n, t, 0, &value_shares, &double_shares).unwrap();

        let result = shamir_t.reconstruct(&result_shares).unwrap();
        assert_eq!(result, expected, "50! mod p mismatch");
    }

    #[test]
    fn test_verify_king_broadcast_honest() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let n = 5;
        let t = 1;
        let shamir_t = Shamir::new(n, t).unwrap();

        let x = Fp::new(7);
        let y = Fp::new(6);
        let x_shares = shamir_t.share(x, &mut rng);
        let y_shares = shamir_t.share(y, &mut rng);

        let params = RanDouShaParams::new(n, t, 1).unwrap();
        let party_ds = RanDouShaProtocol::new(params).generate_local(&mut rng).unwrap();
        let ds: Vec<DoubleShare> = (0..n).map(|p| party_ds[p][0].clone()).collect();

        let masked_shares: Vec<Share> = (0..n)
            .map(|i| DnMultiply::compute_masked_share(&x_shares[i], &y_shares[i], &ds[i]))
            .collect();

        let dn = DnMultiply::new(n, t, 0).unwrap();
        let opened = dn.king_reconstruct(&masked_shares).unwrap();

        dn.verify_king_broadcast(&masked_shares, opened).unwrap();
    }

    #[test]
    fn test_verify_king_broadcast_detects_lying_king() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let n = 5;
        let t = 1;
        let shamir_t = Shamir::new(n, t).unwrap();

        let x = Fp::new(7);
        let y = Fp::new(6);
        let x_shares = shamir_t.share(x, &mut rng);
        let y_shares = shamir_t.share(y, &mut rng);

        let params = RanDouShaParams::new(n, t, 1).unwrap();
        let party_ds = RanDouShaProtocol::new(params).generate_local(&mut rng).unwrap();
        let ds: Vec<DoubleShare> = (0..n).map(|p| party_ds[p][0].clone()).collect();

        let masked_shares: Vec<Share> = (0..n)
            .map(|i| DnMultiply::compute_masked_share(&x_shares[i], &y_shares[i], &ds[i]))
            .collect();

        let dn = DnMultiply::new(n, t, 0).unwrap();
        let opened = dn.king_reconstruct(&masked_shares).unwrap();

        let wrong_value = opened + Fp::new(1);
        let result = dn.verify_king_broadcast(&masked_shares, wrong_value);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("king broadcast verification failed"));
    }

    #[test]
    fn test_invalid_params() {
        assert!(DnMultiply::new(5, 1, 5).is_err());
        assert!(DnMultiply::new(2, 1, 0).is_err());
    }
}
