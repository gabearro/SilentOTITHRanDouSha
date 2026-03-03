use crate::error::{ProtocolError, Result};
use crate::field::Fp;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Share {
    pub point: Fp,
    pub value: Fp,
}

pub struct Shamir {
    pub degree: usize,
    pub n: usize,
    pub eval_points: Vec<Fp>,
    lagrange_all_at_zero: Vec<Fp>,
}

impl Shamir {
    pub fn new(n: usize, degree: usize) -> Result<Self> {
        if n <= degree {
            return Err(ProtocolError::InvalidParams(format!(
                "Shamir requires n > degree, got n={}, degree={}",
                n, degree
            )));
        }
        let eval_points: Vec<Fp> = (1..=n as u64).map(Fp::new).collect();
        let lagrange_all_at_zero = lagrange_coefficients_at_zero(&eval_points);
        Ok(Shamir {
            degree,
            n,
            eval_points,
            lagrange_all_at_zero,
        })
    }

    pub fn share<R: Rng>(&self, secret: Fp, rng: &mut R) -> Vec<Share> {
        let mut coeffs = Vec::with_capacity(self.degree + 1);
        coeffs.push(secret);
        for _ in 0..self.degree {
            coeffs.push(Fp::random(rng));
        }
        self.eval_points
            .iter()
            .map(|&point| {
                let value = eval_poly(&coeffs, point);
                Share { point, value }
            })
            .collect()
    }

    pub fn share_with_poly(&self, coeffs: &[Fp]) -> Result<Vec<Share>> {
        if coeffs.len() != self.degree + 1 {
            return Err(ProtocolError::InvalidParams(format!(
                "expected {} coefficients, got {}",
                self.degree + 1,
                coeffs.len()
            )));
        }
        Ok(self
            .eval_points
            .iter()
            .map(|&point| {
                let value = eval_poly(coeffs, point);
                Share { point, value }
            })
            .collect())
    }

    pub fn reconstruct(&self, shares: &[Share]) -> Result<Fp> {
        if shares.len() < self.degree + 1 {
            return Err(ProtocolError::ShamirError(format!(
                "need at least {} shares, got {}",
                self.degree + 1,
                shares.len()
            )));
        }
        lagrange_interpolate_at_zero(shares)
    }

    #[inline]
    pub fn lagrange_coefficients(&self) -> &[Fp] {
        &self.lagrange_all_at_zero
    }

    #[inline]
    pub fn reconstruct_all_values(&self, values: &[Fp]) -> Fp {
        debug_assert_eq!(values.len(), self.n);
        values
            .iter()
            .zip(&self.lagrange_all_at_zero)
            .map(|(&v, &c)| v * c)
            .sum()
    }
}

fn eval_poly(coeffs: &[Fp], x: Fp) -> Fp {
    let mut result = Fp::ZERO;
    for &c in coeffs.iter().rev() {
        result = result * x + c;
    }
    result
}

pub fn lagrange_interpolate_at_zero(shares: &[Share]) -> Result<Fp> {
    let n = shares.len();

    for i in 0..n {
        for j in (i + 1)..n {
            if shares[i].point == shares[j].point {
                return Err(ProtocolError::MaliciousParty(format!(
                    "duplicate evaluation point {} at indices {} and {}",
                    shares[i].point, i, j
                )));
            }
        }
    }

    let mut result = Fp::ZERO;
    for i in 0..n {
        let xi = shares[i].point;
        let yi = shares[i].value;

        let mut num = Fp::ONE;
        let mut den = Fp::ONE;
        for j in 0..n {
            if i == j {
                continue;
            }
            let xj = shares[j].point;
            num *= -xj;
            den *= xi - xj;
        }
        result += yi * num * den.inv();
    }

    Ok(result)
}

pub fn lagrange_interpolate_at(shares: &[Share], target: Fp) -> Result<Fp> {
    let n = shares.len();

    for i in 0..n {
        for j in (i + 1)..n {
            if shares[i].point == shares[j].point {
                return Err(ProtocolError::MaliciousParty(format!(
                    "duplicate evaluation point {} at indices {} and {}",
                    shares[i].point, i, j
                )));
            }
        }
    }

    let mut result = Fp::ZERO;

    for i in 0..n {
        let xi = shares[i].point;
        let yi = shares[i].value;

        let mut num = Fp::ONE;
        let mut den = Fp::ONE;
        for j in 0..n {
            if i == j {
                continue;
            }
            let xj = shares[j].point;
            num *= target - xj;
            den *= xi - xj;
        }
        result += yi * num * den.inv();
    }

    Ok(result)
}

pub fn lagrange_coefficients_at_zero(points: &[Fp]) -> Vec<Fp> {
    let n = points.len();
    let mut coeffs = Vec::with_capacity(n);

    for i in 0..n {
        let mut num = Fp::ONE;
        let mut den = Fp::ONE;
        for j in 0..n {
            if i == j {
                continue;
            }
            num *= -points[j];
            den *= points[i] - points[j];
        }
        coeffs.push(num * den.inv());
    }

    coeffs
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn test_share_and_reconstruct() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let secret = Fp::new(12345);
        let shamir = Shamir::new(5, 2).unwrap();

        let shares = shamir.share(secret, &mut rng);
        assert_eq!(shares.len(), 5);

        let recovered = shamir.reconstruct(&shares).unwrap();
        assert_eq!(recovered, secret);

        let recovered = shamir.reconstruct(&shares[0..3]).unwrap();
        assert_eq!(recovered, secret);
    }

    #[test]
    fn test_share_and_reconstruct_degree1() {
        let mut rng = ChaCha20Rng::seed_from_u64(99);
        let secret = Fp::new(999);
        let shamir = Shamir::new(5, 1).unwrap();

        let shares = shamir.share(secret, &mut rng);
        let recovered = shamir.reconstruct(&shares[0..2]).unwrap();
        assert_eq!(recovered, secret);
    }

    #[test]
    fn test_additive_homomorphism() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let a = Fp::new(100);
        let b = Fp::new(200);
        let shamir = Shamir::new(5, 2).unwrap();

        let shares_a = shamir.share(a, &mut rng);
        let shares_b = shamir.share(b, &mut rng);

        let shares_sum: Vec<Share> = shares_a
            .iter()
            .zip(shares_b.iter())
            .map(|(sa, sb)| Share {
                point: sa.point,
                value: sa.value + sb.value,
            })
            .collect();

        let recovered = shamir.reconstruct(&shares_sum).unwrap();
        assert_eq!(recovered, a + b);
    }

    #[test]
    fn test_multiplicative_degree_doubling() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let a = Fp::new(7);
        let b = Fp::new(6);
        let shamir_t = Shamir::new(5, 1).unwrap();
        let shamir_2t = Shamir::new(5, 2).unwrap();

        let shares_a = shamir_t.share(a, &mut rng);
        let shares_b = shamir_t.share(b, &mut rng);

        let shares_prod: Vec<Share> = shares_a
            .iter()
            .zip(shares_b.iter())
            .map(|(sa, sb)| Share {
                point: sa.point,
                value: sa.value * sb.value,
            })
            .collect();

        let recovered = shamir_2t.reconstruct(&shares_prod).unwrap();
        assert_eq!(recovered, a * b);
    }

    #[test]
    fn test_lagrange_coefficients() {
        let points: Vec<Fp> = (1..=5u64).map(Fp::new).collect();
        let coeffs = lagrange_coefficients_at_zero(&points[0..3]);

        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let secret = Fp::new(54321);
        let shamir = Shamir::new(5, 2).unwrap();
        let shares = shamir.share(secret, &mut rng);

        let mut recovered = Fp::ZERO;
        for i in 0..3 {
            recovered += coeffs[i] * shares[i].value;
        }
        assert_eq!(recovered, secret);
    }

    #[test]
    fn test_invalid_params() {
        assert!(Shamir::new(2, 2).is_err());
        assert!(Shamir::new(1, 1).is_err());
        assert!(Shamir::new(3, 2).is_ok());
    }

    #[test]
    fn test_insufficient_shares() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let shamir = Shamir::new(5, 2).unwrap();
        let shares = shamir.share(Fp::new(42), &mut rng);
        assert!(shamir.reconstruct(&shares[0..2]).is_err());
    }

    #[test]
    fn test_duplicate_points_detected() {
        let shares = vec![
            Share { point: Fp::new(1), value: Fp::new(10) },
            Share { point: Fp::new(1), value: Fp::new(20) },
            Share { point: Fp::new(3), value: Fp::new(30) },
        ];
        let result = lagrange_interpolate_at_zero(&shares);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("duplicate"));
    }
}
