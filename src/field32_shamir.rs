//! Shamir secret sharing over Fp32 (p = 2^31-1).

use crate::error::{ProtocolError, Result};
use crate::field32::Fp32;
use rand::Rng;

#[derive(Clone, Copy, Debug)]
pub struct Share32 {
    pub point: Fp32,
    pub value: Fp32,
}

pub struct Shamir32 {
    pub degree: usize,
    pub n: usize,
    pub eval_points: Vec<Fp32>,
    lag_at_zero: Vec<Fp32>,
}

impl Shamir32 {
    pub fn new(n: usize, degree: usize) -> Result<Self> {
        if n <= degree {
            return Err(ProtocolError::InvalidParams(format!(
                "Shamir32 requires n > degree, got n={}, degree={}",
                n, degree
            )));
        }
        let eval_points: Vec<Fp32> = (1..=n as u32).map(Fp32::new).collect();
        let lag_at_zero = lagrange_coefficients_at_zero_32(&eval_points);
        Ok(Shamir32 {
            degree,
            n,
            eval_points,
            lag_at_zero,
        })
    }

    pub fn share<R: Rng>(&self, secret: Fp32, rng: &mut R) -> Vec<Share32> {
        let mut coeffs = Vec::with_capacity(self.degree + 1);
        coeffs.push(secret);
        for _ in 0..self.degree {
            coeffs.push(Fp32::random(rng));
        }
        self.eval_points
            .iter()
            .map(|&point| Share32 {
                point,
                value: eval_poly_32(&coeffs, point),
            })
            .collect()
    }

    pub fn reconstruct(&self, shares: &[Share32]) -> Result<Fp32> {
        if shares.len() < self.degree + 1 {
            return Err(ProtocolError::ShamirError(format!(
                "need at least {} shares, got {}",
                self.degree + 1,
                shares.len()
            )));
        }
        lagrange_interpolate_at_zero_32(shares)
    }

    #[inline]
    pub fn reconstruct_raw(&self, values: &[Fp32]) -> Fp32 {
        debug_assert_eq!(values.len(), self.n);
        let mut result = Fp32::ZERO;
        for i in 0..self.n {
            result = result + values[i] * self.lag_at_zero[i];
        }
        result
    }

    #[inline]
    pub fn lagrange_coefficients(&self) -> &[Fp32] {
        &self.lag_at_zero
    }
}

fn eval_poly_32(coeffs: &[Fp32], x: Fp32) -> Fp32 {
    let mut result = Fp32::ZERO;
    for &c in coeffs.iter().rev() {
        result = result * x + c;
    }
    result
}

fn lagrange_coefficients_at_zero_32(points: &[Fp32]) -> Vec<Fp32> {
    let n = points.len();
    let mut coeffs = Vec::with_capacity(n);
    for i in 0..n {
        let mut num = Fp32::ONE;
        let mut den = Fp32::ONE;
        for j in 0..n {
            if i == j {
                continue;
            }
            num = num * (-points[j]);
            den = den * (points[i] - points[j]);
        }
        coeffs.push(num * den.inv());
    }
    coeffs
}

fn lagrange_interpolate_at_zero_32(shares: &[Share32]) -> Result<Fp32> {
    let n = shares.len();
    let mut result = Fp32::ZERO;
    for i in 0..n {
        let xi = shares[i].point;
        let yi = shares[i].value;
        let mut num = Fp32::ONE;
        let mut den = Fp32::ONE;
        for j in 0..n {
            if i == j {
                continue;
            }
            let xj = shares[j].point;
            num = num * (-xj);
            den = den * (xi - xj);
        }
        result = result + yi * num * den.inv();
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn test_shamir32_share_reconstruct() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let shamir = Shamir32::new(5, 1).unwrap();

        for _ in 0..100 {
            let secret = Fp32::random(&mut rng);
            let shares = shamir.share(secret, &mut rng);
            let recovered = shamir.reconstruct(&shares).unwrap();
            assert_eq!(secret, recovered);
        }
    }

    #[test]
    fn test_shamir32_reconstruct_raw() {
        let mut rng = ChaCha20Rng::seed_from_u64(43);
        let shamir = Shamir32::new(5, 1).unwrap();

        for _ in 0..100 {
            let secret = Fp32::random(&mut rng);
            let shares = shamir.share(secret, &mut rng);
            let values: Vec<Fp32> = shares.iter().map(|s| s.value).collect();
            let recovered = shamir.reconstruct_raw(&values);
            assert_eq!(secret, recovered);
        }
    }

    #[test]
    fn test_shamir32_linearity() {
        let mut rng = ChaCha20Rng::seed_from_u64(44);
        let shamir = Shamir32::new(5, 1).unwrap();

        let a = Fp32::random(&mut rng);
        let b = Fp32::random(&mut rng);
        let shares_a = shamir.share(a, &mut rng);
        let shares_b = shamir.share(b, &mut rng);

        // Sum of shares should reconstruct to sum of secrets
        let shares_sum: Vec<Share32> = shares_a
            .iter()
            .zip(&shares_b)
            .map(|(sa, sb)| Share32 {
                point: sa.point,
                value: sa.value + sb.value,
            })
            .collect();
        let recovered = shamir.reconstruct(&shares_sum).unwrap();
        assert_eq!(recovered, a + b);
    }
}
