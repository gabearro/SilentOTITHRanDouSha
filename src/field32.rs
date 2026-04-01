use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

pub const PRIME32: u32 = (1u32 << 31) - 1;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct Fp32(u32);

impl Fp32 {
    pub const ZERO: Fp32 = Fp32(0);
    pub const ONE: Fp32 = Fp32(1);

    #[inline]
    pub fn new(v: u32) -> Self {
        Fp32(Self::reduce(v))
    }

    #[inline]
    pub fn from_raw(v: u32) -> Self {
        assert!(v < PRIME32, "from_raw: value {} >= PRIME32", v);
        Fp32(v)
    }

    #[inline]
    pub fn val(self) -> u32 {
        self.0
    }

    #[inline]
    pub fn raw(self) -> u32 {
        self.0
    }

    #[inline]
    pub fn from_reduced(v: u32) -> Self {
        debug_assert!(v < PRIME32, "from_reduced: value {} >= PRIME32", v);
        Fp32(v)
    }

    #[inline]
    pub fn reduce(x: u32) -> u32 {
        let mut r = (x >> 31) + (x & PRIME32);
        if r >= PRIME32 {
            r -= PRIME32;
        }
        r
    }

    #[inline]
    pub fn reduce_wide(x: u64) -> u32 {
        let lo = (x & PRIME32 as u64) as u32;
        let hi = (x >> 31) as u32;
        let mut r = lo.wrapping_add(hi);
        // After adding lo (<2^31) + hi (could be up to ~2^33 for products of reduced values),
        // we may need two reduction steps.
        r = (r >> 31) + (r & PRIME32);
        if r >= PRIME32 {
            r -= PRIME32;
        }
        r
    }

    #[inline]
    pub fn mul_raw(a: u32, b: u32) -> u32 {
        Self::reduce_wide((a as u64) * (b as u64))
    }

    #[inline]
    pub fn add_raw(a: u32, b: u32) -> u32 {
        let r = a + b;
        if r >= PRIME32 {
            r - PRIME32
        } else {
            r
        }
    }

    #[inline]
    pub fn sub_raw(a: u32, b: u32) -> u32 {
        if a >= b {
            a - b
        } else {
            PRIME32 - b + a
        }
    }

    pub fn inv(self) -> Self {
        assert_ne!(self.0, 0, "cannot invert zero in Fp32");
        self.pow(PRIME32 - 2)
    }

    pub fn pow(self, mut exp: u32) -> Self {
        let mut base = self;
        let mut result = Fp32::ONE;
        while exp > 0 {
            if exp & 1 == 1 {
                result *= base;
            }
            base = base * base;
            exp >>= 1;
        }
        result
    }

    pub fn random<R: Rng>(rng: &mut R) -> Self {
        loop {
            let v: u32 = rng.gen::<u32>() >> 1;
            if v < PRIME32 {
                return Fp32(v);
            }
        }
    }
}

impl fmt::Debug for Fp32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Fp32({})", self.0)
    }
}

impl fmt::Display for Fp32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<u32> for Fp32 {
    fn from(v: u32) -> Self {
        Fp32::new(v)
    }
}

impl Sum for Fp32 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Fp32::ZERO, |acc, x| acc + x)
    }
}

impl Add for Fp32 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Fp32(Self::add_raw(self.0, rhs.0))
    }
}

impl AddAssign for Fp32 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Fp32 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Fp32(Self::sub_raw(self.0, rhs.0))
    }
}

impl SubAssign for Fp32 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul for Fp32 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Fp32(Self::mul_raw(self.0, rhs.0))
    }
}

impl MulAssign for Fp32 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Neg for Fp32 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        if self.0 == 0 {
            self
        } else {
            Fp32(PRIME32 - self.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_arithmetic() {
        let a = Fp32::new(100);
        let b = Fp32::new(200);
        assert_eq!((a + b).val(), 300);
        assert_eq!((b - a).val(), 100);
        assert_eq!((a * b).val(), 20000);
    }

    #[test]
    fn test_reduce() {
        assert_eq!(Fp32::reduce(0), 0);
        assert_eq!(Fp32::reduce(PRIME32), 0);
        assert_eq!(Fp32::reduce(PRIME32 + 1), 1);
        assert_eq!(Fp32::reduce(u32::MAX), Fp32::reduce(u32::MAX));
    }

    #[test]
    fn test_reduce_wide() {
        assert_eq!(Fp32::reduce_wide(0), 0);
        assert_eq!(Fp32::reduce_wide(PRIME32 as u64), 0);
        assert_eq!(Fp32::reduce_wide((PRIME32 as u64) * (PRIME32 as u64)), 0);
        // (PRIME32-1)^2 mod PRIME32 should equal 1
        let p_minus_1 = (PRIME32 - 1) as u64;
        assert_eq!(Fp32::reduce_wide(p_minus_1 * p_minus_1), 1);
    }

    #[test]
    fn test_mul_raw() {
        // Small values
        assert_eq!(Fp32::mul_raw(3, 7), 21);
        // (PRIME32-1) * (PRIME32-1) = (-1)*(-1) = 1
        assert_eq!(Fp32::mul_raw(PRIME32 - 1, PRIME32 - 1), 1);
        // x * 0 = 0
        assert_eq!(Fp32::mul_raw(12345, 0), 0);
        // x * 1 = x
        assert_eq!(Fp32::mul_raw(12345, 1), 12345);
    }

    #[test]
    fn test_inverse() {
        let a = Fp32::new(42);
        let a_inv = a.inv();
        assert_eq!(a * a_inv, Fp32::ONE);

        let b = Fp32::new(PRIME32 - 1);
        let b_inv = b.inv();
        assert_eq!(b * b_inv, Fp32::ONE);
    }

    #[test]
    fn test_negation() {
        let a = Fp32::new(42);
        assert_eq!(a + (-a), Fp32::ZERO);
        assert_eq!((-Fp32::ZERO), Fp32::ZERO);
    }

    #[test]
    fn test_pow() {
        let a = Fp32::new(2);
        assert_eq!(a.pow(10).val(), 1024);
        assert_eq!(a.pow(0), Fp32::ONE);
    }

    #[test]
    fn test_random_in_range() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha20Rng;
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        for _ in 0..10000 {
            let v = Fp32::random(&mut rng);
            assert!(v.val() < PRIME32);
        }
    }

    #[test]
    fn test_sum_trait() {
        let vals = vec![Fp32::new(1), Fp32::new(2), Fp32::new(3)];
        let s: Fp32 = vals.into_iter().sum();
        assert_eq!(s.val(), 6);
    }

    #[test]
    #[should_panic]
    fn test_from_raw_panics() {
        Fp32::from_raw(PRIME32);
    }
}
