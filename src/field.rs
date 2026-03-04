use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

pub const PRIME: u64 = (1u64 << 61) - 1;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct Fp(u64);

impl Fp {
    pub const ZERO: Fp = Fp(0);
    pub const ONE: Fp = Fp(1);

    #[inline]
    pub fn new(v: u64) -> Self {
        Fp(Self::reduce(v))
    }

    #[inline]
    pub fn from_raw(v: u64) -> Self {
        assert!(v < PRIME, "from_raw: value {} >= PRIME", v);
        Fp(v)
    }

    #[inline]
    pub fn val(self) -> u64 {
        self.0
    }

    /// Return the inner u64 value (already reduced mod p).
    #[inline]
    pub fn raw(self) -> u64 {
        self.0
    }

    /// Construct from a value that is *already* reduced (i.e. < PRIME).
    /// No reduction is performed — caller must guarantee `v < PRIME`.
    #[inline]
    pub fn from_reduced(v: u64) -> Self {
        debug_assert!(v < PRIME, "from_reduced: value {} >= PRIME", v);
        Fp(v)
    }

    /// Multiply two already-reduced u64 values mod p, returning a reduced u64.
    #[inline]
    pub fn mul_raw(a: u64, b: u64) -> u64 {
        Self::reduce_wide((a as u128) * (b as u128))
    }

    /// Add two already-reduced u64 values mod p, returning a reduced u64.
    #[inline]
    pub fn add_raw(a: u64, b: u64) -> u64 {
        let r = a + b;
        if r >= PRIME { r - PRIME } else { r }
    }

    /// Subtract two already-reduced u64 values mod p, returning a reduced u64.
    #[inline]
    pub fn sub_raw(a: u64, b: u64) -> u64 {
        if a >= b { a - b } else { PRIME - b + a }
    }

    #[inline]
    pub fn reduce(x: u64) -> u64 {
        let mut r = (x >> 61) + (x & PRIME);
        if r >= PRIME {
            r -= PRIME;
        }
        r
    }

    /// Reduce a wide 128-bit product mod p = 2^61 - 1.
    #[inline]
    pub fn reduce_wide(x: u128) -> u64 {
        let lo = (x & (PRIME as u128)) as u64;
        let mid = ((x >> 61) & (PRIME as u128)) as u64;
        let hi = (x >> 122) as u64;
        let mut r = lo + mid + hi;
        r = (r >> 61) + (r & PRIME);
        if r >= PRIME {
            r -= PRIME;
        }
        r
    }

    pub fn inv(self) -> Self {
        assert_ne!(self.0, 0, "cannot invert zero in Fp");
        self.pow(PRIME - 2)
    }

    pub fn try_inv(self) -> Option<Self> {
        if self.0 == 0 {
            None
        } else {
            Some(self.pow(PRIME - 2))
        }
    }

    pub fn pow(self, mut exp: u64) -> Self {
        let mut base = self;
        let mut result = Fp::ONE;
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
            let v: u64 = rng.gen::<u64>() >> 3;
            if v < PRIME {
                return Fp(v);
            }
        }
    }

    #[inline]
    pub fn random_batch_raw<R: Rng>(rng: &mut R, out: &mut [u64]) {
        let byte_len = out.len() * 8;
        let byte_slice = unsafe {
            std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, byte_len)
        };
        rng.fill_bytes(byte_slice);
        for v in out.iter_mut() {
            *v >>= 3;
            if *v >= PRIME {
                *v -= PRIME;
            }
        }
    }
}

impl fmt::Debug for Fp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Fp({})", self.0)
    }
}

impl fmt::Display for Fp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<u64> for Fp {
    fn from(v: u64) -> Self {
        Fp::new(v)
    }
}

impl Sum for Fp {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Fp::ZERO, |acc, x| acc + x)
    }
}

impl<'a> Sum<&'a Fp> for Fp {
    fn sum<I: Iterator<Item = &'a Fp>>(iter: I) -> Self {
        iter.fold(Fp::ZERO, |acc, &x| acc + x)
    }
}

impl Add for Fp {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut r = self.0 + rhs.0;
        if r >= PRIME {
            r -= PRIME;
        }
        Fp(r)
    }
}

impl AddAssign for Fp {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Fp {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let r = if self.0 >= rhs.0 {
            self.0 - rhs.0
        } else {
            PRIME - rhs.0 + self.0
        };
        Fp(r)
    }
}

impl SubAssign for Fp {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul for Fp {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let wide = (self.0 as u128) * (rhs.0 as u128);
        Fp(Self::reduce_wide(wide))
    }
}

impl MulAssign for Fp {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Neg for Fp {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        if self.0 == 0 {
            self
        } else {
            Fp(PRIME - self.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_arithmetic() {
        let a = Fp::new(42);
        let b = Fp::new(17);
        assert_eq!((a + b).val(), 59);
        assert_eq!((a - b).val(), 25);
        assert_eq!((a * b).val(), 714);
    }

    #[test]
    fn test_inverse() {
        let a = Fp::new(42);
        let a_inv = a.inv();
        assert_eq!((a * a_inv).val(), 1);
    }

    #[test]
    fn test_try_inv_zero() {
        assert!(Fp::ZERO.try_inv().is_none());
        assert!(Fp::new(42).try_inv().is_some());
    }

    #[test]
    fn test_negation() {
        let a = Fp::new(42);
        assert_eq!((a + (-a)).val(), 0);
    }

    #[test]
    fn test_zero_one() {
        let a = Fp::new(123456);
        assert_eq!((a + Fp::ZERO).val(), a.val());
        assert_eq!((a * Fp::ONE).val(), a.val());
        assert_eq!((a * Fp::ZERO).val(), 0);
    }

    #[test]
    fn test_large_values() {
        let a = Fp::new(PRIME - 1);
        let b = Fp::new(PRIME - 1);
        assert_eq!((a * b).val(), 1);
    }

    #[test]
    fn test_reduce_identity() {
        assert_eq!(Fp::new(0).val(), 0);
        assert_eq!(Fp::new(PRIME).val(), 0);
        assert_eq!(Fp::new(PRIME + 1).val(), 1);
    }

    #[test]
    fn test_sum_trait() {
        let vals = [Fp::new(1), Fp::new(2), Fp::new(3)];
        let total: Fp = vals.iter().sum();
        assert_eq!(total, Fp::new(6));
    }

    #[test]
    #[should_panic(expected = "from_raw: value")]
    fn test_from_raw_panics_on_invalid() {
        Fp::from_raw(PRIME);
    }

    #[test]
    fn test_random_uniformity() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha20Rng;
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        for _ in 0..1000 {
            let v = Fp::random(&mut rng);
            assert!(v.val() < PRIME);
        }
    }
}
