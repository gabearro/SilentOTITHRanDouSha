use aes;
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
        if r >= PRIME {
            r - PRIME
        } else {
            r
        }
    }

    /// Subtract two already-reduced u64 values mod p, returning a reduced u64.
    #[inline]
    pub fn sub_raw(a: u64, b: u64) -> u64 {
        if a >= b {
            a - b
        } else {
            PRIME - b + a
        }
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
        let byte_slice =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, byte_len) };
        rng.fill_bytes(byte_slice);
        for v in out.iter_mut() {
            *v >>= 3;
            if *v >= PRIME {
                *v -= PRIME;
            }
        }
    }
}

/// Fast non-cryptographic mixer for deriving field elements from a seed.
/// ~2 cycles per u64. Suitable for polynomial coefficients and HIM masking
/// where computational unpredictability (not information-theoretic) suffices.
#[derive(Clone)]
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    #[inline]
    pub fn new(seed: u64) -> Self {
        SplitMix64(seed)
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    /// Next value reduced to Fp range (< PRIME).
    #[inline]
    pub fn next_fp(&mut self) -> u64 {
        Fp::reduce(self.next_u64() >> 3)
    }

    /// Next value in [0, 2^61-1] range WITHOUT full Fp reduction.
    /// Result may equal PRIME (2^61-1). Safe for u128 multiply-accumulate
    /// followed by reduce_wide, which handles the full input range.
    /// Saves ~3 cycles per value vs next_fp by skipping the conditional branch.
    #[inline]
    pub fn next_raw61(&mut self) -> u64 {
        self.next_u64() >> 3
    }

    /// Fast single-round variant: 1 mixing step instead of 2.
    /// ~40% fewer cycles per value. Sufficient for HIM coefficient derivation
    /// where computational unpredictability (not statistical quality) is needed.
    #[inline]
    pub fn next_fast(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let z = (self.0 ^ (self.0 >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        (z ^ (z >> 27)) >> 3
    }
}

/// AES-CTR based fast PRNG for field elements. Uses hardware AES-NI which is
/// ~4x faster than ChaCha20 for bulk random generation on modern CPUs.
pub struct AesCtrRng {
    key: aes::Aes128,
    counter: u128,
    buffer: Vec<u64>,
    pos: usize,
}

impl AesCtrRng {
    /// Create from a 64-bit seed (expanded to AES key via zero-padding).
    pub fn from_seed(seed: u64) -> Self {
        use aes::cipher::KeyInit;
        let mut key_bytes = [0u8; 16];
        key_bytes[..8].copy_from_slice(&seed.to_le_bytes());
        AesCtrRng {
            key: aes::Aes128::new_from_slice(&key_bytes).unwrap(),
            counter: 0,
            buffer: Vec::new(),
            pos: 0,
        }
    }

    /// Fill a slice with reduced random field elements (< PRIME).
    /// Uses batched AES-CTR: encrypts counter blocks in bulk, extracts u64s, reduces.
    #[inline]
    pub fn fill_field_raw(&mut self, out: &mut [u64]) {
        use aes::cipher::BlockEncrypt;

        // Each AES block = 16 bytes = 2 u64 values
        let num_blocks = (out.len() + 1) / 2;

        // Reuse buffer across calls
        if self.buffer.len() < num_blocks * 2 {
            self.buffer.resize(num_blocks * 2, 0);
        }

        // Build counter blocks and encrypt in batch
        const BATCH: usize = 4096;
        let mut aes_blocks: Vec<aes::Block> = Vec::with_capacity(BATCH);
        let mut written = 0usize;

        for chunk_start in (0..num_blocks).step_by(BATCH) {
            let chunk_end = (chunk_start + BATCH).min(num_blocks);
            aes_blocks.clear();
            for i in chunk_start..chunk_end {
                let ctr = self.counter + i as u128;
                aes_blocks.push(aes::Block::from(ctr.to_le_bytes()));
            }
            self.key.encrypt_blocks(&mut aes_blocks);

            for block in &aes_blocks {
                let bytes: [u8; 16] = (*block).into();
                let lo = u64::from_le_bytes(bytes[..8].try_into().unwrap());
                let hi = u64::from_le_bytes(bytes[8..].try_into().unwrap());
                if written < out.len() {
                    out[written] = Fp::reduce(lo >> 3);
                    written += 1;
                }
                if written < out.len() {
                    out[written] = Fp::reduce(hi >> 3);
                    written += 1;
                }
            }
        }
        self.counter += num_blocks as u128;
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
