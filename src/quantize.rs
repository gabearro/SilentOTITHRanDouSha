//! f32 ↔ Fp32 quantization for bridging ANE (fp16/f32) and MPC (Fp32 field).
//!
//! Scale factor 512: maps typical activation range [-32, 32] to integer range
//! [-16384, 16384]. Dot products of 64 scaled values: 64 × 512² × 25 = 419M < p = 2.15B.

use crate::field32::{Fp32, PRIME32};

const SCALE: f32 = 512.0;
const INV_SCALE: f32 = 1.0 / SCALE;
const HALF_P: u32 = PRIME32 / 2;

/// Quantize an f32 value to Fp32. Scales, rounds, maps negative values to p - |v|.
#[inline]
pub fn quantize(val: f32) -> Fp32 {
    let scaled = (val * SCALE).round() as i64;
    let reduced = scaled.rem_euclid(PRIME32 as i64) as u32;
    Fp32::from_reduced(reduced)
}

/// Dequantize Fp32 back to f32. Values > p/2 are treated as negative.
#[inline]
pub fn dequantize(val: Fp32) -> f32 {
    let raw = val.raw();
    if raw <= HALF_P {
        raw as f32 * INV_SCALE
    } else {
        -((PRIME32 - raw) as f32) * INV_SCALE
    }
}

pub fn quantize_vec(data: &[f32]) -> Vec<Fp32> {
    data.iter().map(|&v| quantize(v)).collect()
}

pub fn dequantize_vec(data: &[Fp32]) -> Vec<f32> {
    data.iter().map(|&v| dequantize(v)).collect()
}

/// Dequantize a product of two quantized values.
/// Products are in SCALE² domain (x_q * y_q = x*y*SCALE²), so we divide by SCALE².
#[inline]
pub fn dequantize_product(val: Fp32) -> f32 {
    dequantize(val) * INV_SCALE
}

/// Quantize model weights (same operation, named for clarity).
pub fn quantize_weights(weights: &[f32]) -> Vec<Fp32> {
    quantize_vec(weights)
}

pub fn scale_factor() -> f32 {
    SCALE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_round_trip() {
        let values = [-5.0, -1.0, -0.5, 0.0, 0.5, 1.0, 5.0, 10.0, -10.0];
        for &v in &values {
            let q = quantize(v);
            let d = dequantize(q);
            assert!(
                (d - v).abs() < 0.01,
                "round-trip failed: {} → {:?} → {}",
                v,
                q,
                d
            );
        }
    }

    #[test]
    fn test_quantize_preserves_addition() {
        let a = 3.5f32;
        let b = -1.25f32;
        let qa = quantize(a);
        let qb = quantize(b);
        let sum = dequantize(qa + qb);
        assert!(
            (sum - (a + b)).abs() < 0.01,
            "addition: {} + {} = {}, got {}",
            a,
            b,
            a + b,
            sum
        );
    }

    #[test]
    fn test_quantize_preserves_multiplication() {
        let a = 2.0f32;
        let b = 3.5f32;
        let qa = quantize(a);
        let qb = quantize(b);
        // Product is in SCALE² domain → use dequantize_product
        let product_f32 = dequantize_product(qa * qb);
        assert!(
            (product_f32 - a * b).abs() < 0.1,
            "mul: {} × {} = {}, got {}",
            a,
            b,
            a * b,
            product_f32
        );
    }

    #[test]
    fn test_quantize_vec() {
        let data = vec![1.0, -2.0, 0.5, -0.25];
        let q = quantize_vec(&data);
        let d = dequantize_vec(&q);
        for i in 0..data.len() {
            assert!((d[i] - data[i]).abs() < 0.01);
        }
    }

    #[test]
    fn test_quantize_overflow_check() {
        // Verify dot product of 64 terms stays within field
        let max_val = 5.0f32;
        let q = quantize(max_val);
        let product = q * q; // SCALE² × max_val²
                             // Sum 64 of these
        let mut sum = Fp32::ZERO;
        for _ in 0..64 {
            sum = sum + product;
        }
        // Should not wrap (419M < 2.15B)
        let raw = sum.raw();
        assert!(
            raw < PRIME32 / 2,
            "overflow: dot product sum {} >= p/2",
            raw
        );
    }
}
