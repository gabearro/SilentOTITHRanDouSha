//! Rotary Position Embeddings (RoPE) with YaRN context extension.
//!
//! Applied on CPU after QKV projection, before attention.
//! Position-dependent: not suitable for ANE static graphs.

use super::config::GptOssConfig;

pub struct RopeTable {
    /// inv_freq[i] for i in 0..head_dim/2, pre-scaled by YaRN
    inv_freq: Box<[f64]>,
    head_dim: usize,
    /// Attention scaling factor from YaRN
    attn_factor: f64,
}

impl RopeTable {
    pub fn new(config: &GptOssConfig) -> Self {
        let hd = config.head_dim;
        let half = hd / 2;
        let base = config.rope_theta;

        let inv_freq: Vec<f64> = if let Some(ref scaling) = config.rope_scaling {
            if scaling.rope_type == "yarn" {
                Self::yarn_inv_freq(
                    half,
                    base,
                    scaling.factor,
                    scaling.original_max_position_embeddings,
                    scaling.beta_fast,
                    scaling.beta_slow,
                )
            } else {
                Self::base_inv_freq(half, base)
            }
        } else {
            Self::base_inv_freq(half, base)
        };

        let attn_factor = if let Some(ref scaling) = config.rope_scaling {
            if scaling.rope_type == "yarn" {
                // YaRN attention scaling: 0.1 * ln(factor) + 1.0
                0.1 * (scaling.factor as f64).ln() + 1.0
            } else {
                1.0
            }
        } else {
            1.0
        };

        RopeTable {
            inv_freq: inv_freq.into_boxed_slice(),
            head_dim: hd,
            attn_factor,
        }
    }

    fn base_inv_freq(half: usize, base: f64) -> Vec<f64> {
        (0..half)
            .map(|i| 1.0 / base.powf(2.0 * i as f64 / (2 * half) as f64))
            .collect()
    }

    /// YaRN frequency scaling: interpolates between original and extended context.
    /// Low frequencies (long wavelengths) get interpolation, high frequencies stay unchanged.
    fn yarn_inv_freq(
        half: usize,
        base: f64,
        factor: f64,
        original_max: usize,
        beta_fast: f64,
        beta_slow: f64,
    ) -> Vec<f64> {
        let dim = 2 * half;
        // Wavelength thresholds
        let low_freq_wavelen = original_max as f64 / beta_slow;
        let high_freq_wavelen = original_max as f64 / beta_fast;

        (0..half)
            .map(|i| {
                let freq = 1.0 / base.powf(2.0 * i as f64 / dim as f64);
                let wavelen = 2.0 * std::f64::consts::PI / freq;

                if wavelen < high_freq_wavelen {
                    // High frequency: no interpolation
                    freq
                } else if wavelen > low_freq_wavelen {
                    // Low frequency: full interpolation
                    freq / factor
                } else {
                    // Medium frequency: smooth interpolation
                    let smooth = (low_freq_wavelen / wavelen - 1.0)
                        / (low_freq_wavelen / high_freq_wavelen - 1.0);
                    (1.0 - smooth) * freq / factor + smooth * freq
                }
            })
            .collect()
    }

    /// Apply RoPE to a single head's Q or K vector at position `pos`.
    /// `data` is [head_dim] f32 values, modified in-place.
    ///
    /// The YaRN attention_factor is baked into cos/sin (matching HF transformers).
    /// This means Q·K naturally gets attention_factor² in the dot product.
    #[inline]
    pub fn apply(&self, data: &mut [f32], pos: usize) {
        let half = self.head_dim / 2;
        let af = self.attn_factor as f32;
        for i in 0..half {
            let theta = pos as f64 * self.inv_freq[i];
            let cos = theta.cos() as f32 * af;
            let sin = theta.sin() as f32 * af;

            let x0 = data[i];
            let x1 = data[i + half];
            data[i] = x0 * cos - x1 * sin;
            data[i + half] = x0 * sin + x1 * cos;
        }
    }

    /// Get the precomputed inverse frequency table (head_dim/2 values).
    pub fn inv_freq(&self) -> &[f64] {
        &self.inv_freq
    }

    /// Attention scaling factor from YaRN (baked into cos/sin).
    pub fn attn_scale(&self) -> f32 {
        self.attn_factor as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpt_oss::config::{GptOssConfig, RopeScaling};

    fn test_config() -> GptOssConfig {
        serde_json::from_str(
            r#"{
            "hidden_size": 2880, "num_hidden_layers": 2,
            "num_attention_heads": 64, "num_key_value_heads": 8,
            "head_dim": 64, "intermediate_size": 2880,
            "vocab_size": 201088, "num_local_experts": 32,
            "num_experts_per_tok": 4, "layer_types": ["sliding_attention", "full_attention"],
            "rope_theta": 150000.0,
            "rope_scaling": {
                "rope_type": "yarn", "factor": 32.0,
                "original_max_position_embeddings": 4096,
                "beta_fast": 32.0, "beta_slow": 1.0
            }
        }"#,
        )
        .unwrap()
    }

    #[test]
    fn test_rope_scaling_at_pos_zero() {
        let config = test_config();
        let rope = RopeTable::new(&config);
        let mut data = vec![1.0f32; 64];
        rope.apply(&mut data, 0);
        // At pos=0, cos=attn_factor, sin=0 → data is scaled by attn_factor
        let af = rope.attn_scale();
        for i in 0..64 {
            assert!(
                (data[i] - af).abs() < 1e-5,
                "pos=0 should scale by attn_factor, dim {i}: got {} expected {}",
                data[i],
                af
            );
        }
    }

    #[test]
    fn test_rope_rotates_at_pos_1() {
        let config = test_config();
        let rope = RopeTable::new(&config);
        let mut data = vec![1.0f32; 64];
        rope.apply(&mut data, 1);
        // Should differ from identity
        let sum: f32 = data.iter().map(|&v| (v - 1.0).abs()).sum();
        assert!(sum > 0.01, "pos=1 should differ from identity");
    }

    #[test]
    fn test_yarn_attn_scale() {
        let config = test_config();
        let rope = RopeTable::new(&config);
        let scale = rope.attn_scale();
        // 0.1 * ln(32) + 1.0 ≈ 1.347
        assert!((scale - 1.347).abs() < 0.01, "YaRN attn scale: {scale}");
    }
}
