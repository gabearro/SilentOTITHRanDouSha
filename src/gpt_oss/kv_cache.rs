//! KV cache for GPT-OSS-20B with optional TurboQuant compression.
//!
//! K and V use `kv_dim = num_kv_heads * head_dim` per position.
//! Each KV head is shared by `gqa_ratio` query heads.

use super::turboquant::{packed_bits_len, MseScratch, ProdScratch, TurboQuantMse, TurboQuantProd};

#[derive(Debug, Clone, Copy)]
pub struct TurboQuantConfig {
    /// Total bit-width for keys (TurboQuant_prod).
    pub key_bits: u8,
    /// Bit-width for values (TurboQuant_mse).
    pub value_bits: u8,
    /// Seed used to instantiate random rotation/projection matrices.
    pub seed: u64,
}

impl Default for TurboQuantConfig {
    fn default() -> Self {
        // 3-bit KV was fast but produced severe quality degradation in real prompts.
        // Keep 4-bit as default and allow lower settings via env for experiments.
        TurboQuantConfig {
            key_bits: 4,
            value_bits: 4,
            seed: 0xC0DE_5EED_2026_0001,
        }
    }
}

impl TurboQuantConfig {
    pub fn from_env() -> Option<Self> {
        let enabled = std::env::var("GPT_OSS_TURBOQUANT").ok()?;
        if !parse_bool(&enabled) {
            return None;
        }

        let mut cfg = TurboQuantConfig::default();

        if let Ok(v) = std::env::var("GPT_OSS_TQ_KEY_BITS") {
            if let Ok(bits) = v.trim().parse::<u8>() {
                cfg.key_bits = bits.clamp(2, 8);
            }
        }
        if let Ok(v) = std::env::var("GPT_OSS_TQ_VALUE_BITS") {
            if let Ok(bits) = v.trim().parse::<u8>() {
                cfg.value_bits = bits.clamp(1, 8);
            }
        }
        if let Ok(v) = std::env::var("GPT_OSS_TQ_SEED") {
            if let Some(seed) = parse_u64_auto(&v) {
                cfg.seed = seed;
            }
        }

        Some(cfg)
    }
}

fn parse_bool(v: &str) -> bool {
    matches!(
        v.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn parse_u64_auto(v: &str) -> Option<u64> {
    let s = v.trim();
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).ok()
    } else {
        s.parse::<u64>().ok()
    }
}

enum CacheStorage {
    Full {
        /// keys[layer][pos * kv_dim + dim]
        keys: Vec<Vec<f32>>,
        /// values[layer][pos * kv_dim + dim]
        values: Vec<Vec<f32>>,
    },
    Turbo(TurboStorage),
}

struct TurboStorage {
    cfg: TurboQuantConfig,

    key_quant: TurboQuantProd,
    value_quant: TurboQuantMse,
    qjl_bytes_per_head: usize,

    // Keys: one quantized entry per (layer, pos, kv_head)
    key_idx: Vec<Vec<u8>>,     // [layer][slots * head_dim]
    key_qjl: Vec<Vec<u8>>,     // [layer][slots * qjl_bytes_per_head]
    key_norms: Vec<Vec<f32>>,  // [layer][slots]
    key_gammas: Vec<Vec<f32>>, // [layer][slots]

    // Values: MSE-only quantized entries
    value_idx: Vec<Vec<u8>>,    // [layer][slots * head_dim]
    value_norms: Vec<Vec<f32>>, // [layer][slots]

    // Reused scratch for write-path quantization.
    key_scratch: ProdScratch,
    value_scratch: MseScratch,
}

pub struct KvCache {
    storage: CacheStorage,
    pub kv_dim: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq: usize,
    pub position: usize,
}

impl KvCache {
    pub fn new(num_layers: usize, num_kv_heads: usize, head_dim: usize, max_seq: usize) -> Self {
        let kv_dim = num_kv_heads * head_dim;
        KvCache {
            storage: CacheStorage::Full {
                keys: (0..num_layers)
                    .map(|_| vec![0.0; max_seq * kv_dim])
                    .collect(),
                values: (0..num_layers)
                    .map(|_| vec![0.0; max_seq * kv_dim])
                    .collect(),
            },
            kv_dim,
            num_kv_heads,
            head_dim,
            max_seq,
            position: 0,
        }
    }

    pub fn new_turboquant(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        cfg: TurboQuantConfig,
    ) -> Self {
        let kv_dim = num_kv_heads * head_dim;
        let slots_per_layer = max_seq * num_kv_heads;
        let qjl_bytes = packed_bits_len(head_dim);

        let key_quant = TurboQuantProd::new(head_dim, cfg.key_bits, cfg.seed ^ 0xAA01);
        let value_quant = TurboQuantMse::new(head_dim, cfg.value_bits, cfg.seed ^ 0x55FE);

        let storage = CacheStorage::Turbo(TurboStorage {
            cfg,
            key_quant,
            value_quant,
            qjl_bytes_per_head: qjl_bytes,
            key_idx: (0..num_layers)
                .map(|_| vec![0u8; slots_per_layer * head_dim])
                .collect(),
            key_qjl: (0..num_layers)
                .map(|_| vec![0u8; slots_per_layer * qjl_bytes])
                .collect(),
            key_norms: (0..num_layers)
                .map(|_| vec![0.0f32; slots_per_layer])
                .collect(),
            key_gammas: (0..num_layers)
                .map(|_| vec![0.0f32; slots_per_layer])
                .collect(),
            value_idx: (0..num_layers)
                .map(|_| vec![0u8; slots_per_layer * head_dim])
                .collect(),
            value_norms: (0..num_layers)
                .map(|_| vec![0.0f32; slots_per_layer])
                .collect(),
            key_scratch: ProdScratch::new(head_dim),
            value_scratch: MseScratch::new(head_dim),
        });

        KvCache {
            storage,
            kv_dim,
            num_kv_heads,
            head_dim,
            max_seq,
            position: 0,
        }
    }

    #[inline]
    pub fn turboquant_config(&self) -> Option<TurboQuantConfig> {
        match &self.storage {
            CacheStorage::Full { .. } => None,
            CacheStorage::Turbo(t) => Some(t.cfg),
        }
    }

    #[inline]
    fn slot_index(&self, pos: usize, kv_head: usize) -> usize {
        debug_assert!(pos < self.max_seq);
        debug_assert!(kv_head < self.num_kv_heads);
        pos * self.num_kv_heads + kv_head
    }

    /// Write K and V for a single position.
    #[inline]
    pub fn write(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) {
        debug_assert_eq!(k.len(), self.kv_dim);
        debug_assert_eq!(v.len(), self.kv_dim);

        let kv_dim = self.kv_dim;
        let num_kv_heads = self.num_kv_heads;
        let head_dim = self.head_dim;

        match &mut self.storage {
            CacheStorage::Full { keys, values } => {
                let base = pos * kv_dim;
                keys[layer][base..base + kv_dim].copy_from_slice(k);
                values[layer][base..base + kv_dim].copy_from_slice(v);
            }
            CacheStorage::Turbo(t) => {
                for kv_h in 0..num_kv_heads {
                    let slot = pos * num_kv_heads + kv_h;
                    let src_base = kv_h * head_dim;
                    let k_head = &k[src_base..src_base + head_dim];
                    let v_head = &v[src_base..src_base + head_dim];

                    let key_idx_base = slot * head_dim;
                    let key_qjl_base = slot * t.qjl_bytes_per_head;
                    t.key_quant.quantize_into(
                        k_head,
                        &mut t.key_idx[layer][key_idx_base..key_idx_base + head_dim],
                        &mut t.key_qjl[layer][key_qjl_base..key_qjl_base + t.qjl_bytes_per_head],
                        &mut t.key_norms[layer][slot],
                        &mut t.key_gammas[layer][slot],
                        &mut t.key_scratch,
                    );

                    let value_idx_base = slot * head_dim;
                    t.value_quant.quantize_into(
                        v_head,
                        &mut t.value_idx[layer][value_idx_base..value_idx_base + head_dim],
                        &mut t.value_norms[layer][slot],
                        &mut t.value_scratch,
                    );
                }
            }
        }
    }

    /// Precompute query-head transforms for key dot-products.
    ///
    /// Full precision mode:
    /// - `prepared_mse` is just `q_head` copied through.
    /// - `prepared_qjl` is ignored.
    ///
    /// TurboQuant mode:
    /// - `prepared_mse = Pi * q_head`
    /// - `prepared_qjl = S * q_head`
    pub fn prepare_key_query_head(
        &self,
        q_head: &[f32],
        prepared_mse: &mut [f32],
        prepared_qjl: &mut [f32],
    ) {
        debug_assert_eq!(q_head.len(), self.head_dim);
        debug_assert_eq!(prepared_mse.len(), self.head_dim);
        debug_assert_eq!(prepared_qjl.len(), self.head_dim);

        match &self.storage {
            CacheStorage::Full { .. } => {
                prepared_mse.copy_from_slice(q_head);
                prepared_qjl.fill(0.0);
            }
            CacheStorage::Turbo(t) => {
                t.key_quant
                    .prepare_query(q_head, prepared_mse, prepared_qjl);
            }
        }
    }

    /// Dot product between prepared query head and cached key head.
    pub fn key_dot_head_from_prepared(
        &self,
        layer: usize,
        pos: usize,
        kv_head: usize,
        prepared_mse: &[f32],
        prepared_qjl: &[f32],
    ) -> f32 {
        debug_assert_eq!(prepared_mse.len(), self.head_dim);
        debug_assert_eq!(prepared_qjl.len(), self.head_dim);

        let slot = self.slot_index(pos, kv_head);
        match &self.storage {
            CacheStorage::Full { keys, .. } => {
                let base = pos * self.kv_dim + kv_head * self.head_dim;
                let k_head = &keys[layer][base..base + self.head_dim];
                prepared_mse.iter().zip(k_head).map(|(a, b)| a * b).sum()
            }
            CacheStorage::Turbo(t) => {
                let idx_base = slot * self.head_dim;
                let qjl_base = slot * t.qjl_bytes_per_head;
                t.key_quant.dot_from_quantized_prepared(
                    prepared_mse,
                    prepared_qjl,
                    &t.key_idx[layer][idx_base..idx_base + self.head_dim],
                    &t.key_qjl[layer][qjl_base..qjl_base + t.qjl_bytes_per_head],
                    t.key_norms[layer][slot],
                    t.key_gammas[layer][slot],
                )
            }
        }
    }

    /// Accumulate weighted value head into `acc`.
    ///
    /// Full precision mode accumulates in original domain.
    /// TurboQuant mode accumulates in rotated domain.
    pub fn add_weighted_value_head(
        &self,
        layer: usize,
        pos: usize,
        kv_head: usize,
        weight: f32,
        acc: &mut [f32],
    ) {
        debug_assert_eq!(acc.len(), self.head_dim);

        let slot = self.slot_index(pos, kv_head);
        match &self.storage {
            CacheStorage::Full { values, .. } => {
                let base = pos * self.kv_dim + kv_head * self.head_dim;
                let v_head = &values[layer][base..base + self.head_dim];
                for d in 0..self.head_dim {
                    acc[d] += weight * v_head[d];
                }
            }
            CacheStorage::Turbo(t) => {
                let idx_base = slot * self.head_dim;
                let coeff = weight * t.value_norms[layer][slot];
                t.value_quant.accumulate_rotated_from_indices(
                    &t.value_idx[layer][idx_base..idx_base + self.head_dim],
                    coeff,
                    acc,
                );
            }
        }
    }

    /// Finalize value head after accumulation.
    ///
    /// Full precision mode copies `acc` to `out`.
    /// TurboQuant mode applies inverse rotation (`Pi^T`).
    pub fn finalize_value_head(&self, acc: &[f32], out: &mut [f32]) {
        debug_assert_eq!(acc.len(), self.head_dim);
        debug_assert_eq!(out.len(), self.head_dim);
        match &self.storage {
            CacheStorage::Full { .. } => out.copy_from_slice(acc),
            CacheStorage::Turbo(t) => t.value_quant.finalize_rotated(acc, out),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    fn rand_vec(rng: &mut ChaCha8Rng, n: usize) -> Vec<f32> {
        (0..n).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()
    }

    #[test]
    fn turbo_key_dot_tracks_full_precision() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let num_layers = 1usize;
        let nkv = 2usize;
        let hd = 16usize;
        let kvd = nkv * hd;
        let pos = 0usize;
        let layer = 0usize;
        let kv_head = 1usize;

        let cfg = TurboQuantConfig {
            key_bits: 8,
            value_bits: 8,
            seed: 123,
        };
        let mut tq = KvCache::new_turboquant(num_layers, nkv, hd, 8, cfg);
        let mut fp = KvCache::new(num_layers, nkv, hd, 8);

        let k = rand_vec(&mut rng, kvd);
        let v = rand_vec(&mut rng, kvd);
        let q = rand_vec(&mut rng, hd);

        tq.write(layer, pos, &k, &v);
        fp.write(layer, pos, &k, &v);

        let mut tq_q_mse = vec![0.0f32; hd];
        let mut tq_q_qjl = vec![0.0f32; hd];
        tq.prepare_key_query_head(&q, &mut tq_q_mse, &mut tq_q_qjl);
        let tq_dot = tq.key_dot_head_from_prepared(layer, pos, kv_head, &tq_q_mse, &tq_q_qjl);

        let mut fp_q_mse = vec![0.0f32; hd];
        let mut fp_q_qjl = vec![0.0f32; hd];
        fp.prepare_key_query_head(&q, &mut fp_q_mse, &mut fp_q_qjl);
        let fp_dot = fp.key_dot_head_from_prepared(layer, pos, kv_head, &fp_q_mse, &fp_q_qjl);

        let err = (tq_dot - fp_dot).abs();
        assert!(
            err < 0.4,
            "dot mismatch too large: tq={tq_dot}, fp={fp_dot}, err={err}"
        );
    }

    #[test]
    fn turbo_value_accum_tracks_full_precision() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let num_layers = 1usize;
        let nkv = 2usize;
        let hd = 16usize;
        let kvd = nkv * hd;
        let layer = 0usize;
        let kv_head = 0usize;

        let cfg = TurboQuantConfig {
            key_bits: 8,
            value_bits: 8,
            seed: 999,
        };
        let mut tq = KvCache::new_turboquant(num_layers, nkv, hd, 8, cfg);
        let mut fp = KvCache::new(num_layers, nkv, hd, 8);

        for pos in 0..3usize {
            let k = rand_vec(&mut rng, kvd);
            let v = rand_vec(&mut rng, kvd);
            tq.write(layer, pos, &k, &v);
            fp.write(layer, pos, &k, &v);
        }

        let weights = [0.2f32, 0.5, 0.3];
        let mut tq_acc = vec![0.0f32; hd];
        let mut fp_acc = vec![0.0f32; hd];

        for (pos, &w) in weights.iter().enumerate() {
            tq.add_weighted_value_head(layer, pos, kv_head, w, &mut tq_acc);
            fp.add_weighted_value_head(layer, pos, kv_head, w, &mut fp_acc);
        }

        let mut tq_out = vec![0.0f32; hd];
        tq.finalize_value_head(&tq_acc, &mut tq_out);

        let mut fp_out = vec![0.0f32; hd];
        fp.finalize_value_head(&fp_acc, &mut fp_out);

        let max_err = tq_out
            .iter()
            .zip(&fp_out)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 0.4, "value mismatch too large: max_err={max_err}");
    }
}
