use ane::{Shape, TensorData};

use super::executables::{CompiledExecutables, DECODE_SPATIAL_WIDTH};
use super::kv_cache::KvCache;
use super::model::CompiledModel;

pub struct Session<'model> {
    model: &'model CompiledModel,
    kv_cache: KvCache,
    prefill_hidden: TensorData,
    prefill_attn_delta: TensorData,
    prefill_ffn_delta: TensorData,
    decode_hidden: TensorData,
    decode_attn_delta: TensorData,
    decode_ffn_delta: TensorData,
    decode_mask: TensorData,
    lm_head_output: TensorData,
    logits: Vec<f32>,
    position: usize,
    /// Running count of Beaver triples that private attention would consume.
    pub triples_consumed: u64,
}

impl<'model> Session<'model> {
    pub fn new(model: &'model CompiledModel, padded_prompt_length: usize) -> Self {
        let ed = model.config.n_embd;
        let max_seq = model.max_sequence_length;

        let phs = Shape::spatial(ed, 1, padded_prompt_length);
        let pas = Shape::spatial(3 * ed, 1, padded_prompt_length);
        let dhs = Shape::spatial(ed, 1, DECODE_SPATIAL_WIDTH);
        let das = Shape::spatial(3 * ed, 1, DECODE_SPATIAL_WIDTH);
        let dms = Shape {
            batch: 1,
            channels: 1,
            height: DECODE_SPATIAL_WIDTH,
            width: max_seq,
        };
        let lhs = Shape::spatial(model.config.vocab_size, 1, DECODE_SPATIAL_WIDTH);

        Self {
            model,
            kv_cache: KvCache::new(model.config.n_layer, ed, max_seq),
            prefill_hidden: TensorData::new(phs),
            prefill_attn_delta: TensorData::new(pas),
            prefill_ffn_delta: TensorData::new(phs),
            decode_hidden: TensorData::new(dhs),
            decode_attn_delta: TensorData::new(das),
            decode_ffn_delta: TensorData::new(dhs),
            decode_mask: TensorData::new(dms),
            lm_head_output: TensorData::new(lhs),
            logits: vec![0.0; model.config.vocab_size],
            position: 0,
            triples_consumed: 0,
        }
    }

    /// Triples consumed per attention layer for a single decode step.
    fn triples_per_decode_attention(&self) -> u64 {
        // Q·K^T for decode: 1 query token × max_seq keys × head_dim × n_heads
        let hs = self.model.config.head_size() as u64;
        let nh = self.model.config.n_head as u64;
        let ks = self.model.max_sequence_length as u64;
        nh * 1 * ks * hs
    }

    pub fn prefill(&mut self, token_ids: &[u32], real_prompt_length: usize) -> &[f32] {
        let ed = self.model.config.n_embd;
        let seq = token_ids.len();

        {
            let mut s = self.prefill_hidden.as_f32_slice_mut();
            embedding_lookup_into(
                &mut s,
                token_ids,
                &self.model.weights.wte,
                &self.model.weights.wpe,
                ed,
            );
        }

        // Triples for prefill: Q·K^T is seq×seq per head per layer
        let hs = self.model.config.head_size() as u64;
        let nh = self.model.config.n_head as u64;
        let sl = real_prompt_length as u64;
        let triples_per_prefill_attn = nh * sl * sl * hs;

        for (li, layer) in self.model.executables.prefill.iter().enumerate() {
            layer
                .attention
                .run(&[&self.prefill_hidden], &[&self.prefill_attn_delta])
                .unwrap_or_else(|e| panic!("prefill attn {li}: {e}"));

            self.triples_consumed += triples_per_prefill_attn;

            {
                let attn = self.prefill_attn_delta.as_f32_slice();
                let o_size = ed * seq;
                self.kv_cache.write_kv_sequence(
                    li,
                    &attn[o_size..2 * o_size],
                    &attn[2 * o_size..3 * o_size],
                    real_prompt_length,
                    seq,
                );
                let mut hs = self.prefill_hidden.as_f32_slice_mut();
                hs[..o_size].copy_from_slice(&attn[..o_size]);
            }

            layer
                .feed_forward
                .run(&[&self.prefill_hidden], &[&self.prefill_ffn_delta])
                .unwrap_or_else(|e| panic!("prefill ffn {li}: {e}"));
            std::mem::swap(&mut self.prefill_hidden, &mut self.prefill_ffn_delta);
        }

        self.position = real_prompt_length;
        self.kv_cache.position = real_prompt_length;

        {
            let hs = self.prefill_hidden.as_f32_slice();
            let mut lmi = self.decode_hidden.as_f32_slice_mut();
            for d in 0..ed {
                lmi[d * DECODE_SPATIAL_WIDTH] = hs[d * seq + (real_prompt_length - 1)];
            }
        }
        {
            let mut ms = self.decode_mask.as_f32_slice_mut();
            ms.fill(-65504.0);
            for col in 0..self.position {
                ms[col] = 0.0;
            }
        }

        self.run_lm_head()
    }

    pub fn decode_step(&mut self, token: u32) -> &[f32] {
        let ed = self.model.config.n_embd;

        {
            let mut hs = self.decode_hidden.as_f32_slice_mut();
            let ti = token as usize;
            for d in 0..ed {
                hs[d * DECODE_SPATIAL_WIDTH] = self.model.weights.wte[ti * ed + d]
                    + self.model.weights.wpe[self.position * ed + d];
            }
        }
        {
            let mut ms = self.decode_mask.as_f32_slice_mut();
            ms[self.position] = 0.0;
        }

        let triples_per_attn = self.triples_per_decode_attention();

        for (li, layer) in self.model.executables.decode.iter().enumerate() {
            layer
                .attention
                .run(
                    &[
                        &self.decode_hidden,
                        &self.kv_cache.keys[li],
                        &self.kv_cache.values[li],
                        &self.decode_mask,
                    ],
                    &[&self.decode_attn_delta],
                )
                .unwrap_or_else(|e| panic!("decode attn {li}: {e}"));

            self.triples_consumed += triples_per_attn;

            {
                let attn = self.decode_attn_delta.as_f32_slice();
                self.kv_cache
                    .write_kv_from_attn(li, &attn, DECODE_SPATIAL_WIDTH, self.position);
                let mut hs = self.decode_hidden.as_f32_slice_mut();
                hs.copy_from_slice(&attn[..ed * DECODE_SPATIAL_WIDTH]);
            }

            layer
                .feed_forward
                .run(&[&self.decode_hidden], &[&self.decode_ffn_delta])
                .unwrap_or_else(|e| panic!("decode ffn {li}: {e}"));
            std::mem::swap(&mut self.decode_hidden, &mut self.decode_ffn_delta);
        }

        self.position += 1;
        self.kv_cache.position = self.position;
        self.run_lm_head()
    }

    fn run_lm_head(&mut self) -> &[f32] {
        self.model
            .executables
            .lm_head
            .run(&[&self.decode_hidden], &[&self.lm_head_output])
            .expect("lm_head");

        let vs = self.model.config.vocab_size;
        let out = self.lm_head_output.as_f32_slice();
        for v in 0..vs {
            self.logits[v] = out[v * DECODE_SPATIAL_WIDTH];
        }
        &self.logits
    }
}

fn embedding_lookup_into(dst: &mut [f32], token_ids: &[u32], wte: &[f32], wpe: &[f32], ed: usize) {
    let seq = token_ids.len();
    for si in 0..seq {
        let ti = token_ids[si] as usize;
        for d in 0..ed {
            dst[d * seq + si] = wte[ti * ed + d] + wpe[si * ed + d];
        }
    }
}
