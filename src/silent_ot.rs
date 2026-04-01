use crate::error::{ProtocolError, Result};
use crate::field::Fp;
use aes::cipher::{BlockEncrypt, KeyInit};
use aes::Aes128;
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::OnceLock;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Block(pub [u8; 16]);

impl Block {
    pub const ZERO: Block = Block([0u8; 16]);

    pub fn random<R: Rng>(rng: &mut R) -> Self {
        let mut b = [0u8; 16];
        rng.fill(&mut b);
        Block(b)
    }

    #[inline]
    pub fn xor(&self, other: &Block) -> Block {
        let a = u128::from_ne_bytes(self.0);
        let b = u128::from_ne_bytes(other.0);
        Block((a ^ b).to_ne_bytes())
    }

    pub fn to_field_element(&self, domain: u64) -> Fp {
        let domain_bytes = domain.to_le_bytes();
        let mut input = self.0;
        for i in 0..8 {
            input[i] ^= domain_bytes[i];
        }
        let mut aes_block = aes::Block::from(input);
        prg_key_field().encrypt_block(&mut aes_block);
        let encrypted: [u8; 16] = aes_block.into();
        let mut out = [0u8; 16];
        for i in 0..16 {
            out[i] = encrypted[i] ^ input[i];
        }
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&out[..8]);
        Fp::new(u64::from_le_bytes(bytes))
    }

    pub fn commit(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"SilentOT-commit:");
        hasher.update(self.0);
        hasher.finalize().into()
    }

    pub fn commit_with_context(&self, from: usize, to: usize) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"SilentOT-commit:");
        hasher.update(from.to_le_bytes());
        hasher.update(to.to_le_bytes());
        hasher.update(self.0);
        hasher.finalize().into()
    }

    pub fn verify_commitment(&self, commitment: &[u8; 32]) -> bool {
        self.commit() == *commitment
    }

    pub fn verify_commitment_with_context(
        &self,
        commitment: &[u8; 32],
        from: usize,
        to: usize,
    ) -> bool {
        self.commit_with_context(from, to) == *commitment
    }
}

pub fn batch_to_field_elements(blocks: &[Block], count: usize) -> Vec<Fp> {
    const PAR_THRESHOLD: usize = 32768;
    if count >= PAR_THRESHOLD {
        batch_to_field_elements_parallel(blocks, count)
    } else {
        batch_to_field_elements_seq(blocks, count)
    }
}

fn batch_to_field_elements_seq(blocks: &[Block], count: usize) -> Vec<Fp> {
    const CHUNK: usize = 4096;
    let key = prg_key_field();
    let mut results = Vec::with_capacity(count);
    // Reuse buffers across chunks to avoid repeated allocation
    let mut aes_blocks: Vec<aes::Block> = Vec::with_capacity(CHUNK);
    // Store pre-encryption inputs inline — we keep the original bytes in a parallel array
    // but use u128 for fast XOR instead of byte-by-byte
    let mut pre_xor: Vec<u128> = Vec::with_capacity(CHUNK);

    for chunk_start in (0..count).step_by(CHUNK) {
        let chunk_end = (chunk_start + CHUNK).min(count);
        let chunk_len = chunk_end - chunk_start;

        aes_blocks.clear();
        pre_xor.clear();

        for k in chunk_start..chunk_end {
            let mut input = blocks[k].0;
            let domain_bytes = (k as u64).to_le_bytes();
            // XOR domain into low 8 bytes
            for i in 0..8 {
                input[i] ^= domain_bytes[i];
            }
            let input_u128 = u128::from_le_bytes(input);
            pre_xor.push(input_u128);
            aes_blocks.push(aes::Block::from(input));
        }

        key.encrypt_blocks(&mut aes_blocks);

        for idx in 0..chunk_len {
            let enc_u128 = u128::from_le_bytes(aes_blocks[idx].into());
            let xored = enc_u128 ^ pre_xor[idx];
            // Extract low 8 bytes as u64, reduce to Fp
            let val = xored as u64; // low 64 bits
            results.push(Fp::new(val));
        }
    }

    results
}

fn batch_to_field_elements_parallel(blocks: &[Block], count: usize) -> Vec<Fp> {
    const CHUNK: usize = 16384;
    let num_chunks = count.div_ceil(CHUNK);

    let chunks: Vec<Vec<Fp>> = (0..num_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let key = prg_key_field();
            let chunk_start = chunk_idx * CHUNK;
            let chunk_end = (chunk_start + CHUNK).min(count);
            let chunk_len = chunk_end - chunk_start;

            let mut pre_xor: Vec<u128> = Vec::with_capacity(chunk_len);
            let mut aes_blocks: Vec<aes::Block> = Vec::with_capacity(chunk_len);

            for k in chunk_start..chunk_end {
                let mut input = blocks[k].0;
                let domain_bytes = (k as u64).to_le_bytes();
                for i in 0..8 {
                    input[i] ^= domain_bytes[i];
                }
                pre_xor.push(u128::from_le_bytes(input));
                aes_blocks.push(aes::Block::from(input));
            }

            key.encrypt_blocks(&mut aes_blocks);

            let mut out = Vec::with_capacity(chunk_len);
            for idx in 0..chunk_len {
                let enc_u128 = u128::from_le_bytes(aes_blocks[idx].into());
                let val = (enc_u128 ^ pre_xor[idx]) as u64;
                out.push(Fp::new(val));
            }
            out
        })
        .collect();

    let mut results = Vec::with_capacity(count);
    for chunk in chunks {
        results.extend(chunk);
    }
    results
}

impl std::fmt::Debug for Block {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Block({:02x}{:02x}..)", self.0[0], self.0[1])
    }
}

const PRG_KEY_LEFT: [u8; 16] = [
    0x6a, 0x09, 0xe6, 0x67, 0xbb, 0x67, 0xae, 0x85, 0x3c, 0x6e, 0xf3, 0x72, 0xa5, 0x4f, 0xf5, 0x3a,
];
const PRG_KEY_RIGHT: [u8; 16] = [
    0x51, 0x0e, 0x52, 0x7f, 0xad, 0xe6, 0x82, 0xd1, 0x9b, 0x05, 0x68, 0x8c, 0x2b, 0x3e, 0x6c, 0x1f,
];
const PRG_KEY_FIELD: [u8; 16] = [
    0x42, 0x8a, 0x2f, 0x98, 0x71, 0x37, 0x44, 0x91, 0xb5, 0xc0, 0xfb, 0xcf, 0xe9, 0xb5, 0xdb, 0xa5,
];

fn prg_key_left() -> &'static Aes128 {
    static KEY: OnceLock<Aes128> = OnceLock::new();
    KEY.get_or_init(|| Aes128::new_from_slice(&PRG_KEY_LEFT).unwrap())
}

fn prg_key_right() -> &'static Aes128 {
    static KEY: OnceLock<Aes128> = OnceLock::new();
    KEY.get_or_init(|| Aes128::new_from_slice(&PRG_KEY_RIGHT).unwrap())
}

fn prg_key_field() -> &'static Aes128 {
    static KEY: OnceLock<Aes128> = OnceLock::new();
    KEY.get_or_init(|| Aes128::new_from_slice(&PRG_KEY_FIELD).unwrap())
}

// 4-ary PRG keys: 4 fixed AES keys for child derivation
const PRG_KEY_C0: [u8; 16] = [
    0x6a, 0x09, 0xe6, 0x67, 0xbb, 0x67, 0xae, 0x85, 0x3c, 0x6e, 0xf3, 0x72, 0xa5, 0x4f, 0xf5, 0x3a,
];
const PRG_KEY_C1: [u8; 16] = [
    0x51, 0x0e, 0x52, 0x7f, 0xad, 0xe6, 0x82, 0xd1, 0x9b, 0x05, 0x68, 0x8c, 0x2b, 0x3e, 0x6c, 0x1f,
];
const PRG_KEY_C2: [u8; 16] = [
    0x1f, 0x83, 0xd9, 0xab, 0x5b, 0xe0, 0xcd, 0x19, 0x13, 0x7e, 0x21, 0x79, 0x14, 0x31, 0xd7, 0x68,
];
const PRG_KEY_C3: [u8; 16] = [
    0xa5, 0x4f, 0xf5, 0x3a, 0x38, 0x0c, 0x7a, 0xd5, 0xd4, 0x9b, 0x30, 0x16, 0x6f, 0x16, 0x8a, 0x92,
];

fn prg_keys_4ary() -> &'static [Aes128; 4] {
    static KEYS: OnceLock<[Aes128; 4]> = OnceLock::new();
    KEYS.get_or_init(|| {
        [
            Aes128::new_from_slice(&PRG_KEY_C0).unwrap(),
            Aes128::new_from_slice(&PRG_KEY_C1).unwrap(),
            Aes128::new_from_slice(&PRG_KEY_C2).unwrap(),
            Aes128::new_from_slice(&PRG_KEY_C3).unwrap(),
        ]
    })
}

#[inline]
pub fn prg_expand(seed: &Block) -> (Block, Block) {
    let mut block_l = aes::Block::from(seed.0);
    let mut block_r = aes::Block::from(seed.0);
    prg_key_left().encrypt_block(&mut block_l);
    prg_key_right().encrypt_block(&mut block_r);

    let left = Block(block_l.into()).xor(seed);
    let right = Block(block_r.into()).xor(seed);
    (left, right)
}

pub struct GgmTree {
    pub depth: usize,
}

impl GgmTree {
    pub fn new(depth: usize) -> Self {
        GgmTree { depth }
    }

    pub fn num_leaves(&self) -> usize {
        1 << self.depth
    }

    pub fn expand_full(&self, root: &Block) -> Vec<Block> {
        let key_l = prg_key_left();
        let key_r = prg_key_right();
        let num_leaves = 1usize << self.depth;
        let mut current_level = vec![*root];
        let mut next_level = Vec::with_capacity(num_leaves);
        // Pre-allocate two AES buffers: one for left, one for right.
        // Both are filled in a single pass over current_level, then encrypted separately,
        // then written to next_level in a single pass — 2 reads of current_level instead of 3.
        let max_level = if self.depth > 0 {
            1usize << (self.depth - 1)
        } else {
            1
        };
        let mut buf_l: Vec<aes::Block> = Vec::with_capacity(max_level);
        let mut buf_r: Vec<aes::Block> = Vec::with_capacity(max_level);

        for _ in 0..self.depth {
            let len = current_level.len();
            if len >= 64 {
                // Single pass: fill both L and R buffers from current_level
                buf_l.clear();
                buf_r.clear();
                for seed in &current_level {
                    let b = aes::Block::from(seed.0);
                    buf_l.push(b);
                    buf_r.push(b);
                }

                // Encrypt both batches
                key_l.encrypt_blocks(&mut buf_l);
                key_r.encrypt_blocks(&mut buf_r);

                // Single pass: write interleaved children to next_level
                next_level.clear();
                next_level.resize(len * 2, Block::ZERO);
                for (i, seed) in current_level.iter().enumerate() {
                    let seed_u128 = u128::from_ne_bytes(seed.0);
                    let l: [u8; 16] = buf_l[i].into();
                    let r: [u8; 16] = buf_r[i].into();
                    next_level[2 * i] = Block((u128::from_ne_bytes(l) ^ seed_u128).to_ne_bytes());
                    next_level[2 * i + 1] =
                        Block((u128::from_ne_bytes(r) ^ seed_u128).to_ne_bytes());
                }

                std::mem::swap(&mut current_level, &mut next_level);
            } else {
                next_level.clear();
                next_level.reserve(len * 2);
                for node in &current_level {
                    let (left, right) = prg_expand(node);
                    next_level.push(left);
                    next_level.push(right);
                }
                std::mem::swap(&mut current_level, &mut next_level);
            }
        }
        current_level
    }

    pub fn compute_sibling_path(&self, root: &Block, puncture_idx: usize) -> Result<Vec<Block>> {
        if puncture_idx >= self.num_leaves() {
            return Err(ProtocolError::InvalidParams(format!(
                "puncture index {} out of range [0, {})",
                puncture_idx,
                self.num_leaves()
            )));
        }

        let mut sibling_path = Vec::with_capacity(self.depth);
        let mut current = *root;

        for level in 0..self.depth {
            let puncture_bit = (puncture_idx >> (self.depth - 1 - level)) & 1;
            let (left, right) = prg_expand(&current);
            if puncture_bit == 0 {
                sibling_path.push(right);
                current = left;
            } else {
                sibling_path.push(left);
                current = right;
            }
        }

        Ok(sibling_path)
    }

    pub fn reconstruct_from_siblings(
        &self,
        sibling_path: &[Block],
        puncture_idx: usize,
    ) -> Result<Vec<Block>> {
        if sibling_path.len() != self.depth {
            return Err(ProtocolError::MaliciousParty(format!(
                "expected sibling path of length {}, got {}",
                self.depth,
                sibling_path.len()
            )));
        }
        if puncture_idx >= self.num_leaves() {
            return Err(ProtocolError::InvalidParams(format!(
                "puncture index {} out of range [0, {})",
                puncture_idx,
                self.num_leaves()
            )));
        }

        let n = self.num_leaves();

        let subtree_info: Vec<(usize, usize, &Block)> = sibling_path
            .iter()
            .enumerate()
            .map(|(level, sibling)| {
                let subtree_depth = self.depth - 1 - level;
                let subtree_size = 1usize << subtree_depth;
                let parent_idx = puncture_idx >> (self.depth - level);
                let puncture_bit = (puncture_idx >> (self.depth - 1 - level)) & 1;
                let sibling_start = if puncture_bit == 0 {
                    (parent_idx * 2 + 1) * subtree_size
                } else {
                    (parent_idx * 2) * subtree_size
                };
                (subtree_depth, sibling_start, sibling)
            })
            .collect();

        const PAR_THRESHOLD: usize = 10;
        let (large, small): (Vec<_>, Vec<_>) = subtree_info
            .into_iter()
            .partition(|&(depth, _, _)| depth >= PAR_THRESHOLD);

        let mut leaves = vec![Block::ZERO; n];

        // Large subtrees in parallel
        let expanded: Vec<(usize, Vec<Block>)> = large
            .par_iter()
            .map(|&(subtree_depth, sibling_start, sibling)| {
                let subtree_leaves = GgmTree::new(subtree_depth).expand_full(sibling);
                (sibling_start, subtree_leaves)
            })
            .collect();
        for (sibling_start, subtree_leaves) in expanded {
            for (i, leaf) in subtree_leaves.into_iter().enumerate() {
                let global_idx = sibling_start + i;
                if global_idx < n {
                    leaves[global_idx] = leaf;
                }
            }
        }

        // Small subtrees sequentially
        for &(subtree_depth, sibling_start, sibling) in &small {
            let subtree_leaves = GgmTree::new(subtree_depth).expand_full(sibling);
            for (i, leaf) in subtree_leaves.into_iter().enumerate() {
                let global_idx = sibling_start + i;
                if global_idx < n {
                    leaves[global_idx] = leaf;
                }
            }
        }

        Ok(leaves)
    }

    pub fn expand_full_to_field_elements(&self, root: &Block, count: usize) -> Vec<Fp> {
        self.expand_full_to_u64(root, count)
            .into_iter()
            .map(Fp::from_reduced)
            .collect()
    }

    pub fn expand_full_to_u64(&self, root: &Block, count: usize) -> Vec<u64> {
        if self.depth == 0 {
            return batch_to_u64(&[*root], count);
        }

        let penultimate_tree = GgmTree::new(self.depth - 1);
        let penultimate = penultimate_tree.expand_full(root);

        let pen_len = penultimate.len();

        const CHUNK: usize = 8192;
        let num_chunks = pen_len.div_ceil(CHUNK);

        if num_chunks <= 1 {
            let leaves = self.expand_full(root);
            return batch_to_u64(&leaves, count);
        }

        // Parallel chunk processing: each chunk independently expands last level
        // and converts to field elements. Pre-allocate output, write via raw pointers.
        let mut results = vec![0u64; count];
        let results_base = results.as_mut_ptr() as usize;

        // Only parallelize if enough chunks to amortize rayon overhead
        if num_chunks >= 4 {
            (0..num_chunks).into_par_iter().for_each(|ci| {
                let pen_start = ci * CHUNK;
                let pen_end = (pen_start + CHUNK).min(pen_len);
                let pen_chunk_len = pen_end - pen_start;
                let leaf_start = pen_start * 2;
                if leaf_start >= count {
                    return;
                }

                let key_l = prg_key_left();
                let key_r = prg_key_right();
                let key_f = prg_key_field();

                let mut buf_l: Vec<aes::Block> = Vec::with_capacity(pen_chunk_len);
                let mut buf_r: Vec<aes::Block> = Vec::with_capacity(pen_chunk_len);
                let mut seeds_u128: Vec<u128> = Vec::with_capacity(pen_chunk_len);

                for i in pen_start..pen_end {
                    let b = aes::Block::from(penultimate[i].0);
                    buf_l.push(b);
                    buf_r.push(b);
                    seeds_u128.push(u128::from_ne_bytes(penultimate[i].0));
                }

                key_l.encrypt_blocks(&mut buf_l);
                key_r.encrypt_blocks(&mut buf_r);

                let mut field_buf: Vec<aes::Block> = Vec::with_capacity(pen_chunk_len * 2);
                let mut pre_xor: Vec<u128> = Vec::with_capacity(pen_chunk_len * 2);
                let mut out_indices: Vec<usize> = Vec::with_capacity(pen_chunk_len * 2);

                for i in 0..pen_chunk_len {
                    let seed_u128 = seeds_u128[i];
                    let left_u128 = u128::from_ne_bytes(buf_l[i].into()) ^ seed_u128;
                    let right_u128 = u128::from_ne_bytes(buf_r[i].into()) ^ seed_u128;

                    let global_left = leaf_start + i * 2;
                    let global_right = global_left + 1;

                    if global_left < count {
                        let leaf_le = u128::from_le_bytes(left_u128.to_ne_bytes());
                        let input_u128 = leaf_le ^ (global_left as u128);
                        pre_xor.push(input_u128);
                        field_buf.push(aes::Block::from(input_u128.to_le_bytes()));
                        out_indices.push(global_left);
                    }

                    if global_right < count {
                        let leaf_le = u128::from_le_bytes(right_u128.to_ne_bytes());
                        let input_u128 = leaf_le ^ (global_right as u128);
                        pre_xor.push(input_u128);
                        field_buf.push(aes::Block::from(input_u128.to_le_bytes()));
                        out_indices.push(global_right);
                    }
                }

                key_f.encrypt_blocks(&mut field_buf);

                // SAFETY: each chunk writes to disjoint leaf indices [leaf_start..leaf_end)
                let ptr = results_base as *mut u64;
                for idx in 0..field_buf.len() {
                    let enc_u128 = u128::from_le_bytes(field_buf[idx].into());
                    let val = (enc_u128 ^ pre_xor[idx]) as u64;
                    unsafe {
                        *ptr.add(out_indices[idx]) = Fp::reduce(val);
                    }
                }
            });
        } else {
            // Sequential for small trees
            let key_l = prg_key_left();
            let key_r = prg_key_right();
            let key_f = prg_key_field();
            let mut buf_l: Vec<aes::Block> = Vec::with_capacity(CHUNK);
            let mut buf_r: Vec<aes::Block> = Vec::with_capacity(CHUNK);
            let mut seeds_u128: Vec<u128> = Vec::with_capacity(CHUNK);
            let mut field_buf: Vec<aes::Block> = Vec::with_capacity(CHUNK * 2);
            let mut pre_xor: Vec<u128> = Vec::with_capacity(CHUNK * 2);
            let mut write_idx = 0usize;

            for ci in 0..num_chunks {
                let pen_start = ci * CHUNK;
                let pen_end = (pen_start + CHUNK).min(pen_len);
                let pen_chunk_len = pen_end - pen_start;
                let leaf_start = pen_start * 2;
                if leaf_start >= count {
                    break;
                }

                buf_l.clear();
                buf_r.clear();
                seeds_u128.clear();
                for i in pen_start..pen_end {
                    let b = aes::Block::from(penultimate[i].0);
                    buf_l.push(b);
                    buf_r.push(b);
                    seeds_u128.push(u128::from_ne_bytes(penultimate[i].0));
                }
                key_l.encrypt_blocks(&mut buf_l);
                key_r.encrypt_blocks(&mut buf_r);

                field_buf.clear();
                pre_xor.clear();
                for i in 0..pen_chunk_len {
                    let seed_u128 = seeds_u128[i];
                    let left_u128 = u128::from_ne_bytes(buf_l[i].into()) ^ seed_u128;
                    let right_u128 = u128::from_ne_bytes(buf_r[i].into()) ^ seed_u128;
                    let global_left = leaf_start + i * 2;
                    let global_right = global_left + 1;
                    if global_left < count {
                        let leaf_le = u128::from_le_bytes(left_u128.to_ne_bytes());
                        let input_u128 = leaf_le ^ (global_left as u128);
                        pre_xor.push(input_u128);
                        field_buf.push(aes::Block::from(input_u128.to_le_bytes()));
                    }
                    if global_right < count {
                        let leaf_le = u128::from_le_bytes(right_u128.to_ne_bytes());
                        let input_u128 = leaf_le ^ (global_right as u128);
                        pre_xor.push(input_u128);
                        field_buf.push(aes::Block::from(input_u128.to_le_bytes()));
                    }
                }
                key_f.encrypt_blocks(&mut field_buf);
                for idx in 0..field_buf.len() {
                    let enc_u128 = u128::from_le_bytes(field_buf[idx].into());
                    let val = (enc_u128 ^ pre_xor[idx]) as u64;
                    if write_idx < count {
                        results[write_idx] = Fp::reduce(val);
                        write_idx += 1;
                    }
                }
            }
        }

        results
    }

    /// Like expand_full_to_u64 but derives 2 field elements per leaf (dual-half extraction).
    ///
    /// Instead of k separate AES calls per leaf, uses 1 AES call and extracts both the low
    /// and high 64-bit halves of the Davies-Meyer output. This halves field conversion AES.
    /// `k` must be 2. Tree has `2^depth` leaves producing `2 * 2^depth` values.
    pub fn expand_full_to_u64_multi(&self, root: &Block, count: usize, k: usize) -> Vec<u64> {
        if k == 1 {
            return self.expand_full_to_u64(root, count);
        }
        debug_assert_eq!(k, 2, "dual-half extraction only supports k=2");
        if self.depth == 0 {
            return blocks_to_multi_u64(&[*root], k, count);
        }

        let penultimate_tree = GgmTree::new(self.depth - 1);
        let penultimate = penultimate_tree.expand_full(root);
        let pen_len = penultimate.len();

        let mut results = vec![0u64; count];
        let results_base = results.as_mut_ptr() as usize;

        // Same chunk size as k=1 path since we do 1 AES per leaf (not 2)
        const CHUNK: usize = 8192;
        let num_chunks = pen_len.div_ceil(CHUNK);

        if num_chunks >= 4 {
            (0..num_chunks).into_par_iter().for_each(|ci| {
                let pen_start = ci * CHUNK;
                let pen_end = (pen_start + CHUNK).min(pen_len);
                let pen_chunk_len = pen_end - pen_start;
                let leaf_start = pen_start * 2;
                if leaf_start * 2 >= count {
                    return;
                }

                let key_l = prg_key_left();
                let key_r = prg_key_right();
                let key_f = prg_key_field();

                let mut buf_l: Vec<aes::Block> = Vec::with_capacity(pen_chunk_len);
                let mut buf_r: Vec<aes::Block> = Vec::with_capacity(pen_chunk_len);
                let mut seeds_u128: Vec<u128> = Vec::with_capacity(pen_chunk_len);

                for i in pen_start..pen_end {
                    let b = aes::Block::from(penultimate[i].0);
                    buf_l.push(b);
                    buf_r.push(b);
                    seeds_u128.push(u128::from_ne_bytes(penultimate[i].0));
                }

                key_l.encrypt_blocks(&mut buf_l);
                key_r.encrypt_blocks(&mut buf_r);

                // 1 field AES per leaf (same count as k=1), but each produces 2 values
                let mut field_buf: Vec<aes::Block> = Vec::with_capacity(pen_chunk_len * 2);
                let mut pre_xor: Vec<u128> = Vec::with_capacity(pen_chunk_len * 2);
                let mut leaf_indices: Vec<usize> = Vec::with_capacity(pen_chunk_len * 2);

                for i in 0..pen_chunk_len {
                    let seed_u128 = seeds_u128[i];
                    let left_u128 = u128::from_ne_bytes(buf_l[i].into()) ^ seed_u128;
                    let right_u128 = u128::from_ne_bytes(buf_r[i].into()) ^ seed_u128;

                    let left_leaf = leaf_start + i * 2;
                    let right_leaf = left_leaf + 1;

                    // Left leaf: 1 AES → 2 values at indices [left_leaf*2, left_leaf*2+1]
                    if left_leaf * 2 < count {
                        let leaf_le = u128::from_le_bytes(left_u128.to_ne_bytes());
                        let input = leaf_le ^ (left_leaf as u128);
                        pre_xor.push(input);
                        field_buf.push(aes::Block::from(input.to_le_bytes()));
                        leaf_indices.push(left_leaf);
                    }

                    // Right leaf: 1 AES → 2 values
                    if right_leaf * 2 < count {
                        let leaf_le = u128::from_le_bytes(right_u128.to_ne_bytes());
                        let input = leaf_le ^ (right_leaf as u128);
                        pre_xor.push(input);
                        field_buf.push(aes::Block::from(input.to_le_bytes()));
                        leaf_indices.push(right_leaf);
                    }
                }

                key_f.encrypt_blocks(&mut field_buf);

                let ptr = results_base as *mut u64;
                for idx in 0..field_buf.len() {
                    let enc_u128 = u128::from_le_bytes(field_buf[idx].into());
                    let xored = enc_u128 ^ pre_xor[idx];
                    let leaf = leaf_indices[idx];
                    let vidx_lo = leaf * 2;
                    let vidx_hi = vidx_lo + 1;
                    unsafe {
                        if vidx_lo < count {
                            *ptr.add(vidx_lo) = Fp::reduce(xored as u64);
                        }
                        if vidx_hi < count {
                            *ptr.add(vidx_hi) = Fp::reduce((xored >> 64) as u64);
                        }
                    }
                }
            });
        } else {
            // Sequential for small trees
            let key_l = prg_key_left();
            let key_r = prg_key_right();
            let key_f = prg_key_field();
            let mut buf_l: Vec<aes::Block> = Vec::with_capacity(CHUNK);
            let mut buf_r: Vec<aes::Block> = Vec::with_capacity(CHUNK);
            let mut seeds_u128: Vec<u128> = Vec::with_capacity(CHUNK);
            let mut field_buf: Vec<aes::Block> = Vec::with_capacity(CHUNK * 2);
            let mut pre_xor: Vec<u128> = Vec::with_capacity(CHUNK * 2);
            let mut leaf_indices: Vec<usize> = Vec::with_capacity(CHUNK * 2);

            for ci in 0..num_chunks {
                let pen_start = ci * CHUNK;
                let pen_end = (pen_start + CHUNK).min(pen_len);
                let pen_chunk_len = pen_end - pen_start;
                let leaf_start = pen_start * 2;
                if leaf_start * 2 >= count {
                    break;
                }

                buf_l.clear();
                buf_r.clear();
                seeds_u128.clear();
                for i in pen_start..pen_end {
                    let b = aes::Block::from(penultimate[i].0);
                    buf_l.push(b);
                    buf_r.push(b);
                    seeds_u128.push(u128::from_ne_bytes(penultimate[i].0));
                }
                key_l.encrypt_blocks(&mut buf_l);
                key_r.encrypt_blocks(&mut buf_r);

                field_buf.clear();
                pre_xor.clear();
                leaf_indices.clear();
                for i in 0..pen_chunk_len {
                    let seed_u128 = seeds_u128[i];
                    let left_u128 = u128::from_ne_bytes(buf_l[i].into()) ^ seed_u128;
                    let right_u128 = u128::from_ne_bytes(buf_r[i].into()) ^ seed_u128;
                    let left_leaf = leaf_start + i * 2;
                    let right_leaf = left_leaf + 1;

                    if left_leaf * 2 < count {
                        let leaf_le = u128::from_le_bytes(left_u128.to_ne_bytes());
                        let input = leaf_le ^ (left_leaf as u128);
                        pre_xor.push(input);
                        field_buf.push(aes::Block::from(input.to_le_bytes()));
                        leaf_indices.push(left_leaf);
                    }
                    if right_leaf * 2 < count {
                        let leaf_le = u128::from_le_bytes(right_u128.to_ne_bytes());
                        let input = leaf_le ^ (right_leaf as u128);
                        pre_xor.push(input);
                        field_buf.push(aes::Block::from(input.to_le_bytes()));
                        leaf_indices.push(right_leaf);
                    }
                }
                key_f.encrypt_blocks(&mut field_buf);
                for idx in 0..field_buf.len() {
                    let enc_u128 = u128::from_le_bytes(field_buf[idx].into());
                    let xored = enc_u128 ^ pre_xor[idx];
                    let leaf = leaf_indices[idx];
                    let vidx_lo = leaf * 2;
                    let vidx_hi = vidx_lo + 1;
                    if vidx_lo < count {
                        results[vidx_lo] = Fp::reduce(xored as u64);
                    }
                    if vidx_hi < count {
                        results[vidx_hi] = Fp::reduce((xored >> 64) as u64);
                    }
                }
            }
        }

        results
    }

    /// Like reconstruct_accumulate_u64 but derives k field elements per leaf.
    pub fn reconstruct_accumulate_u64_multi(
        &self,
        sibling_path: &[Block],
        puncture_idx: usize,
        count: usize,
        accum: &mut [u64],
        k: usize,
    ) -> Result<()> {
        if k == 1 {
            return self.reconstruct_accumulate_u64(sibling_path, puncture_idx, count, accum);
        }
        if sibling_path.len() != self.depth {
            return Err(ProtocolError::MaliciousParty(format!(
                "expected sibling path of length {}, got {}",
                self.depth,
                sibling_path.len()
            )));
        }
        if puncture_idx >= self.num_leaves() {
            return Err(ProtocolError::InvalidParams(format!(
                "puncture index {} out of range [0, {})",
                puncture_idx,
                self.num_leaves()
            )));
        }

        let subtree_info: Vec<(usize, usize, &Block)> = sibling_path
            .iter()
            .enumerate()
            .map(|(level, sibling)| {
                let subtree_depth = self.depth - 1 - level;
                let subtree_size = 1usize << subtree_depth;
                let parent_idx = puncture_idx >> (self.depth - level);
                let puncture_bit = (puncture_idx >> (self.depth - 1 - level)) & 1;
                let sibling_start = if puncture_bit == 0 {
                    (parent_idx * 2 + 1) * subtree_size
                } else {
                    (parent_idx * 2) * subtree_size
                };
                (subtree_depth, sibling_start, sibling)
            })
            .collect();

        const PAR_THRESHOLD: usize = 10;
        let (large, small): (Vec<_>, Vec<_>) = subtree_info
            .into_iter()
            .partition(|&(depth, _, _)| depth >= PAR_THRESHOLD);

        // Large subtrees in parallel
        let large_results: Vec<(usize, Vec<u64>)> = large
            .par_iter()
            .map(|&(subtree_depth, sibling_start, sibling)| {
                let value_start = sibling_start * k;
                let subtree_leaves = 1usize << subtree_depth;
                let subtree_values = subtree_leaves * k;
                let subtree_count = if value_start < count {
                    subtree_values.min(count - value_start)
                } else {
                    0
                };
                let field_elems = if subtree_count > 0 {
                    GgmTree::new(subtree_depth).expand_full_to_u64_multi(sibling, subtree_count, k)
                } else {
                    Vec::new()
                };
                (value_start, field_elems)
            })
            .collect();

        for (value_start, field_elems) in large_results {
            for (i, &val) in field_elems.iter().enumerate() {
                let global_idx = value_start + i;
                if global_idx < count {
                    // Skip values belonging to the punctured leaf
                    let leaf_idx = global_idx / k;
                    if leaf_idx != puncture_idx {
                        accum[global_idx] = Fp::add_raw(accum[global_idx], val);
                    }
                }
            }
        }

        // Small subtrees sequentially
        for &(subtree_depth, sibling_start, sibling) in &small {
            let value_start = sibling_start * k;
            let subtree_leaves = 1usize << subtree_depth;
            let subtree_values = subtree_leaves * k;
            let subtree_count = if value_start < count {
                subtree_values.min(count - value_start)
            } else {
                0
            };
            if subtree_count > 0 {
                let field_elems =
                    GgmTree::new(subtree_depth).expand_full_to_u64_multi(sibling, subtree_count, k);
                for (i, &val) in field_elems.iter().enumerate() {
                    let global_idx = value_start + i;
                    if global_idx < count {
                        let leaf_idx = global_idx / k;
                        if leaf_idx != puncture_idx {
                            accum[global_idx] = Fp::add_raw(accum[global_idx], val);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    pub fn reconstruct_to_field_elements(
        &self,
        sibling_path: &[Block],
        puncture_idx: usize,
        count: usize,
    ) -> Result<Vec<Fp>> {
        if sibling_path.len() != self.depth {
            return Err(ProtocolError::MaliciousParty(format!(
                "expected sibling path of length {}, got {}",
                self.depth,
                sibling_path.len()
            )));
        }
        if puncture_idx >= self.num_leaves() {
            return Err(ProtocolError::InvalidParams(format!(
                "puncture index {} out of range [0, {})",
                puncture_idx,
                self.num_leaves()
            )));
        }

        let subtree_info: Vec<(usize, usize, &Block)> = sibling_path
            .iter()
            .enumerate()
            .map(|(level, sibling)| {
                let subtree_depth = self.depth - 1 - level;
                let subtree_size = 1usize << subtree_depth;
                let parent_idx = puncture_idx >> (self.depth - level);
                let puncture_bit = (puncture_idx >> (self.depth - 1 - level)) & 1;
                let sibling_start = if puncture_bit == 0 {
                    (parent_idx * 2 + 1) * subtree_size
                } else {
                    (parent_idx * 2) * subtree_size
                };
                (subtree_depth, sibling_start, sibling)
            })
            .collect();

        const PAR_THRESHOLD: usize = 10;
        let (large, small): (Vec<_>, Vec<_>) = subtree_info
            .into_iter()
            .partition(|&(depth, _, _)| depth >= PAR_THRESHOLD);

        let mut result = vec![Fp::ZERO; count];

        // Large subtrees in parallel
        let large_results: Vec<(usize, Vec<Fp>)> = large
            .par_iter()
            .map(|&(subtree_depth, sibling_start, sibling)| {
                let subtree_count = if sibling_start < count {
                    (1usize << subtree_depth).min(count - sibling_start)
                } else {
                    0
                };
                let field_elems = if subtree_count > 0 {
                    expand_full_to_field_at_offset(
                        subtree_depth,
                        sibling,
                        sibling_start,
                        subtree_count,
                    )
                } else {
                    Vec::new()
                };
                (sibling_start, field_elems)
            })
            .collect();

        for (sibling_start, field_elems) in large_results {
            for (i, &val) in field_elems.iter().enumerate() {
                let global_idx = sibling_start + i;
                if global_idx < count {
                    result[global_idx] = val;
                }
            }
        }

        // Small subtrees sequentially
        for &(subtree_depth, sibling_start, sibling) in &small {
            let subtree_count = if sibling_start < count {
                (1usize << subtree_depth).min(count - sibling_start)
            } else {
                0
            };
            if subtree_count > 0 {
                let field_elems = expand_full_to_field_at_offset(
                    subtree_depth,
                    sibling,
                    sibling_start,
                    subtree_count,
                );
                for (i, &val) in field_elems.iter().enumerate() {
                    let global_idx = sibling_start + i;
                    if global_idx < count {
                        result[global_idx] = val;
                    }
                }
            }
        }

        Ok(result)
    }
    pub fn reconstruct_accumulate_field_elements(
        &self,
        sibling_path: &[Block],
        puncture_idx: usize,
        count: usize,
        accum: &mut [Fp],
    ) -> Result<()> {
        let mut u64_accum: Vec<u64> = accum.iter().map(|fp| fp.raw()).collect();
        self.reconstruct_accumulate_u64(sibling_path, puncture_idx, count, &mut u64_accum)?;
        for (i, &v) in u64_accum.iter().enumerate() {
            accum[i] = Fp::from_reduced(v);
        }
        Ok(())
    }

    pub fn reconstruct_accumulate_u64(
        &self,
        sibling_path: &[Block],
        puncture_idx: usize,
        count: usize,
        accum: &mut [u64],
    ) -> Result<()> {
        if sibling_path.len() != self.depth {
            return Err(ProtocolError::MaliciousParty(format!(
                "expected sibling path of length {}, got {}",
                self.depth,
                sibling_path.len()
            )));
        }
        if puncture_idx >= self.num_leaves() {
            return Err(ProtocolError::InvalidParams(format!(
                "puncture index {} out of range [0, {})",
                puncture_idx,
                self.num_leaves()
            )));
        }

        let subtree_info: Vec<(usize, usize, &Block)> = sibling_path
            .iter()
            .enumerate()
            .map(|(level, sibling)| {
                let subtree_depth = self.depth - 1 - level;
                let subtree_size = 1usize << subtree_depth;
                let parent_idx = puncture_idx >> (self.depth - level);
                let puncture_bit = (puncture_idx >> (self.depth - 1 - level)) & 1;
                let sibling_start = if puncture_bit == 0 {
                    (parent_idx * 2 + 1) * subtree_size
                } else {
                    (parent_idx * 2) * subtree_size
                };
                (subtree_depth, sibling_start, sibling)
            })
            .collect();

        // Split subtrees into large (parallelize) and small (sequential).
        // Rayon scheduling overhead dominates for subtrees with < 1024 leaves.
        const PAR_THRESHOLD: usize = 10; // depth threshold: 2^10 = 1024 leaves
        let (large, small): (Vec<_>, Vec<_>) = subtree_info
            .into_iter()
            .partition(|&(depth, _, _)| depth >= PAR_THRESHOLD);

        // Process large subtrees in parallel
        let large_results: Vec<(usize, Vec<u64>)> = large
            .par_iter()
            .map(|&(subtree_depth, sibling_start, sibling)| {
                let subtree_count = if sibling_start < count {
                    (1usize << subtree_depth).min(count - sibling_start)
                } else {
                    0
                };
                let field_elems = if subtree_count > 0 {
                    expand_full_to_u64_at_offset(
                        subtree_depth,
                        sibling,
                        sibling_start,
                        subtree_count,
                    )
                } else {
                    Vec::new()
                };
                (sibling_start, field_elems)
            })
            .collect();

        for (sibling_start, field_elems) in large_results {
            for (i, &val) in field_elems.iter().enumerate() {
                let global_idx = sibling_start + i;
                if global_idx < count && global_idx != puncture_idx {
                    accum[global_idx] = Fp::add_raw(accum[global_idx], val);
                }
            }
        }

        // Process small subtrees sequentially (avoids rayon overhead)
        for &(subtree_depth, sibling_start, sibling) in &small {
            let subtree_count = if sibling_start < count {
                (1usize << subtree_depth).min(count - sibling_start)
            } else {
                0
            };
            if subtree_count > 0 {
                let field_elems = expand_full_to_u64_at_offset(
                    subtree_depth,
                    sibling,
                    sibling_start,
                    subtree_count,
                );
                for (i, &val) in field_elems.iter().enumerate() {
                    let global_idx = sibling_start + i;
                    if global_idx < count && global_idx != puncture_idx {
                        accum[global_idx] = Fp::add_raw(accum[global_idx], val);
                    }
                }
            }
        }

        Ok(())
    }
}

fn expand_full_to_field_at_offset(
    depth: usize,
    root: &Block,
    offset: usize,
    count: usize,
) -> Vec<Fp> {
    expand_full_to_u64_at_offset(depth, root, offset, count)
        .into_iter()
        .map(Fp::from_reduced)
        .collect()
}

fn expand_full_to_u64_at_offset(
    depth: usize,
    root: &Block,
    offset: usize,
    count: usize,
) -> Vec<u64> {
    if depth <= 1 {
        let leaves = GgmTree::new(depth).expand_full(root);
        return batch_to_u64_at_offset(&leaves, offset, count);
    }

    let penultimate = GgmTree::new(depth - 1).expand_full(root);
    let key_l = prg_key_left();
    let key_r = prg_key_right();
    let key_f = prg_key_field();
    let pen_len = penultimate.len();

    const CHUNK: usize = 4096;
    let mut results = Vec::with_capacity(count);
    let mut buf_l: Vec<aes::Block> = Vec::with_capacity(CHUNK);
    let mut buf_r: Vec<aes::Block> = Vec::with_capacity(CHUNK);
    let mut seeds_u128: Vec<u128> = Vec::with_capacity(CHUNK);
    let mut field_buf: Vec<aes::Block> = Vec::with_capacity(CHUNK * 2);
    let mut pre_xor: Vec<u128> = Vec::with_capacity(CHUNK * 2);

    for ci in 0..pen_len.div_ceil(CHUNK) {
        let pen_start = ci * CHUNK;
        let pen_end = (pen_start + CHUNK).min(pen_len);
        let pen_chunk_len = pen_end - pen_start;
        let leaf_start = pen_start * 2;
        if leaf_start >= count {
            break;
        }

        buf_l.clear();
        buf_r.clear();
        seeds_u128.clear();
        for i in pen_start..pen_end {
            let b = aes::Block::from(penultimate[i].0);
            buf_l.push(b);
            buf_r.push(b);
            seeds_u128.push(u128::from_ne_bytes(penultimate[i].0));
        }
        key_l.encrypt_blocks(&mut buf_l);
        key_r.encrypt_blocks(&mut buf_r);

        field_buf.clear();
        pre_xor.clear();
        for i in 0..pen_chunk_len {
            let seed_u128 = seeds_u128[i];
            let left_u128 = u128::from_ne_bytes(buf_l[i].into()) ^ seed_u128;
            let right_u128 = u128::from_ne_bytes(buf_r[i].into()) ^ seed_u128;

            let global_left = leaf_start + i * 2;
            let global_right = global_left + 1;

            if global_left < count {
                let leaf_le = u128::from_le_bytes(left_u128.to_ne_bytes());
                let input_u128 = leaf_le ^ ((offset + global_left) as u128);
                pre_xor.push(input_u128);
                field_buf.push(aes::Block::from(input_u128.to_le_bytes()));
            }
            if global_right < count {
                let leaf_le = u128::from_le_bytes(right_u128.to_ne_bytes());
                let input_u128 = leaf_le ^ ((offset + global_right) as u128);
                pre_xor.push(input_u128);
                field_buf.push(aes::Block::from(input_u128.to_le_bytes()));
            }
        }

        key_f.encrypt_blocks(&mut field_buf);
        for idx in 0..field_buf.len() {
            let enc_u128 = u128::from_le_bytes(field_buf[idx].into());
            let val = (enc_u128 ^ pre_xor[idx]) as u64;
            results.push(Fp::reduce(val));
        }
    }

    results.truncate(count);
    results
}

fn batch_to_u64_at_offset(blocks: &[Block], offset: usize, count: usize) -> Vec<u64> {
    const CHUNK: usize = 4096;
    let key = prg_key_field();
    let mut results = Vec::with_capacity(count);
    let mut aes_blocks: Vec<aes::Block> = Vec::with_capacity(CHUNK);
    let mut pre_xor: Vec<u128> = Vec::with_capacity(CHUNK);

    for chunk_start in (0..count).step_by(CHUNK) {
        let chunk_end = (chunk_start + CHUNK).min(count);
        let chunk_len = chunk_end - chunk_start;

        aes_blocks.clear();
        pre_xor.clear();

        for local_k in chunk_start..chunk_end {
            let global_k = offset + local_k;
            let mut input = blocks[local_k].0;
            let domain_bytes = (global_k as u64).to_le_bytes();
            for i in 0..8 {
                input[i] ^= domain_bytes[i];
            }
            pre_xor.push(u128::from_le_bytes(input));
            aes_blocks.push(aes::Block::from(input));
        }

        key.encrypt_blocks(&mut aes_blocks);

        for idx in 0..chunk_len {
            let enc_u128 = u128::from_le_bytes(aes_blocks[idx].into());
            let val = (enc_u128 ^ pre_xor[idx]) as u64;
            results.push(Fp::reduce(val));
        }
    }

    results
}

fn batch_to_u64(blocks: &[Block], count: usize) -> Vec<u64> {
    batch_to_u64_at_offset(blocks, 0, count)
}

/// Convert leaf blocks to field elements, deriving `k` values per leaf using domain separation.
/// `count` = total number of field elements to produce.
/// Domain for leaf `i`, sub-value `s` = `i * k + s`.
fn blocks_to_multi_u64(blocks: &[Block], k: usize, count: usize) -> Vec<u64> {
    let key = prg_key_field();
    let num_blocks = blocks.len().min(count.div_ceil(k));
    let mut results = Vec::with_capacity(count);

    const CHUNK: usize = 4096;
    let mut aes_buf: Vec<aes::Block> = Vec::with_capacity(CHUNK);
    let mut pre_xor: Vec<u128> = Vec::with_capacity(CHUNK);

    for leaf_idx in 0..num_blocks {
        let block_le = u128::from_le_bytes(blocks[leaf_idx].0);
        for sub in 0..k {
            let value_idx = leaf_idx * k + sub;
            if value_idx >= count {
                break;
            }
            let domain = value_idx as u128;
            let input = block_le ^ domain;
            pre_xor.push(input);
            aes_buf.push(aes::Block::from(input.to_le_bytes()));

            if aes_buf.len() >= CHUNK {
                key.encrypt_blocks(&mut aes_buf);
                for i in 0..aes_buf.len() {
                    let enc = u128::from_le_bytes(aes_buf[i].into());
                    let val = (enc ^ pre_xor[i]) as u64;
                    results.push(Fp::reduce(val));
                }
                aes_buf.clear();
                pre_xor.clear();
            }
        }
    }

    // Flush remaining
    if !aes_buf.is_empty() {
        key.encrypt_blocks(&mut aes_buf);
        for i in 0..aes_buf.len() {
            let enc = u128::from_le_bytes(aes_buf[i].into());
            let val = (enc ^ pre_xor[i]) as u64;
            results.push(Fp::reduce(val));
        }
    }

    results
}

/// Like `blocks_to_multi_u64` but accumulates into `accum` instead of allocating.
/// Skips all k values derived from the leaf at `skip_leaf_idx`.
fn blocks_accumulate_multi_u64(
    blocks: &[Block],
    k: usize,
    count: usize,
    accum: &mut [u64],
    skip_leaf_idx: usize,
) {
    let key = prg_key_field();
    let num_blocks = blocks.len().min(count.div_ceil(k));

    const CHUNK: usize = 4096;
    let mut aes_buf: Vec<aes::Block> = Vec::with_capacity(CHUNK);
    let mut pre_xor: Vec<u128> = Vec::with_capacity(CHUNK);
    let mut out_indices: Vec<usize> = Vec::with_capacity(CHUNK);

    for leaf_idx in 0..num_blocks {
        if leaf_idx == skip_leaf_idx {
            continue;
        }
        let block_le = u128::from_le_bytes(blocks[leaf_idx].0);
        for sub in 0..k {
            let value_idx = leaf_idx * k + sub;
            if value_idx >= count {
                break;
            }
            let domain = value_idx as u128;
            let input = block_le ^ domain;
            pre_xor.push(input);
            aes_buf.push(aes::Block::from(input.to_le_bytes()));
            out_indices.push(value_idx);

            if aes_buf.len() >= CHUNK {
                key.encrypt_blocks(&mut aes_buf);
                for i in 0..aes_buf.len() {
                    let enc = u128::from_le_bytes(aes_buf[i].into());
                    let val = Fp::reduce((enc ^ pre_xor[i]) as u64);
                    accum[out_indices[i]] = Fp::add_raw(accum[out_indices[i]], val);
                }
                aes_buf.clear();
                pre_xor.clear();
                out_indices.clear();
            }
        }
    }

    // Flush remaining
    if !aes_buf.is_empty() {
        key.encrypt_blocks(&mut aes_buf);
        for i in 0..aes_buf.len() {
            let enc = u128::from_le_bytes(aes_buf[i].into());
            let val = Fp::reduce((enc ^ pre_xor[i]) as u64);
            accum[out_indices[i]] = Fp::add_raw(accum[out_indices[i]], val);
        }
    }
}

// ── 4-ary GGM Tree ────────────────────────────────────────────────────
//
// Each node produces 4 children (vs 2 for binary). Halves tree depth for
// the same leaf count, reducing total AES ops by ~33%.

pub struct GgmTree4 {
    pub depth: usize,
}

impl GgmTree4 {
    pub fn new(depth: usize) -> Self {
        GgmTree4 { depth }
    }

    pub fn num_leaves(&self) -> usize {
        4usize.pow(self.depth as u32)
    }

    /// Expand the full 4-ary tree, returning all leaves.
    pub fn expand_full(&self, root: &Block) -> Vec<Block> {
        let keys = prg_keys_4ary();
        let num_leaves = self.num_leaves();
        let mut current_level = vec![*root];
        let mut next_level = Vec::with_capacity(num_leaves);
        let mut bufs: [Vec<aes::Block>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];

        for _ in 0..self.depth {
            let len = current_level.len();
            if len >= 16 {
                for buf in &mut bufs {
                    buf.clear();
                    buf.reserve(len);
                }
                for seed in &current_level {
                    let b = aes::Block::from(seed.0);
                    for buf in &mut bufs {
                        buf.push(b);
                    }
                }
                for (i, buf) in bufs.iter_mut().enumerate() {
                    keys[i].encrypt_blocks(buf);
                }
                next_level.clear();
                next_level.resize(len * 4, Block::ZERO);
                for (i, seed) in current_level.iter().enumerate() {
                    let s = u128::from_ne_bytes(seed.0);
                    for c in 0..4 {
                        let enc: [u8; 16] = bufs[c][i].into();
                        next_level[4 * i + c] = Block((u128::from_ne_bytes(enc) ^ s).to_ne_bytes());
                    }
                }
                std::mem::swap(&mut current_level, &mut next_level);
            } else {
                next_level.clear();
                next_level.reserve(len * 4);
                for node in &current_level {
                    let s = u128::from_ne_bytes(node.0);
                    for key in keys {
                        let mut b = aes::Block::from(node.0);
                        key.encrypt_block(&mut b);
                        let enc: [u8; 16] = b.into();
                        next_level.push(Block((u128::from_ne_bytes(enc) ^ s).to_ne_bytes()));
                    }
                }
                std::mem::swap(&mut current_level, &mut next_level);
            }
        }
        current_level
    }

    /// Compute the sibling path for puncturing at `puncture_idx`.
    /// Returns `depth` entries, each containing 3 sibling Blocks.
    pub fn compute_sibling_path(
        &self,
        root: &Block,
        puncture_idx: usize,
    ) -> Result<Vec<[Block; 3]>> {
        if puncture_idx >= self.num_leaves() {
            return Err(ProtocolError::InvalidParams(format!(
                "puncture index {} out of range [0, {})",
                puncture_idx,
                self.num_leaves()
            )));
        }
        let keys = prg_keys_4ary();
        let mut path = Vec::with_capacity(self.depth);
        let mut current = *root;

        for level in 0..self.depth {
            // Which of the 4 children is on the puncture path?
            let child_idx = (puncture_idx >> (2 * (self.depth - 1 - level))) & 3;
            // Expand all 4 children
            let s = u128::from_ne_bytes(current.0);
            let mut children = [Block::ZERO; 4];
            for c in 0..4 {
                let mut b = aes::Block::from(current.0);
                keys[c].encrypt_block(&mut b);
                let enc: [u8; 16] = b.into();
                children[c] = Block((u128::from_ne_bytes(enc) ^ s).to_ne_bytes());
            }
            // Record the 3 siblings
            let mut siblings = [Block::ZERO; 3];
            let mut si = 0;
            for c in 0..4 {
                if c != child_idx {
                    siblings[si] = children[c];
                    si += 1;
                }
            }
            path.push(siblings);
            current = children[child_idx];
        }
        Ok(path)
    }

    /// Reconstruct all leaves except the punctured one from a sibling path.
    pub fn reconstruct_from_siblings(
        &self,
        sibling_path: &[[Block; 3]],
        puncture_idx: usize,
    ) -> Result<Vec<Block>> {
        if sibling_path.len() != self.depth {
            return Err(ProtocolError::MaliciousParty(format!(
                "expected 4-ary sibling path of length {}, got {}",
                self.depth,
                sibling_path.len()
            )));
        }
        if puncture_idx >= self.num_leaves() {
            return Err(ProtocolError::InvalidParams(format!(
                "puncture index {} out of range [0, {})",
                puncture_idx,
                self.num_leaves()
            )));
        }

        let n = self.num_leaves();
        let mut leaves = vec![Block::ZERO; n];

        for (level, siblings) in sibling_path.iter().enumerate() {
            let child_idx = (puncture_idx >> (2 * (self.depth - 1 - level))) & 3;
            let parent_idx = puncture_idx >> (2 * (self.depth - level));
            // The 3 siblings cover children of the parent, excluding child_idx
            let mut si = 0;
            for c in 0..4usize {
                if c == child_idx {
                    continue;
                }
                let subtree_root = siblings[si];
                si += 1;
                let subtree_depth = self.depth - 1 - level;
                let subtree_start = (parent_idx * 4 + c) * 4usize.pow(subtree_depth as u32);
                if subtree_depth == 0 {
                    if subtree_start < n {
                        leaves[subtree_start] = subtree_root;
                    }
                } else {
                    let subtree = GgmTree4::new(subtree_depth).expand_full(&subtree_root);
                    for (i, leaf) in subtree.into_iter().enumerate() {
                        let global = subtree_start + i;
                        if global < n {
                            leaves[global] = leaf;
                        }
                    }
                }
            }
        }
        Ok(leaves)
    }

    /// Expand full 4-ary tree and convert leaves to u64 field elements.
    /// Fused last-level: batched 4-child PRG + batched field conversion in one pass.
    /// Parallel chunk processing with raw pointer writes for large trees.
    pub fn expand_full_to_u64(&self, root: &Block, count: usize) -> Vec<u64> {
        if self.depth == 0 {
            return batch_to_u64(&[*root], count.min(1));
        }
        if self.depth == 1 {
            let leaves = self.expand_full(root);
            return batch_to_u64(&leaves, count.min(leaves.len()));
        }

        // Expand to penultimate level (depth-1), then fuse last level
        let penultimate = GgmTree4::new(self.depth - 1).expand_full(root);
        let pen_len = penultimate.len();

        let mut results = vec![0u64; count];
        let results_base = results.as_mut_ptr() as usize;

        // Each pen node produces 4 children → 4 field elements.
        // Scale chunk so working set stays reasonable (~512KB field buffers).
        const CHUNK: usize = 4096;
        let num_chunks = pen_len.div_ceil(CHUNK);

        if num_chunks >= 4 {
            (0..num_chunks).into_par_iter().for_each(|ci| {
                let pen_start = ci * CHUNK;
                let pen_end = (pen_start + CHUNK).min(pen_len);
                let pen_chunk_len = pen_end - pen_start;
                let leaf_start = pen_start * 4; // 4 children per pen node
                if leaf_start >= count {
                    return;
                }

                let keys = prg_keys_4ary();
                let key_f = prg_key_field();

                // Batch PRG: 4 AES encryptions per pen node
                let mut bufs: [Vec<aes::Block>; 4] = [
                    Vec::with_capacity(pen_chunk_len),
                    Vec::with_capacity(pen_chunk_len),
                    Vec::with_capacity(pen_chunk_len),
                    Vec::with_capacity(pen_chunk_len),
                ];
                let mut seeds_u128: Vec<u128> = Vec::with_capacity(pen_chunk_len);

                for i in pen_start..pen_end {
                    let b = aes::Block::from(penultimate[i].0);
                    for buf in &mut bufs {
                        buf.push(b);
                    }
                    seeds_u128.push(u128::from_ne_bytes(penultimate[i].0));
                }

                for (k, buf) in bufs.iter_mut().enumerate() {
                    keys[k].encrypt_blocks(buf);
                }

                // Fused field conversion: 1 AES per leaf
                let max_field = pen_chunk_len * 4;
                let mut field_buf: Vec<aes::Block> = Vec::with_capacity(max_field);
                let mut pre_xor: Vec<u128> = Vec::with_capacity(max_field);
                let mut out_indices: Vec<usize> = Vec::with_capacity(max_field);

                for i in 0..pen_chunk_len {
                    let seed_u128 = seeds_u128[i];
                    for c in 0..4usize {
                        let global_leaf = leaf_start + i * 4 + c;
                        if global_leaf >= count {
                            continue;
                        }
                        let enc: [u8; 16] = bufs[c][i].into();
                        let child_u128 = u128::from_ne_bytes(enc) ^ seed_u128;
                        let leaf_le = u128::from_le_bytes(child_u128.to_ne_bytes());
                        let input = leaf_le ^ (global_leaf as u128);
                        pre_xor.push(input);
                        field_buf.push(aes::Block::from(input.to_le_bytes()));
                        out_indices.push(global_leaf);
                    }
                }

                key_f.encrypt_blocks(&mut field_buf);

                let ptr = results_base as *mut u64;
                for idx in 0..field_buf.len() {
                    let enc_u128 = u128::from_le_bytes(field_buf[idx].into());
                    let val = (enc_u128 ^ pre_xor[idx]) as u64;
                    unsafe {
                        *ptr.add(out_indices[idx]) = Fp::reduce(val);
                    }
                }
            });
        } else {
            // Sequential for small trees
            let keys = prg_keys_4ary();
            let key_f = prg_key_field();
            let mut bufs: [Vec<aes::Block>; 4] = [
                Vec::with_capacity(CHUNK),
                Vec::with_capacity(CHUNK),
                Vec::with_capacity(CHUNK),
                Vec::with_capacity(CHUNK),
            ];
            let mut seeds_u128: Vec<u128> = Vec::with_capacity(CHUNK);
            let cap = CHUNK * 4;
            let mut field_buf: Vec<aes::Block> = Vec::with_capacity(cap);
            let mut pre_xor: Vec<u128> = Vec::with_capacity(cap);
            let mut out_indices: Vec<usize> = Vec::with_capacity(cap);

            for ci in 0..num_chunks {
                let pen_start = ci * CHUNK;
                let pen_end = (pen_start + CHUNK).min(pen_len);
                let pen_chunk_len = pen_end - pen_start;
                let leaf_start = pen_start * 4;
                if leaf_start >= count {
                    break;
                }

                for buf in &mut bufs {
                    buf.clear();
                }
                seeds_u128.clear();
                for i in pen_start..pen_end {
                    let b = aes::Block::from(penultimate[i].0);
                    for buf in &mut bufs {
                        buf.push(b);
                    }
                    seeds_u128.push(u128::from_ne_bytes(penultimate[i].0));
                }
                for (k, buf) in bufs.iter_mut().enumerate() {
                    keys[k].encrypt_blocks(buf);
                }

                field_buf.clear();
                pre_xor.clear();
                out_indices.clear();
                for i in 0..pen_chunk_len {
                    let seed_u128 = seeds_u128[i];
                    for c in 0..4usize {
                        let global_leaf = leaf_start + i * 4 + c;
                        if global_leaf >= count {
                            continue;
                        }
                        let enc: [u8; 16] = bufs[c][i].into();
                        let child_u128 = u128::from_ne_bytes(enc) ^ seed_u128;
                        let leaf_le = u128::from_le_bytes(child_u128.to_ne_bytes());
                        let input = leaf_le ^ (global_leaf as u128);
                        pre_xor.push(input);
                        field_buf.push(aes::Block::from(input.to_le_bytes()));
                        out_indices.push(global_leaf);
                    }
                }
                key_f.encrypt_blocks(&mut field_buf);
                for idx in 0..field_buf.len() {
                    let enc_u128 = u128::from_le_bytes(field_buf[idx].into());
                    let val = (enc_u128 ^ pre_xor[idx]) as u64;
                    results[out_indices[idx]] = Fp::reduce(val);
                }
            }
        }

        results
    }

    /// Expand full 4-ary tree to u64 with a global domain offset.
    /// Used by reconstruction to ensure domain separation matches the sender's full tree.
    fn expand_full_to_u64_at_offset(&self, root: &Block, offset: usize, count: usize) -> Vec<u64> {
        if self.depth <= 1 {
            let leaves = self.expand_full(root);
            return batch_to_u64_at_offset(&leaves, offset, count.min(leaves.len()));
        }

        let penultimate = GgmTree4::new(self.depth - 1).expand_full(root);
        let pen_len = penultimate.len();
        let keys = prg_keys_4ary();
        let key_f = prg_key_field();

        let mut results = Vec::with_capacity(count);
        const CHUNK: usize = 4096;
        let mut bufs: [Vec<aes::Block>; 4] = [
            Vec::with_capacity(CHUNK),
            Vec::with_capacity(CHUNK),
            Vec::with_capacity(CHUNK),
            Vec::with_capacity(CHUNK),
        ];
        let mut seeds_u128: Vec<u128> = Vec::with_capacity(CHUNK);
        let cap = CHUNK * 4;
        let mut field_buf: Vec<aes::Block> = Vec::with_capacity(cap);
        let mut pre_xor: Vec<u128> = Vec::with_capacity(cap);

        for ci in 0..pen_len.div_ceil(CHUNK) {
            let pen_start = ci * CHUNK;
            let pen_end = (pen_start + CHUNK).min(pen_len);
            let pen_chunk_len = pen_end - pen_start;
            let leaf_start = pen_start * 4;
            if leaf_start >= count {
                break;
            }

            for buf in &mut bufs {
                buf.clear();
            }
            seeds_u128.clear();
            for i in pen_start..pen_end {
                let b = aes::Block::from(penultimate[i].0);
                for buf in &mut bufs {
                    buf.push(b);
                }
                seeds_u128.push(u128::from_ne_bytes(penultimate[i].0));
            }
            for (k, buf) in bufs.iter_mut().enumerate() {
                keys[k].encrypt_blocks(buf);
            }

            field_buf.clear();
            pre_xor.clear();
            for i in 0..pen_chunk_len {
                let seed_u128 = seeds_u128[i];
                for c in 0..4usize {
                    let local_leaf = leaf_start + i * 4 + c;
                    if local_leaf >= count {
                        continue;
                    }
                    let enc: [u8; 16] = bufs[c][i].into();
                    let child_u128 = u128::from_ne_bytes(enc) ^ seed_u128;
                    let leaf_le = u128::from_le_bytes(child_u128.to_ne_bytes());
                    // Domain uses global offset for consistency with sender
                    let input = leaf_le ^ ((offset + local_leaf) as u128);
                    pre_xor.push(input);
                    field_buf.push(aes::Block::from(input.to_le_bytes()));
                }
            }
            key_f.encrypt_blocks(&mut field_buf);
            for idx in 0..field_buf.len() {
                let enc_u128 = u128::from_le_bytes(field_buf[idx].into());
                let val = (enc_u128 ^ pre_xor[idx]) as u64;
                results.push(Fp::reduce(val));
            }
        }

        results.truncate(count);
        results
    }

    /// Reconstruct from sibling path and accumulate into `accum` as u64.
    /// Uses fused last-level expansion for large subtrees.
    pub fn reconstruct_accumulate_u64(
        &self,
        sibling_path: &[[Block; 3]],
        puncture_idx: usize,
        count: usize,
        accum: &mut [u64],
    ) -> Result<()> {
        if sibling_path.len() != self.depth {
            return Err(ProtocolError::MaliciousParty(format!(
                "expected 4-ary sibling path of length {}, got {}",
                self.depth,
                sibling_path.len()
            )));
        }
        if puncture_idx >= self.num_leaves() {
            return Err(ProtocolError::InvalidParams(format!(
                "puncture index {} out of range [0, {})",
                puncture_idx,
                self.num_leaves()
            )));
        }

        for (level, siblings) in sibling_path.iter().enumerate() {
            let child_idx = (puncture_idx >> (2 * (self.depth - 1 - level))) & 3;
            let parent_idx = puncture_idx >> (2 * (self.depth - level));

            let mut si = 0;
            for c in 0..4usize {
                if c == child_idx {
                    continue;
                }
                let subtree_root = &siblings[si];
                si += 1;
                let subtree_depth = self.depth - 1 - level;
                let subtree_start = (parent_idx * 4 + c) * 4usize.pow(subtree_depth as u32);
                if subtree_start >= count {
                    continue;
                }
                let subtree_count = if subtree_depth == 0 {
                    1
                } else {
                    4usize.pow(subtree_depth as u32).min(count - subtree_start)
                };
                // Use fused expansion with correct global offset
                let vals = if subtree_depth == 0 {
                    batch_to_u64_at_offset(&[*subtree_root], subtree_start, 1)
                } else {
                    GgmTree4::new(subtree_depth).expand_full_to_u64_at_offset(
                        subtree_root,
                        subtree_start,
                        subtree_count,
                    )
                };
                for (i, &val) in vals.iter().enumerate() {
                    let global = subtree_start + i;
                    if global < count {
                        accum[global] = Fp::add_raw(accum[global], val);
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SilentOtParams {
    pub n: usize,
    pub t: usize,
    pub num_ots: usize,
    pub tree_depth: usize,
    pub values_per_leaf: usize,
    /// Tree branching factor: 2 (binary, default) or 4 (4-ary, halves depth).
    pub tree_arity: usize,
}

impl SilentOtParams {
    pub fn new(n: usize, t: usize, num_ots: usize) -> Result<Self> {
        // Auto-select arity: 4-ary for large trees (halves depth, fewer AES ops),
        // binary for small trees (simpler, less overhead).
        let arity = if num_ots > 256 { 4 } else { 2 };
        Self::with_options(n, t, num_ots, 1, arity)
    }

    pub fn with_values_per_leaf(
        n: usize,
        t: usize,
        num_ots: usize,
        values_per_leaf: usize,
    ) -> Result<Self> {
        Self::with_options(n, t, num_ots, values_per_leaf, 2)
    }

    pub fn with_arity(n: usize, t: usize, num_ots: usize, tree_arity: usize) -> Result<Self> {
        Self::with_options(n, t, num_ots, 1, tree_arity)
    }

    pub fn with_options(
        n: usize,
        t: usize,
        num_ots: usize,
        values_per_leaf: usize,
        tree_arity: usize,
    ) -> Result<Self> {
        if n == 0 {
            return Err(ProtocolError::InvalidParams("n must be > 0".into()));
        }
        if num_ots == 0 {
            return Err(ProtocolError::InvalidParams("num_ots must be > 0".into()));
        }
        if values_per_leaf == 0 {
            return Err(ProtocolError::InvalidParams(
                "values_per_leaf must be > 0".into(),
            ));
        }
        if tree_arity != 2 && tree_arity != 4 {
            return Err(ProtocolError::InvalidParams(format!(
                "tree_arity must be 2 or 4, got {}",
                tree_arity
            )));
        }
        if n <= 2 * t {
            return Err(ProtocolError::InvalidParams(format!(
                "need n > 2t for honest majority, got n={}, t={}",
                n, t
            )));
        }
        let num_leaves_needed = num_ots.div_ceil(values_per_leaf);
        let tree_depth = if num_leaves_needed <= 1 {
            1
        } else if tree_arity == 4 {
            // ceil(log4(num_leaves_needed)) = ceil(log2(num_leaves_needed) / 2)
            let bits = usize::BITS as usize - (num_leaves_needed - 1).leading_zeros() as usize;
            bits.div_ceil(2)
        } else {
            usize::BITS as usize - (num_leaves_needed - 1).leading_zeros() as usize
        };
        Ok(SilentOtParams {
            n,
            t,
            num_ots,
            tree_depth,
            values_per_leaf,
            tree_arity,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PartySetupState {
    pub party_id: usize,
    pub n: usize,
    pub tree_depth: usize,
    pub num_ots: usize,
    pub values_per_leaf: usize,
    pub tree_arity: usize,
    pub my_seeds: Vec<Option<Block>>,
    pub my_commitments: Vec<Option<[u8; 32]>>,
    pub their_commitments: Vec<Option<[u8; 32]>>,
    pub my_puncture_indices: Vec<Option<usize>>,
    pub received_sibling_paths: Vec<Option<Vec<Block>>>,
    pub their_puncture_indices: Vec<Option<usize>>,
    pub revealed_seeds: Vec<Option<Block>>,
}

impl PartySetupState {
    /// Expected sibling path length in Blocks: depth for binary, 3*depth for 4-ary.
    pub fn expected_path_len(&self) -> usize {
        if self.tree_arity == 4 {
            self.tree_depth * 3
        } else {
            self.tree_depth
        }
    }

    pub fn num_leaves(&self) -> usize {
        if self.tree_arity == 4 {
            4usize.pow(self.tree_depth as u32)
        } else {
            1usize << self.tree_depth
        }
    }
}

pub struct DistributedSilentOt {
    pub params: SilentOtParams,
}

impl DistributedSilentOt {
    pub fn new(params: SilentOtParams) -> Self {
        DistributedSilentOt { params }
    }

    pub fn init_party<R: Rng>(&self, party_id: usize, rng: &mut R) -> PartySetupState {
        let n = self.params.n;
        let num_leaves = if self.params.tree_arity == 4 {
            4usize.pow(self.params.tree_depth as u32)
        } else {
            1usize << self.params.tree_depth
        };

        let mut my_seeds = vec![None; n];
        let mut my_commitments = vec![None; n];
        let mut my_puncture_indices = vec![None; n];

        for j in 0..n {
            if j == party_id {
                continue;
            }
            let seed = Block::random(rng);
            my_commitments[j] = Some(seed.commit_with_context(party_id, j));
            my_seeds[j] = Some(seed);
            my_puncture_indices[j] = Some(rng.gen_range(0..num_leaves));
        }

        PartySetupState {
            party_id,
            n,
            tree_depth: self.params.tree_depth,
            num_ots: self.params.num_ots,
            values_per_leaf: self.params.values_per_leaf,
            tree_arity: self.params.tree_arity,
            my_seeds,
            my_commitments,
            their_commitments: vec![None; n],
            my_puncture_indices,
            received_sibling_paths: vec![None; n],
            their_puncture_indices: vec![None; n],
            revealed_seeds: vec![None; n],
        }
    }

    /// Returns commitments to send: Vec<(recipient_id, commitment_hash)>
    pub fn round0_commitments(state: &PartySetupState) -> Vec<(usize, [u8; 32])> {
        let mut msgs = Vec::with_capacity(state.n - 1);
        for j in 0..state.n {
            if j == state.party_id {
                continue;
            }
            if let Some(commitment) = state.my_commitments[j] {
                msgs.push((j, commitment));
            }
        }
        msgs
    }

    /// Process received commitments: &[(sender_id, commitment_hash)]
    pub fn process_round0(
        state: &mut PartySetupState,
        commitments: &[(usize, [u8; 32])],
    ) -> Result<()> {
        // Pre-validate all messages before mutating state (atomic update)
        let mut seen = vec![false; state.n];
        for &(from, _) in commitments {
            if from >= state.n || from == state.party_id {
                return Err(ProtocolError::MaliciousParty(format!(
                    "invalid sender {} in commitment (n={})",
                    from, state.n
                )));
            }
            if state.their_commitments[from].is_some() || seen[from] {
                return Err(ProtocolError::MaliciousParty(format!(
                    "duplicate commitment from party {}",
                    from
                )));
            }
            seen[from] = true;
        }
        // All validated — apply atomically
        for &(from, commitment) in commitments {
            state.their_commitments[from] = Some(commitment);
        }
        Ok(())
    }

    /// Returns puncture choices to send: Vec<(recipient_id, puncture_index)>
    pub fn round1_puncture_choices(state: &PartySetupState) -> Vec<(usize, usize)> {
        let mut msgs = Vec::with_capacity(state.n - 1);
        for j in 0..state.n {
            if j == state.party_id {
                continue;
            }
            if let Some(idx) = state.my_puncture_indices[j] {
                msgs.push((j, idx));
            }
        }
        msgs
    }

    /// Process received puncture choices: &[(sender_id, puncture_index)]
    pub fn process_round1(state: &mut PartySetupState, choices: &[(usize, usize)]) -> Result<()> {
        // Pre-validate all messages before mutating state (atomic update)
        let num_leaves = state.num_leaves();
        let mut seen = vec![false; state.n];
        for &(from, index) in choices {
            if from >= state.n || from == state.party_id {
                return Err(ProtocolError::MaliciousParty(format!(
                    "invalid sender {} in puncture choice (n={})",
                    from, state.n
                )));
            }
            if index >= num_leaves {
                return Err(ProtocolError::MaliciousParty(format!(
                    "party {} sent puncture index {} >= num_leaves {}",
                    from, index, num_leaves
                )));
            }
            if state.their_puncture_indices[from].is_some() || seen[from] {
                return Err(ProtocolError::MaliciousParty(format!(
                    "duplicate puncture choice from party {}",
                    from
                )));
            }
            seen[from] = true;
        }
        // All validated — apply atomically
        for &(from, index) in choices {
            state.their_puncture_indices[from] = Some(index);
        }
        Ok(())
    }

    /// Returns sibling paths to send: Vec<(recipient_id, sibling_path)>
    pub fn round2_sibling_paths(state: &PartySetupState) -> Result<Vec<(usize, Vec<Block>)>> {
        let mut msgs = Vec::with_capacity(state.n - 1);

        for j in 0..state.n {
            if j == state.party_id {
                continue;
            }
            if let (Some(seed), Some(puncture_idx)) =
                (state.my_seeds[j], state.their_puncture_indices[j])
            {
                let flat_path = if state.tree_arity == 4 {
                    let tree4 = GgmTree4::new(state.tree_depth);
                    let path = tree4.compute_sibling_path(&seed, puncture_idx)?;
                    path.iter().flat_map(|s| s.iter().copied()).collect()
                } else {
                    let tree = GgmTree::new(state.tree_depth);
                    tree.compute_sibling_path(&seed, puncture_idx)?
                };
                msgs.push((j, flat_path));
            }
        }

        Ok(msgs)
    }

    /// Process received sibling paths: &[(sender_id, sibling_path)]
    pub fn process_round2(
        state: &mut PartySetupState,
        paths: &[(usize, Vec<Block>)],
    ) -> Result<()> {
        let expected_len = state.expected_path_len();
        let mut seen = vec![false; state.n];
        for (from, sibling_path) in paths {
            let from = *from;
            if from >= state.n || from == state.party_id {
                return Err(ProtocolError::MaliciousParty(format!(
                    "invalid sender {} in sibling path (n={})",
                    from, state.n
                )));
            }
            if sibling_path.len() != expected_len {
                return Err(ProtocolError::MaliciousParty(format!(
                    "party {} sent sibling path of length {}, expected {}",
                    from,
                    sibling_path.len(),
                    expected_len
                )));
            }
            if state.received_sibling_paths[from].is_some() || seen[from] {
                return Err(ProtocolError::MaliciousParty(format!(
                    "duplicate sibling path from party {}",
                    from
                )));
            }
            seen[from] = true;
        }
        // All validated — apply atomically
        for (from, path) in paths {
            state.received_sibling_paths[*from] = Some(path.clone());
        }
        Ok(())
    }

    /// Returns seed reveals to send: Vec<(recipient_id, seed)>
    pub fn round3_seed_reveals(state: &PartySetupState) -> Vec<(usize, Block)> {
        let mut msgs = Vec::with_capacity(state.n - 1);
        for j in 0..state.n {
            if j == state.party_id {
                continue;
            }
            if let Some(seed) = state.my_seeds[j] {
                msgs.push((j, seed));
            }
        }
        msgs
    }

    /// Process received seed reveals: &[(sender_id, seed)]
    pub fn process_round3(state: &mut PartySetupState, reveals: &[(usize, Block)]) -> Result<()> {
        // Pre-validate all messages before mutating state (atomic update)
        let mut seen = vec![false; state.n];
        for &(from, ref seed) in reveals {
            if from >= state.n || from == state.party_id {
                return Err(ProtocolError::MaliciousParty(format!(
                    "invalid sender {} in seed reveal (n={})",
                    from, state.n
                )));
            }
            if state.revealed_seeds[from].is_some() || seen[from] {
                return Err(ProtocolError::MaliciousParty(format!(
                    "duplicate seed reveal from party {}",
                    from
                )));
            }
            match state.their_commitments[from] {
                Some(commitment) => {
                    if !seed.verify_commitment_with_context(&commitment, from, state.party_id) {
                        return Err(ProtocolError::MaliciousParty(format!(
                            "party {} revealed seed that doesn't match commitment",
                            from
                        )));
                    }
                }
                None => {
                    return Err(ProtocolError::MaliciousParty(format!(
                        "seed reveal from party {} without prior commitment",
                        from
                    )));
                }
            }
            seen[from] = true;
        }
        // All validated — apply atomically
        for &(from, seed) in reveals {
            state.revealed_seeds[from] = Some(seed);
        }
        Ok(())
    }

    // ── 2-Round Protocol ─────────────────────────────────────────────
    //
    // Merges the 4-round protocol into 2 rounds:
    //   Round A = Round 0 + Round 1 (commitments + puncture choices)
    //   Round B = Round 2 + Round 3 (sibling paths + seed reveals)
    //
    // Security: commitment binding is verified in process_round_b (not deferred).
    // The expand() function still verifies sibling path integrity.

    /// 2-round: returns messages for Round A (commitment + puncture choice per peer).
    pub fn round_a_messages(state: &PartySetupState) -> Vec<(usize, [u8; 32], usize)> {
        let mut msgs = Vec::with_capacity(state.n - 1);
        for j in 0..state.n {
            if j == state.party_id {
                continue;
            }
            if let (Some(commitment), Some(punct_idx)) =
                (state.my_commitments[j], state.my_puncture_indices[j])
            {
                msgs.push((j, commitment, punct_idx));
            }
        }
        msgs
    }

    /// 2-round: process received Round A messages (commitment + puncture choice).
    pub fn process_round_a(
        state: &mut PartySetupState,
        messages: &[(usize, [u8; 32], usize)],
    ) -> Result<()> {
        let num_leaves = state.num_leaves();
        let mut seen = vec![false; state.n];
        for &(from, _, index) in messages {
            if from >= state.n || from == state.party_id {
                return Err(ProtocolError::MaliciousParty(format!(
                    "invalid sender {} in round_a (n={})",
                    from, state.n
                )));
            }
            if index >= num_leaves {
                return Err(ProtocolError::MaliciousParty(format!(
                    "party {} sent puncture index {} >= num_leaves {}",
                    from, index, num_leaves
                )));
            }
            if state.their_commitments[from].is_some() || seen[from] {
                return Err(ProtocolError::MaliciousParty(format!(
                    "duplicate round_a message from party {}",
                    from
                )));
            }
            seen[from] = true;
        }
        for &(from, commitment, index) in messages {
            state.their_commitments[from] = Some(commitment);
            state.their_puncture_indices[from] = Some(index);
        }
        Ok(())
    }

    /// 2-round: returns messages for Round B (sibling path + seed reveal per peer).
    /// Requires Round A to have been processed (needs their_puncture_indices).
    pub fn round_b_messages(state: &PartySetupState) -> Result<Vec<(usize, Vec<Block>, Block)>> {
        let mut msgs = Vec::with_capacity(state.n - 1);
        for j in 0..state.n {
            if j == state.party_id {
                continue;
            }
            if let (Some(seed), Some(puncture_idx)) =
                (state.my_seeds[j], state.their_puncture_indices[j])
            {
                let flat_path = if state.tree_arity == 4 {
                    let tree4 = GgmTree4::new(state.tree_depth);
                    let path = tree4.compute_sibling_path(&seed, puncture_idx)?;
                    path.iter().flat_map(|s| s.iter().copied()).collect()
                } else {
                    let tree = GgmTree::new(state.tree_depth);
                    tree.compute_sibling_path(&seed, puncture_idx)?
                };
                msgs.push((j, flat_path, seed));
            }
        }
        Ok(msgs)
    }

    /// 2-round: process received Round B messages (sibling path + seed reveal).
    /// Verifies seed against commitment from Round A immediately.
    pub fn process_round_b(
        state: &mut PartySetupState,
        messages: &[(usize, Vec<Block>, Block)],
    ) -> Result<()> {
        let expected_len = state.expected_path_len();
        let mut seen = vec![false; state.n];
        for (from, sibling_path, seed) in messages {
            let from = *from;
            if from >= state.n || from == state.party_id {
                return Err(ProtocolError::MaliciousParty(format!(
                    "invalid sender {} in round_b (n={})",
                    from, state.n
                )));
            }
            if sibling_path.len() != expected_len {
                return Err(ProtocolError::MaliciousParty(format!(
                    "party {} sent sibling path of length {}, expected {}",
                    from,
                    sibling_path.len(),
                    expected_len
                )));
            }
            if state.received_sibling_paths[from].is_some() || seen[from] {
                return Err(ProtocolError::MaliciousParty(format!(
                    "duplicate round_b message from party {}",
                    from
                )));
            }
            // Verify seed against commitment immediately
            match state.their_commitments[from] {
                Some(commitment) => {
                    if !seed.verify_commitment_with_context(&commitment, from, state.party_id) {
                        return Err(ProtocolError::MaliciousParty(format!(
                            "party {} revealed seed that doesn't match commitment",
                            from
                        )));
                    }
                }
                None => {
                    return Err(ProtocolError::MaliciousParty(format!(
                        "round_b from party {} without prior commitment (round_a missing)",
                        from
                    )));
                }
            }
            seen[from] = true;
        }
        for (from, path, seed) in messages {
            state.received_sibling_paths[*from] = Some(path.clone());
            state.revealed_seeds[*from] = Some(*seed);
        }
        Ok(())
    }

    /// Expand all parties' OT correlations in a single flat parallel pass.
    ///
    /// Instead of nested parallelism (5 parties × 4 peers), this creates
    /// a flat work list of all (party, peer) pairs and processes them in one
    /// par_iter, giving rayon better scheduling control.
    pub fn expand_all(states: &[PartySetupState]) -> Result<Vec<ExpandedCorrelations>> {
        if states.is_empty() {
            return Ok(Vec::new());
        }
        let n = states[0].n;
        let num_ots = states[0].num_ots;
        let tree_depth = states[0].tree_depth;
        let arity = states[0].tree_arity;
        let k = states[0].values_per_leaf;
        let num_leaves = states[0].num_leaves();

        if num_ots > num_leaves * k {
            return Err(ProtocolError::InvalidParams(format!(
                "num_ots ({}) > num_leaves * k ({} * {})",
                num_ots, num_leaves, k
            )));
        }

        for s in states {
            Self::validate_state(s)?;
        }

        let mut work_items: Vec<(usize, usize)> = Vec::with_capacity(states.len() * (n - 1));
        for (pi, state) in states.iter().enumerate() {
            for j in 0..n {
                if j != state.party_id {
                    work_items.push((pi, j));
                }
            }
        }

        let peer_contribs: std::result::Result<Vec<(usize, Vec<u64>)>, ProtocolError> = work_items
            .par_iter()
            .map(|&(pi, j)| {
                let state = &states[pi];

                // Verify sibling path
                if let (Some(revealed_seed), Some(received_path), Some(puncture_idx)) = (
                    state.revealed_seeds[j],
                    &state.received_sibling_paths[j],
                    state.my_puncture_indices[j],
                ) {
                    if arity == 4 {
                        let tree4 = GgmTree4::new(tree_depth);
                        let expected = tree4.compute_sibling_path(&revealed_seed, puncture_idx)?;
                        let expected_flat: Vec<Block> =
                            expected.iter().flat_map(|s| s.iter().copied()).collect();
                        for (i, (e, r)) in
                            expected_flat.iter().zip(received_path.iter()).enumerate()
                        {
                            if e != r {
                                return Err(ProtocolError::MaliciousParty(format!(
                                    "party {} sent fake sibling path at block {}",
                                    j, i
                                )));
                            }
                        }
                    } else {
                        let tree = GgmTree::new(tree_depth);
                        let expected = tree.compute_sibling_path(&revealed_seed, puncture_idx)?;
                        for (level, (e, r)) in expected.iter().zip(received_path.iter()).enumerate()
                        {
                            if e != r {
                                return Err(ProtocolError::MaliciousParty(format!(
                                    "party {} sent fake sibling path at level {}",
                                    j, level
                                )));
                            }
                        }
                    }
                }

                if arity == 4 {
                    let tree4 = GgmTree4::new(tree_depth);
                    let mut contrib = if let Some(seed) = state.my_seeds[j] {
                        tree4.expand_full_to_u64(&seed, num_ots)
                    } else {
                        vec![0u64; num_ots]
                    };
                    if let (Some(received_path), Some(puncture_idx)) = (
                        &state.received_sibling_paths[j],
                        state.my_puncture_indices[j],
                    ) {
                        let path_4ary: Vec<[Block; 3]> = received_path
                            .chunks_exact(3)
                            .map(|c| [c[0], c[1], c[2]])
                            .collect();
                        tree4.reconstruct_accumulate_u64(
                            &path_4ary,
                            puncture_idx,
                            num_ots,
                            &mut contrib,
                        )?;
                    }
                    Ok((pi, contrib))
                } else {
                    let tree = GgmTree::new(tree_depth);
                    let mut contrib = if let Some(seed) = state.my_seeds[j] {
                        tree.expand_full_to_u64_multi(&seed, num_ots, k)
                    } else {
                        vec![0u64; num_ots]
                    };
                    if let (Some(received_path), Some(puncture_idx)) = (
                        &state.received_sibling_paths[j],
                        state.my_puncture_indices[j],
                    ) {
                        tree.reconstruct_accumulate_u64_multi(
                            received_path,
                            puncture_idx,
                            num_ots,
                            &mut contrib,
                            k,
                        )?;
                    }
                    Ok((pi, contrib))
                }
            })
            .collect();

        let peer_contribs = peer_contribs?;

        // Sum contributions per party using chunked reduction for cache locality
        let mut results: Vec<Vec<u64>> = (0..states.len()).map(|_| vec![0u64; num_ots]).collect();
        const SUM_CHUNK: usize = 32768;
        let num_sum_chunks = num_ots.div_ceil(SUM_CHUNK);
        for ci in 0..num_sum_chunks {
            let start = ci * SUM_CHUNK;
            let end = (start + SUM_CHUNK).min(num_ots);
            for (pi, contrib) in &peer_contribs {
                for idx in start..end {
                    results[*pi][idx] = Fp::add_raw(results[*pi][idx], contrib[idx]);
                }
            }
        }

        Ok(results
            .into_iter()
            .enumerate()
            .map(|(i, raw_values)| ExpandedCorrelations {
                party_id: states[i].party_id,
                num_ots,
                random_values: raw_values,
            })
            .collect())
    }

    pub fn validate_state(state: &PartySetupState) -> Result<()> {
        for j in 0..state.n {
            if j == state.party_id {
                continue;
            }
            if state.their_commitments[j].is_none() {
                return Err(ProtocolError::MaliciousParty(format!(
                    "missing commitment from party {}",
                    j
                )));
            }
            if state.their_puncture_indices[j].is_none() {
                return Err(ProtocolError::MaliciousParty(format!(
                    "missing puncture index from party {}",
                    j
                )));
            }
            if state.received_sibling_paths[j].is_none() {
                return Err(ProtocolError::MaliciousParty(format!(
                    "missing sibling path from party {}",
                    j
                )));
            }
            if state.revealed_seeds[j].is_none() {
                return Err(ProtocolError::MaliciousParty(format!(
                    "missing revealed seed from party {}",
                    j
                )));
            }
        }
        Ok(())
    }

    pub fn expand(state: &PartySetupState) -> Result<ExpandedCorrelations> {
        Self::validate_state(state)?;

        let n = state.n;
        let num_ots = state.num_ots;
        let tree_depth = state.tree_depth;
        let arity = state.tree_arity;
        let k = state.values_per_leaf;
        let num_leaves = state.num_leaves();

        if num_ots > num_leaves * k {
            return Err(ProtocolError::InvalidParams(format!(
                "num_ots ({}) > num_leaves * k ({} * {}): would reuse GGM leaves",
                num_ots, num_leaves, k
            )));
        }

        let peers: Vec<usize> = (0..n).filter(|&j| j != state.party_id).collect();

        let contributions: std::result::Result<Vec<Vec<u64>>, ProtocolError> = peers
            .par_iter()
            .map(|&j| {
                // Verify sibling path against revealed seed
                if let (Some(revealed_seed), Some(received_path), Some(puncture_idx)) = (
                    state.revealed_seeds[j],
                    &state.received_sibling_paths[j],
                    state.my_puncture_indices[j],
                ) {
                    if arity == 4 {
                        let tree4 = GgmTree4::new(tree_depth);
                        let expected = tree4.compute_sibling_path(&revealed_seed, puncture_idx)?;
                        let expected_flat: Vec<Block> =
                            expected.iter().flat_map(|s| s.iter().copied()).collect();
                        for (i, (e, r)) in
                            expected_flat.iter().zip(received_path.iter()).enumerate()
                        {
                            if e != r {
                                return Err(ProtocolError::MaliciousParty(format!(
                                    "party {} sent fake sibling path: mismatch at block {}",
                                    j, i
                                )));
                            }
                        }
                    } else {
                        let tree = GgmTree::new(tree_depth);
                        let expected = tree.compute_sibling_path(&revealed_seed, puncture_idx)?;
                        for (level, (e, r)) in expected.iter().zip(received_path.iter()).enumerate()
                        {
                            if e != r {
                                return Err(ProtocolError::MaliciousParty(format!(
                                    "party {} sent fake sibling path: mismatch at level {}",
                                    j, level
                                )));
                            }
                        }
                    }
                }

                if arity == 4 {
                    let tree4 = GgmTree4::new(tree_depth);
                    let mut contrib = if let Some(seed) = state.my_seeds[j] {
                        tree4.expand_full_to_u64(&seed, num_ots)
                    } else {
                        vec![0u64; num_ots]
                    };
                    if let (Some(received_path), Some(puncture_idx)) = (
                        &state.received_sibling_paths[j],
                        state.my_puncture_indices[j],
                    ) {
                        // Unflatten: Vec<Block> → Vec<[Block; 3]>
                        let path_4ary: Vec<[Block; 3]> = received_path
                            .chunks_exact(3)
                            .map(|c| [c[0], c[1], c[2]])
                            .collect();
                        tree4.reconstruct_accumulate_u64(
                            &path_4ary,
                            puncture_idx,
                            num_ots,
                            &mut contrib,
                        )?;
                    }
                    Ok(contrib)
                } else {
                    let tree = GgmTree::new(tree_depth);
                    let mut contrib = if let Some(seed) = state.my_seeds[j] {
                        tree.expand_full_to_u64_multi(&seed, num_ots, k)
                    } else {
                        vec![0u64; num_ots]
                    };
                    if let (Some(received_path), Some(puncture_idx)) = (
                        &state.received_sibling_paths[j],
                        state.my_puncture_indices[j],
                    ) {
                        tree.reconstruct_accumulate_u64_multi(
                            received_path,
                            puncture_idx,
                            num_ots,
                            &mut contrib,
                            k,
                        )?;
                    }
                    Ok(contrib)
                }
            })
            .collect();

        let contributions = contributions?;

        // Sum all peer contributions using chunked reduction for cache locality.
        let random_values: Vec<u64> = if contributions.len() <= 1 {
            contributions
                .into_iter()
                .next()
                .unwrap_or_else(|| vec![0u64; num_ots])
        } else {
            const SUM_CHUNK: usize = 32768;
            let num_peers = contributions.len();
            let num_sum_chunks = num_ots.div_ceil(SUM_CHUNK);
            let mut result = vec![0u64; num_ots];
            for ci in 0..num_sum_chunks {
                let start = ci * SUM_CHUNK;
                let end = (start + SUM_CHUNK).min(num_ots);
                for p in 0..num_peers {
                    for idx in start..end {
                        result[idx] = Fp::add_raw(result[idx], contributions[p][idx]);
                    }
                }
            }
            result
        };

        Ok(ExpandedCorrelations {
            party_id: state.party_id,
            num_ots,
            random_values,
        })
    }
}

#[derive(Clone, Debug)]
pub struct ExpandedCorrelations {
    pub party_id: usize,
    num_ots: usize,
    random_values: Vec<u64>,
}

impl ExpandedCorrelations {
    #[inline]
    pub fn get_random(&self, index: usize) -> Fp {
        assert!(
            index < self.num_ots,
            "get_random: index {} out of range (num_ots={})",
            index,
            self.num_ots
        );
        Fp::from_reduced(self.random_values[index])
    }

    #[inline]
    pub fn get_random_raw(&self, index: usize) -> u64 {
        assert!(
            index < self.num_ots,
            "get_random_raw: index {} out of range (num_ots={})",
            index,
            self.num_ots
        );
        self.random_values[index]
    }

    /// Number of OT correlations available.
    #[inline]
    pub fn num_ots(&self) -> usize {
        self.num_ots
    }

    /// Unchecked raw access for validated hot loops.
    /// SAFETY: caller must ensure `index < self.num_ots`.
    #[inline]
    pub unsafe fn get_random_raw_unchecked(&self, index: usize) -> u64 {
        debug_assert!(index < self.num_ots);
        *self.random_values.get_unchecked(index)
    }

    /// Create ExpandedCorrelations directly from random values (no OT protocol).
    /// For local simulation / benchmarking where OT security properties are not needed.
    /// Uses fast SplitMix64 since these are not security-critical in local mode.
    pub fn from_random<R: Rng>(party_id: usize, num_ots: usize, rng: &mut R) -> Self {
        let seed: u64 = rng.gen();
        let mut mix = crate::field::SplitMix64::new(seed);
        let random_values: Vec<u64> = (0..num_ots).map(|_| mix.next_fp()).collect();
        ExpandedCorrelations {
            party_id,
            num_ots,
            random_values,
        }
    }

    /// Direct access to the raw u64 random values slice.
    #[inline]
    pub fn raw_values(&self) -> &[u64] {
        &self.random_values
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    /// Pre-bucket round messages by recipient and dispatch to each party.
    fn run_all_rounds(states: &mut [PartySetupState]) {
        let n = states.len();

        // Round 0: commitments
        let mut r0: Vec<Vec<(usize, [u8; 32])>> = vec![Vec::new(); n];
        for s in states.iter() {
            for (to, c) in DistributedSilentOt::round0_commitments(s) {
                r0[to].push((s.party_id, c));
            }
        }
        for (i, s) in states.iter_mut().enumerate() {
            DistributedSilentOt::process_round0(s, &r0[i]).unwrap();
        }

        // Round 1: puncture choices
        let mut r1: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
        for s in states.iter() {
            for (to, idx) in DistributedSilentOt::round1_puncture_choices(s) {
                r1[to].push((s.party_id, idx));
            }
        }
        for (i, s) in states.iter_mut().enumerate() {
            DistributedSilentOt::process_round1(s, &r1[i]).unwrap();
        }

        // Round 2: sibling paths
        let mut r2: Vec<Vec<(usize, Vec<Block>)>> = vec![Vec::new(); n];
        for s in states.iter() {
            for (to, path) in DistributedSilentOt::round2_sibling_paths(s).unwrap() {
                r2[to].push((s.party_id, path));
            }
        }
        for (i, s) in states.iter_mut().enumerate() {
            DistributedSilentOt::process_round2(s, &r2[i]).unwrap();
        }

        // Round 3: seed reveals
        let mut r3: Vec<Vec<(usize, Block)>> = vec![Vec::new(); n];
        for s in states.iter() {
            for (to, seed) in DistributedSilentOt::round3_seed_reveals(s) {
                r3[to].push((s.party_id, seed));
            }
        }
        for (i, s) in states.iter_mut().enumerate() {
            DistributedSilentOt::process_round3(s, &r3[i]).unwrap();
        }
    }

    /// Run all rounds but return the per-party sibling path buckets
    /// so the caller can tamper before processing round 2.
    fn run_rounds_0_1(states: &mut [PartySetupState]) {
        let n = states.len();

        let mut r0: Vec<Vec<(usize, [u8; 32])>> = vec![Vec::new(); n];
        for s in states.iter() {
            for (to, c) in DistributedSilentOt::round0_commitments(s) {
                r0[to].push((s.party_id, c));
            }
        }
        for (i, s) in states.iter_mut().enumerate() {
            DistributedSilentOt::process_round0(s, &r0[i]).unwrap();
        }

        let mut r1: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
        for s in states.iter() {
            for (to, idx) in DistributedSilentOt::round1_puncture_choices(s) {
                r1[to].push((s.party_id, idx));
            }
        }
        for (i, s) in states.iter_mut().enumerate() {
            DistributedSilentOt::process_round1(s, &r1[i]).unwrap();
        }
    }

    #[test]
    fn test_prg_deterministic() {
        let seed = Block([1u8; 16]);
        let (l1, r1) = prg_expand(&seed);
        let (l2, r2) = prg_expand(&seed);
        assert_eq!(l1, l2);
        assert_eq!(r1, r2);
        assert_ne!(l1, r1);
    }

    #[test]
    fn test_ggm_tree_expansion() {
        let tree = GgmTree::new(4);
        let seed = Block([42u8; 16]);
        let leaves = tree.expand_full(&seed);
        assert_eq!(leaves.len(), 16);
        for i in 0..16 {
            for j in (i + 1)..16 {
                assert_ne!(leaves[i], leaves[j]);
            }
        }
    }

    #[test]
    fn test_sibling_path_reconstruction() {
        let tree = GgmTree::new(4);
        let seed = Block([42u8; 16]);
        let full = tree.expand_full(&seed);

        for puncture_idx in 0..16 {
            let sibling_path = tree.compute_sibling_path(&seed, puncture_idx).unwrap();
            let reconstructed = tree
                .reconstruct_from_siblings(&sibling_path, puncture_idx)
                .unwrap();

            for i in 0..16 {
                if i == puncture_idx {
                    assert_eq!(reconstructed[i], Block::ZERO);
                } else {
                    assert_eq!(reconstructed[i], full[i]);
                }
            }
        }
    }

    #[test]
    fn test_commitment_verification() {
        let seed = Block([42u8; 16]);
        let commitment = seed.commit();
        assert!(seed.verify_commitment(&commitment));

        let bad_seed = Block([43u8; 16]);
        assert!(!bad_seed.verify_commitment(&commitment));
    }

    #[test]
    fn test_distributed_protocol_produces_random_values() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let params = SilentOtParams::new(5, 1, 16).unwrap();
        let protocol = DistributedSilentOt::new(params);

        let mut states: Vec<PartySetupState> =
            (0..5).map(|i| protocol.init_party(i, &mut rng)).collect();

        run_all_rounds(&mut states);

        let correlations: Vec<ExpandedCorrelations> = states
            .iter()
            .map(|s| DistributedSilentOt::expand(s).unwrap())
            .collect();

        // Each party should produce non-trivial random values
        for corr in &correlations {
            let vals: Vec<Fp> = (0..16).map(|k| corr.get_random(k)).collect();
            let all_zero = vals.iter().all(|v| *v == Fp::ZERO);
            assert!(
                !all_zero,
                "party {} produced all-zero values",
                corr.party_id
            );
        }

        // Different parties should get different random values (different puncture sets)
        let v0: Vec<Fp> = (0..16).map(|k| correlations[0].get_random(k)).collect();
        let v1: Vec<Fp> = (0..16).map(|k| correlations[1].get_random(k)).collect();
        assert_ne!(
            v0, v1,
            "different parties should have different random values"
        );
    }

    #[test]
    fn test_invalid_params() {
        assert!(SilentOtParams::new(0, 0, 10).is_err());
        assert!(SilentOtParams::new(5, 0, 0).is_err());
        assert!(SilentOtParams::new(2, 1, 10).is_err());
        assert!(SilentOtParams::new(3, 1, 10).is_ok());
    }

    #[test]
    fn test_invalid_sibling_path_length() {
        let tree = GgmTree::new(4);
        let short_path = vec![Block::ZERO; 3];
        let result = tree.reconstruct_from_siblings(&short_path, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_out_of_range_puncture_index() {
        let tree = GgmTree::new(4);
        let seed = Block([1u8; 16]);
        assert!(tree.compute_sibling_path(&seed, 16).is_err());
        assert!(tree.compute_sibling_path(&seed, 15).is_ok());
    }

    #[test]
    fn test_validate_state_detects_missing_messages() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let params = SilentOtParams::new(5, 1, 16).unwrap();
        let protocol = DistributedSilentOt::new(params);
        let state = protocol.init_party(0, &mut rng);

        let result = DistributedSilentOt::validate_state(&state);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("missing commitment"));
    }

    #[test]
    fn test_validate_state_passes_after_all_rounds() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let params = SilentOtParams::new(5, 1, 16).unwrap();
        let protocol = DistributedSilentOt::new(params);

        let mut states: Vec<PartySetupState> =
            (0..5).map(|i| protocol.init_party(i, &mut rng)).collect();

        run_all_rounds(&mut states);

        for state in &states {
            DistributedSilentOt::validate_state(state).unwrap();
        }
    }

    #[test]
    fn test_duplicate_commitment_detected() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let params = SilentOtParams::new(5, 1, 16).unwrap();
        let protocol = DistributedSilentOt::new(params);
        let mut state = protocol.init_party(0, &mut rng);

        let seed1 = Block([1u8; 16]);
        let seed2 = Block([2u8; 16]);
        let messages = vec![(1, seed1.commit()), (1, seed2.commit())];
        let result = DistributedSilentOt::process_round0(&mut state, &messages);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("duplicate commitment"));
    }

    #[test]
    fn test_commitment_prevents_equivocation() {
        let seed_a = Block([1u8; 16]);
        let seed_b = Block([2u8; 16]);
        let commitment = seed_a.commit();
        assert!(!seed_b.verify_commitment(&commitment));
    }

    // ===== REGRESSION TESTS FOR SECURITY FINDINGS =====

    #[test]
    fn test_oob_from_party_index_round0() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let params = SilentOtParams::new(5, 1, 16).unwrap();
        let protocol = DistributedSilentOt::new(params);
        let mut state = protocol.init_party(0, &mut rng);

        let result = DistributedSilentOt::process_round0(&mut state, &[(99, [0u8; 32])]);
        assert!(
            result.is_err(),
            "out-of-bounds from index should return error, not panic"
        );
    }

    #[test]
    fn test_oob_from_party_index_round1() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let params = SilentOtParams::new(5, 1, 16).unwrap();
        let protocol = DistributedSilentOt::new(params);
        let mut state = protocol.init_party(0, &mut rng);

        let result = DistributedSilentOt::process_round1(&mut state, &[(99, 0)]);
        assert!(
            result.is_err(),
            "out-of-bounds from index should return error, not panic"
        );
    }

    #[test]
    fn test_oob_from_party_index_round2() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let params = SilentOtParams::new(5, 1, 16).unwrap();
        let protocol = DistributedSilentOt::new(params);
        let mut state = protocol.init_party(0, &mut rng);

        let result = DistributedSilentOt::process_round2(&mut state, &[(99, vec![Block::ZERO; 4])]);
        assert!(
            result.is_err(),
            "out-of-bounds from index should return error, not panic"
        );
    }

    #[test]
    fn test_oob_from_party_index_round3() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let params = SilentOtParams::new(5, 1, 16).unwrap();
        let protocol = DistributedSilentOt::new(params);
        let mut state = protocol.init_party(0, &mut rng);

        let result = DistributedSilentOt::process_round3(&mut state, &[(99, Block::ZERO)]);
        assert!(
            result.is_err(),
            "out-of-bounds from index should return error, not panic"
        );
    }

    #[test]
    fn test_commitment_lacks_context_binding() {
        let seed = Block([42u8; 16]);
        let commitment_0_to_1 = seed.commit_with_context(0, 1);
        let commitment_0_to_2 = seed.commit_with_context(0, 2);
        let commitment_1_to_0 = seed.commit_with_context(1, 0);

        assert_ne!(
            commitment_0_to_1, commitment_0_to_2,
            "commitments with different targets should differ"
        );
        assert_ne!(
            commitment_0_to_1, commitment_1_to_0,
            "commitments with swapped parties should differ"
        );
    }

    #[test]
    fn test_puncture_index_always_in_leaf_range() {
        let params = SilentOtParams::new(5, 1, 17).unwrap();
        assert_eq!(params.tree_depth, 5);
        let num_leaves = 1usize << params.tree_depth;
        assert_eq!(num_leaves, 32);

        let protocol = DistributedSilentOt::new(params);
        for seed_val in 0..100u64 {
            let mut rng_iter = ChaCha20Rng::seed_from_u64(seed_val);
            let state = protocol.init_party(0, &mut rng_iter);
            for j in 0..5 {
                if j == 0 {
                    continue;
                }
                if let Some(idx) = state.my_puncture_indices[j] {
                    assert!(
                        idx < num_leaves,
                        "puncture index {} >= num_leaves {} (seed={})",
                        idx,
                        num_leaves,
                        seed_val
                    );
                }
            }
        }
    }

    #[test]
    fn test_receiver_punctured_leaf_excluded_from_random() {
        // Verify the GGM puncture mechanism: reconstructed tree has Block::ZERO at
        // the punctured position, so the receiver's field element there differs from
        // the sender's — ensuring the OT correlation gap.
        let tree = GgmTree::new(4);
        let seed = Block([42u8; 16]);
        let full_leaves = tree.expand_full(&seed);

        for punct in 0..16 {
            let path = tree.compute_sibling_path(&seed, punct).unwrap();
            let recon = tree.reconstruct_from_siblings(&path, punct).unwrap();

            // Punctured position: receiver gets Block::ZERO, NOT the real leaf
            assert_eq!(recon[punct], Block::ZERO);
            let sender_fp = full_leaves[punct].to_field_element(punct as u64);
            let receiver_fp = Block::ZERO.to_field_element(punct as u64);
            assert_ne!(sender_fp, receiver_fp, "punctured position should differ");

            // All other positions match
            for k in 0..16 {
                if k != punct {
                    assert_eq!(recon[k], full_leaves[k]);
                }
            }
        }
    }

    #[test]
    fn test_self_message_rejected() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let params = SilentOtParams::new(5, 1, 16).unwrap();
        let protocol = DistributedSilentOt::new(params);
        let mut state = protocol.init_party(0, &mut rng);

        let result = DistributedSilentOt::process_round0(&mut state, &[(0, [0u8; 32])]);
        assert!(result.is_err(), "self-messages should be rejected");
        assert!(format!("{}", result.unwrap_err()).contains("invalid sender"));
    }

    #[test]
    fn test_seed_reveal_without_commitment_rejected() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let params = SilentOtParams::new(5, 1, 16).unwrap();
        let protocol = DistributedSilentOt::new(params);
        let mut state = protocol.init_party(0, &mut rng);

        let seed = Block([42u8; 16]);
        let result = DistributedSilentOt::process_round3(&mut state, &[(1, seed)]);
        assert!(
            result.is_err(),
            "seed reveal without prior commitment should be rejected"
        );
        assert!(
            format!("{}", result.unwrap_err()).contains("without prior commitment"),
            "error should mention missing commitment"
        );
    }

    #[test]
    fn test_puncture_index_outside_leaf_range_rejected() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let params = SilentOtParams::new(5, 1, 17).unwrap();
        assert_eq!(params.tree_depth, 5);
        let num_leaves = 1usize << params.tree_depth; // 32
        let protocol = DistributedSilentOt::new(params);
        let mut state = protocol.init_party(0, &mut rng);

        // puncture index 32 >= num_leaves (32)
        let result = DistributedSilentOt::process_round1(&mut state, &[(1, num_leaves)]);
        assert!(
            result.is_err(),
            "puncture index >= num_leaves should be rejected"
        );
        assert!(format!("{}", result.unwrap_err()).contains("num_leaves"));

        // puncture at num_leaves - 1 (last valid) should be accepted
        assert!(DistributedSilentOt::process_round1(&mut state, &[(1, num_leaves - 1)]).is_ok());
    }

    #[test]
    fn test_fake_sibling_path_detected() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let params = SilentOtParams::new(5, 1, 16).unwrap();
        let protocol = DistributedSilentOt::new(params);
        let n = 5;

        let mut states: Vec<PartySetupState> =
            (0..n).map(|i| protocol.init_party(i, &mut rng)).collect();

        run_rounds_0_1(&mut states);

        // Collect sibling paths, tamper with the one from party 1 → party 0
        let mut r2: Vec<Vec<(usize, Vec<Block>)>> = vec![Vec::new(); n];
        for s in states.iter() {
            for (to, path) in DistributedSilentOt::round2_sibling_paths(s).unwrap() {
                r2[to].push((s.party_id, path));
            }
        }
        // Tamper: modify the sibling path from party 1 in party 0's bucket
        for (from, ref mut path) in &mut r2[0] {
            if *from == 1 {
                path[0] = Block([0xff; 16]);
            }
        }
        for (i, s) in states.iter_mut().enumerate() {
            DistributedSilentOt::process_round2(s, &r2[i]).unwrap();
        }

        // Round 3
        let mut r3: Vec<Vec<(usize, Block)>> = vec![Vec::new(); n];
        for s in states.iter() {
            for (to, seed) in DistributedSilentOt::round3_seed_reveals(s) {
                r3[to].push((s.party_id, seed));
            }
        }
        for (i, s) in states.iter_mut().enumerate() {
            DistributedSilentOt::process_round3(s, &r3[i]).unwrap();
        }

        let result = DistributedSilentOt::expand(&states[0]);
        assert!(result.is_err(), "fake sibling path should be detected");
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("fake sibling path"), "error: {}", err_msg);

        for i in 1..n {
            assert!(
                DistributedSilentOt::expand(&states[i]).is_ok(),
                "party {} should expand successfully",
                i
            );
        }
    }

    #[test]
    fn test_process_round0_atomic_on_error() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let params = SilentOtParams::new(5, 1, 16).unwrap();
        let protocol = DistributedSilentOt::new(params);
        let mut state = protocol.init_party(0, &mut rng);

        let seed1 = Block([1u8; 16]);
        let seed2 = Block([2u8; 16]);
        let messages = vec![
            (1, seed1.commit_with_context(1, 0)),
            (1, seed2.commit_with_context(1, 0)),
        ];
        let result = DistributedSilentOt::process_round0(&mut state, &messages);
        assert!(result.is_err(), "duplicate should cause error");
        assert!(
            state.their_commitments[1].is_none(),
            "state should not be partially modified after error"
        );
    }

    #[test]
    fn test_process_round1_atomic_on_error() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let params = SilentOtParams::new(5, 1, 16).unwrap();
        let protocol = DistributedSilentOt::new(params);
        let mut state = protocol.init_party(0, &mut rng);

        let messages = vec![(1, 5), (99, 3)];
        let result = DistributedSilentOt::process_round1(&mut state, &messages);
        assert!(result.is_err());
        assert!(
            state.their_puncture_indices[1].is_none(),
            "state should not be partially modified after error"
        );
    }

    #[test]
    fn test_process_round2_atomic_on_error() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let params = SilentOtParams::new(5, 1, 16).unwrap();
        let protocol = DistributedSilentOt::new(params);
        let mut state = protocol.init_party(0, &mut rng);

        let messages = vec![
            (1, vec![Block::ZERO; 4]), // correct depth
            (2, vec![Block::ZERO; 3]), // wrong depth
        ];
        let result = DistributedSilentOt::process_round2(&mut state, &messages);
        assert!(result.is_err());
        assert!(
            state.received_sibling_paths[1].is_none(),
            "state should not be partially modified after error"
        );
    }

    #[test]
    fn test_process_round3_atomic_on_error() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let params = SilentOtParams::new(5, 1, 16).unwrap();
        let protocol = DistributedSilentOt::new(params);
        let mut state = protocol.init_party(0, &mut rng);

        let seed1 = Block([1u8; 16]);
        DistributedSilentOt::process_round0(&mut state, &[(1, seed1.commit_with_context(1, 0))])
            .unwrap();

        let messages = vec![(1, seed1), (2, Block([2u8; 16]))];
        let result = DistributedSilentOt::process_round3(&mut state, &messages);
        assert!(result.is_err());
        assert!(
            state.revealed_seeds[1].is_none(),
            "state should not be partially modified after error"
        );
    }

    #[test]
    #[should_panic(expected = "get_random: index")]
    fn test_get_random_out_of_range_panics() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let params = SilentOtParams::new(5, 1, 16).unwrap();
        let protocol = DistributedSilentOt::new(params);

        let mut states: Vec<PartySetupState> =
            (0..5).map(|i| protocol.init_party(i, &mut rng)).collect();

        run_all_rounds(&mut states);

        let correlations = DistributedSilentOt::expand(&states[0]).unwrap();
        let _should_panic = correlations.get_random(9999);
    }

    #[test]
    #[ignore]
    fn bench_raw_aes_throughput() {
        let key = prg_key_left();
        let n = 524_288usize; // 2^19
        let mut blocks: Vec<aes::Block> = (0..n)
            .map(|i| aes::Block::from((i as u128).to_ne_bytes()))
            .collect();

        // Warmup
        key.encrypt_blocks(&mut blocks);

        let iters = 20;
        let start = std::time::Instant::now();
        for _ in 0..iters {
            key.encrypt_blocks(&mut blocks);
        }
        let elapsed = start.elapsed();
        let total_blocks = n as u64 * iters;
        eprintln!(
            "AES batch: {}x encrypt_blocks({} blocks) = {:?}",
            iters, n, elapsed
        );
        eprintln!(
            "per call: {:.2?}, per block: {:.1}ns, throughput: {:.1} Mblocks/s",
            elapsed / iters as u32,
            elapsed.as_nanos() as f64 / total_blocks as f64,
            total_blocks as f64 / elapsed.as_secs_f64() / 1e6
        );
    }

    // ── 2-Round Protocol Tests ──────────────────────────────────

    fn run_2round_protocol(n: usize, t: usize, num_ots: usize) -> Vec<PartySetupState> {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let params = SilentOtParams::new(n, t, num_ots).unwrap();
        let protocol = DistributedSilentOt::new(params);
        let mut states: Vec<_> = (0..n).map(|i| protocol.init_party(i, &mut rng)).collect();

        // Round A
        let mut ra = vec![Vec::new(); n];
        for s in states.iter() {
            for (to, c, idx) in DistributedSilentOt::round_a_messages(s) {
                ra[to].push((s.party_id, c, idx));
            }
        }
        for (i, s) in states.iter_mut().enumerate() {
            DistributedSilentOt::process_round_a(s, &ra[i]).unwrap();
        }

        // Round B
        let mut rb = vec![Vec::new(); n];
        for s in states.iter() {
            for (to, path, seed) in DistributedSilentOt::round_b_messages(s).unwrap() {
                rb[to].push((s.party_id, path, seed));
            }
        }
        for (i, s) in states.iter_mut().enumerate() {
            DistributedSilentOt::process_round_b(s, &rb[i]).unwrap();
        }

        states
    }

    #[test]
    fn test_2round_produces_random_values() {
        let states = run_2round_protocol(5, 1, 16);
        let correlations: Vec<_> = states
            .iter()
            .map(|s| DistributedSilentOt::expand(s).unwrap())
            .collect();
        assert_eq!(correlations.len(), 5);
        for c in &correlations {
            assert_eq!(c.num_ots(), 16);
        }
        let any_nonzero = (0..16).any(|i| correlations[0].get_random(i).raw() != 0);
        assert!(any_nonzero);
    }

    #[test]
    fn test_2round_fake_path_detected() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let params = SilentOtParams::new(5, 1, 16).unwrap();
        let protocol = DistributedSilentOt::new(params);
        let n = 5;
        let mut states: Vec<_> = (0..n).map(|i| protocol.init_party(i, &mut rng)).collect();

        // Round A (honest)
        let mut ra = vec![Vec::new(); n];
        for s in states.iter() {
            for (to, c, idx) in DistributedSilentOt::round_a_messages(s) {
                ra[to].push((s.party_id, c, idx));
            }
        }
        for (i, s) in states.iter_mut().enumerate() {
            DistributedSilentOt::process_round_a(s, &ra[i]).unwrap();
        }

        // Round B: tamper party 1→0 path
        let mut rb = vec![Vec::new(); n];
        for s in states.iter() {
            for (to, mut path, seed) in DistributedSilentOt::round_b_messages(s).unwrap() {
                if s.party_id == 1 && to == 0 && !path.is_empty() {
                    path[0] = Block::random(&mut rng); // tamper
                }
                rb[to].push((s.party_id, path, seed));
            }
        }
        for (i, s) in states.iter_mut().enumerate() {
            DistributedSilentOt::process_round_b(s, &rb[i]).unwrap();
        }

        // Expand should detect the fake path
        let result = DistributedSilentOt::expand(&states[0]);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("fake sibling path"));
    }

    #[test]
    fn test_2round_commitment_binding() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let params = SilentOtParams::new(5, 1, 16).unwrap();
        let protocol = DistributedSilentOt::new(params);
        let mut state = protocol.init_party(0, &mut rng);

        // Round A from party 1
        let commitment = Block::random(&mut rng).commit_with_context(1, 0);
        DistributedSilentOt::process_round_a(&mut state, &[(1, commitment, 0)]).unwrap();

        // Round B: wrong seed
        let wrong_seed = Block::random(&mut rng);
        let tree = GgmTree::new(state.tree_depth);
        let path = tree.compute_sibling_path(&wrong_seed, 0).unwrap();
        let result = DistributedSilentOt::process_round_b(&mut state, &[(1, path, wrong_seed)]);
        assert!(result.is_err(), "mismatched seed should be rejected");
    }

    #[test]
    fn test_2round_duplicate_rejected() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let params = SilentOtParams::new(5, 1, 16).unwrap();
        let protocol = DistributedSilentOt::new(params);
        let mut state = protocol.init_party(0, &mut rng);

        let c1 = Block::random(&mut rng).commit_with_context(1, 0);
        let c2 = Block::random(&mut rng).commit_with_context(1, 0);
        let result = DistributedSilentOt::process_round_a(&mut state, &[(1, c1, 0), (1, c2, 1)]);
        assert!(
            result.is_err(),
            "duplicate sender in round_a should be rejected"
        );
    }

    // ── 4-ary GGM Tree Tests ───────────────────────────────────

    #[test]
    fn test_4ary_expand_leaf_count() {
        for depth in 1..=4 {
            let tree = GgmTree4::new(depth);
            let root = Block::random(&mut ChaCha20Rng::seed_from_u64(42));
            let leaves = tree.expand_full(&root);
            assert_eq!(leaves.len(), 4usize.pow(depth as u32));
        }
    }

    #[test]
    fn test_4ary_expand_deterministic() {
        let root = Block::random(&mut ChaCha20Rng::seed_from_u64(42));
        let t1 = GgmTree4::new(3).expand_full(&root);
        let t2 = GgmTree4::new(3).expand_full(&root);
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_4ary_sibling_path_roundtrip() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let tree = GgmTree4::new(3); // 64 leaves
        let root = Block::random(&mut rng);
        let all_leaves = tree.expand_full(&root);

        for punct in [0, 1, 4, 15, 32, 63] {
            let path = tree.compute_sibling_path(&root, punct).unwrap();
            assert_eq!(path.len(), 3); // depth entries
            let reconstructed = tree.reconstruct_from_siblings(&path, punct).unwrap();
            assert_eq!(reconstructed.len(), all_leaves.len());
            for (i, (orig, recon)) in all_leaves.iter().zip(reconstructed.iter()).enumerate() {
                if i == punct {
                    assert_eq!(*recon, Block::ZERO, "punctured leaf should be zero");
                } else {
                    assert_eq!(orig, recon, "leaf {} mismatch for puncture {}", i, punct);
                }
            }
        }
    }

    #[test]
    fn test_4ary_accumulate_u64() {
        let mut rng = ChaCha20Rng::seed_from_u64(99);
        let tree = GgmTree4::new(2); // 16 leaves
        let root = Block::random(&mut rng);
        let count = 16;

        let full_vals = tree.expand_full_to_u64(&root, count);
        assert_eq!(full_vals.len(), count);

        // Reconstruct and accumulate into zeros
        let path = tree.compute_sibling_path(&root, 5).unwrap();
        let mut accum = vec![0u64; count];
        tree.reconstruct_accumulate_u64(&path, 5, count, &mut accum)
            .unwrap();

        // All non-punctured values should match
        for i in 0..count {
            if i == 5 {
                assert_eq!(accum[i], 0, "punctured index should be 0 in accum");
            } else {
                assert_eq!(accum[i], full_vals[i], "leaf {} mismatch", i);
            }
        }
    }

    #[test]
    fn test_4ary_protocol_produces_random_values() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let params = SilentOtParams::with_arity(5, 1, 16, 4).unwrap();
        let protocol = DistributedSilentOt::new(params);
        let mut states: Vec<_> = (0..5).map(|i| protocol.init_party(i, &mut rng)).collect();

        // 2-round protocol
        let mut ra = vec![Vec::new(); 5];
        for s in states.iter() {
            for (to, c, idx) in DistributedSilentOt::round_a_messages(s) {
                ra[to].push((s.party_id, c, idx));
            }
        }
        for (i, s) in states.iter_mut().enumerate() {
            DistributedSilentOt::process_round_a(s, &ra[i]).unwrap();
        }
        let mut rb = vec![Vec::new(); 5];
        for s in states.iter() {
            for (to, path, seed) in DistributedSilentOt::round_b_messages(s).unwrap() {
                rb[to].push((s.party_id, path, seed));
            }
        }
        for (i, s) in states.iter_mut().enumerate() {
            DistributedSilentOt::process_round_b(s, &rb[i]).unwrap();
        }

        let correlations: Vec<_> = states
            .iter()
            .map(|s| DistributedSilentOt::expand(s).unwrap())
            .collect();
        assert_eq!(correlations.len(), 5);
        let any_nonzero = (0..16).any(|i| correlations[0].get_random(i).raw() != 0);
        assert!(
            any_nonzero,
            "4-ary OT should produce non-zero random values"
        );
    }
}
