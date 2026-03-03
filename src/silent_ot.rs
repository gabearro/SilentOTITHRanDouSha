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
    const CHUNK: usize = 4096;
    let key = prg_key_field();
    let mut results = Vec::with_capacity(count);

    for chunk_start in (0..count).step_by(CHUNK) {
        let chunk_end = (chunk_start + CHUNK).min(count);
        let chunk_len = chunk_end - chunk_start;

        let mut inputs: Vec<[u8; 16]> = Vec::with_capacity(chunk_len);
        let mut aes_blocks: Vec<aes::Block> = Vec::with_capacity(chunk_len);

        for k in chunk_start..chunk_end {
            let domain_bytes = (k as u64).to_le_bytes();
            let mut input = blocks[k].0;
            for i in 0..8 {
                input[i] ^= domain_bytes[i];
            }
            inputs.push(input);
            aes_blocks.push(aes::Block::from(input));
        }

        key.encrypt_blocks(&mut aes_blocks);

        for (idx, _k) in (chunk_start..chunk_end).enumerate() {
            let enc: [u8; 16] = aes_blocks[idx].into();
            let val = u64::from_le_bytes([
                enc[0] ^ inputs[idx][0],
                enc[1] ^ inputs[idx][1],
                enc[2] ^ inputs[idx][2],
                enc[3] ^ inputs[idx][3],
                enc[4] ^ inputs[idx][4],
                enc[5] ^ inputs[idx][5],
                enc[6] ^ inputs[idx][6],
                enc[7] ^ inputs[idx][7],
            ]);
            results.push(Fp::new(val));
        }
    }

    results
}

impl std::fmt::Debug for Block {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Block({:02x}{:02x}..)", self.0[0], self.0[1])
    }
}

const PRG_KEY_LEFT: [u8; 16] = [
    0x6a, 0x09, 0xe6, 0x67, 0xbb, 0x67, 0xae, 0x85, 0x3c, 0x6e, 0xf3, 0x72, 0xa5, 0x4f, 0xf5,
    0x3a,
];
const PRG_KEY_RIGHT: [u8; 16] = [
    0x51, 0x0e, 0x52, 0x7f, 0xad, 0xe6, 0x82, 0xd1, 0x9b, 0x05, 0x68, 0x8c, 0x2b, 0x3e, 0x6c,
    0x1f,
];
const PRG_KEY_FIELD: [u8; 16] = [
    0x42, 0x8a, 0x2f, 0x98, 0x71, 0x37, 0x44, 0x91, 0xb5, 0xc0, 0xfb, 0xcf, 0xe9, 0xb5, 0xdb,
    0xa5,
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
        let mut current_level = vec![*root];
        let mut next_level = Vec::new();

        for _ in 0..self.depth {
            let len = current_level.len();
            if len >= 64 {
                let mut blocks_l: Vec<aes::Block> =
                    current_level.iter().map(|s| aes::Block::from(s.0)).collect();
                let mut blocks_r: Vec<aes::Block> =
                    current_level.iter().map(|s| aes::Block::from(s.0)).collect();

                key_l.encrypt_blocks(&mut blocks_l);
                key_r.encrypt_blocks(&mut blocks_r);

                next_level.clear();
                next_level.resize(len * 2, Block::ZERO);
                for (i, seed) in current_level.iter().enumerate() {
                    next_level[2 * i] = Block(blocks_l[i].into()).xor(seed);
                    next_level[2 * i + 1] = Block(blocks_r[i].into()).xor(seed);
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

        let expanded: Vec<(usize, Vec<Block>)> = subtree_info
            .par_iter()
            .map(|&(subtree_depth, sibling_start, sibling)| {
                let subtree_leaves = GgmTree::new(subtree_depth).expand_full(sibling);
                (sibling_start, subtree_leaves)
            })
            .collect();

        let mut leaves = vec![Block::ZERO; n];
        for (sibling_start, subtree_leaves) in expanded {
            for (i, leaf) in subtree_leaves.into_iter().enumerate() {
                let global_idx = sibling_start + i;
                if global_idx < n {
                    leaves[global_idx] = leaf;
                }
            }
        }

        Ok(leaves)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SilentOtParams {
    pub n: usize,
    pub t: usize,
    pub num_ots: usize,
    pub tree_depth: usize,
}

impl SilentOtParams {
    pub fn new(n: usize, t: usize, num_ots: usize) -> Result<Self> {
        if n == 0 {
            return Err(ProtocolError::InvalidParams("n must be > 0".into()));
        }
        if num_ots == 0 {
            return Err(ProtocolError::InvalidParams("num_ots must be > 0".into()));
        }
        if n <= 2 * t {
            return Err(ProtocolError::InvalidParams(format!(
                "need n > 2t for honest majority, got n={}, t={}",
                n, t
            )));
        }
        let tree_depth = if num_ots <= 1 {
            1
        } else {
            usize::BITS as usize - (num_ots - 1).leading_zeros() as usize
        };
        Ok(SilentOtParams {
            n,
            t,
            num_ots,
            tree_depth,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PartySetupState {
    pub party_id: usize,
    pub n: usize,
    pub tree_depth: usize,
    pub num_ots: usize,
    pub my_seeds: Vec<Option<Block>>,
    pub my_commitments: Vec<Option<[u8; 32]>>,
    pub their_commitments: Vec<Option<[u8; 32]>>,
    pub my_puncture_indices: Vec<Option<usize>>,
    pub received_sibling_paths: Vec<Option<Vec<Block>>>,
    pub their_puncture_indices: Vec<Option<usize>>,
    pub revealed_seeds: Vec<Option<Block>>,
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
        // tree_depth and num_ots validated in SilentOtParams::new

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
            my_puncture_indices[j] = Some(rng.gen_range(0..self.params.num_ots));
        }

        PartySetupState {
            party_id,
            n,
            tree_depth: self.params.tree_depth,
            num_ots: self.params.num_ots,
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
    pub fn process_round0(state: &mut PartySetupState, commitments: &[(usize, [u8; 32])]) -> Result<()> {
        // Pre-validate all messages before mutating state (atomic update)
        let mut seen = vec![false; state.n];
        for &(from, _) in commitments {
            if from >= state.n || from == state.party_id {
                return Err(ProtocolError::MaliciousParty(format!(
                    "invalid sender {} in commitment (n={})", from, state.n
                )));
            }
            if state.their_commitments[from].is_some() || seen[from] {
                return Err(ProtocolError::MaliciousParty(format!(
                    "duplicate commitment from party {}", from
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
        let mut seen = vec![false; state.n];
        for &(from, index) in choices {
            if from >= state.n || from == state.party_id {
                return Err(ProtocolError::MaliciousParty(format!(
                    "invalid sender {} in puncture choice (n={})", from, state.n
                )));
            }
            if index >= state.num_ots {
                return Err(ProtocolError::MaliciousParty(format!(
                    "party {} sent puncture index {} >= num_ots {}",
                    from, index, state.num_ots
                )));
            }
            if state.their_puncture_indices[from].is_some() || seen[from] {
                return Err(ProtocolError::MaliciousParty(format!(
                    "duplicate puncture choice from party {}", from
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
        let tree = GgmTree::new(state.tree_depth);
        let mut msgs = Vec::with_capacity(state.n - 1);

        for j in 0..state.n {
            if j == state.party_id {
                continue;
            }
            if let (Some(seed), Some(puncture_idx)) =
                (state.my_seeds[j], state.their_puncture_indices[j])
            {
                let sibling_path = tree.compute_sibling_path(&seed, puncture_idx)?;
                msgs.push((j, sibling_path));
            }
        }

        Ok(msgs)
    }

    /// Process received sibling paths: &[(sender_id, sibling_path)]
    pub fn process_round2(state: &mut PartySetupState, paths: &[(usize, Vec<Block>)]) -> Result<()> {
        // Pre-validate all messages before mutating state (atomic update)
        let mut seen = vec![false; state.n];
        for (from, sibling_path) in paths {
            let from = *from;
            if from >= state.n || from == state.party_id {
                return Err(ProtocolError::MaliciousParty(format!(
                    "invalid sender {} in sibling path (n={})", from, state.n
                )));
            }
            if sibling_path.len() != state.tree_depth {
                return Err(ProtocolError::MaliciousParty(format!(
                    "party {} sent sibling path of length {}, expected {}",
                    from,
                    sibling_path.len(),
                    state.tree_depth
                )));
            }
            if state.received_sibling_paths[from].is_some() || seen[from] {
                return Err(ProtocolError::MaliciousParty(format!(
                    "duplicate sibling path from party {}", from
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
                    "invalid sender {} in seed reveal (n={})", from, state.n
                )));
            }
            if state.revealed_seeds[from].is_some() || seen[from] {
                return Err(ProtocolError::MaliciousParty(format!(
                    "duplicate seed reveal from party {}", from
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
                        "seed reveal from party {} without prior commitment", from
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
        let num_leaves = 1 << tree_depth;

        if num_ots > num_leaves {
            return Err(ProtocolError::InvalidParams(format!(
                "num_ots ({}) > num_leaves ({}): would reuse GGM leaves, weakening security",
                num_ots, num_leaves
            )));
        }

        let peers: Vec<usize> = (0..n).filter(|&j| j != state.party_id).collect();

        // For each peer: verify sibling path, compute sender/receiver values,
        // and pre-sum the per-peer contribution — all in one parallel pass.
        let contributions: std::result::Result<Vec<Vec<Fp>>, ProtocolError> = peers
            .par_iter()
            .map(|&j| {
                let tree = GgmTree::new(tree_depth);

                // Verify sibling path against revealed seed
                if let (Some(revealed_seed), Some(received_path), Some(puncture_idx)) = (
                    state.revealed_seeds[j],
                    &state.received_sibling_paths[j],
                    state.my_puncture_indices[j],
                ) {
                    let expected_path = tree.compute_sibling_path(&revealed_seed, puncture_idx)?;
                    for (level, (expected, received)) in
                        expected_path.iter().zip(received_path.iter()).enumerate()
                    {
                        if expected != received {
                            return Err(ProtocolError::MaliciousParty(format!(
                                "party {} sent fake sibling path: mismatch at level {}",
                                j, level
                            )));
                        }
                    }
                }

                // Compute sender values (my tree for peer j)
                let sender_vals = if let Some(seed) = state.my_seeds[j] {
                    let leaves = tree.expand_full(&seed);
                    batch_to_field_elements(&leaves, num_ots)
                } else {
                    vec![Fp::ZERO; num_ots]
                };

                // Compute receiver values (peer j's tree for me)
                let receiver_vals = if let (Some(received_path), Some(puncture_idx)) = (
                    &state.received_sibling_paths[j],
                    state.my_puncture_indices[j],
                ) {
                    let leaves = tree.reconstruct_from_siblings(received_path, puncture_idx)?;
                    batch_to_field_elements(&leaves, num_ots)
                } else {
                    vec![Fp::ZERO; num_ots]
                };

                // Pre-sum this peer's contribution
                let puncture_idx = state.my_puncture_indices[j];
                let mut contrib = Vec::with_capacity(num_ots);
                for k in 0..num_ots {
                    let mut val = sender_vals[k];
                    if puncture_idx != Some(k) {
                        val += receiver_vals[k];
                    }
                    contrib.push(val);
                }
                Ok(contrib)
            })
            .collect();

        let contributions = contributions?;

        // Sum all peer contributions into final random values
        let mut random_values = vec![Fp::ZERO; num_ots];
        for contrib in &contributions {
            for (k, val) in contrib.iter().enumerate() {
                random_values[k] += *val;
            }
        }

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
    random_values: Vec<Fp>,
}

impl ExpandedCorrelations {
    #[inline]
    pub fn get_random(&self, index: usize) -> Fp {
        assert!(
            index < self.num_ots,
            "get_random: index {} out of range (num_ots={})",
            index, self.num_ots
        );
        self.random_values[index]
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

        let mut states: Vec<PartySetupState> = (0..5)
            .map(|i| protocol.init_party(i, &mut rng))
            .collect();

        run_all_rounds(&mut states);

        let correlations: Vec<ExpandedCorrelations> = states
            .iter()
            .map(|s| DistributedSilentOt::expand(s).unwrap())
            .collect();

        // Each party should produce non-trivial random values
        for corr in &correlations {
            let vals: Vec<Fp> = (0..16).map(|k| corr.get_random(k)).collect();
            let all_zero = vals.iter().all(|v| *v == Fp::ZERO);
            assert!(!all_zero, "party {} produced all-zero values", corr.party_id);
        }

        // Different parties should get different random values (different puncture sets)
        let v0: Vec<Fp> = (0..16).map(|k| correlations[0].get_random(k)).collect();
        let v1: Vec<Fp> = (0..16).map(|k| correlations[1].get_random(k)).collect();
        assert_ne!(v0, v1, "different parties should have different random values");
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

        let mut states: Vec<PartySetupState> = (0..5)
            .map(|i| protocol.init_party(i, &mut rng))
            .collect();

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
        let messages = vec![
            (1, seed1.commit()),
            (1, seed2.commit()),
        ];
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
        assert!(result.is_err(), "out-of-bounds from index should return error, not panic");
    }

    #[test]
    fn test_oob_from_party_index_round1() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let params = SilentOtParams::new(5, 1, 16).unwrap();
        let protocol = DistributedSilentOt::new(params);
        let mut state = protocol.init_party(0, &mut rng);

        let result = DistributedSilentOt::process_round1(&mut state, &[(99, 0)]);
        assert!(result.is_err(), "out-of-bounds from index should return error, not panic");
    }

    #[test]
    fn test_oob_from_party_index_round2() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let params = SilentOtParams::new(5, 1, 16).unwrap();
        let protocol = DistributedSilentOt::new(params);
        let mut state = protocol.init_party(0, &mut rng);

        let result = DistributedSilentOt::process_round2(
            &mut state,
            &[(99, vec![Block::ZERO; 4])],
        );
        assert!(result.is_err(), "out-of-bounds from index should return error, not panic");
    }

    #[test]
    fn test_oob_from_party_index_round3() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let params = SilentOtParams::new(5, 1, 16).unwrap();
        let protocol = DistributedSilentOt::new(params);
        let mut state = protocol.init_party(0, &mut rng);

        let result = DistributedSilentOt::process_round3(&mut state, &[(99, Block::ZERO)]);
        assert!(result.is_err(), "out-of-bounds from index should return error, not panic");
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
    fn test_puncture_index_always_in_ot_range() {
        let params = SilentOtParams::new(5, 1, 17).unwrap();
        assert_eq!(params.tree_depth, 5);
        assert_eq!(1 << params.tree_depth, 32);

        let protocol = DistributedSilentOt::new(params);
        for seed_val in 0..100u64 {
            let mut rng_iter = ChaCha20Rng::seed_from_u64(seed_val);
            let state = protocol.init_party(0, &mut rng_iter);
            for j in 0..5 {
                if j == 0 { continue; }
                if let Some(idx) = state.my_puncture_indices[j] {
                    assert!(
                        idx < params.num_ots,
                        "puncture index {} >= num_ots {} (seed={})",
                        idx, params.num_ots, seed_val
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
    fn test_puncture_index_outside_ot_range_rejected() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let params = SilentOtParams::new(5, 1, 17).unwrap();
        assert_eq!(params.tree_depth, 5);
        let protocol = DistributedSilentOt::new(params);
        let mut state = protocol.init_party(0, &mut rng);

        // puncture index 17 >= num_ots (17)
        let result = DistributedSilentOt::process_round1(&mut state, &[(1, 17)]);
        assert!(result.is_err(), "puncture index >= num_ots should be rejected");
        assert!(format!("{}", result.unwrap_err()).contains("num_ots"));

        // puncture at 16 (last valid) should be accepted
        assert!(DistributedSilentOt::process_round1(&mut state, &[(1, 16)]).is_ok());
    }

    #[test]
    fn test_fake_sibling_path_detected() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let params = SilentOtParams::new(5, 1, 16).unwrap();
        let protocol = DistributedSilentOt::new(params);
        let n = 5;

        let mut states: Vec<PartySetupState> = (0..n)
            .map(|i| protocol.init_party(i, &mut rng))
            .collect();

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
        DistributedSilentOt::process_round0(
            &mut state,
            &[(1, seed1.commit_with_context(1, 0))],
        )
        .unwrap();

        let messages = vec![
            (1, seed1),
            (2, Block([2u8; 16])),
        ];
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

        let mut states: Vec<PartySetupState> = (0..5)
            .map(|i| protocol.init_party(i, &mut rng))
            .collect();

        run_all_rounds(&mut states);

        let correlations = DistributedSilentOt::expand(&states[0]).unwrap();
        let _should_panic = correlations.get_random(9999);
    }
}
