//! Distributed MPC runtime for Fp32 shares using explicit party networking.
//!
//! This module provides a party-local API: each party holds only its own shares,
//! communicates with peers to open masked values, and computes Beaver
//! multiplication outputs locally.
//!
//! Unlike local simulation helpers that can see all shares at once, this runtime
//! is organized around `PartyNetwork` message passing and round-tagged openings.

use crate::error::{ProtocolError, Result};
use crate::field32::Fp32;
use crate::field32_shamir::Shamir32;
use crate::network::{setup_channel_network, PartyNetwork};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc as std_mpsc;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};

#[derive(Clone, Copy, Debug)]
pub struct BeaverTripleShare32 {
    pub a: Fp32,
    pub b: Fp32,
    pub c: Fp32,
}

#[derive(Clone, Copy, Debug)]
pub struct DaBitShare32 {
    pub arithmetic: Fp32,
    pub bit: Fp32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct OpenMsg32 {
    round: u64,
    values: Vec<Fp32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ShareMsg32 {
    round: u64,
    values: Vec<Fp32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
enum PartyMsg32 {
    Open(OpenMsg32),
    Share(ShareMsg32),
}

/// Party-local distributed MPC runtime for Fp32.
///
/// Each instance corresponds to one party with one network endpoint.
pub struct DistributedMpcParty32 {
    party_id: usize,
    n: usize,
    t: usize,
    lag_coeffs: Vec<Fp32>,
    eval_points: Vec<Fp32>,
    net: PartyNetwork,
    round_ctr: u64,
    pending: Vec<(usize, PartyMsg32)>,
}

enum PartyRequest32 {
    BeaverBatch {
        x_shares: Vec<Fp32>,
        y_shares: Vec<Fp32>,
        triples: Vec<BeaverTripleShare32>,
        resp: oneshot::Sender<Result<Vec<Fp32>>>,
    },
    OpenBatch {
        shares: Vec<Fp32>,
        resp: oneshot::Sender<Result<Vec<Fp32>>>,
    },
    PrandBitBatch {
        bit_count: usize,
        xor_triples: Vec<Vec<BeaverTripleShare32>>,
        resp: oneshot::Sender<Result<Vec<Fp32>>>,
    },
}

impl DistributedMpcParty32 {
    pub fn new(net: PartyNetwork, t: usize) -> Result<Self> {
        let n = net.n;
        if n <= 2 * t {
            return Err(ProtocolError::InvalidParams(format!(
                "need n > 2t, got n={}, t={}",
                n, t
            )));
        }
        let shamir = Shamir32::new(n, t)?;
        let eval_points = shamir.eval_points.clone();
        Ok(DistributedMpcParty32 {
            party_id: net.party_id,
            n,
            t,
            lag_coeffs: shamir.lagrange_coefficients().to_vec(),
            eval_points,
            net,
            round_ctr: 0,
            pending: Vec::new(),
        })
    }

    #[inline]
    pub fn party_id(&self) -> usize {
        self.party_id
    }

    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    #[inline]
    pub fn t(&self) -> usize {
        self.t
    }

    #[inline]
    fn reconstruct_with_lagrange(&self, vals: &[Fp32]) -> Fp32 {
        debug_assert_eq!(vals.len(), self.n);
        let mut out = Fp32::ZERO;
        for p in 0..self.n {
            out = out + vals[p] * self.lag_coeffs[p];
        }
        out
    }

    #[inline]
    fn next_round(&mut self) -> u64 {
        let r = self.round_ctr;
        self.round_ctr += 1;
        r
    }

    async fn recv_open_round_message(&mut self, round: u64) -> Result<(usize, OpenMsg32)> {
        for i in 0..self.pending.len() {
            if let PartyMsg32::Open(msg) = &self.pending[i].1 {
                if msg.round == round {
                    let (from, pending) = self.pending.remove(i);
                    if let PartyMsg32::Open(open_msg) = pending {
                        return Ok((from, open_msg));
                    }
                }
            }
        }

        loop {
            let (from, msg): (usize, PartyMsg32) = self
                .net
                .recv()
                .await
                .map_err(|e| ProtocolError::NetworkError(format!("recv failed: {}", e)))?;
            match msg {
                PartyMsg32::Open(open_msg) if open_msg.round == round => {
                    return Ok((from, open_msg));
                }
                other => self.pending.push((from, other)),
            }
        }
    }

    async fn recv_share_round_message(&mut self, round: u64) -> Result<(usize, ShareMsg32)> {
        for i in 0..self.pending.len() {
            if let PartyMsg32::Share(msg) = &self.pending[i].1 {
                if msg.round == round {
                    let (from, pending) = self.pending.remove(i);
                    if let PartyMsg32::Share(share_msg) = pending {
                        return Ok((from, share_msg));
                    }
                }
            }
        }

        loop {
            let (from, msg): (usize, PartyMsg32) = self
                .net
                .recv()
                .await
                .map_err(|e| ProtocolError::NetworkError(format!("recv failed: {}", e)))?;
            match msg {
                PartyMsg32::Share(share_msg) if share_msg.round == round => {
                    return Ok((from, share_msg));
                }
                other => self.pending.push((from, other)),
            }
        }
    }

    /// Open one share value (degree-t polynomial) to all parties.
    pub async fn open_value(&mut self, my_share: Fp32) -> Result<Fp32> {
        Ok(self.open_batch(&[my_share]).await?[0])
    }

    /// Open multiple values in one communication round.
    pub async fn open_batch(&mut self, my_shares: &[Fp32]) -> Result<Vec<Fp32>> {
        let round = self.next_round();
        let msg = OpenMsg32 {
            round,
            values: my_shares.to_vec(),
        };
        self.net
            .broadcast(&PartyMsg32::Open(msg))
            .await
            .map_err(|e| ProtocolError::NetworkError(format!("broadcast failed: {}", e)))?;

        let k = my_shares.len();
        let mut per_party = vec![vec![Fp32::ZERO; k]; self.n];
        per_party[self.party_id].copy_from_slice(my_shares);

        for _ in 0..self.n - 1 {
            let (from, recv_msg) = self.recv_open_round_message(round).await?;
            if recv_msg.values.len() != k {
                return Err(ProtocolError::InvalidParams(format!(
                    "round {} value length mismatch from party {}: got {}, expected {}",
                    round,
                    from,
                    recv_msg.values.len(),
                    k
                )));
            }
            per_party[from] = recv_msg.values;
        }

        let mut opened = vec![Fp32::ZERO; k];
        for i in 0..k {
            let mut vals = vec![Fp32::ZERO; self.n];
            for p in 0..self.n {
                vals[p] = per_party[p][i];
            }
            opened[i] = self.reconstruct_with_lagrange(&vals);
        }
        Ok(opened)
    }

    /// Beaver multiplication for one shared value pair.
    ///
    /// Inputs are this party's local shares `(x_i, y_i)` and triple share
    /// `(a_i, b_i, c_i)`. Returns this party's output share `z_i`.
    pub async fn beaver_multiply(
        &mut self,
        x_share: Fp32,
        y_share: Fp32,
        triple: BeaverTripleShare32,
    ) -> Result<Fp32> {
        let d_i = x_share - triple.a;
        let e_i = y_share - triple.b;
        let opened = self.open_batch(&[d_i, e_i]).await?;
        let d = opened[0];
        let e = opened[1];
        Ok(triple.c + e * x_share + d * y_share - d * e)
    }

    /// Beaver multiplication for many independent shared value pairs.
    ///
    /// Opens all masked `(d, e)` values in a single round.
    pub async fn beaver_multiply_batch(
        &mut self,
        x_shares: &[Fp32],
        y_shares: &[Fp32],
        triples: &[BeaverTripleShare32],
    ) -> Result<Vec<Fp32>> {
        let k = x_shares.len();
        if y_shares.len() != k || triples.len() != k {
            return Err(ProtocolError::InvalidParams(format!(
                "batch length mismatch: x={}, y={}, triples={}",
                x_shares.len(),
                y_shares.len(),
                triples.len()
            )));
        }

        let mut masked = vec![Fp32::ZERO; 2 * k];
        for i in 0..k {
            masked[2 * i] = x_shares[i] - triples[i].a;
            masked[2 * i + 1] = y_shares[i] - triples[i].b;
        }

        let opened = self.open_batch(&masked).await?;
        let mut out = vec![Fp32::ZERO; k];
        for i in 0..k {
            let d = opened[2 * i];
            let e = opened[2 * i + 1];
            out[i] = triples[i].c + e * x_shares[i] + d * y_shares[i] - d * e;
        }
        Ok(out)
    }

    fn local_bit_shares_for_all_parties(&self, bit_count: usize) -> Vec<Vec<Fp32>> {
        let mut rng = rand::thread_rng();
        let mut shares = vec![vec![Fp32::ZERO; bit_count]; self.n];
        if self.t == 1 {
            for b in 0..bit_count {
                let bit = if (rng.gen::<u8>() & 1) == 0 {
                    Fp32::ZERO
                } else {
                    Fp32::ONE
                };
                let slope = Fp32::random(&mut rng);
                for p in 0..self.n {
                    shares[p][b] = bit + slope * self.eval_points[p];
                }
            }
            return shares;
        }

        let mut x_pows = vec![vec![Fp32::ONE; self.t + 1]; self.n];
        for p in 0..self.n {
            for d in 1..=self.t {
                x_pows[p][d] = x_pows[p][d - 1] * self.eval_points[p];
            }
        }
        for b in 0..bit_count {
            let mut coeffs = vec![Fp32::ZERO; self.t + 1];
            coeffs[0] = if (rng.gen::<u8>() & 1) == 0 {
                Fp32::ZERO
            } else {
                Fp32::ONE
            };
            for d in 1..=self.t {
                coeffs[d] = Fp32::random(&mut rng);
            }
            for p in 0..self.n {
                let mut val = Fp32::ZERO;
                for d in 0..=self.t {
                    val = val + coeffs[d] * x_pows[p][d];
                }
                shares[p][b] = val;
            }
        }
        shares
    }

    /// Generate `bit_count` PRandBits as arithmetic shares with no openings.
    ///
    /// Uses per-party random bit sharings, then combines them with Beaver-based
    /// batched XOR (`a xor b = a + b - 2ab`).
    pub async fn prandbit_batch(
        &mut self,
        bit_count: usize,
        xor_triples: &[Vec<BeaverTripleShare32>],
    ) -> Result<Vec<Fp32>> {
        if bit_count == 0 {
            return Ok(Vec::new());
        }
        if xor_triples.len() != self.n - 1 {
            return Err(ProtocolError::InvalidParams(format!(
                "xor triple rounds mismatch: got {}, expected {}",
                xor_triples.len(),
                self.n - 1
            )));
        }
        for (r, ts) in xor_triples.iter().enumerate() {
            if ts.len() != bit_count {
                return Err(ProtocolError::InvalidParams(format!(
                    "xor triple round {} len mismatch: got {}, expected {}",
                    r,
                    ts.len(),
                    bit_count
                )));
            }
        }

        let round = self.next_round();
        let local_shares = self.local_bit_shares_for_all_parties(bit_count);

        for to in 0..self.n {
            if to == self.party_id {
                continue;
            }
            let msg = ShareMsg32 {
                round,
                values: local_shares[to].clone(),
            };
            self.net
                .send_to(to, &PartyMsg32::Share(msg))
                .await
                .map_err(|e| ProtocolError::NetworkError(format!("send_to failed: {}", e)))?;
        }

        let mut per_party_bits = vec![vec![Fp32::ZERO; bit_count]; self.n];
        per_party_bits[self.party_id].copy_from_slice(&local_shares[self.party_id]);
        for _ in 0..self.n - 1 {
            let (from, msg) = self.recv_share_round_message(round).await?;
            if msg.values.len() != bit_count {
                return Err(ProtocolError::InvalidParams(format!(
                    "share round {} length mismatch from party {}: got {}, expected {}",
                    round,
                    from,
                    msg.values.len(),
                    bit_count
                )));
            }
            per_party_bits[from] = msg.values;
        }

        let two = Fp32::new(2);
        let mut acc = per_party_bits[0].clone();
        for i in 1..self.n {
            let prod = self
                .beaver_multiply_batch(&acc, &per_party_bits[i], &xor_triples[i - 1])
                .await?;
            for b in 0..bit_count {
                acc[b] = acc[b] + per_party_bits[i][b] - two * prod[b];
            }
        }
        Ok(acc)
    }
}

/// Long-lived in-process executor for distributed Beaver batch multiplications.
///
/// This keeps party workers, channels, and async runtime alive across many batch
/// calls so callers avoid per-batch network/runtime construction overhead.
pub struct InProcessBeaverExecutor32 {
    n: usize,
    t: usize,
    rt: tokio::runtime::Runtime,
    party_txs: Vec<mpsc::UnboundedSender<PartyRequest32>>,
}

impl InProcessBeaverExecutor32 {
    pub fn new(n: usize, t: usize) -> Result<Self> {
        if n <= 2 * t {
            return Err(ProtocolError::InvalidParams(format!(
                "need n > 2t, got n={}, t={}",
                n, t
            )));
        }
        let networks = setup_channel_network(n);
        let worker_threads = n.clamp(2, 8);
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(worker_threads)
            .enable_all()
            .build()
            .map_err(|e| ProtocolError::NetworkError(format!("runtime build failed: {}", e)))?;

        let mut party_txs = Vec::with_capacity(n);
        rt.block_on(async {
            for net in networks {
                let (tx, mut rx) = mpsc::unbounded_channel::<PartyRequest32>();
                party_txs.push(tx);
                tokio::spawn(async move {
                    let mut party = match DistributedMpcParty32::new(net, t) {
                        Ok(p) => p,
                        Err(_) => return,
                    };
                    while let Some(req) = rx.recv().await {
                        match req {
                            PartyRequest32::BeaverBatch {
                                x_shares,
                                y_shares,
                                triples,
                                resp,
                            } => {
                                let res = party
                                    .beaver_multiply_batch(&x_shares, &y_shares, &triples)
                                    .await;
                                let _ = resp.send(res);
                            }
                            PartyRequest32::OpenBatch { shares, resp } => {
                                let res = party.open_batch(&shares).await;
                                let _ = resp.send(res);
                            }
                            PartyRequest32::PrandBitBatch {
                                bit_count,
                                xor_triples,
                                resp,
                            } => {
                                let res = party.prandbit_batch(bit_count, &xor_triples).await;
                                let _ = resp.send(res);
                            }
                        }
                    }
                });
            }
        });

        Ok(InProcessBeaverExecutor32 {
            n,
            t,
            rt,
            party_txs,
        })
    }

    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    #[inline]
    pub fn t(&self) -> usize {
        self.t
    }

    /// Open `k` shared values from a flat `[k][n]` buffer.
    ///
    /// Returns cleartext values in value-major order `[k]`.
    pub fn open_batch_flat(&self, shares_flat: &[Fp32]) -> Result<Vec<Fp32>> {
        if shares_flat.len() % self.n != 0 {
            return Err(ProtocolError::InvalidParams(format!(
                "flat buffer length {} is not divisible by n={}",
                shares_flat.len(),
                self.n
            )));
        }
        let k = shares_flat.len() / self.n;
        let mut recvs = Vec::with_capacity(self.n);

        for p in 0..self.n {
            let mut shares = Vec::with_capacity(k);
            for i in 0..k {
                shares.push(shares_flat[i * self.n + p]);
            }
            let (tx, rx) = oneshot::channel::<Result<Vec<Fp32>>>();
            self.party_txs[p]
                .send(PartyRequest32::OpenBatch { shares, resp: tx })
                .map_err(|e| {
                    ProtocolError::NetworkError(format!("party {} send failed: {}", p, e))
                })?;
            recvs.push(rx);
        }

        self.rt.block_on(async {
            let mut first: Option<Vec<Fp32>> = None;
            for (p, rx) in recvs.into_iter().enumerate() {
                let opened = rx.await.map_err(|e| {
                    ProtocolError::NetworkError(format!("party {} recv failed: {}", p, e))
                })??;
                if opened.len() != k {
                    return Err(ProtocolError::InvalidParams(format!(
                        "party {} opened len {} != expected {}",
                        p,
                        opened.len(),
                        k
                    )));
                }
                if let Some(ref baseline) = first {
                    if baseline != &opened {
                        return Err(ProtocolError::NetworkError(format!(
                            "open mismatch across parties at party {}",
                            p
                        )));
                    }
                } else {
                    first = Some(opened);
                }
            }
            Ok::<Vec<Fp32>, ProtocolError>(first.unwrap_or_default())
        })
    }

    /// Run `k` independent Beaver multiplications from party-interleaved flat buffers.
    ///
    /// Inputs:
    /// - `x_flat`, `y_flat`: layout `[k][n]` flattened
    /// - `triples_by_party[p][i]`: party `p` triple for multiplication `i`
    ///
    /// Returns `out[i][p]`: party share of i-th product.
    pub fn run_batch(
        &self,
        x_flat: &[Fp32],
        y_flat: &[Fp32],
        triples_by_party: &[Vec<BeaverTripleShare32>],
    ) -> Result<Vec<Vec<Fp32>>> {
        if x_flat.len() != y_flat.len() {
            return Err(ProtocolError::InvalidParams(format!(
                "x/y flat size mismatch: x={}, y={}",
                x_flat.len(),
                y_flat.len()
            )));
        }
        if x_flat.len() % self.n != 0 {
            return Err(ProtocolError::InvalidParams(format!(
                "flat buffer length {} is not divisible by n={}",
                x_flat.len(),
                self.n
            )));
        }
        if triples_by_party.len() != self.n {
            return Err(ProtocolError::InvalidParams(format!(
                "triples_by_party len {} != n {}",
                triples_by_party.len(),
                self.n
            )));
        }
        let k = x_flat.len() / self.n;
        for p in 0..self.n {
            if triples_by_party[p].len() != k {
                return Err(ProtocolError::InvalidParams(format!(
                    "party {} triple len {} != k {}",
                    p,
                    triples_by_party[p].len(),
                    k
                )));
            }
        }

        let mut recvs = Vec::with_capacity(self.n);
        for p in 0..self.n {
            let mut x_shares = Vec::with_capacity(k);
            let mut y_shares = Vec::with_capacity(k);
            for i in 0..k {
                x_shares.push(x_flat[i * self.n + p]);
                y_shares.push(y_flat[i * self.n + p]);
            }
            let triples = triples_by_party[p].clone();
            let (tx, rx) = oneshot::channel::<Result<Vec<Fp32>>>();
            self.party_txs[p]
                .send(PartyRequest32::BeaverBatch {
                    x_shares,
                    y_shares,
                    triples,
                    resp: tx,
                })
                .map_err(|e| {
                    ProtocolError::NetworkError(format!("party {} send failed: {}", p, e))
                })?;
            recvs.push(rx);
        }

        let party_major = self.rt.block_on(async {
            let mut vals = vec![Vec::<Fp32>::new(); self.n];
            for (p, rx) in recvs.into_iter().enumerate() {
                let out = rx.await.map_err(|e| {
                    ProtocolError::NetworkError(format!("party {} recv failed: {}", p, e))
                })??;
                vals[p] = out;
            }
            Ok::<Vec<Vec<Fp32>>, ProtocolError>(vals)
        })?;

        let mut out = vec![vec![Fp32::ZERO; self.n]; k];
        for p in 0..self.n {
            for i in 0..k {
                out[i][p] = party_major[p][i];
            }
        }
        Ok(out)
    }

    fn sample_xor_triples_by_party(
        &self,
        bit_count: usize,
    ) -> Result<Vec<Vec<Vec<BeaverTripleShare32>>>> {
        let rounds = self.n.saturating_sub(1);
        let shamir = Shamir32::new(self.n, self.t)?;
        let mut rng = rand::thread_rng();
        let mut by_party =
            vec![vec![Vec::<BeaverTripleShare32>::with_capacity(bit_count); rounds]; self.n];

        for r in 0..rounds {
            for _ in 0..bit_count {
                let a = Fp32::random(&mut rng);
                let b = Fp32::random(&mut rng);
                let c = a * b;
                let a_sh = shamir.share(a, &mut rng);
                let b_sh = shamir.share(b, &mut rng);
                let c_sh = shamir.share(c, &mut rng);
                for p in 0..self.n {
                    by_party[p][r].push(BeaverTripleShare32 {
                        a: a_sh[p].value,
                        b: b_sh[p].value,
                        c: c_sh[p].value,
                    });
                }
            }
        }
        Ok(by_party)
    }

    /// Batched PRandBit generation.
    ///
    /// Returns `out[i][p]`: party `p` share of i-th random bit.
    pub fn prandbit_batch(&self, bit_count: usize) -> Result<Vec<Vec<Fp32>>> {
        if bit_count == 0 {
            return Ok(Vec::new());
        }
        let xor_triples_by_party = self.sample_xor_triples_by_party(bit_count)?;
        let mut recvs = Vec::with_capacity(self.n);
        for p in 0..self.n {
            let (tx, rx) = oneshot::channel::<Result<Vec<Fp32>>>();
            self.party_txs[p]
                .send(PartyRequest32::PrandBitBatch {
                    bit_count,
                    xor_triples: xor_triples_by_party[p].clone(),
                    resp: tx,
                })
                .map_err(|e| {
                    ProtocolError::NetworkError(format!("party {} send failed: {}", p, e))
                })?;
            recvs.push(rx);
        }

        let party_major = self.rt.block_on(async {
            let mut vals = vec![Vec::<Fp32>::new(); self.n];
            for (p, rx) in recvs.into_iter().enumerate() {
                let out = rx.await.map_err(|e| {
                    ProtocolError::NetworkError(format!("party {} recv failed: {}", p, e))
                })??;
                if out.len() != bit_count {
                    return Err(ProtocolError::InvalidParams(format!(
                        "party {} returned {} bits, expected {}",
                        p,
                        out.len(),
                        bit_count
                    )));
                }
                vals[p] = out;
            }
            Ok::<Vec<Vec<Fp32>>, ProtocolError>(vals)
        })?;

        let mut out = vec![vec![Fp32::ZERO; self.n]; bit_count];
        for p in 0..self.n {
            for i in 0..bit_count {
                out[i][p] = party_major[p][i];
            }
        }
        Ok(out)
    }

    /// Batched daBit generation (arithmetic and bit shares over the same random bit).
    pub fn dabit_batch(&self, bit_count: usize) -> Result<Vec<Vec<DaBitShare32>>> {
        let bits = self.prandbit_batch(bit_count)?;
        Ok(bits
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|s| DaBitShare32 {
                        arithmetic: s,
                        bit: s,
                    })
                    .collect()
            })
            .collect())
    }
}

/// Background PRandBit preprocessor that aggressively batches and pipelines generation.
///
/// Internally keeps a dedicated in-process distributed runtime and fills a bounded
/// queue with precomputed PRandBit batches.
pub struct PipelinedPrandBitProvider32 {
    rx: std_mpsc::Receiver<Result<Vec<Vec<Fp32>>>>,
    stop: Arc<AtomicBool>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl PipelinedPrandBitProvider32 {
    pub fn new(n: usize, t: usize, batch_size: usize, queue_depth: usize) -> Result<Self> {
        if batch_size == 0 {
            return Err(ProtocolError::InvalidParams(
                "batch_size must be > 0".to_string(),
            ));
        }
        let queue_depth = queue_depth.max(1);
        let super_batch_size = batch_size.checked_mul(queue_depth).ok_or_else(|| {
            ProtocolError::InvalidParams("batch_size * queue_depth overflow".to_string())
        })?;

        let (tx, rx) = std_mpsc::sync_channel(queue_depth);
        let stop = Arc::new(AtomicBool::new(false));
        let stop_w = stop.clone();

        let worker = std::thread::spawn(move || {
            let exec = match InProcessBeaverExecutor32::new(n, t) {
                Ok(v) => v,
                Err(e) => {
                    let _ = tx.send(Err(e));
                    return;
                }
            };
            while !stop_w.load(Ordering::Relaxed) {
                match exec.prandbit_batch(super_batch_size) {
                    Ok(rows) => {
                        let mut chunk = Vec::with_capacity(batch_size);
                        for row in rows {
                            if stop_w.load(Ordering::Relaxed) {
                                return;
                            }
                            chunk.push(row);
                            if chunk.len() == batch_size {
                                if tx.send(Ok(chunk)).is_err() {
                                    return;
                                }
                                chunk = Vec::with_capacity(batch_size);
                            }
                        }
                        if !chunk.is_empty() && tx.send(Ok(chunk)).is_err() {
                            return;
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(e));
                        return;
                    }
                }
            }
        });

        Ok(PipelinedPrandBitProvider32 {
            rx,
            stop,
            worker: Some(worker),
        })
    }

    pub fn next_batch(&self) -> Result<Vec<Vec<Fp32>>> {
        self.rx.recv().map_err(|_| {
            ProtocolError::NetworkError("PRandBit pipeline channel closed".to_string())
        })?
    }
}

impl Drop for PipelinedPrandBitProvider32 {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let (_tx, dummy_rx) = std_mpsc::sync_channel(1);
        let old_rx = std::mem::replace(&mut self.rx, dummy_rx);
        drop(old_rx);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// Background daBit preprocessor that aggressively batches and pipelines generation.
///
/// Internally keeps a dedicated in-process distributed runtime and fills a bounded
/// queue with precomputed daBit batches.
pub struct PipelinedDaBitProvider32 {
    rx: std_mpsc::Receiver<Result<Vec<Vec<DaBitShare32>>>>,
    stop: Arc<AtomicBool>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl PipelinedDaBitProvider32 {
    pub fn new(n: usize, t: usize, batch_size: usize, queue_depth: usize) -> Result<Self> {
        if batch_size == 0 {
            return Err(ProtocolError::InvalidParams(
                "batch_size must be > 0".to_string(),
            ));
        }
        let queue_depth = queue_depth.max(1);
        let super_batch_size = batch_size.checked_mul(queue_depth).ok_or_else(|| {
            ProtocolError::InvalidParams("batch_size * queue_depth overflow".to_string())
        })?;

        let (tx, rx) = std_mpsc::sync_channel(queue_depth);
        let stop = Arc::new(AtomicBool::new(false));
        let stop_w = stop.clone();

        let worker = std::thread::spawn(move || {
            let exec = match InProcessBeaverExecutor32::new(n, t) {
                Ok(v) => v,
                Err(e) => {
                    let _ = tx.send(Err(e));
                    return;
                }
            };
            while !stop_w.load(Ordering::Relaxed) {
                match exec.dabit_batch(super_batch_size) {
                    Ok(rows) => {
                        let mut chunk = Vec::with_capacity(batch_size);
                        for row in rows {
                            if stop_w.load(Ordering::Relaxed) {
                                return;
                            }
                            chunk.push(row);
                            if chunk.len() == batch_size {
                                if tx.send(Ok(chunk)).is_err() {
                                    return;
                                }
                                chunk = Vec::with_capacity(batch_size);
                            }
                        }
                        if !chunk.is_empty() && tx.send(Ok(chunk)).is_err() {
                            return;
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(e));
                        return;
                    }
                }
            }
        });

        Ok(PipelinedDaBitProvider32 {
            rx,
            stop,
            worker: Some(worker),
        })
    }

    pub fn next_batch(&self) -> Result<Vec<Vec<DaBitShare32>>> {
        self.rx
            .recv()
            .map_err(|_| ProtocolError::NetworkError("daBit pipeline channel closed".to_string()))?
    }
}

impl Drop for PipelinedDaBitProvider32 {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let (_tx, dummy_rx) = std_mpsc::sync_channel(1);
        let old_rx = std::mem::replace(&mut self.rx, dummy_rx);
        drop(old_rx);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field32_shamir::{Shamir32, Share32};
    use crate::network::setup_channel_network;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    fn share_values_of(shares: &[Share32]) -> Vec<Fp32> {
        shares.iter().map(|s| s.value).collect()
    }

    #[tokio::test]
    async fn test_distributed_open_batch() {
        let n = 5usize;
        let t = 1usize;
        let shamir = Shamir32::new(n, t).unwrap();
        let mut rng = ChaCha20Rng::seed_from_u64(123);

        let secrets: Vec<Fp32> = (0..12).map(|_| Fp32::random(&mut rng)).collect();
        let shared: Vec<Vec<Share32>> =
            secrets.iter().map(|&s| shamir.share(s, &mut rng)).collect();

        let networks = setup_channel_network(n);
        let mut handles = Vec::with_capacity(n);
        for (party_id, net) in networks.into_iter().enumerate() {
            let my_shares: Vec<Fp32> = shared.iter().map(|v| v[party_id].value).collect();
            handles.push(tokio::spawn(async move {
                let mut party = DistributedMpcParty32::new(net, t).unwrap();
                party.open_batch(&my_shares).await.unwrap()
            }));
        }

        let mut opened_by_party = Vec::with_capacity(n);
        for h in handles {
            opened_by_party.push(h.await.unwrap());
        }

        for opened in opened_by_party {
            assert_eq!(opened, secrets);
        }
    }

    #[tokio::test]
    async fn test_distributed_beaver_multiply() {
        let n = 5usize;
        let t = 1usize;
        let shamir = Shamir32::new(n, t).unwrap();
        let mut rng = ChaCha20Rng::seed_from_u64(456);

        let x = Fp32::random(&mut rng);
        let y = Fp32::random(&mut rng);
        let a = Fp32::random(&mut rng);
        let b = Fp32::random(&mut rng);
        let c = a * b;

        let x_sh = shamir.share(x, &mut rng);
        let y_sh = shamir.share(y, &mut rng);
        let a_sh = shamir.share(a, &mut rng);
        let b_sh = shamir.share(b, &mut rng);
        let c_sh = shamir.share(c, &mut rng);

        let networks = setup_channel_network(n);
        let mut handles = Vec::with_capacity(n);
        for (party_id, net) in networks.into_iter().enumerate() {
            let my_x = x_sh[party_id].value;
            let my_y = y_sh[party_id].value;
            let my_triple = BeaverTripleShare32 {
                a: a_sh[party_id].value,
                b: b_sh[party_id].value,
                c: c_sh[party_id].value,
            };
            handles.push(tokio::spawn(async move {
                let mut party = DistributedMpcParty32::new(net, t).unwrap();
                party.beaver_multiply(my_x, my_y, my_triple).await.unwrap()
            }));
        }

        let mut z_vals = Vec::with_capacity(n);
        for h in handles {
            z_vals.push(h.await.unwrap());
        }

        let eval_points = shamir.eval_points.clone();
        let z_shares: Vec<Share32> = (0..n)
            .map(|p| Share32 {
                point: eval_points[p],
                value: z_vals[p],
            })
            .collect();
        let z = shamir.reconstruct(&z_shares).unwrap();
        assert_eq!(z, x * y);
    }

    #[tokio::test]
    async fn test_distributed_beaver_multiply_batch() {
        let n = 5usize;
        let t = 1usize;
        let k = 24usize;
        let shamir = Shamir32::new(n, t).unwrap();
        let mut rng = ChaCha20Rng::seed_from_u64(789);

        let xs: Vec<Fp32> = (0..k).map(|_| Fp32::random(&mut rng)).collect();
        let ys: Vec<Fp32> = (0..k).map(|_| Fp32::random(&mut rng)).collect();

        let x_shares: Vec<Vec<Share32>> = xs.iter().map(|&v| shamir.share(v, &mut rng)).collect();
        let y_shares: Vec<Vec<Share32>> = ys.iter().map(|&v| shamir.share(v, &mut rng)).collect();

        let mut a_shares = Vec::with_capacity(k);
        let mut b_shares = Vec::with_capacity(k);
        let mut c_shares = Vec::with_capacity(k);
        for _ in 0..k {
            let a = Fp32::random(&mut rng);
            let b = Fp32::random(&mut rng);
            let c = a * b;
            a_shares.push(shamir.share(a, &mut rng));
            b_shares.push(shamir.share(b, &mut rng));
            c_shares.push(shamir.share(c, &mut rng));
        }

        let networks = setup_channel_network(n);
        let mut handles = Vec::with_capacity(n);
        for (party_id, net) in networks.into_iter().enumerate() {
            let my_x: Vec<Fp32> = x_shares.iter().map(|v| v[party_id].value).collect();
            let my_y: Vec<Fp32> = y_shares.iter().map(|v| v[party_id].value).collect();
            let my_triples: Vec<BeaverTripleShare32> = (0..k)
                .map(|i| BeaverTripleShare32 {
                    a: a_shares[i][party_id].value,
                    b: b_shares[i][party_id].value,
                    c: c_shares[i][party_id].value,
                })
                .collect();
            handles.push(tokio::spawn(async move {
                let mut party = DistributedMpcParty32::new(net, t).unwrap();
                party
                    .beaver_multiply_batch(&my_x, &my_y, &my_triples)
                    .await
                    .unwrap()
            }));
        }

        let mut z_by_party = Vec::with_capacity(n);
        for h in handles {
            z_by_party.push(h.await.unwrap());
        }

        for i in 0..k {
            let mut z_sh = Vec::with_capacity(n);
            for p in 0..n {
                z_sh.push(Share32 {
                    point: shamir.eval_points[p],
                    value: z_by_party[p][i],
                });
            }
            let got = shamir.reconstruct(&z_sh).unwrap();
            assert_eq!(got, xs[i] * ys[i], "batch idx {}", i);
        }

        // Sanity: shares are not all identical across parties.
        for p in 1..n {
            assert_ne!(z_by_party[0], z_by_party[p]);
        }

        // Keep helper used so clippy doesn't complain when tests get filtered.
        let _ = share_values_of(&x_shares[0]);
    }

    #[test]
    fn test_inprocess_beaver_executor_batch() {
        let n = 5usize;
        let t = 1usize;
        let k = 16usize;
        let shamir = Shamir32::new(n, t).unwrap();
        let mut rng = ChaCha20Rng::seed_from_u64(8080);
        let exec = InProcessBeaverExecutor32::new(n, t).unwrap();

        let xs: Vec<Fp32> = (0..k).map(|_| Fp32::random(&mut rng)).collect();
        let ys: Vec<Fp32> = (0..k).map(|_| Fp32::random(&mut rng)).collect();
        let x_shares: Vec<Vec<Share32>> = xs.iter().map(|&v| shamir.share(v, &mut rng)).collect();
        let y_shares: Vec<Vec<Share32>> = ys.iter().map(|&v| shamir.share(v, &mut rng)).collect();

        let mut triples_by_party = vec![Vec::<BeaverTripleShare32>::with_capacity(k); n];
        for _ in 0..k {
            let a = Fp32::random(&mut rng);
            let b = Fp32::random(&mut rng);
            let c = a * b;
            let a_sh = shamir.share(a, &mut rng);
            let b_sh = shamir.share(b, &mut rng);
            let c_sh = shamir.share(c, &mut rng);
            for p in 0..n {
                triples_by_party[p].push(BeaverTripleShare32 {
                    a: a_sh[p].value,
                    b: b_sh[p].value,
                    c: c_sh[p].value,
                });
            }
        }

        let mut x_flat = vec![Fp32::ZERO; k * n];
        let mut y_flat = vec![Fp32::ZERO; k * n];
        for i in 0..k {
            for p in 0..n {
                x_flat[i * n + p] = x_shares[i][p].value;
                y_flat[i * n + p] = y_shares[i][p].value;
            }
        }

        let out = exec.run_batch(&x_flat, &y_flat, &triples_by_party).unwrap();
        for i in 0..k {
            let mut shares = Vec::with_capacity(n);
            for p in 0..n {
                shares.push(Share32 {
                    point: shamir.eval_points[p],
                    value: out[i][p],
                });
            }
            let got = shamir.reconstruct(&shares).unwrap();
            assert_eq!(got, xs[i] * ys[i], "idx {}", i);
        }
    }

    #[test]
    fn test_inprocess_beaver_executor_open_batch_flat() {
        let n = 5usize;
        let t = 1usize;
        let k = 19usize;
        let shamir = Shamir32::new(n, t).unwrap();
        let mut rng = ChaCha20Rng::seed_from_u64(9090);
        let exec = InProcessBeaverExecutor32::new(n, t).unwrap();

        let secrets: Vec<Fp32> = (0..k).map(|_| Fp32::random(&mut rng)).collect();
        let shares: Vec<Vec<Share32>> =
            secrets.iter().map(|&s| shamir.share(s, &mut rng)).collect();

        let mut flat = vec![Fp32::ZERO; k * n];
        for i in 0..k {
            for p in 0..n {
                flat[i * n + p] = shares[i][p].value;
            }
        }

        let opened = exec.open_batch_flat(&flat).unwrap();
        assert_eq!(opened, secrets);
    }

    #[test]
    fn test_inprocess_prandbit_batch_binary_and_balanced() {
        let n = 5usize;
        let t = 1usize;
        let k = 256usize;
        let shamir = Shamir32::new(n, t).unwrap();
        let exec = InProcessBeaverExecutor32::new(n, t).unwrap();
        let bits = exec.prandbit_batch(k).unwrap();
        assert_eq!(bits.len(), k);
        let mut ones = 0usize;
        for row in bits {
            assert_eq!(row.len(), n);
            let mut shares = Vec::with_capacity(n);
            for p in 0..n {
                shares.push(Share32 {
                    point: shamir.eval_points[p],
                    value: row[p],
                });
            }
            let rec = shamir.reconstruct(&shares).unwrap();
            let raw = rec.raw();
            assert!(
                raw == 0 || raw == 1,
                "PRandBit reconstructed non-bit value: {}",
                raw
            );
            if raw == 1 {
                ones += 1;
            }
        }
        // Very loose balance check.
        assert!(
            ones > 64 && ones < 192,
            "bit imbalance: {} ones out of {}",
            ones,
            k
        );
    }

    #[test]
    fn test_inprocess_dabit_batch_consistency() {
        let n = 5usize;
        let t = 1usize;
        let k = 128usize;
        let shamir = Shamir32::new(n, t).unwrap();
        let exec = InProcessBeaverExecutor32::new(n, t).unwrap();
        let dabits = exec.dabit_batch(k).unwrap();
        assert_eq!(dabits.len(), k);

        for row in dabits {
            assert_eq!(row.len(), n);
            let mut arith_sh = Vec::with_capacity(n);
            let mut bit_sh = Vec::with_capacity(n);
            for p in 0..n {
                arith_sh.push(Share32 {
                    point: shamir.eval_points[p],
                    value: row[p].arithmetic,
                });
                bit_sh.push(Share32 {
                    point: shamir.eval_points[p],
                    value: row[p].bit,
                });
            }
            let a = shamir.reconstruct(&arith_sh).unwrap();
            let b = shamir.reconstruct(&bit_sh).unwrap();
            assert_eq!(a, b);
            let raw = a.raw();
            assert!(
                raw == 0 || raw == 1,
                "daBit reconstructed non-bit value: {}",
                raw
            );
        }
    }

    #[test]
    fn test_pipelined_prandbit_provider() {
        let n = 5usize;
        let t = 1usize;
        let batch_size = 32usize;
        let shamir = Shamir32::new(n, t).unwrap();
        let provider = PipelinedPrandBitProvider32::new(n, t, batch_size, 4).unwrap();

        for _ in 0..2 {
            let batch = provider.next_batch().unwrap();
            assert_eq!(batch.len(), batch_size);
            for row in &batch {
                assert_eq!(row.len(), n);
                let mut shares = Vec::with_capacity(n);
                for p in 0..n {
                    shares.push(Share32 {
                        point: shamir.eval_points[p],
                        value: row[p],
                    });
                }
                let rec = shamir.reconstruct(&shares).unwrap().raw();
                assert!(
                    rec == 0 || rec == 1,
                    "PRandBit pipeline non-bit value: {}",
                    rec
                );
            }
        }
    }

    #[test]
    fn test_pipelined_dabit_provider() {
        let n = 5usize;
        let t = 1usize;
        let batch_size = 32usize;
        let shamir = Shamir32::new(n, t).unwrap();
        let provider = PipelinedDaBitProvider32::new(n, t, batch_size, 4).unwrap();

        for _ in 0..2 {
            let batch = provider.next_batch().unwrap();
            assert_eq!(batch.len(), batch_size);
            for row in &batch {
                assert_eq!(row.len(), n);
                let mut arith_sh = Vec::with_capacity(n);
                let mut bit_sh = Vec::with_capacity(n);
                for p in 0..n {
                    arith_sh.push(Share32 {
                        point: shamir.eval_points[p],
                        value: row[p].arithmetic,
                    });
                    bit_sh.push(Share32 {
                        point: shamir.eval_points[p],
                        value: row[p].bit,
                    });
                }
                let a = shamir.reconstruct(&arith_sh).unwrap();
                let b = shamir.reconstruct(&bit_sh).unwrap();
                assert_eq!(a, b);
                let raw = a.raw();
                assert!(
                    raw == 0 || raw == 1,
                    "daBit pipeline non-bit value: {}",
                    raw
                );
            }
        }
    }
}
