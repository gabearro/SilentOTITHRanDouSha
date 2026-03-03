use silent_ot_randousha::field::Fp;
use silent_ot_randousha::multiply::DnMultiply;
use silent_ot_randousha::network::{setup_tcp_network, PartyNetwork};
use silent_ot_randousha::randousha::{DoubleShare, RanDouShaParams, RanDouShaProtocol};
use silent_ot_randousha::shamir::{Shamir, Share};
use silent_ot_randousha::silent_ot::{
    Block, DistributedMessage, DistributedSilentOt, SilentOtParams,
};

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

const N: usize = 5;
const T: usize = 1;
const KING: usize = 0;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TaggedMessage {
    phase: u32,
    round: u32,
    payload: Vec<u8>,
}

struct BufferedNet {
    inner: PartyNetwork,
    buffer: Vec<(usize, TaggedMessage)>,
}

impl BufferedNet {
    fn new(net: PartyNetwork) -> Self {
        BufferedNet {
            inner: net,
            buffer: Vec::new(),
        }
    }

    async fn send<T: Serialize>(
        &self,
        to: usize,
        phase: u32,
        round: u32,
        msg: &T,
    ) -> Result<(), String> {
        let payload = bincode::serialize(msg).map_err(|e| e.to_string())?;
        self.inner
            .send_to(
                to,
                &TaggedMessage {
                    phase,
                    round,
                    payload,
                },
            )
            .await
    }

    async fn broadcast<T: Serialize>(
        &self,
        phase: u32,
        round: u32,
        msg: &T,
    ) -> Result<(), String> {
        let payload = bincode::serialize(msg).map_err(|e| e.to_string())?;
        self.inner
            .broadcast(&TaggedMessage {
                phase,
                round,
                payload,
            })
            .await
    }

    async fn recv<T: for<'de> Deserialize<'de>>(
        &mut self,
        phase: u32,
        round: u32,
    ) -> Result<(usize, T), String> {
        for i in 0..self.buffer.len() {
            if self.buffer[i].1.phase == phase && self.buffer[i].1.round == round {
                let (from, tagged) = self.buffer.remove(i);
                let msg: T = bincode::deserialize(&tagged.payload).map_err(|e| e.to_string())?;
                return Ok((from, msg));
            }
        }
        loop {
            let (from, tagged): (usize, TaggedMessage) = self.inner.recv().await?;
            if tagged.phase == phase && tagged.round == round {
                let msg: T =
                    bincode::deserialize(&tagged.payload).map_err(|e| e.to_string())?;
                return Ok((from, msg));
            }
            self.buffer.push((from, tagged));
        }
    }

    async fn recv_from_all<T: for<'de> Deserialize<'de>>(
        &mut self,
        phase: u32,
        round: u32,
    ) -> Result<HashMap<usize, T>, String> {
        let n = self.inner.n;
        let mut results = HashMap::new();
        for _ in 0..n - 1 {
            let (from, msg) = self.recv::<T>(phase, round).await?;
            results.insert(from, msg);
        }
        Ok(results)
    }
}

async fn run_silent_ot_setup(
    net: &mut BufferedNet,
    party_id: usize,
    rng: &mut ChaCha20Rng,
) -> Vec<Fp> {
    let ot_params = SilentOtParams::new(N, T, 128).unwrap();
    let protocol = DistributedSilentOt::new(ot_params);
    let mut ot_state = protocol.init_party(party_id, rng);

    let r0_msgs = DistributedSilentOt::round0_commitments(&ot_state);
    for msg in &r0_msgs {
        if let DistributedMessage::Commitment {
            to, commitment, ..
        } = msg
        {
            net.send(*to, 0, 0, commitment).await.unwrap();
        }
    }
    let commitments: HashMap<usize, [u8; 32]> = net.recv_from_all(0, 0).await.unwrap();
    let dist_r0: Vec<DistributedMessage> = commitments
        .iter()
        .map(|(from, c)| DistributedMessage::Commitment {
            from: *from,
            to: party_id,
            commitment: *c,
        })
        .collect();
    DistributedSilentOt::process_round0(&mut ot_state, &dist_r0).unwrap();

    let r1_msgs = DistributedSilentOt::round1_puncture_choices(&ot_state);
    for msg in &r1_msgs {
        if let DistributedMessage::PunctureChoice { to, index, .. } = msg {
            net.send(*to, 0, 1, index).await.unwrap();
        }
    }
    let choices: HashMap<usize, usize> = net.recv_from_all(0, 1).await.unwrap();
    let dist_r1: Vec<DistributedMessage> = choices
        .iter()
        .map(|(from, idx)| DistributedMessage::PunctureChoice {
            from: *from,
            to: party_id,
            index: *idx,
        })
        .collect();
    DistributedSilentOt::process_round1(&mut ot_state, &dist_r1).unwrap();

    let r2_msgs = DistributedSilentOt::round2_sibling_paths(&ot_state).unwrap();
    for msg in &r2_msgs {
        if let DistributedMessage::SiblingPathMsg {
            to, sibling_path, ..
        } = msg
        {
            net.send(*to, 0, 2, sibling_path).await.unwrap();
        }
    }
    let paths: HashMap<usize, Vec<Block>> = net.recv_from_all(0, 2).await.unwrap();
    let dist_r2: Vec<DistributedMessage> = paths
        .into_iter()
        .map(|(from, sp)| DistributedMessage::SiblingPathMsg {
            from,
            to: party_id,
            sibling_path: sp,
        })
        .collect();
    DistributedSilentOt::process_round2(&mut ot_state, &dist_r2).unwrap();

    let r3_msgs = DistributedSilentOt::round3_seed_reveals(&ot_state);
    for msg in &r3_msgs {
        if let DistributedMessage::SeedReveal { to, seed, .. } = msg {
            net.send(*to, 0, 3, seed).await.unwrap();
        }
    }
    let reveals: HashMap<usize, Block> = net.recv_from_all(0, 3).await.unwrap();
    let dist_r3: Vec<DistributedMessage> = reveals
        .into_iter()
        .map(|(from, seed)| DistributedMessage::SeedReveal {
            from,
            to: party_id,
            seed,
        })
        .collect();
    DistributedSilentOt::process_round3(&ot_state, &dist_r3).unwrap();

    let correlations = DistributedSilentOt::expand(&ot_state).unwrap();

    let num_randoms = 128;
    (0..num_randoms)
        .map(|k| correlations.get_random(k))
        .collect()
}

async fn run_party(net: PartyNetwork, party_id: usize) -> Option<Fp> {
    let mut rng = ChaCha20Rng::seed_from_u64(party_id as u64 + 1000);
    let mut net = BufferedNet::new(net);

    let ot_randoms = run_silent_ot_setup(&mut net, party_id, &mut rng).await;

    let shamir_t = Shamir::new(N, T).unwrap();
    let shamir_2t = Shamir::new(N, 2 * T).unwrap();
    let num_double_shares = 3;

    let mut my_contribs: Vec<(Vec<Share>, Vec<Share>)> = Vec::new();
    for k in 0..num_double_shares {
        let secret = ot_randoms[k];
        let shares_t = shamir_t.share(secret, &mut rng);
        let shares_2t = shamir_2t.share(secret, &mut rng);

        for j in 0..N {
            if j == party_id {
                continue;
            }
            let msg: (Share, Share) = (shares_t[j], shares_2t[j]);
            net.send(j, 1, k as u32, &msg).await.unwrap();
        }
        my_contribs.push((shares_t, shares_2t));
    }

    let mut my_double_shares: Vec<DoubleShare> = Vec::with_capacity(num_double_shares);
    for k in 0..num_double_shares {
        let received: HashMap<usize, (Share, Share)> =
            net.recv_from_all(1, k as u32).await.unwrap();

        let mut val_t = my_contribs[k].0[party_id];
        let mut val_2t = my_contribs[k].1[party_id];
        for (_, (st, s2t)) in &received {
            val_t.value = val_t.value + st.value;
            val_2t.value = val_2t.value + s2t.value;
        }
        my_double_shares.push(DoubleShare {
            share_t: val_t,
            share_2t: val_2t,
        });
    }

    let values = vec![Fp::new(3), Fp::new(5), Fp::new(7), Fp::new(11)];
    let num_values = values.len();
    let mut my_input_shares: Vec<Share> = Vec::with_capacity(num_values);

    if party_id == 0 {
        for (v_idx, v) in values.iter().enumerate() {
            let shares = shamir_t.share(*v, &mut rng);
            for j in 1..N {
                net.send(j, 2, v_idx as u32, &shares[j]).await.unwrap();
            }
            my_input_shares.push(shares[0]);
        }
    } else {
        for v_idx in 0..num_values {
            let (_, share): (usize, Share) = net.recv(2, v_idx as u32).await.unwrap();
            my_input_shares.push(share);
        }
    }

    let mut current_share = my_input_shares[0];

    for mult_idx in 0..(num_values - 1) {
        let next_share = my_input_shares[mult_idx + 1];
        let ds = &my_double_shares[mult_idx];

        let masked =
            DnMultiply::compute_masked_share(&current_share, &next_share, ds);

        let phase = 3;
        let round = (mult_idx * 2) as u32;

        if party_id != KING {
            net.send(KING, phase, round, &masked).await.unwrap();
        }

        let opened_value: Fp;
        if party_id == KING {
            let received: HashMap<usize, Share> =
                net.recv_from_all(phase, round).await.unwrap();
            let mut all_masked = vec![masked];
            for (_, s) in received {
                all_masked.push(s);
            }

            let dn = DnMultiply::new(N, T, KING).unwrap();
            opened_value = dn.king_reconstruct(&all_masked).unwrap();

            net.broadcast(phase, round + 1, &opened_value)
                .await
                .unwrap();
        } else {
            let (_, val): (usize, Fp) = net.recv(phase, round + 1).await.unwrap();
            opened_value = val;
        }

        current_share = DnMultiply::compute_output_share(opened_value, ds);
    }

    if party_id != 0 {
        net.send(0, 4, 0, &current_share).await.unwrap();
        None
    } else {
        let received: HashMap<usize, Share> = net.recv_from_all(4, 0).await.unwrap();
        let mut all_shares = vec![current_share];
        for (_, s) in received {
            all_shares.push(s);
        }
        let result = shamir_t.reconstruct(&all_shares).unwrap();
        Some(result)
    }
}

#[tokio::test]
async fn test_networked_silent_ot_with_commitments() {
    let networks = setup_tcp_network(N, 17100).await;

    let mut handles = Vec::new();
    for (i, net) in networks.into_iter().enumerate() {
        handles.push(tokio::spawn(async move {
            let mut rng = ChaCha20Rng::seed_from_u64(i as u64 + 2000);
            let mut bnet = BufferedNet::new(net);
            let randoms = run_silent_ot_setup(&mut bnet, i, &mut rng).await;
            assert!(!randoms.is_empty());
            true
        }));
    }

    for handle in handles {
        assert!(handle.await.unwrap());
    }
}

#[tokio::test]
async fn test_networked_randousha_verification() {
    let networks = setup_tcp_network(N, 17200).await;

    let mut handles = Vec::new();
    for (i, net) in networks.into_iter().enumerate() {
        handles.push(tokio::spawn(async move {
            let mut rng = ChaCha20Rng::seed_from_u64(i as u64 + 3000);
            let mut bnet = BufferedNet::new(net);
            let shamir_t = Shamir::new(N, T).unwrap();
            let shamir_2t = Shamir::new(N, 2 * T).unwrap();
            let num_ds = 5;

            let ot_randoms = run_silent_ot_setup(&mut bnet, i, &mut rng).await;

            let mut my_contribs: Vec<(Vec<Share>, Vec<Share>)> = Vec::new();
            for k in 0..num_ds {
                let secret = ot_randoms[k];
                let st = shamir_t.share(secret, &mut rng);
                let s2t = shamir_2t.share(secret, &mut rng);
                for j in 0..N {
                    if j == i {
                        continue;
                    }
                    let msg: (Share, Share) = (st[j], s2t[j]);
                    bnet.send(j, 1, k as u32, &msg).await.unwrap();
                }
                my_contribs.push((st, s2t));
            }

            let mut double_shares = Vec::new();
            for k in 0..num_ds {
                let received: HashMap<usize, (Share, Share)> =
                    bnet.recv_from_all(1, k as u32).await.unwrap();
                let mut val_t = my_contribs[k].0[i];
                let mut val_2t = my_contribs[k].1[i];
                for (_, (st, s2t)) in &received {
                    val_t.value = val_t.value + st.value;
                    val_2t.value = val_2t.value + s2t.value;
                }
                double_shares.push(DoubleShare {
                    share_t: val_t,
                    share_2t: val_2t,
                });
            }

            if i != 0 {
                bnet.send(0, 2, 0, &double_shares).await.unwrap();
            }

            if i == 0 {
                let mut all_party_ds: Vec<Vec<DoubleShare>> = vec![Vec::new(); N];
                all_party_ds[0] = double_shares;
                let received: HashMap<usize, Vec<DoubleShare>> =
                    bnet.recv_from_all(2, 0).await.unwrap();
                for (from, ds) in received {
                    all_party_ds[from] = ds;
                }

                for k in 0..num_ds {
                    let shares_t: Vec<Share> =
                        (0..N).map(|p| all_party_ds[p][k].share_t).collect();
                    let shares_2t: Vec<Share> =
                        (0..N).map(|p| all_party_ds[p][k].share_2t).collect();
                    let secret_t = shamir_t.reconstruct(&shares_t).unwrap();
                    let secret_2t = shamir_2t.reconstruct(&shares_2t).unwrap();
                    assert_eq!(secret_t, secret_2t, "double share {} mismatch", k);
                }
            }
            true
        }));
    }

    for handle in handles {
        assert!(handle.await.unwrap());
    }
}

#[tokio::test]
async fn test_networked_multiplication_t1_n5() {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .try_init();

    eprintln!("=== Networked Integration Test: t=1, n=5 ===");
    eprintln!("Computing: 3 * 5 * 7 * 11 = 1155");

    let networks = setup_tcp_network(N, 17300).await;

    let mut handles = Vec::new();
    for (i, net) in networks.into_iter().enumerate() {
        handles.push(tokio::spawn(run_party(net, i)));
    }

    let mut results = Vec::new();
    for handle in handles {
        results.push(handle.await.unwrap());
    }

    let result = results[0].expect("Party 0 should produce a result");
    let expected = Fp::new(3) * Fp::new(5) * Fp::new(7) * Fp::new(11);
    assert_eq!(expected, Fp::new(1155));
    assert_eq!(result, expected, "got {} expected {}", result, expected);
    eprintln!("=== PASSED: {} == {} ===", result, expected);
}

#[tokio::test]
async fn test_networked_mass_chained_multiplication() {
    let num_values = 10;
    let values: Vec<Fp> = (2..2 + num_values as u64).map(Fp::new).collect();
    let expected: Fp = values.iter().copied().reduce(|a, b| a * b).unwrap();
    let num_mults = num_values - 1;

    let networks = setup_tcp_network(N, 17400).await;

    let mut handles = Vec::new();
    for (party_id, net) in networks.into_iter().enumerate() {
        let values = values.clone();
        handles.push(tokio::spawn(async move {
            let mut rng = ChaCha20Rng::seed_from_u64(party_id as u64 + 5000);
            let mut bnet = BufferedNet::new(net);
            let shamir_t = Shamir::new(N, T).unwrap();
            let shamir_2t = Shamir::new(N, 2 * T).unwrap();

            let ot_randoms = run_silent_ot_setup(&mut bnet, party_id, &mut rng).await;

            let mut my_contribs: Vec<(Vec<Share>, Vec<Share>)> = Vec::new();
            for k in 0..num_mults {
                let secret = ot_randoms[k];
                let st = shamir_t.share(secret, &mut rng);
                let s2t = shamir_2t.share(secret, &mut rng);
                for j in 0..N {
                    if j == party_id {
                        continue;
                    }
                    let msg: (Share, Share) = (st[j], s2t[j]);
                    bnet.send(j, 1, k as u32, &msg).await.unwrap();
                }
                my_contribs.push((st, s2t));
            }

            let mut double_shares: Vec<DoubleShare> = Vec::with_capacity(num_mults);
            for k in 0..num_mults {
                let received: HashMap<usize, (Share, Share)> =
                    bnet.recv_from_all(1, k as u32).await.unwrap();
                let mut val_t = my_contribs[k].0[party_id];
                let mut val_2t = my_contribs[k].1[party_id];
                for (_, (st, s2t)) in &received {
                    val_t.value = val_t.value + st.value;
                    val_2t.value = val_2t.value + s2t.value;
                }
                double_shares.push(DoubleShare {
                    share_t: val_t,
                    share_2t: val_2t,
                });
            }

            let mut my_input_shares: Vec<Share> = Vec::with_capacity(num_values);
            if party_id == 0 {
                for (v_idx, v) in values.iter().enumerate() {
                    let shares = shamir_t.share(*v, &mut rng);
                    for j in 1..N {
                        bnet.send(j, 2, v_idx as u32, &shares[j]).await.unwrap();
                    }
                    my_input_shares.push(shares[0]);
                }
            } else {
                for v_idx in 0..num_values {
                    let (_, share): (usize, Share) =
                        bnet.recv(2, v_idx as u32).await.unwrap();
                    my_input_shares.push(share);
                }
            }

            let mut current = my_input_shares[0];
            for mult_idx in 0..num_mults {
                let next = my_input_shares[mult_idx + 1];
                let ds = &double_shares[mult_idx];
                let masked =
                    DnMultiply::compute_masked_share(&current, &next, ds);

                let phase = 3;
                let round = (mult_idx * 2) as u32;

                if party_id != KING {
                    bnet.send(KING, phase, round, &masked).await.unwrap();
                }

                let opened: Fp;
                if party_id == KING {
                    let received: HashMap<usize, Share> =
                        bnet.recv_from_all(phase, round).await.unwrap();
                    let mut all_masked = vec![masked];
                    for (_, s) in received {
                        all_masked.push(s);
                    }
                    let dn = DnMultiply::new(N, T, KING).unwrap();
                    opened = dn.king_reconstruct(&all_masked).unwrap();
                    bnet.broadcast(phase, round + 1, &opened).await.unwrap();
                } else {
                    let (_, val): (usize, Fp) =
                        bnet.recv(phase, round + 1).await.unwrap();
                    opened = val;
                }

                current = DnMultiply::compute_output_share(opened, ds);
            }

            if party_id != 0 {
                bnet.send(0, 4, 0, &current).await.unwrap();
                None
            } else {
                let received: HashMap<usize, Share> =
                    bnet.recv_from_all(4, 0).await.unwrap();
                let mut all_shares = vec![current];
                for (_, s) in received {
                    all_shares.push(s);
                }
                Some(shamir_t.reconstruct(&all_shares).unwrap())
            }
        }));
    }

    let mut results = Vec::new();
    for handle in handles {
        results.push(handle.await.unwrap());
    }

    let result = results[0].expect("Party 0 should produce a result");
    assert_eq!(
        result, expected,
        "chained multiplication of {} values: got {} expected {}",
        num_values, result, expected
    );
    eprintln!(
        "=== Mass chained multiplication PASSED: {} values, result = {} ===",
        num_values, result
    );
}

async fn run_offline_online_multiply(
    net: PartyNetwork,
    party_id: usize,
    values: Vec<Fp>,
    ot_count: usize,
) -> Option<Fp> {
    let mut rng = ChaCha20Rng::seed_from_u64(party_id as u64 + 7000);
    let mut bnet = BufferedNet::new(net);
    let shamir_t = Shamir::new(N, T).unwrap();
    let shamir_2t = Shamir::new(N, 2 * T).unwrap();
    let num_mults = values.len() - 1;

    let ot_randoms = run_silent_ot_setup(&mut bnet, party_id, &mut rng).await;

    let mut my_contribs: Vec<(Vec<Share>, Vec<Share>)> = Vec::new();
    for k in 0..num_mults {
        let secret = ot_randoms[k % ot_count];
        let st = shamir_t.share(secret, &mut rng);
        let s2t = shamir_2t.share(secret, &mut rng);
        for j in 0..N {
            if j == party_id {
                continue;
            }
            let msg: (Share, Share) = (st[j], s2t[j]);
            bnet.send(j, 1, k as u32, &msg).await.unwrap();
        }
        my_contribs.push((st, s2t));
    }

    let mut double_shares: Vec<DoubleShare> = Vec::with_capacity(num_mults);
    for k in 0..num_mults {
        let received: HashMap<usize, (Share, Share)> =
            bnet.recv_from_all(1, k as u32).await.unwrap();
        let mut val_t = my_contribs[k].0[party_id];
        let mut val_2t = my_contribs[k].1[party_id];
        for (_, (st, s2t)) in &received {
            val_t.value = val_t.value + st.value;
            val_2t.value = val_2t.value + s2t.value;
        }
        double_shares.push(DoubleShare {
            share_t: val_t,
            share_2t: val_2t,
        });
    }

    let num_values = values.len();
    let mut my_input_shares: Vec<Share> = Vec::with_capacity(num_values);
    if party_id == 0 {
        for (v_idx, v) in values.iter().enumerate() {
            let shares = shamir_t.share(*v, &mut rng);
            for j in 1..N {
                bnet.send(j, 2, v_idx as u32, &shares[j]).await.unwrap();
            }
            my_input_shares.push(shares[0]);
        }
    } else {
        for v_idx in 0..num_values {
            let (_, share): (usize, Share) = bnet.recv(2, v_idx as u32).await.unwrap();
            my_input_shares.push(share);
        }
    }

    let mut current = my_input_shares[0];
    for mult_idx in 0..num_mults {
        let next = my_input_shares[mult_idx + 1];
        let ds = &double_shares[mult_idx];
        let masked = DnMultiply::compute_masked_share(&current, &next, ds);

        let phase = 3;
        let round = (mult_idx * 2) as u32;

        if party_id != KING {
            bnet.send(KING, phase, round, &masked).await.unwrap();
        }

        let opened: Fp;
        if party_id == KING {
            let received: HashMap<usize, Share> =
                bnet.recv_from_all(phase, round).await.unwrap();
            let mut all_masked = vec![masked];
            for (_, s) in received {
                all_masked.push(s);
            }
            let dn = DnMultiply::new(N, T, KING).unwrap();
            opened = dn.king_reconstruct(&all_masked).unwrap();
            bnet.broadcast(phase, round + 1, &opened).await.unwrap();
        } else {
            let (_, val): (usize, Fp) = bnet.recv(phase, round + 1).await.unwrap();
            opened = val;
        }

        current = DnMultiply::compute_output_share(opened, ds);
    }

    if party_id != 0 {
        bnet.send(0, 4, 0, &current).await.unwrap();
        None
    } else {
        let received: HashMap<usize, Share> = bnet.recv_from_all(4, 0).await.unwrap();
        let mut all_shares = vec![current];
        for (_, s) in received {
            all_shares.push(s);
        }
        Some(shamir_t.reconstruct(&all_shares).unwrap())
    }
}

#[tokio::test]
async fn test_mass_chained_offline_multiply_with_reveal_20() {
    let num_values = 20;
    let values: Vec<Fp> = (2..2 + num_values as u64).map(Fp::new).collect();
    let expected: Fp = values.iter().copied().reduce(|a, b| a * b).unwrap();

    eprintln!(
        "=== Mass offline multiply+reveal: {} values (2*3*...*{}) ===",
        num_values,
        num_values + 1
    );

    let networks = setup_tcp_network(N, 17500).await;

    let mut handles = Vec::new();
    for (party_id, net) in networks.into_iter().enumerate() {
        let vals = values.clone();
        handles.push(tokio::spawn(
            run_offline_online_multiply(net, party_id, vals, 128),
        ));
    }

    let mut results = Vec::new();
    for handle in handles {
        results.push(handle.await.unwrap());
    }

    let result = results[0].expect("Party 0 should produce result");
    assert_eq!(result, expected, "20-value chained multiply: {} != {}", result, expected);
    eprintln!("=== PASSED: 20-value chained offline multiply+reveal = {} ===", result);
}

#[tokio::test]
async fn test_mass_chained_offline_multiply_with_reveal_50() {
    let num_values = 50;
    let values: Vec<Fp> = (1..=num_values as u64).map(Fp::new).collect();
    let expected: Fp = values.iter().copied().reduce(|a, b| a * b).unwrap();

    eprintln!("=== Mass offline multiply+reveal: {} values ({}!) ===", num_values, num_values);

    let networks = setup_tcp_network(N, 17600).await;

    let mut handles = Vec::new();
    for (party_id, net) in networks.into_iter().enumerate() {
        let vals = values.clone();
        handles.push(tokio::spawn(
            run_offline_online_multiply(net, party_id, vals, 128),
        ));
    }

    let mut results = Vec::new();
    for handle in handles {
        results.push(handle.await.unwrap());
    }

    let result = results[0].expect("Party 0 should produce result");
    assert_eq!(result, expected, "50!: {} != {}", result, expected);
    eprintln!("=== PASSED: 50-value chained offline multiply+reveal = {} ===", result);
}

#[tokio::test]
async fn test_offline_online_separation_with_reveal() {
    eprintln!("=== Testing explicit offline/online phase separation ===");

    let networks = setup_tcp_network(N, 17700).await;

    let values = vec![Fp::new(13), Fp::new(17), Fp::new(19), Fp::new(23), Fp::new(29)];
    let expected: Fp = values.iter().copied().reduce(|a, b| a * b).unwrap();
    let expected_plain = 13u64 * 17 * 19 * 23 * 29;
    assert_eq!(expected, Fp::new(expected_plain));
    eprintln!("Computing: 13 * 17 * 19 * 23 * 29 = {}", expected_plain);

    let mut handles = Vec::new();
    for (party_id, net) in networks.into_iter().enumerate() {
        let vals = values.clone();
        handles.push(tokio::spawn(
            run_offline_online_multiply(net, party_id, vals, 128),
        ));
    }

    let mut results = Vec::new();
    for handle in handles {
        results.push(handle.await.unwrap());
    }

    let result = results[0].expect("Party 0 should produce result");
    assert_eq!(result, expected);
    eprintln!("=== PASSED: offline/online separation test = {} ===", result);
}

#[tokio::test]
async fn test_malicious_double_share_detection() {
    eprintln!("=== Testing malicious double-share detection ===");

    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let n = N;
    let t = T;
    let params = RanDouShaParams::new(n, t, 5).unwrap();
    let protocol = RanDouShaProtocol::new(params);

    let mut party_shares = protocol.generate_local(&mut rng).unwrap();

    assert!(RanDouShaProtocol::verify(&party_shares, n, t).unwrap());
    eprintln!("  Honest double-shares: VERIFIED");

    let original = party_shares[1][0].share_t.value;
    party_shares[1][0].share_t.value = original + Fp::new(999);

    let result = RanDouShaProtocol::verify(&party_shares, n, t);
    assert!(result.is_err(), "tampered shares must be detected");
    eprintln!("  Tampered degree-t share: DETECTED");

    party_shares[1][0].share_t.value = original;
    party_shares[1][0].share_2t.value = party_shares[1][0].share_2t.value + Fp::new(42);

    let result = RanDouShaProtocol::verify(&party_shares, n, t);
    assert!(result.is_err(), "tampered 2t shares must be detected");
    eprintln!("  Tampered degree-2t share: DETECTED");

    eprintln!("=== PASSED: malicious double-share detection ===");
}

#[tokio::test]
async fn test_malicious_king_detection() {
    eprintln!("=== Testing malicious king detection ===");

    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let n = N;
    let t = T;
    let shamir_t = Shamir::new(n, t).unwrap();

    let x = Fp::new(42);
    let y = Fp::new(17);
    let x_shares = shamir_t.share(x, &mut rng);
    let y_shares = shamir_t.share(y, &mut rng);

    let params = RanDouShaParams::new(n, t, 1).unwrap();
    let party_ds = RanDouShaProtocol::new(params)
        .generate_local(&mut rng)
        .unwrap();
    let ds: Vec<DoubleShare> = (0..n).map(|p| party_ds[p][0].clone()).collect();

    let masked_shares: Vec<Share> = (0..n)
        .map(|i| DnMultiply::compute_masked_share(&x_shares[i], &y_shares[i], &ds[i]))
        .collect();

    let dn = DnMultiply::new(n, t, 0).unwrap();
    let honest_opened = dn.king_reconstruct(&masked_shares).unwrap();

    dn.verify_king_broadcast(&masked_shares, honest_opened).unwrap();
    eprintln!("  Honest king broadcast: VERIFIED");

    let malicious_opened = honest_opened + Fp::new(1);
    let result = dn.verify_king_broadcast(&masked_shares, malicious_opened);
    assert!(result.is_err());
    eprintln!("  Malicious king broadcast: DETECTED");

    let result_shares: Vec<Share> = (0..n)
        .map(|i| DnMultiply::compute_output_share(honest_opened, &ds[i]))
        .collect();
    let product = shamir_t.reconstruct(&result_shares).unwrap();
    assert_eq!(product, x * y);
    eprintln!("  Correct product: {} * {} = {} (verified)", x, y, product);

    eprintln!("=== PASSED: malicious king detection ===");
}

#[tokio::test]
async fn test_mass_chained_offline_multiply_with_reveal_100() {
    let num_values = 100;
    let values: Vec<Fp> = (0..num_values).map(|i| Fp::new((i % 7 + 2) as u64)).collect();
    let expected: Fp = values.iter().copied().reduce(|a, b| a * b).unwrap();

    eprintln!("=== Mass offline multiply+reveal: {} values ===", num_values);

    let networks = setup_tcp_network(N, 17800).await;

    let mut handles = Vec::new();
    for (party_id, net) in networks.into_iter().enumerate() {
        let vals = values.clone();
        handles.push(tokio::spawn(
            run_offline_online_multiply(net, party_id, vals, 128),
        ));
    }

    let mut results = Vec::new();
    for handle in handles {
        results.push(handle.await.unwrap());
    }

    let result = results[0].expect("Party 0 should produce result");
    assert_eq!(result, expected, "100-value multiply failed: {} != {}", result, expected);
    eprintln!("=== PASSED: 100-value chained offline multiply+reveal = {} ===", result);
}

#[tokio::test]
async fn test_chained_multiply_with_intermediate_reveals() {
    let values = vec![Fp::new(2), Fp::new(3), Fp::new(5), Fp::new(7), Fp::new(11)];
    let num_values = values.len();
    let num_mults = num_values - 1;

    eprintln!("=== Chained multiply with intermediate reveals ===");
    eprintln!("Values: 2, 3, 5, 7, 11");

    let networks = setup_tcp_network(N, 17900).await;

    let mut handles = Vec::new();
    for (party_id, net) in networks.into_iter().enumerate() {
        let vals = values.clone();
        handles.push(tokio::spawn(async move {
            let mut rng = ChaCha20Rng::seed_from_u64(party_id as u64 + 9000);
            let mut bnet = BufferedNet::new(net);
            let shamir_t = Shamir::new(N, T).unwrap();
            let shamir_2t = Shamir::new(N, 2 * T).unwrap();

            let ot_randoms = run_silent_ot_setup(&mut bnet, party_id, &mut rng).await;
            let mut my_contribs: Vec<(Vec<Share>, Vec<Share>)> = Vec::new();
            for k in 0..num_mults {
                let secret = ot_randoms[k];
                let st = shamir_t.share(secret, &mut rng);
                let s2t = shamir_2t.share(secret, &mut rng);
                for j in 0..N {
                    if j == party_id {
                        continue;
                    }
                    bnet.send(j, 1, k as u32, &(st[j], s2t[j])).await.unwrap();
                }
                my_contribs.push((st, s2t));
            }

            let mut double_shares: Vec<DoubleShare> = Vec::with_capacity(num_mults);
            for k in 0..num_mults {
                let received: HashMap<usize, (Share, Share)> =
                    bnet.recv_from_all(1, k as u32).await.unwrap();
                let mut val_t = my_contribs[k].0[party_id];
                let mut val_2t = my_contribs[k].1[party_id];
                for (_, (st, s2t)) in &received {
                    val_t.value = val_t.value + st.value;
                    val_2t.value = val_2t.value + s2t.value;
                }
                double_shares.push(DoubleShare {
                    share_t: val_t,
                    share_2t: val_2t,
                });
            }

            let mut my_input_shares: Vec<Share> = Vec::with_capacity(num_values);
            if party_id == 0 {
                for (v_idx, v) in vals.iter().enumerate() {
                    let shares = shamir_t.share(*v, &mut rng);
                    for j in 1..N {
                        bnet.send(j, 2, v_idx as u32, &shares[j]).await.unwrap();
                    }
                    my_input_shares.push(shares[0]);
                }
            } else {
                for v_idx in 0..num_values {
                    let (_, share): (usize, Share) =
                        bnet.recv(2, v_idx as u32).await.unwrap();
                    my_input_shares.push(share);
                }
            }

            let mut current = my_input_shares[0];
            let mut intermediate_results: Vec<Fp> = Vec::new();

            for mult_idx in 0..num_mults {
                let next = my_input_shares[mult_idx + 1];
                let ds = &double_shares[mult_idx];
                let masked = DnMultiply::compute_masked_share(&current, &next, ds);

                let phase = 3;
                let round = (mult_idx * 3) as u32;

                if party_id != KING {
                    bnet.send(KING, phase, round, &masked).await.unwrap();
                }

                let opened: Fp;
                if party_id == KING {
                    let received: HashMap<usize, Share> =
                        bnet.recv_from_all(phase, round).await.unwrap();
                    let mut all_masked = vec![masked];
                    for (_, s) in received {
                        all_masked.push(s);
                    }
                    let dn = DnMultiply::new(N, T, KING).unwrap();
                    opened = dn.king_reconstruct(&all_masked).unwrap();
                    bnet.broadcast(phase, round + 1, &opened).await.unwrap();
                } else {
                    let (_, val): (usize, Fp) =
                        bnet.recv(phase, round + 1).await.unwrap();
                    opened = val;
                }

                current = DnMultiply::compute_output_share(opened, ds);

                let reveal_round = round + 2;
                if party_id != 0 {
                    bnet.send(0, phase, reveal_round, &current).await.unwrap();
                }
                if party_id == 0 {
                    let received: HashMap<usize, Share> =
                        bnet.recv_from_all(phase, reveal_round).await.unwrap();
                    let mut all_shares = vec![current];
                    for (_, s) in received {
                        all_shares.push(s);
                    }
                    let intermediate = shamir_t.reconstruct(&all_shares).unwrap();
                    intermediate_results.push(intermediate);
                }
            }

            if party_id == 0 {
                Some(intermediate_results)
            } else {
                None
            }
        }));
    }

    let mut party_results = Vec::new();
    for handle in handles {
        party_results.push(handle.await.unwrap());
    }

    let intermediates = party_results[0].as_ref().unwrap();
    let expected_intermediates = vec![
        Fp::new(2 * 3),                    // 6
        Fp::new(2 * 3 * 5),                // 30
        Fp::new(2 * 3 * 5 * 7),            // 210
        Fp::new(2 * 3 * 5 * 7 * 11),       // 2310
    ];

    for (i, (got, exp)) in intermediates.iter().zip(expected_intermediates.iter()).enumerate() {
        assert_eq!(
            *got, *exp,
            "intermediate result {} mismatch: {} != {}",
            i, got, exp
        );
        eprintln!("  Step {}: product = {} (expected {})", i + 1, got, exp);
    }
    eprintln!("=== PASSED: all intermediate reveals correct ===");
}
