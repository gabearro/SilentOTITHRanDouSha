#![allow(clippy::needless_range_loop)]

use silent_ot_randousha::field::Fp;
use silent_ot_randousha::multiply::DnMultiply;
use silent_ot_randousha::network::{setup_tcp_network, PartyNetwork};
use silent_ot_randousha::randousha::{
    DoubleShare, HyperInvertibleMatrix, RanDouShaParams, RanDouShaProtocol,
};
use silent_ot_randousha::shamir::{Shamir, Share};
use silent_ot_randousha::silent_ot::{Block, DistributedSilentOt, SilentOtParams};

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

    async fn broadcast<T: Serialize>(&self, phase: u32, round: u32, msg: &T) -> Result<(), String> {
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
                let msg: T = bincode::deserialize(&tagged.payload).map_err(|e| e.to_string())?;
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
    run_silent_ot_setup_n(net, party_id, rng, 128).await
}

async fn run_silent_ot_setup_n(
    net: &mut BufferedNet,
    party_id: usize,
    rng: &mut ChaCha20Rng,
    num_randoms: usize,
) -> Vec<Fp> {
    let ot_params = SilentOtParams::new(N, T, std::cmp::max(num_randoms, 16)).unwrap();
    let protocol = DistributedSilentOt::new(ot_params);
    let mut ot_state = protocol.init_party(party_id, rng);

    // Round 0: commitments
    for (to, commitment) in DistributedSilentOt::round0_commitments(&ot_state) {
        net.send(to, 0, 0, &commitment).await.unwrap();
    }
    let commitments: HashMap<usize, [u8; 32]> = net.recv_from_all(0, 0).await.unwrap();
    let r0: Vec<(usize, [u8; 32])> = commitments.into_iter().collect();
    DistributedSilentOt::process_round0(&mut ot_state, &r0).unwrap();

    // Round 1: puncture choices
    for (to, index) in DistributedSilentOt::round1_puncture_choices(&ot_state) {
        net.send(to, 0, 1, &index).await.unwrap();
    }
    let choices: HashMap<usize, usize> = net.recv_from_all(0, 1).await.unwrap();
    let r1: Vec<(usize, usize)> = choices.into_iter().collect();
    DistributedSilentOt::process_round1(&mut ot_state, &r1).unwrap();

    // Round 2: sibling paths
    for (to, sibling_path) in DistributedSilentOt::round2_sibling_paths(&ot_state).unwrap() {
        net.send(to, 0, 2, &sibling_path).await.unwrap();
    }
    let paths: HashMap<usize, Vec<Block>> = net.recv_from_all(0, 2).await.unwrap();
    let r2: Vec<(usize, Vec<Block>)> = paths.into_iter().collect();
    DistributedSilentOt::process_round2(&mut ot_state, &r2).unwrap();

    // Round 3: seed reveals
    for (to, seed) in DistributedSilentOt::round3_seed_reveals(&ot_state) {
        net.send(to, 0, 3, &seed).await.unwrap();
    }
    let reveals: HashMap<usize, Block> = net.recv_from_all(0, 3).await.unwrap();
    let r3: Vec<(usize, Block)> = reveals.into_iter().collect();
    DistributedSilentOt::process_round3(&mut ot_state, &r3).unwrap();

    let correlations = DistributedSilentOt::expand(&ot_state).unwrap();

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
        for (st, s2t) in received.values() {
            val_t.value += st.value;
            val_2t.value += s2t.value;
        }
        my_double_shares.push(DoubleShare {
            share_t: val_t,
            share_2t: val_2t,
        });
    }

    let values = [Fp::new(3), Fp::new(5), Fp::new(7), Fp::new(11)];
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

        let masked = DnMultiply::compute_masked_share(&current_share, &next_share, ds);

        let phase = 3;
        let round = (mult_idx * 2) as u32;

        if party_id != KING {
            net.send(KING, phase, round, &masked).await.unwrap();
        }

        let opened_value: Fp;
        if party_id == KING {
            let received: HashMap<usize, Share> = net.recv_from_all(phase, round).await.unwrap();
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
                for (st, s2t) in received.values() {
                    val_t.value += st.value;
                    val_2t.value += s2t.value;
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
                    let shares_t: Vec<Share> = (0..N).map(|p| all_party_ds[p][k].share_t).collect();
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
                for (st, s2t) in received.values() {
                    val_t.value += st.value;
                    val_2t.value += s2t.value;
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
    _ot_count: usize,
) -> Option<Fp> {
    let mut rng = ChaCha20Rng::seed_from_u64(party_id as u64 + 7000);
    let mut bnet = BufferedNet::new(net);
    let shamir_t = Shamir::new(N, T).unwrap();
    let shamir_2t = Shamir::new(N, 2 * T).unwrap();
    let num_mults = values.len() - 1;
    let sharings_per_round = N - 2 * T;
    let num_him_rounds = num_mults.div_ceil(sharings_per_round);
    let him = HyperInvertibleMatrix::new(N);

    let ot_randoms = run_silent_ot_setup_n(&mut bnet, party_id, &mut rng, num_him_rounds).await;

    let mut double_shares: Vec<DoubleShare> = Vec::with_capacity(num_mults);

    for him_round in 0..num_him_rounds {
        let secret = ot_randoms[him_round];
        let st = shamir_t.share(secret, &mut rng);
        let s2t = shamir_2t.share(secret, &mut rng);

        for j in 0..N {
            if j == party_id {
                continue;
            }
            let msg: (Share, Share) = (st[j], s2t[j]);
            bnet.send(j, 1, him_round as u32, &msg).await.unwrap();
        }

        let received: HashMap<usize, (Share, Share)> =
            bnet.recv_from_all(1, him_round as u32).await.unwrap();

        let mut input_t = vec![Fp::ZERO; N];
        let mut input_2t = vec![Fp::ZERO; N];
        input_t[party_id] = st[party_id].value;
        input_2t[party_id] = s2t[party_id].value;
        for (&from, (s, s2)) in &received {
            input_t[from] = s.value;
            input_2t[from] = s2.value;
        }

        let out_t = him.mul_vec(&input_t);
        let out_2t = him.mul_vec(&input_2t);

        for check_idx in sharings_per_round..N {
            let my_check_t = out_t[check_idx];
            let my_check_2t = out_2t[check_idx];
            let check_msg: (Fp, Fp) = (my_check_t, my_check_2t);
            bnet.broadcast(10, (him_round * N + check_idx) as u32, &check_msg)
                .await
                .unwrap();
            let check_received: HashMap<usize, (Fp, Fp)> = bnet
                .recv_from_all(10, (him_round * N + check_idx) as u32)
                .await
                .unwrap();

            let mut check_shares_t = vec![Share {
                point: shamir_t.eval_points[party_id],
                value: my_check_t,
            }];
            let mut check_shares_2t = vec![Share {
                point: shamir_2t.eval_points[party_id],
                value: my_check_2t,
            }];
            for (&from, &(ct, c2t)) in &check_received {
                check_shares_t.push(Share {
                    point: shamir_t.eval_points[from],
                    value: ct,
                });
                check_shares_2t.push(Share {
                    point: shamir_2t.eval_points[from],
                    value: c2t,
                });
            }
            let secret_t = shamir_t.reconstruct(&check_shares_t).unwrap();
            let secret_2t = shamir_2t.reconstruct(&check_shares_2t).unwrap();
            assert_eq!(
                secret_t, secret_2t,
                "HIM check row {} failed in round {}: t={} vs 2t={}",
                check_idx, him_round, secret_t, secret_2t
            );
        }

        for j in 0..sharings_per_round {
            if double_shares.len() >= num_mults {
                break;
            }
            double_shares.push(DoubleShare {
                share_t: Share {
                    point: shamir_t.eval_points[party_id],
                    value: out_t[j],
                },
                share_2t: Share {
                    point: shamir_2t.eval_points[party_id],
                    value: out_2t[j],
                },
            });
        }
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

    let dn = DnMultiply::new(N, T, KING).unwrap();
    let mut current = my_input_shares[0];
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
            let received: HashMap<usize, Share> = bnet.recv_from_all(phase, round).await.unwrap();
            let mut all_masked = vec![masked];
            for (_, s) in received {
                all_masked.push(s);
            }
            opened = dn.king_reconstruct(&all_masked).unwrap();
            bnet.broadcast(phase, round + 1, &opened).await.unwrap();
        } else {
            let (_, val): (usize, Fp) = bnet.recv(phase, round + 1).await.unwrap();
            opened = val;
        }

        bnet.broadcast(phase, round + 2, &masked).await.unwrap();
        let verify_received: HashMap<usize, Share> =
            bnet.recv_from_all(phase, round + 2).await.unwrap();
        let mut verify_shares = vec![masked];
        for (_, s) in verify_received {
            verify_shares.push(s);
        }
        dn.verify_king_broadcast(&verify_shares, opened).unwrap();

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
        handles.push(tokio::spawn(run_offline_online_multiply(
            net, party_id, vals, 128,
        )));
    }

    let mut results = Vec::new();
    for handle in handles {
        results.push(handle.await.unwrap());
    }

    let result = results[0].expect("Party 0 should produce result");
    assert_eq!(
        result, expected,
        "20-value chained multiply: {} != {}",
        result, expected
    );
    eprintln!(
        "=== PASSED: 20-value chained offline multiply+reveal = {} ===",
        result
    );
}

#[tokio::test]
async fn test_mass_chained_offline_multiply_with_reveal_50() {
    let num_values = 50;
    let values: Vec<Fp> = (1..=num_values as u64).map(Fp::new).collect();
    let expected: Fp = values.iter().copied().reduce(|a, b| a * b).unwrap();

    eprintln!(
        "=== Mass offline multiply+reveal: {} values ({}!) ===",
        num_values, num_values
    );

    let networks = setup_tcp_network(N, 17600).await;

    let mut handles = Vec::new();
    for (party_id, net) in networks.into_iter().enumerate() {
        let vals = values.clone();
        handles.push(tokio::spawn(run_offline_online_multiply(
            net, party_id, vals, 128,
        )));
    }

    let mut results = Vec::new();
    for handle in handles {
        results.push(handle.await.unwrap());
    }

    let result = results[0].expect("Party 0 should produce result");
    assert_eq!(result, expected, "50!: {} != {}", result, expected);
    eprintln!(
        "=== PASSED: 50-value chained offline multiply+reveal = {} ===",
        result
    );
}

#[tokio::test]
async fn test_offline_online_separation_with_reveal() {
    eprintln!("=== Testing explicit offline/online phase separation ===");

    let networks = setup_tcp_network(N, 17700).await;

    let values = vec![
        Fp::new(13),
        Fp::new(17),
        Fp::new(19),
        Fp::new(23),
        Fp::new(29),
    ];
    let expected: Fp = values.iter().copied().reduce(|a, b| a * b).unwrap();
    let expected_plain = 13u64 * 17 * 19 * 23 * 29;
    assert_eq!(expected, Fp::new(expected_plain));
    eprintln!("Computing: 13 * 17 * 19 * 23 * 29 = {}", expected_plain);

    let mut handles = Vec::new();
    for (party_id, net) in networks.into_iter().enumerate() {
        let vals = values.clone();
        handles.push(tokio::spawn(run_offline_online_multiply(
            net, party_id, vals, 128,
        )));
    }

    let mut results = Vec::new();
    for handle in handles {
        results.push(handle.await.unwrap());
    }

    let result = results[0].expect("Party 0 should produce result");
    assert_eq!(result, expected);
    eprintln!(
        "=== PASSED: offline/online separation test = {} ===",
        result
    );
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
    party_shares[1][0].share_2t.value += Fp::new(42);

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

    dn.verify_king_broadcast(&masked_shares, honest_opened)
        .unwrap();
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
    let values: Vec<Fp> = (0..num_values)
        .map(|i| Fp::new((i % 7 + 2) as u64))
        .collect();
    let expected: Fp = values.iter().copied().reduce(|a, b| a * b).unwrap();

    eprintln!(
        "=== Mass offline multiply+reveal: {} values ===",
        num_values
    );

    let networks = setup_tcp_network(N, 17800).await;

    let mut handles = Vec::new();
    for (party_id, net) in networks.into_iter().enumerate() {
        let vals = values.clone();
        handles.push(tokio::spawn(run_offline_online_multiply(
            net, party_id, vals, 128,
        )));
    }

    let mut results = Vec::new();
    for handle in handles {
        results.push(handle.await.unwrap());
    }

    let result = results[0].expect("Party 0 should produce result");
    assert_eq!(
        result, expected,
        "100-value multiply failed: {} != {}",
        result, expected
    );
    eprintln!(
        "=== PASSED: 100-value chained offline multiply+reveal = {} ===",
        result
    );
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
                for (st, s2t) in received.values() {
                    val_t.value += st.value;
                    val_2t.value += s2t.value;
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
                    let (_, share): (usize, Share) = bnet.recv(2, v_idx as u32).await.unwrap();
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
                    let (_, val): (usize, Fp) = bnet.recv(phase, round + 1).await.unwrap();
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
    let expected_intermediates = [Fp::new(6), Fp::new(30), Fp::new(210), Fp::new(2310)];

    for (i, (got, exp)) in intermediates
        .iter()
        .zip(expected_intermediates.iter())
        .enumerate()
    {
        assert_eq!(
            *got, *exp,
            "intermediate result {} mismatch: {} != {}",
            i, got, exp
        );
        eprintln!("  Step {}: product = {} (expected {})", i + 1, got, exp);
    }
    eprintln!("=== PASSED: all intermediate reveals correct ===");
}

#[tokio::test]
async fn test_mass_chained_offline_multiply_200() {
    let num_values = 200;
    let values: Vec<Fp> = (0..num_values)
        .map(|i| Fp::new((i % 7 + 2) as u64))
        .collect();
    let expected: Fp = values.iter().copied().reduce(|a, b| a * b).unwrap();

    eprintln!(
        "=== Mass offline multiply+reveal: {} values ===",
        num_values
    );

    let networks = setup_tcp_network(N, 18000).await;
    let mut handles = Vec::new();
    for (party_id, net) in networks.into_iter().enumerate() {
        let vals = values.clone();
        handles.push(tokio::spawn(run_offline_online_multiply(
            net, party_id, vals, 0,
        )));
    }

    let mut results = Vec::new();
    for handle in handles {
        results.push(handle.await.unwrap());
    }

    let result = results[0].expect("Party 0 should produce result");
    assert_eq!(
        result, expected,
        "200-value multiply: {} != {}",
        result, expected
    );
    eprintln!("=== PASSED: 200-value chained offline multiply+reveal ===");
}

#[tokio::test]
async fn test_mass_chained_offline_multiply_500() {
    let num_values = 500;
    let values: Vec<Fp> = (0..num_values)
        .map(|i| Fp::new((i % 11 + 2) as u64))
        .collect();
    let expected: Fp = values.iter().copied().reduce(|a, b| a * b).unwrap();

    eprintln!(
        "=== Mass offline multiply+reveal: {} values ===",
        num_values
    );

    let networks = setup_tcp_network(N, 18100).await;
    let mut handles = Vec::new();
    for (party_id, net) in networks.into_iter().enumerate() {
        let vals = values.clone();
        handles.push(tokio::spawn(run_offline_online_multiply(
            net, party_id, vals, 0,
        )));
    }

    let mut results = Vec::new();
    for handle in handles {
        results.push(handle.await.unwrap());
    }

    let result = results[0].expect("Party 0 should produce result");
    assert_eq!(
        result, expected,
        "500-value multiply: {} != {}",
        result, expected
    );
    eprintln!("=== PASSED: 500-value chained offline multiply+reveal ===");
}

#[tokio::test]
async fn test_mass_chained_offline_multiply_1000() {
    let num_values = 1000;
    let values: Vec<Fp> = (0..num_values)
        .map(|i| Fp::new((i % 13 + 2) as u64))
        .collect();
    let expected: Fp = values.iter().copied().reduce(|a, b| a * b).unwrap();

    eprintln!(
        "=== Mass offline multiply+reveal: {} values ===",
        num_values
    );

    let networks = setup_tcp_network(N, 18200).await;
    let mut handles = Vec::new();
    for (party_id, net) in networks.into_iter().enumerate() {
        let vals = values.clone();
        handles.push(tokio::spawn(run_offline_online_multiply(
            net, party_id, vals, 0,
        )));
    }

    let mut results = Vec::new();
    for handle in handles {
        results.push(handle.await.unwrap());
    }

    let result = results[0].expect("Party 0 should produce result");
    assert_eq!(
        result, expected,
        "1000-value multiply: {} != {}",
        result, expected
    );
    eprintln!("=== PASSED: 1000-value chained offline multiply+reveal ===");
}

#[tokio::test]
async fn test_him_check_row_verification_networked() {
    eprintln!("=== Testing HIM check row verification over network ===");

    let num_values = 10;
    let values: Vec<Fp> = (1..=num_values as u64).map(Fp::new).collect();
    let expected: Fp = values.iter().copied().reduce(|a, b| a * b).unwrap();

    let networks = setup_tcp_network(N, 18300).await;
    let mut handles = Vec::new();
    for (party_id, net) in networks.into_iter().enumerate() {
        let vals = values.clone();
        handles.push(tokio::spawn(run_offline_online_multiply(
            net, party_id, vals, 0,
        )));
    }

    let mut results = Vec::new();
    for handle in handles {
        results.push(handle.await.unwrap());
    }

    let result = results[0].expect("Party 0 should produce result");
    assert_eq!(result, expected, "10!: {} != {}", result, expected);
    eprintln!(
        "=== PASSED: HIM check row verification over network, 10! = {} ===",
        result
    );
}

// ── Distributed Fibonacci with Streaming Triples ───────────────────────
//
// Each party provides a private input. The inputs are summed to produce
// the starting value x = Σ input_p. Then we compute a multiplicative
// Fibonacci sequence: F(0)=1, F(1)=x, F(k) = F(k-1) * F(k-2).
//
// The entire computation is done on secret-shared values:
//   - Addition of shares is free (no communication)
//   - Multiplication uses Beaver triples from the streaming generator
//   - Only the final result is revealed
//
// Phases:
//   0: Silent OT setup (2-round protocol)
//   1: Input sharing (each party shares its input with all others)
//   2: Online multiplications (Beaver multiply with streamed triples)
//   3: Reveal final result to party 0

/// A single party's role in the distributed private Fibonacci computation.
///
/// Fully peer-to-peer: no king, no single point of trust. Every party is
/// symmetric. Each party runs this function independently, communicating
/// with all other parties over TCP.
///
/// Protocol:
///   Phase 0 (offline): Silent OT (2-round) → HIM double share generation
///                       → convert double shares to Beaver triples (p2p)
///   Phase 1 (input):   Each party secret-shares its input to all others
///   Phase 2 (compute): Multiplicative Fibonacci using p2p Beaver multiply
///                       (all-to-all broadcast of d_p, e_p — no king)
///   Phase 3 (reveal):  All parties broadcast final shares, all reconstruct
async fn run_fibonacci_party(
    net: PartyNetwork,
    party_id: usize,
    my_input: Fp,
    fib_steps: usize,
) -> Option<Fp> {
    let start = std::time::Instant::now();
    let mut rng = ChaCha20Rng::seed_from_u64(party_id as u64 + 9000);
    let mut bnet = BufferedNet::new(net);
    let shamir_t = Shamir::new(N, T).unwrap();
    let shamir_2t = Shamir::new(N, 2 * T).unwrap();
    let num_mults = fib_steps.saturating_sub(1);
    let spr = N - 2 * T;
    let num_him_rounds = if num_mults > 0 {
        num_mults.div_ceil(spr)
    } else {
        1
    };
    let him = HyperInvertibleMatrix::new(N);

    eprintln!(
        "[Party {}] starting with private input={}, {} Fibonacci steps",
        party_id, my_input, fib_steps
    );

    // ── Phase 0: Offline — Silent OT (2-round) + Beaver triples ─────
    let offline_start = std::time::Instant::now();

    let ot_params = SilentOtParams::new(N, T, std::cmp::max(num_him_rounds, 16)).unwrap();
    let protocol = DistributedSilentOt::new(ot_params);
    let mut ot_state = protocol.init_party(party_id, &mut rng);

    // OT Round A: each party sends (commitment, puncture_index) to every peer
    for (to, commitment, punct_idx) in DistributedSilentOt::round_a_messages(&ot_state) {
        bnet.send(to, 0, 0, &(commitment, punct_idx)).await.unwrap();
    }
    let ra_recv: HashMap<usize, ([u8; 32], usize)> = bnet.recv_from_all(0, 0).await.unwrap();
    let ra: Vec<_> = ra_recv.into_iter().map(|(f, (c, i))| (f, c, i)).collect();
    DistributedSilentOt::process_round_a(&mut ot_state, &ra).unwrap();

    // OT Round B: each party sends (sibling_path, seed) to every peer
    for (to, path, seed) in DistributedSilentOt::round_b_messages(&ot_state).unwrap() {
        bnet.send(to, 0, 1, &(path, seed)).await.unwrap();
    }
    let rb_recv: HashMap<usize, (Vec<Block>, Block)> = bnet.recv_from_all(0, 1).await.unwrap();
    let rb: Vec<_> = rb_recv.into_iter().map(|(f, (p, s))| (f, p, s)).collect();
    DistributedSilentOt::process_round_b(&mut ot_state, &rb).unwrap();

    // Expand OT correlations locally (no communication)
    let correlations = DistributedSilentOt::expand(&ot_state).unwrap();
    let ot_randoms: Vec<Fp> = (0..num_him_rounds)
        .map(|k| correlations.get_random(k))
        .collect();

    // Generate double shares via distributed HIM mixing (all-to-all per round)
    let mut double_shares: Vec<DoubleShare> = Vec::with_capacity(num_mults);
    for him_round in 0..num_him_rounds {
        let secret = ot_randoms[him_round];
        let st = shamir_t.share(secret, &mut rng);
        let s2t = shamir_2t.share(secret, &mut rng);

        for j in 0..N {
            if j != party_id {
                bnet.send(j, 1, him_round as u32, &(st[j], s2t[j]))
                    .await
                    .unwrap();
            }
        }
        let received: HashMap<usize, (Share, Share)> =
            bnet.recv_from_all(1, him_round as u32).await.unwrap();

        let mut input_t = vec![Fp::ZERO; N];
        let mut input_2t = vec![Fp::ZERO; N];
        input_t[party_id] = st[party_id].value;
        input_2t[party_id] = s2t[party_id].value;
        for (&from, (s, s2)) in &received {
            input_t[from] = s.value;
            input_2t[from] = s2.value;
        }

        let out_t = him.mul_vec(&input_t);
        let out_2t = him.mul_vec(&input_2t);

        for j in 0..spr {
            if double_shares.len() >= num_mults {
                break;
            }
            double_shares.push(DoubleShare {
                share_t: Share {
                    point: shamir_t.eval_points[party_id],
                    value: out_t[j],
                },
                share_2t: Share {
                    point: shamir_2t.eval_points[party_id],
                    value: out_2t[j],
                },
            });
        }
    }

    let my_point = shamir_t.eval_points[party_id];

    eprintln!(
        "[Party {}] offline phase complete: {} double shares in {:.2?}",
        party_id,
        double_shares.len(),
        offline_start.elapsed()
    );

    // ── Phase 1: Each party secret-shares its private input ─────────
    let my_shares = shamir_t.share(my_input, &mut rng);
    for j in 0..N {
        if j != party_id {
            bnet.send(j, 2, 0, &my_shares[j]).await.unwrap();
        }
    }
    let input_received: HashMap<usize, Share> = bnet.recv_from_all(2, 0).await.unwrap();
    let mut all_input_shares = vec![my_shares[party_id]; N];
    for (&from, &share) in &input_received {
        all_input_shares[from] = share;
    }

    // Sum all inputs locally (addition is free — no communication needed)
    let x_share = Share {
        point: shamir_t.eval_points[party_id],
        value: all_input_shares
            .iter()
            .map(|s| s.value)
            .fold(Fp::ZERO, |a, b| a + b),
    };

    eprintln!(
        "[Party {}] received shares from all peers, summed inputs locally",
        party_id
    );

    // ── Phase 2: P2P multiplication (no king) ─────────────────────
    // F(0) = 1, F(1) = x, F(k) = F(k-1) * F(k-2)
    //
    // Each multiplication uses a double share and all-to-all broadcast:
    //   1. Party p computes: masked_p = F_curr_p * F_prev_p + r_2t_p
    //   2. Party p broadcasts masked_p to ALL peers
    //   3. Every party locally reconstructs: opened = Σ lag[p] * masked_p
    //   4. Party p computes: F_next_p = opened - r_t_p
    //
    // Fully symmetric — no king, no single point of trust.

    let one_share = Share {
        point: my_point,
        value: Fp::ONE,
    };

    let lagrange_2t = shamir_2t.lagrange_coefficients();
    let mut f_prev = one_share;
    let mut f_curr = x_share;

    let online_start = std::time::Instant::now();
    for k in 0..num_mults {
        let ds = &double_shares[k];

        // Masked product: degree-2t share of F_curr * F_prev
        let my_masked = f_curr.value * f_prev.value + ds.share_2t.value;

        // Broadcast to ALL peers (no king)
        bnet.broadcast(3, k as u32, &my_masked).await.unwrap();
        let masked_recv: HashMap<usize, Fp> = bnet.recv_from_all(3, k as u32).await.unwrap();

        // Every party reconstructs the opened value locally via Lagrange
        let mut opened = lagrange_2t[party_id] * my_masked;
        for (&from, &val) in &masked_recv {
            opened += lagrange_2t[from] * val;
        }

        // Output share: subtract degree-t component
        let f_next = Share {
            point: my_point,
            value: opened - ds.share_t.value,
        };

        f_prev = f_curr;
        f_curr = f_next;
    }

    eprintln!(
        "[Party {}] computed F({}) via {} p2p multiplications in {:.2?}",
        party_id,
        fib_steps,
        num_mults,
        online_start.elapsed()
    );

    // ── Phase 3: Reveal final result (all-to-all, everyone learns it) ──
    bnet.broadcast(4, 0, &f_curr).await.unwrap();
    let reveal_received: HashMap<usize, Share> = bnet.recv_from_all(4, 0).await.unwrap();
    let mut all_shares = vec![f_curr];
    for (_, s) in reveal_received {
        all_shares.push(s);
    }
    let result = shamir_t.reconstruct(&all_shares).unwrap();

    eprintln!(
        "[Party {}] revealed result: F({}) = {}, total time: {:.2?}",
        party_id,
        fib_steps,
        result,
        start.elapsed()
    );
    Some(result)
}

/// Distributed private Fibonacci computation over TCP peer-to-peer network.
///
/// 5 parties on separate TCP ports, each with a private input:
///   Party 0: 3, Party 1: 5, Party 2: 7, Party 3: 11, Party 4: 13
///
/// The protocol privately computes:
///   x = 3 + 5 + 7 + 11 + 13 = 39  (sum of all inputs)
///   F(0) = 1, F(1) = 39
///   F(k) = F(k-1) * F(k-2)  (multiplicative Fibonacci)
///
/// No party learns any intermediate value. Only the final F(n) is revealed.
#[tokio::test]
async fn test_distributed_fibonacci_streaming() {
    let inputs = vec![Fp::new(3), Fp::new(5), Fp::new(7), Fp::new(11), Fp::new(13)];
    let fib_steps = 8;

    // Compute expected result in plaintext for verification
    let x = inputs.iter().copied().fold(Fp::ZERO, |a, b| a + b);
    let mut fp = Fp::ONE;
    let mut fc = x;
    for _ in 0..fib_steps - 1 {
        let next = fc * fp;
        fp = fc;
        fc = next;
    }
    let expected = fc;

    eprintln!("╔══════════════════════════════════════════════════════╗");
    eprintln!("║  Distributed Private Fibonacci over TCP P2P Network ║");
    eprintln!("╠══════════════════════════════════════════════════════╣");
    eprintln!(
        "║  Parties: {} (threshold t={})                        ║",
        N, T
    );
    eprintln!(
        "║  Inputs: {:?}            ║",
        inputs.iter().map(|f| f.val()).collect::<Vec<_>>()
    );
    eprintln!("║  Sum: x = {}                                        ║", x);
    eprintln!(
        "║  Computing: F({}) where F(k)=F(k-1)*F(k-2)          ║",
        fib_steps
    );
    eprintln!("║  Expected: F({}) = {}          ║", fib_steps, expected);
    eprintln!("╚══════════════════════════════════════════════════════╝");

    // Create TCP mesh network: 5 parties on ports 18400-18404
    let networks = setup_tcp_network(N, 18400).await;

    // Launch each party as an independent async task (simulates separate machines)
    let mut handles = Vec::new();
    for (party_id, net) in networks.into_iter().enumerate() {
        let input = inputs[party_id];
        handles.push(tokio::spawn(run_fibonacci_party(
            net, party_id, input, fib_steps,
        )));
    }

    // Wait for all parties to complete — every party learns the result
    let mut results = Vec::new();
    for handle in handles {
        results.push(handle.await.unwrap());
    }

    // Every party should have reconstructed the same result
    for (p, result) in results.iter().enumerate() {
        let val = result.expect(&format!("Party {} should have result", p));
        assert_eq!(
            val, expected,
            "Party {} got wrong result: {} expected {}",
            p, val, expected
        );
    }

    eprintln!("╔══════════════════════════════════════════════════════╗");
    eprintln!(
        "║  PASSED: All {} parties agree F({}) = {:>16} ║",
        N, fib_steps, expected
    );
    eprintln!("╚══════════════════════════════════════════════════════╝");
}
