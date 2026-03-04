use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;

use silent_ot_randousha::field::Fp;
use silent_ot_randousha::multiply::multiply_sequence_party_indexed;
use silent_ot_randousha::randousha::{DoubleShare, HyperInvertibleMatrix, RanDouShaParams, RanDouShaProtocol};
use silent_ot_randousha::shamir::{Shamir, Share};
use silent_ot_randousha::silent_ot::{
    batch_to_field_elements, prg_expand, Block, DistributedSilentOt, GgmTree, SilentOtParams,
};

fn bench_ggm_tree_expand(c: &mut Criterion) {
    let mut group = c.benchmark_group("ggm_tree_expand");
    group.sample_size(10);

    for depth in [10, 15, 20] {
        let tree = GgmTree::new(depth);
        let seed = Block::random(&mut ChaCha20Rng::seed_from_u64(42));
        group.bench_with_input(BenchmarkId::new("expand_full", depth), &depth, |b, _| {
            b.iter(|| black_box(tree.expand_full(&seed)))
        });
    }

    group.finish();
}

fn bench_ggm_tree_reconstruct(c: &mut Criterion) {
    let mut group = c.benchmark_group("ggm_tree_reconstruct");
    group.sample_size(10);

    for depth in [10, 15, 20] {
        let tree = GgmTree::new(depth);
        let seed = Block::random(&mut ChaCha20Rng::seed_from_u64(42));
        let puncture_idx = 42;
        let sibling_path = tree.compute_sibling_path(&seed, puncture_idx).unwrap();

        group.bench_with_input(
            BenchmarkId::new("reconstruct_from_siblings", depth),
            &depth,
            |b, _| {
                b.iter(|| black_box(tree.reconstruct_from_siblings(&sibling_path, puncture_idx)))
            },
        );
    }

    group.finish();
}

fn bench_batch_to_field_elements(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_to_field_elements");
    group.sample_size(10);

    for count in [1024, 65536, 666_667] {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let blocks: Vec<Block> = (0..count).map(|_| Block::random(&mut rng)).collect();
        group.bench_with_input(
            BenchmarkId::new("convert", count),
            &count,
            |b, &count| b.iter(|| black_box(batch_to_field_elements(&blocks, count))),
        );
    }

    group.finish();
}

fn bench_shamir_share(c: &mut Criterion) {
    let mut group = c.benchmark_group("shamir_share");

    let shamir_t = Shamir::new(5, 1).unwrap();
    let shamir_2t = Shamir::new(5, 2).unwrap();

    group.bench_function("degree_1_n5", |b| {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        b.iter(|| black_box(shamir_t.share(Fp::new(42), &mut rng)))
    });

    group.bench_function("degree_2_n5", |b| {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        b.iter(|| black_box(shamir_2t.share(Fp::new(42), &mut rng)))
    });

    group.finish();
}

fn bench_him_multiply(c: &mut Criterion) {
    let mut group = c.benchmark_group("him_multiply");
    let him = HyperInvertibleMatrix::new(5);
    let v: Vec<Fp> = (1..=5u64).map(Fp::new).collect();

    group.bench_function("n5", |b| {
        b.iter(|| black_box(him.mul_vec(&v)))
    });

    group.finish();
}

fn bench_silent_ot_full(c: &mut Criterion) {
    let mut group = c.benchmark_group("silent_ot_full");
    group.sample_size(10);

    let n = 5;
    let t = 1;

    for num_ots in [1024, 16384] {
        group.bench_with_input(
            BenchmarkId::new("ot_protocol", num_ots),
            &num_ots,
            |b, &num_ots| {
                b.iter(|| {
                    let mut rng = ChaCha20Rng::seed_from_u64(42);
                    let ot_params = SilentOtParams::new(n, t, num_ots).unwrap();
                    let protocol = DistributedSilentOt::new(ot_params);
                    let mut states: Vec<_> =
                        (0..n).map(|i| protocol.init_party(i, &mut rng)).collect();

                    let mut r0 = vec![Vec::new(); n];
                    for s in states.iter() {
                        for (to, c) in DistributedSilentOt::round0_commitments(s) {
                            r0[to].push((s.party_id, c));
                        }
                    }
                    for (i, s) in states.iter_mut().enumerate() {
                        DistributedSilentOt::process_round0(s, &r0[i]).unwrap();
                    }

                    let mut r1 = vec![Vec::new(); n];
                    for s in states.iter() {
                        for (to, idx) in DistributedSilentOt::round1_puncture_choices(s) {
                            r1[to].push((s.party_id, idx));
                        }
                    }
                    for (i, s) in states.iter_mut().enumerate() {
                        DistributedSilentOt::process_round1(s, &r1[i]).unwrap();
                    }

                    let mut r2 = vec![Vec::new(); n];
                    for s in states.iter() {
                        for (to, path) in DistributedSilentOt::round2_sibling_paths(s).unwrap() {
                            r2[to].push((s.party_id, path));
                        }
                    }
                    for (i, s) in states.iter_mut().enumerate() {
                        DistributedSilentOt::process_round2(s, &r2[i]).unwrap();
                    }

                    let mut r3 = vec![Vec::new(); n];
                    for s in states.iter() {
                        for (to, seed) in DistributedSilentOt::round3_seed_reveals(s) {
                            r3[to].push((s.party_id, seed));
                        }
                    }
                    for (i, s) in states.iter_mut().enumerate() {
                        DistributedSilentOt::process_round3(s, &r3[i]).unwrap();
                    }

                    let _correlations: Vec<_> = states
                        .par_iter()
                        .map(|s| DistributedSilentOt::expand(s).unwrap())
                        .collect();
                    black_box(())
                })
            },
        );
    }

    group.finish();
}

fn bench_silent_ot_expand_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("silent_ot_expand_only");
    group.sample_size(10);

    let n = 5;
    let t = 1;

    for num_ots in [1024, 16384, 666_667] {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let ot_params = SilentOtParams::new(n, t, num_ots).unwrap();
        let protocol = DistributedSilentOt::new(ot_params);
        let mut states: Vec<_> = (0..n).map(|i| protocol.init_party(i, &mut rng)).collect();

        // Run all rounds
        let mut r0 = vec![Vec::new(); n];
        for s in states.iter() {
            for (to, c) in DistributedSilentOt::round0_commitments(s) {
                r0[to].push((s.party_id, c));
            }
        }
        for (i, s) in states.iter_mut().enumerate() {
            DistributedSilentOt::process_round0(s, &r0[i]).unwrap();
        }
        let mut r1 = vec![Vec::new(); n];
        for s in states.iter() {
            for (to, idx) in DistributedSilentOt::round1_puncture_choices(s) {
                r1[to].push((s.party_id, idx));
            }
        }
        for (i, s) in states.iter_mut().enumerate() {
            DistributedSilentOt::process_round1(s, &r1[i]).unwrap();
        }
        let mut r2 = vec![Vec::new(); n];
        for s in states.iter() {
            for (to, path) in DistributedSilentOt::round2_sibling_paths(s).unwrap() {
                r2[to].push((s.party_id, path));
            }
        }
        for (i, s) in states.iter_mut().enumerate() {
            DistributedSilentOt::process_round2(s, &r2[i]).unwrap();
        }
        let mut r3 = vec![Vec::new(); n];
        for s in states.iter() {
            for (to, seed) in DistributedSilentOt::round3_seed_reveals(s) {
                r3[to].push((s.party_id, seed));
            }
        }
        for (i, s) in states.iter_mut().enumerate() {
            DistributedSilentOt::process_round3(s, &r3[i]).unwrap();
        }

        group.bench_with_input(
            BenchmarkId::new("expand_all_parties", num_ots),
            &num_ots,
            |b, _| {
                b.iter(|| {
                    let _: Vec<_> = states
                        .par_iter()
                        .map(|s| DistributedSilentOt::expand(s).unwrap())
                        .collect();
                    black_box(())
                })
            },
        );
    }

    group.finish();
}

fn bench_him_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("him_generation");
    group.sample_size(10);

    let n = 5;
    let t = 1;
    let shamir_t = Shamir::new(n, t).unwrap();
    let shamir_2t = Shamir::new(n, 2 * t).unwrap();
    let him = HyperInvertibleMatrix::new(n);
    let sharings_per_round = n - 2 * t;

    let him_rows: Vec<Vec<Fp>> = (0..n)
        .map(|j| (0..n).map(|col| him.get(j, col)).collect())
        .collect();
    let eval_points_t: Vec<Fp> = shamir_t.eval_points.clone();
    let eval_points_2t: Vec<Fp> = shamir_2t.eval_points.clone();

    for num_rounds in [1000, 10000] {
        let random_values: Vec<Vec<Fp>> = {
            let mut rng = ChaCha20Rng::seed_from_u64(42);
            (0..n)
                .map(|_| (0..num_rounds).map(|_| Fp::random(&mut rng)).collect())
                .collect()
        };

        group.bench_with_input(
            BenchmarkId::new("parallel_rounds", num_rounds),
            &num_rounds,
            |b, &num_rounds| {
                b.iter(|| {
                    let round_seeds: Vec<u64> = (0..num_rounds as u64).collect();
                    let _results: Vec<Vec<(Share, Share)>> = (0..num_rounds)
                        .into_par_iter()
                        .map(|round| {
                            let mut local_rng = ChaCha20Rng::seed_from_u64(round_seeds[round]);
                            let secrets: Vec<Fp> =
                                (0..n).map(|i| random_values[i][round]).collect();

                            let mut all_shares_t: Vec<Vec<Share>> = Vec::with_capacity(n);
                            let mut all_shares_2t: Vec<Vec<Share>> = Vec::with_capacity(n);
                            for i in 0..n {
                                all_shares_t.push(shamir_t.share(secrets[i], &mut local_rng));
                                all_shares_2t.push(shamir_2t.share(secrets[i], &mut local_rng));
                            }

                            let mut out = Vec::with_capacity(sharings_per_round * n);
                            for j in 0..sharings_per_round {
                                for p in 0..n {
                                    let val_t: Fp = (0..n)
                                        .map(|i| him_rows[j][i] * all_shares_t[i][p].value)
                                        .sum();
                                    let val_2t: Fp = (0..n)
                                        .map(|i| him_rows[j][i] * all_shares_2t[i][p].value)
                                        .sum();
                                    out.push((
                                        Share {
                                            point: eval_points_t[p],
                                            value: val_t,
                                        },
                                        Share {
                                            point: eval_points_2t[p],
                                            value: val_2t,
                                        },
                                    ));
                                }
                            }
                            out
                        })
                        .collect();
                    black_box(())
                })
            },
        );
    }

    group.finish();
}

fn bench_multiply_sequence(c: &mut Criterion) {
    let mut group = c.benchmark_group("multiply_sequence");
    group.sample_size(10);

    let n = 5;
    let t = 1;
    let shamir_t = Shamir::new(n, t).unwrap();

    for num_mults in [1000, 10000] {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let num_values = num_mults + 1;

        let values: Vec<Fp> = (0..num_values).map(|i| Fp::new((i % 7 + 2) as u64)).collect();
        let value_shares: Vec<Vec<Share>> =
            values.iter().map(|v| shamir_t.share(*v, &mut rng)).collect();

        let params = RanDouShaParams::new(n, t, num_mults).unwrap();
        let protocol = RanDouShaProtocol::new(params);
        let party_shares = protocol.generate_local(&mut rng).unwrap();

        group.bench_with_input(
            BenchmarkId::new("party_indexed", num_mults),
            &num_mults,
            |b, _| {
                b.iter(|| {
                    black_box(
                        multiply_sequence_party_indexed(n, t, &value_shares, &party_shares)
                            .unwrap(),
                    )
                })
            },
        );
    }

    group.finish();
}

fn bench_prg_expand(c: &mut Criterion) {
    let seed = Block::random(&mut ChaCha20Rng::seed_from_u64(42));
    c.bench_function("prg_expand", |b| {
        b.iter(|| black_box(prg_expand(&seed)))
    });
}

fn bench_fp_arithmetic(c: &mut Criterion) {
    let mut group = c.benchmark_group("fp_arithmetic");
    let a = Fp::new(123456789);
    let b = Fp::new(987654321);

    group.bench_function("mul", |b_| {
        b_.iter(|| black_box(a * b))
    });
    group.bench_function("add", |b_| {
        b_.iter(|| black_box(a + b))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_fp_arithmetic,
    bench_prg_expand,
    bench_ggm_tree_expand,
    bench_ggm_tree_reconstruct,
    bench_batch_to_field_elements,
    bench_shamir_share,
    bench_him_multiply,
    bench_him_generation,
    bench_multiply_sequence,
    bench_silent_ot_expand_only,
    bench_silent_ot_full,
);
criterion_main!(benches);
