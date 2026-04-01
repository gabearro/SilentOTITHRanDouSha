#include <metal_stdlib>
using namespace metal;

// Mersenne prime: p = 2^61 - 1
constant ulong PRIME = (1UL << 61) - 1;

// ── Branchless Mersenne field arithmetic ────────────────────────────

inline ulong fp_reduce(ulong x) {
    ulong r = (x >> 61) + (x & PRIME);
    return select(r, r - PRIME, r >= PRIME);
}

inline ulong fp_add(ulong a, ulong b) {
    ulong r = a + b;
    return select(r, r - PRIME, r >= PRIME);
}

inline ulong fp_sub(ulong a, ulong b) {
    return select(a - b, PRIME - b + a, a < b);
}

// 64×64→128 multiply mod p using native mulhi()
inline ulong fp_mul(ulong a, ulong b) {
    ulong lo = a * b;
    ulong hi = mulhi(a, b);
    ulong lo_masked = lo & PRIME;
    ulong mid = ((hi << 3) | (lo >> 61)) & PRIME;
    ulong hi_part = hi >> 58;
    ulong r = lo_masked + mid + hi_part;
    ulong r2 = (r >> 61) + (r & PRIME);
    return select(r2, r2 - PRIME, r2 >= PRIME);
}

// ── Manual 128-bit accumulator (since __uint128_t fails at runtime) ──

struct U128 {
    ulong lo;
    ulong hi;

    // Add a * b to the accumulator with carry propagation
    inline void add_mul(ulong a, ulong b) {
        ulong prod_lo = a * b;
        ulong prod_hi = mulhi(a, b);
        ulong old_lo = lo;
        lo += prod_lo;
        hi += prod_hi + select(0UL, 1UL, lo < old_lo);
    }

    // Add a plain u64 value
    inline void add_u64(ulong v) {
        ulong old_lo = lo;
        lo += v;
        hi += select(0UL, 1UL, lo < old_lo);
    }

    // Reduce (hi:lo) mod PRIME
    inline ulong reduce() {
        ulong lo_masked = lo & PRIME;
        ulong mid = ((hi << 3) | (lo >> 61)) & PRIME;
        ulong hi_part = hi >> 58;
        ulong r = lo_masked + mid + hi_part;
        ulong r2 = (r >> 61) + (r & PRIME);
        return select(r2, r2 - PRIME, r2 >= PRIME);
    }
};

// ── SplitMix64 PRNG (matches Rust implementation exactly) ───────────

struct SplitMix64 {
    ulong state;

    inline ulong next_raw61() {
        state += 0x9e3779b97f4a7c15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9UL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebUL;
        return (z ^ (z >> 31)) >> 3;
    }
};

// ── Precomputed constants passed from CPU ───────────────────────────

struct Constants {
    uint n;
    uint t;
    uint count;
    uint spr;           // sharings_per_round = n - 2*t
    uint num_rounds;
    uint party;         // for single-party kernel
    ulong eval_raw[8];  // evaluation points (padded to 8)
    ulong him_rows[3][8]; // HIM Vandermonde rows (padded)
    ulong lag_sum;
    ulong lag_x_sum;
    ulong lag_xsq_sum;
};

// ── All-party kernel: one thread per round, writes n triples ────────

kernel void beaver_triple_gen(
    device const ulong* ot_secrets [[buffer(0)]],  // interleaved [round*n + party]
    device const ulong* fresh_buf  [[buffer(1)]],  // AES-CTR random: 2*spr per round
    device ulong* a_out            [[buffer(2)]],  // party-major: a_out[p*count + k]
    device ulong* b_out            [[buffer(3)]],
    device ulong* c_out            [[buffer(4)]],
    constant Constants& C          [[buffer(5)]],
    device const ulong* round_seeds [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    uint round = tid;
    if (round >= C.num_rounds) return;

    uint n = C.n;
    uint spr = C.spr;
    uint count = C.count;

    // Load OT secrets for this round (coalesced: consecutive threads read consecutive memory)
    ulong secrets[8];
    uint ot_base = round * n;
    for (uint i = 0; i < n; i++) {
        secrets[i] = ot_secrets[ot_base + i];
    }

    // SplitMix64 for HIM coefficients
    SplitMix64 mix = { round_seeds[round] };
    ulong r_t[8], r1_2t[8], r2_2t[8];
    for (uint i = 0; i < n; i++) {
        r_t[i] = mix.next_raw61();
        r1_2t[i] = mix.next_raw61();
        r2_2t[i] = mix.next_raw61();
    }

    uint fresh_base = round * 2 * spr;

    for (uint j = 0; j < spr; j++) {
        uint triple_idx = round * spr + j;
        if (triple_idx >= count) return;

        // HIM dot products: 4 accumulators
        U128 am = {0, 0}, bt_acc = {0, 0}, b1_acc = {0, 0}, b2_acc = {0, 0};

        if (j == 0) {
            // Row 0 = [1,1,...,1]: just sum (skip multiplies)
            for (uint i = 0; i < n; i++) {
                am.add_u64(secrets[i]);
                bt_acc.add_u64(r_t[i]);
                b1_acc.add_u64(r1_2t[i]);
                b2_acc.add_u64(r2_2t[i]);
            }
        } else {
            for (uint i = 0; i < n; i++) {
                ulong m = C.him_rows[j][i];
                am.add_mul(m, secrets[i]);
                bt_acc.add_mul(m, r_t[i]);
                b1_acc.add_mul(m, r1_2t[i]);
                b2_acc.add_mul(m, r2_2t[i]);
            }
        }

        ulong a_mixed = am.reduce();
        ulong bt = bt_acc.reduce();
        ulong b1 = b1_acc.reduce();
        ulong b2 = b2_acc.reduce();

        // Fresh secrets from AES-CTR buffer
        ulong a0 = fresh_buf[fresh_base + j * 2];
        ulong b0 = fresh_buf[fresh_base + j * 2 + 1];

        // Polynomial coefficients from SplitMix
        ulong ca = mix.next_raw61();
        ulong cb = mix.next_raw61();

        // Analytical delta for t=1
        ulong d0 = fp_sub(fp_mul(a0, b0), a_mixed);
        ulong d1 = fp_sub(fp_add(fp_mul(a0, cb), fp_mul(ca, b0)), b1);
        ulong d2 = fp_sub(fp_mul(ca, cb), b2);

        U128 delta_acc = {0, 0};
        delta_acc.add_mul(d0, C.lag_sum);
        delta_acc.add_mul(d1, C.lag_x_sum);
        delta_acc.add_mul(d2, C.lag_xsq_sum);
        ulong delta = delta_acc.reduce();

        // Evaluate polynomials at each party's point and write output
        for (uint p = 0; p < n; p++) {
            ulong x = C.eval_raw[p];
            ulong a_val = fp_add(a0, fp_mul(ca, x));
            ulong b_val = fp_add(b0, fp_mul(cb, x));
            ulong c_val = fp_add(fp_add(a_mixed, fp_mul(x, bt)), delta);

            uint dst = p * count + triple_idx;
            a_out[dst] = a_val;
            b_out[dst] = b_val;
            c_out[dst] = c_val;
        }
    }
}

// ── Single-party kernel: only evaluates at one point ────────────────

kernel void beaver_triple_gen_single_party(
    device const ulong* ot_secrets [[buffer(0)]],
    device const ulong* fresh_buf  [[buffer(1)]],
    device ulong* a_out            [[buffer(2)]],  // linear: a_out[k]
    device ulong* b_out            [[buffer(3)]],
    device ulong* c_out            [[buffer(4)]],
    constant Constants& C          [[buffer(5)]],
    device const ulong* round_seeds [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    uint round = tid;
    if (round >= C.num_rounds) return;

    uint n = C.n;
    uint spr = C.spr;
    uint count = C.count;
    ulong eval_p = C.eval_raw[C.party];

    // Load OT secrets
    ulong secrets[8];
    uint ot_base = round * n;
    for (uint i = 0; i < n; i++) {
        secrets[i] = ot_secrets[ot_base + i];
    }

    // SplitMix64 for HIM coefficients
    SplitMix64 mix = { round_seeds[round] };
    ulong r_t[8], r1_2t[8], r2_2t[8];
    for (uint i = 0; i < n; i++) {
        r_t[i] = mix.next_raw61();
        r1_2t[i] = mix.next_raw61();
        r2_2t[i] = mix.next_raw61();
    }

    uint fresh_base = round * 2 * spr;

    for (uint j = 0; j < spr; j++) {
        uint triple_idx = round * spr + j;
        if (triple_idx >= count) return;

        // HIM dot products (same as all-party kernel)
        U128 am = {0, 0}, bt_acc = {0, 0}, b1_acc = {0, 0}, b2_acc = {0, 0};

        if (j == 0) {
            for (uint i = 0; i < n; i++) {
                am.add_u64(secrets[i]);
                bt_acc.add_u64(r_t[i]);
                b1_acc.add_u64(r1_2t[i]);
                b2_acc.add_u64(r2_2t[i]);
            }
        } else {
            for (uint i = 0; i < n; i++) {
                ulong m = C.him_rows[j][i];
                am.add_mul(m, secrets[i]);
                bt_acc.add_mul(m, r_t[i]);
                b1_acc.add_mul(m, r1_2t[i]);
                b2_acc.add_mul(m, r2_2t[i]);
            }
        }

        ulong a_mixed = am.reduce();
        ulong bt = bt_acc.reduce();
        ulong b1 = b1_acc.reduce();
        ulong b2 = b2_acc.reduce();

        ulong a0 = fresh_buf[fresh_base + j * 2];
        ulong b0 = fresh_buf[fresh_base + j * 2 + 1];
        ulong ca = mix.next_raw61();
        ulong cb = mix.next_raw61();

        // Delta
        ulong d0 = fp_sub(fp_mul(a0, b0), a_mixed);
        ulong d1 = fp_sub(fp_add(fp_mul(a0, cb), fp_mul(ca, b0)), b1);
        ulong d2 = fp_sub(fp_mul(ca, cb), b2);

        U128 delta_acc = {0, 0};
        delta_acc.add_mul(d0, C.lag_sum);
        delta_acc.add_mul(d1, C.lag_x_sum);
        delta_acc.add_mul(d2, C.lag_xsq_sum);
        ulong delta = delta_acc.reduce();

        // Single party evaluation only
        ulong a_val = fp_add(a0, fp_mul(ca, eval_p));
        ulong b_val = fp_add(b0, fp_mul(cb, eval_p));
        ulong c_val = fp_add(fp_add(a_mixed, fp_mul(eval_p, bt)), delta);

        a_out[triple_idx] = a_val;
        b_out[triple_idx] = b_val;
        c_out[triple_idx] = c_val;
    }
}
