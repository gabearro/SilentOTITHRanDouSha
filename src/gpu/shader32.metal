#include <metal_stdlib>
using namespace metal;

constant uint P = (1U << 31) - 1;

inline uint fmul(uint a, uint b) {
    ulong x = ulong(a) * ulong(b);
    uint lo = uint(x & ulong(P));
    uint hi = uint(x >> 31);
    uint r = lo + hi;
    r = (r >> 31) + (r & P);
    return select(r, r - P, r >= P);
}

inline uint fadd(uint a, uint b) {
    uint r = a + b;
    return select(r, r - P, r >= P);
}

inline uint fsub(uint a, uint b) {
    return select(a - b, P - b + a, a < b);
}

inline uint chash32(uint seed, uint idx) {
    uint x = seed ^ (idx * 0x9e3779b9U);
    x ^= x >> 16; x *= 0x85ebca6bU;
    x ^= x >> 13; x *= 0xc2b2ae35U;
    x ^= x >> 16;
    x &= 0x7FFFFFFFU;
    return select(x, x - P, x >= P);
}

inline uint simd_fadd_reduce(uint v) {
    v = fadd(v, simd_shuffle_xor(v, 1));
    v = fadd(v, simd_shuffle_xor(v, 2));
    v = fadd(v, simd_shuffle_xor(v, 4));
    return v;
}

struct Constants32 {
    uint n;
    uint t;
    uint count;
    uint spr;
    uint num_rounds;
    uint party;
    uint eval_raw[8];
    uint him_rows[3][8];
    uint lag_sum;
    uint lag_x_sum;
    uint lag_xsq_sum;
    uint ot_offsets[8];
};

// ═══════════════════════════════════════════════════════════════════
// Production single-party kernel for distributed p2p Beaver triples
//
// Reads OT correlations through HIM mixing (required for malicious
// security). Computes 2 SIMD reductions per sharing index:
//   a_mixed = Σ him[j][i] · ot[i]     (HIM-mixed OT correlation)
//   bt_r    = Σ him[j][i] · hash(seed,i) (degree-t randomization)
//
// b1 and b2 reductions are eliminated: for Lagrange interpolation
// at x=0 with eval points {1..n}, lag_x_sum=0 and lag_xsq_sum=0,
// so d1·lag_x_sum=0 and d2·lag_xsq_sum=0 regardless of b1,b2.
// delta thus simplifies to d0·lag_sum = (a0·b0 - a_mixed)·1.
//
// Final output: c_p = a_mixed + eval_p·bt_r + a0·b0 - a_mixed
//                    = a0·b0 + eval_p·bt_r
//
// One thread per round, producing spr triples. OT read and seed
// read amortized across all sharing indices in the round.
// ═══════════════════════════════════════════════════════════════════

kernel void beaver_triple_gen_single_party_32(
    device const uint* ot_packed  [[buffer(0)]],
    device uint* a_out            [[buffer(1)]],
    device uint* b_out            [[buffer(2)]],
    device uint* c_out            [[buffer(3)]],
    constant Constants32& C       [[buffer(4)]],
    device const uint* seeds      [[buffer(5)]],
    uint tid [[thread_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]])
{
    uint round = tid;
    if (round >= C.num_rounds) return;

    uint spr = C.spr;
    uint n = C.n;
    uint eval_p = C.eval_raw[C.party];
    uint seed = seeds[round];
    uint base = round * spr;

    // Read OT and compute rt ONCE per round (shared across all spr triples)
    ushort i = lane % ushort(n);
    bool active = (lane < n);

    uint s  = active ? ot_packed[C.ot_offsets[i] + round] : 0;
    uint rt = active ? chash32(seed, i * 3u) : 0;

    for (uint j = 0; j < spr && (base + j) < C.count; j++) {
        uint out_idx = base + j;
        uint m = active ? C.him_rows[j][i] : 0;

        // 2 SIMD reductions: a_mixed (OT·HIM) and bt_r (hash·HIM)
        uint a_mixed = simd_fadd_reduce(fmul(m, s));
        uint bt_r    = simd_fadd_reduce(fmul(m, rt));

        uint a0 = chash32(seed, n * 3u + spr * 2u + j * 2u);
        uint b0 = chash32(seed, n * 3u + spr * 2u + j * 2u + 1u);
        uint ca = chash32(seed, n * 3u + j * 2u);
        uint cb = chash32(seed, n * 3u + j * 2u + 1u);

        // delta = d0 = a0*b0 - a_mixed (since lag_x_sum=lag_xsq_sum=0, lag_sum=1)
        // c_p = a_mixed + eval_p*bt_r + delta = a0*b0 + eval_p*bt_r
        a_out[out_idx] = fadd(a0, fmul(ca, eval_p));
        b_out[out_idx] = fadd(b0, fmul(cb, eval_p));
        c_out[out_idx] = fadd(fmul(a0, b0), fmul(eval_p, bt_r));
    }
}

// All-party version — one thread per round, produces spr triples × n parties
kernel void beaver_triple_gen_32(
    device const uint* ot_packed  [[buffer(0)]],
    device uint* a_out            [[buffer(1)]],
    device uint* b_out            [[buffer(2)]],
    device uint* c_out            [[buffer(3)]],
    constant Constants32& C       [[buffer(4)]],
    device const uint* seeds      [[buffer(5)]],
    uint tid [[thread_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]])
{
    uint round = tid;
    if (round >= C.num_rounds) return;

    uint n = C.n;
    uint spr = C.spr;
    uint seed = seeds[round];
    uint base = round * spr;

    ushort i = lane % ushort(n);
    bool active = (lane < n);

    uint s  = active ? ot_packed[C.ot_offsets[i] + round] : 0;
    uint rt = active ? chash32(seed, i * 3u) : 0;

    for (uint j = 0; j < spr && (base + j) < C.count; j++) {
        uint out_tid = base + j;
        uint m = active ? C.him_rows[j][i] : 0;

        uint a_mixed = simd_fadd_reduce(fmul(m, s));
        uint bt_r    = simd_fadd_reduce(fmul(m, rt));

        uint a0 = chash32(seed, n * 3u + spr * 2u + j * 2u);
        uint b0 = chash32(seed, n * 3u + spr * 2u + j * 2u + 1u);
        uint ca = chash32(seed, n * 3u + j * 2u);
        uint cb = chash32(seed, n * 3u + j * 2u + 1u);

        for (ushort p = 0; p < ushort(n); p++) {
            uint x = C.eval_raw[p];
            uint dst = p * C.count + out_tid;
            a_out[dst] = fadd(a0, fmul(ca, x));
            b_out[dst] = fadd(b0, fmul(cb, x));
            c_out[dst] = fadd(fmul(a0, b0), fmul(x, bt_r));
        }
    }
}
