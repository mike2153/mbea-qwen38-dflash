// T03-M03 drafter attention, replacement "A" (bit-exact). gfx1201 (RDNA4, wave32).
//
// Replaces vLLM's Triton `kernel_unified_attention` as the DFlash2 drafter calls it (2D path, max_seqlen_q = 8 > 1):
// BLOCK_M 16 = BLOCK_Q 4 queries x 4 q heads per KV head, TILE_SIZE 32 (prefill tile, fp8), non-causal, sliding
// window W (2048), fp8 e4m3fn per-tensor KV (KV_QUANT_MODE 1), bf16 q/out, paged cache NHD [blocks, BS, kv, 256].
// Same grid as Triton: (num_tokens / 4 + num_seqs, num_kv_heads) x 128 threads; each program owns one q-block
// (16 rows) of one KV head and walks the same tiles in the same order, so every row's arithmetic is unchanged.
// What changes is only the latency structure: the window's block-table entries are read once into LDS, raw fp8
// K/V tiles are prefetched DA_DEPTH tiles ahead (Triton: 1), and fp8 is decoded with v_cvt_pk_f32_fp8 instead of
// Triton's ~1300-instruction software select chain per tile (the hot-cache cost, 1.8 us/tile).
//
// Arithmetic order was pinned from the production compile (Triton 3.6.0, cache entry FUV6DEJQ... in
// /root/dflash-accel/cache-atlas/triton; TTGIR #mma = amd_wmma v2 isTranspose warpsPerCTA [4,1], acc #mma1 [1,4]).
// Line refs "U:n" = vllm/v1/attention/ops/triton_unified_attention.py, "H:n" = triton_attention_helpers.py,
// "%n" = values in that entry's .llir, "isa:n" = lines in its .amdgcn.
//   1. K, V: fp8 -> f32 (exact) * scale (f32 mul) -> bf16 RNE, NaN -> 0x7FFF (U:54, %1460.., %1490-%1499).
//   2. S = WMMA chain over head-dim chunks c = 0..7 ascending from C = 0, A = K rows (keys), B = Q rows, i.e. the
//      transposed form (%3136-%3159: wmma(Kfrag, Qfrag, acc)); lane l holds S[m = l&15][key = 16t + 8(l>>4) + i].
//   3. S = fma(scale, dot, 0)  (U:542 `S += scale*dot` -> v_fma_f32 s82, d, 0 at isa:5019).
//   4. masks -> -inf (U:547, H:307-337, non-causal + window both ways).
//   5. row max: maxnum over the lane's 16 values, then with the other half-wave (permlanex16) (H:428, %3232-%3250).
//   6. m = maxnum(M, rowmax); m = (m > -inf) ? m : 0 (H:431). M starts at -inf, L at 1.0 (H:129, U:380).
//   7. exp(x): LLVM's ieee-denormal exp2 expansion with the +64 offset fused by FMA (isa:5400-5530):
//      c = x*log2e < -126; y = fma(x, log2e, c ? 64 : 0); r = ldexp(v_exp_f32(y), c ? -64 : 0).
//   8. l = linear sum of the lane's 16 P values (p0+p1, +p2, ... +p15), then + other half (%3302-%3320).
//   9. alpha = exp(M - m); L = fma(L, alpha, l)  (H:437-439 contracted: isa:5774 v_fmac_f32 v170, v137, v85).
//  10. acc = acc * alpha (plain mul), then acc = WMMA(Vt chunk1, P1, WMMA(Vt chunk0, P0, acc)) with
//      A = V^T rows (head dim), B = P rows (keys 16c + 8(l>>4) + i, bf16 RNE) (U:562/584, %3564-%3583).
//  11. V rows outside the q-block's window (dist = ctx + qpos_lo - key, |dist| < W) or >= seq_len are +0 (U:564-578).
//  12. out = bf16_rne(acc / L) with IEEE division (U:644, isa:10047 v_div_scale/fmas/fixup).
// Uncertain until the GPU test: (a) v_cvt_pk_f32_fp8 decoding OCP e4m3fn exactly like Triton's table (build with
// -DDA_SW_FP8 for a bit-level decode identical to Triton's); (b) WMMA results depend only on the operand
// registers (we feed the exact same registers, so this should hold); (c) hipcc keeping the explicit op order
// (no reassociation without fast-math; contraction disabled below, FMAs written explicitly).
#include <hip/hip_runtime.h>
#include <stdint.h>

#pragma clang fp contract(off)

#ifndef DA_DEPTH
#define DA_DEPTH 2  // L-071: 8.27% warm / 8.06% large-working-set gain over depth3.
#endif

#ifndef DA_PARTITIONS
#define DA_PARTITIONS 1 // L-069: two-way output partition is experimental
#endif
static_assert(DA_PARTITIONS == 1 || DA_PARTITIONS == 2, "supported output partitions");
#ifndef DA_TRANSPOSE_V
#define DA_TRANSPOSE_V 0 // L-070: experimental contiguous V operands
#endif

typedef short v8s __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef float v2f __attribute__((ext_vector_type(2)));
typedef uint32_t v2u __attribute__((ext_vector_type(2)));
typedef uint32_t v4u __attribute__((ext_vector_type(4)));  // fragments built as packed bf16 pairs, then bit-cast

namespace {
constexpr int BQ = 4;      // BLOCK_Q
constexpr int QPK = 4;     // num_queries_per_kv
constexpr int HD = 128;    // head_size
constexpr int TS = 32;     // TILE_SIZE
constexpr int VROW = 136;  // LDS V row stride in bf16 (padded: the two half-waves read keys 8 apart)
constexpr int VKEYS = 40;  // transposed key stride, each output column remains 16-byte aligned
constexpr int MAXBT = 32;  // block-table entries cached in LDS (window spans <= 4 blocks of 880)

__device__ __forceinline__ int vkey(int column, int key) {
#if DA_TRANSPOSE_V >= 2
  constexpr int group = DA_TRANSPOSE_V == 2 ? 8 : 4;
  return key ^ (((column >> 4) & (32 / group - 1)) * group);
#else
  return key;
#endif
}

__device__ __forceinline__ uint32_t rne16(float f) {  // Triton's f32 -> bf16 rounding without the NaN select
  const uint32_t u = __float_as_uint(f);
  return (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
}

__device__ __forceinline__ uint32_t bf16_rne(float f) {  // exactly Triton's: RNE, any NaN -> 0x7FFF
  return __builtin_isnan(f) ? 0x7FFFu : rne16(f);
}

template <bool HI>  // bytes 2,3 of w if HI else bytes 0,1
__device__ __forceinline__ v2f fp8x2(uint32_t w) {
#ifdef DA_SW_FP8
  // Triton's decode (%591-%617): normals by shift + bias 8 (exact), subnormals by table (= m * 2^-9), 0x7F -> NaN.
  v2f r;
  for (int k = 0; k < 2; ++k) {
    const uint32_t b = (w >> (8 * (2 * HI + k))) & 0xFFu, em = b & 0x7Fu;
    const float mag = em == 0x7Fu ? __uint_as_float(0x7FC00000u)
                    : em >= 8u    ? __uint_as_float((em << 20) + (120u << 23))
                                  : (float)em * 0x1p-9f;
    r[k] = __uint_as_float(__float_as_uint(mag) | ((b & 0x80u) << 24));
  }
  return r;
#else
  return __builtin_amdgcn_cvt_pk_f32_fp8(w, HI);
#endif
}

// 16 fp8 -> 16 bf16 (two v4u, element order kept): (x.f32 * scale).bf16, U:54. No NaN canonicalisation here:
// a NaN K/V input stays NaN (other payload than Triton's 0x7FFF) and NaN only ever propagates to NaN outputs,
// which both kernels store as 0x7FFF, or is masked identically. UNIT (scale == 1.0f, the production value):
// x * 1 is exact and an fp8 value has <= 4 significant bits, so RNE is plain truncation of the f32.
template <bool UNIT>
__device__ __forceinline__ void dec16(v4u r, float sc, v4u& lo, v4u& hi) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const v2f a = fp8x2<false>(r[i]), b = fp8x2<true>(r[i]);
    uint32_t x, y;
    if (UNIT) {
      x = (__float_as_uint(a.x) >> 16) | (__float_as_uint(a.y) & 0xFFFF0000u);
      y = (__float_as_uint(b.x) >> 16) | (__float_as_uint(b.y) & 0xFFFF0000u);
    } else {
      x = rne16(a.x * sc) | (rne16(a.y * sc) << 16);
      y = rne16(b.x * sc) | (rne16(b.y * sc) << 16);
    }
    if (i < 2) { lo[2 * i] = x; lo[2 * i + 1] = y; } else { hi[2 * i - 4] = x; hi[2 * i - 3] = y; }
  }
}

__device__ __forceinline__ void decode(v4u r, float sc, v4u& lo, v4u& hi) {  // sc is uniform: scalar branch
  if (sc == 1.0f) dec16<true>(r, sc, lo, hi); else dec16<false>(r, sc, lo, hi);
}

__device__ __forceinline__ float exp_tri(float x) {  // tl.exp as compiled (see 7.)
  const float L2E = 0x1.715476p+0f;                  // 0x3fb8aa3b
  const bool c = x * L2E < -126.0f;
  const float y = __builtin_fmaf(x, L2E, c ? 64.0f : 0.0f);
  return __builtin_amdgcn_ldexpf(__builtin_amdgcn_exp2f(y), c ? -64 : 0);
}

// Workgroup barrier ordering LDS only. __syncthreads()' release fence also waits for every global load in flight
// (s_wait_loadcnt 0 on gfx12), which would drain the prefetch ring at each tile.
__device__ __forceinline__ void lds_barrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

__device__ __forceinline__ float other_half(float x) {  // lane l <- lane l^16, as Triton's permlanex16
  return __int_as_float(__builtin_amdgcn_permlanex16(__float_as_int(x), __float_as_int(x), 0x76543210u,
                                                     0xfedcba98u, true, false));
}
}  // namespace

// 16 working programs on 32 WGPs: occupancy is irrelevant, so take the whole register file for the prefetch ring.
extern "C" __global__ void __launch_bounds__(128) __attribute__((amdgpu_waves_per_eu(1, 1)))
drafter_attn_kernel(uint16_t* __restrict__ out, const uint16_t* __restrict__ q, const uint8_t* __restrict__ kc,
                    const uint8_t* __restrict__ vc, const int* __restrict__ block_table,
                    const int* __restrict__ seq_lens, const int* __restrict__ cu_q, const float* __restrict__ k_scale,
                    const float* __restrict__ v_scale, float scale, int num_seqs, int window, int block_size,
                    long long bt_stride, long long qs0, long long qs1, long long os0, long long os1, long long ks0,
                    long long ks1, long long ks2, long long vs0, long long vs1, long long vs2) {
  __shared__ __attribute__((aligned(16))) uint16_t sQ[16 * HD];       // the q-block's 16 rows (frees 32 VGPRs)
  __shared__ __attribute__((aligned(16))) uint16_t sK[2][TS * HD];    // decoded K tile; Q/K 16-B chunks swizzled
  __shared__ __attribute__((aligned(16))) uint16_t sV[2][DA_TRANSPOSE_V ? HD * VKEYS : TS * VROW];
  __shared__ int sBT[MAXBT];

  const int qbg = blockIdx.x, kvh = blockIdx.y, tid = threadIdx.x;

  // resolve_seq_and_query_len / find_seq_idx (H:45-106)
  int lo = 0, hi = num_seqs;
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    if (cu_q[mid] / BQ + mid <= qbg) lo = mid + 1; else hi = mid;
  }
  const int s = lo - 1, q0 = cu_q[s], qlen = cu_q[s + 1] - q0, qb = qbg - (q0 / BQ + s);
  if (qb * BQ >= qlen) return;  // U:319
  const int seq_len = seq_lens[s], ctx = seq_len - qlen;

  // compute_tile_loop_bounds (H:179-236), non-causal + sliding window. C '/' truncates like Triton's divsi.
  const int qlo = qb * BQ, qhi = min(qlo + (16 - 1) / QPK, qlen - 1);
  const int num_tiles = (seq_len + TS - 1) / TS;
  const int t0 = max(0, (ctx + qlo - window + 1) / TS);
  const int t1 = min(min(ctx + qhi + window - 1, seq_len - 1) / TS + 1, num_tiles);

  // The window's block-table entries, read once (the launcher guarantees the span fits in MAXBT). Keeping the
  // address path free of global reads matters: a global load feeding an address makes the compiler wait for
  // every prefetch in flight (s_wait_loadcnt 0).
  const int* bt = block_table + (long long)s * bt_stride;
  const int b_lo = t0 * TS / block_size;
  if (tid < MAXBT && (long long)(b_lo + tid) * block_size < seq_len) sBT[tid] = bt[b_lo + tid];

  const int lane = tid & 31, w = tid >> 5, m = lane & 15, h = lane >> 4;
  const int output_base = (HD / DA_PARTITIONS) * blockIdx.z + (32 / DA_PARTITIONS) * w;
  const int qpos = qlo + (m >> 2), head = kvh * QPK + (m & 3), qabs = ctx + qpos;
  const bool rowok = qpos < qlen;  // query_mask_0 (query_mask_1 is always true: 4 * num_kv_heads q heads)
  const int ld_r = tid >> 3, ld_c = tid & 7;  // loaders: row/key (tid >> 3) [+16], 16-element chunk (tid & 7)

  {  // Q rows -> LDS, rows past the query length 0 (U:369). Row r = 4 * query + head (U:338-341).
    const int qp = qlo + (ld_r >> 2);
    v4u a = (v4u)0, b = (v4u)0;
    if (qp < qlen) {
      const v4u* src = (const v4u*)(q + (long long)(q0 + qp) * qs0 + (long long)(kvh * QPK + (ld_r & 3)) * qs1 +
                                    16 * ld_c);
      a = src[0];
      b = src[1];
    }
    v4u* dst = (v4u*)&sQ[ld_r * HD];
    dst[(2 * ld_c) ^ ld_r] = a;
    dst[(2 * ld_c + 1) ^ ld_r] = b;
  }
  __syncthreads();

  const float ksc = *k_scale, vsc = *v_scale;

  auto issue = [&](int j, v4u (&k)[2], v4u (&v)[2]) {  // raw fp8 rows, unconditional (clamped) loads
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int key = min(j * TS + ld_r + 16 * r, seq_len - 1);  // keys >= seq_len are zeroed in stage()
      const int b = key / block_size, off = key - b * block_size;
      const long long pb = sBT[b - b_lo];
      k[r] = *(const v4u*)(kc + pb * ks0 + off * ks1 + kvh * ks2 + 16 * ld_c);
      v[r] = *(const v4u*)(vc + pb * vs0 + off * vs1 + kvh * vs2 + 16 * ld_c);
    }
  };

  auto stage = [&](int j, const v4u (&k)[2], const v4u (&v)[2], int bi) {
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int kk = ld_r + 16 * r, key = j * TS + kk;
      const bool live = key < seq_len;  // tile_mask: masked K/V load -> 0 (U:481-491)
      v4u a, b;
      decode(live ? k[r] : (v4u)0, ksc, a, b);
      v4u* kr = (v4u*)&sK[bi][kk * HD];
      kr[(2 * ld_c) ^ (kk & 15)] = a;
      kr[(2 * ld_c + 1) ^ (kk & 15)] = b;
      decode(live ? v[r] : (v4u)0, vsc, a, b);
      const int dist = ctx + qlo - key;  // sw_mask_v, non-causal (U:564-578)
      const bool vok = live & (dist < window) & (dist > -window);
#if DA_TRANSPOSE_V
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const uint32_t pair = i < 4 ? a[i] : b[i - 4];
        sV[bi][(16 * ld_c + 2 * i) * VKEYS + vkey(16 * ld_c + 2 * i, kk)] = vok ? (uint16_t)pair : 0;
        sV[bi][(16 * ld_c + 2 * i + 1) * VKEYS + vkey(16 * ld_c + 2 * i + 1, kk)] = vok ? (uint16_t)(pair >> 16) : 0;
      }
#else
      v4u* vr = (v4u*)&sV[bi][kk * VROW];
      vr[2 * ld_c] = vok ? a : (v4u)0;
      vr[2 * ld_c + 1] = vok ? b : (v4u)0;
#endif
    }
  };

  float M = -__builtin_inff(), L = 1.0f;
  v8f acc[2 / DA_PARTITIONS] = {};  // exact independent output columns, unchanged accumulation order

  auto tile = [&](int j, int bi) {
    // B operand chunk c = Q[m][16c + 8h .. +8] (%139-%165); A operand = K[16t + m][16c + 8h .. +8] (%335-%358).
    // The two key-halves t are independent chains; interleaving them hides WMMA and LDS latency.
    const v4u* qr = (const v4u*)&sQ[m * HD];
    const v4u* kr0 = (const v4u*)&sK[bi][m * HD];
    const v4u* kr1 = (const v4u*)&sK[bi][(16 + m) * HD];
    v8f S[2] = {(v8f)0, (v8f)0};
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      const int x = (2 * c + h) ^ m;
      const v8s qc = __builtin_bit_cast(v8s, qr[x]);
      S[0] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(__builtin_bit_cast(v8s, kr0[x]), qc, S[0]);
      S[1] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(__builtin_bit_cast(v8s, kr1[x]), qc, S[1]);
    }
    // Masks (U:547). `full`: every key of the tile is visible to all 16 rows, the select is the identity.
    const bool full = (qlo + BQ <= qlen) & (j * TS > ctx + qlo + BQ - 1 - window) &
                      (j * TS + TS - 1 < min(seq_len, ctx + qlo + window));
    float sv[16];
    if (full) {
#pragma unroll
      for (int r = 0; r < 16; ++r) sv[r] = __builtin_fmaf(scale, S[r >> 3][r & 7], 0.0f);
    } else {
#pragma unroll
      for (int r = 0; r < 16; ++r) {
        const int key = j * TS + 16 * (r >> 3) + 8 * h + (r & 7);
        const bool ok = rowok & (key < seq_len) & (qabs - key < window) & (key - qabs < window);
        sv[r] = ok ? __builtin_fmaf(scale, S[r >> 3][r & 7], 0.0f) : -__builtin_inff();
      }
    }
    float mx = sv[0];
#pragma unroll
    for (int r = 1; r < 16; ++r) mx = __builtin_fmaxf(mx, sv[r]);
    mx = __builtin_fmaxf(mx, other_half(mx));
    float mj = __builtin_fmaxf(M, mx);
    mj = mj > -__builtin_inff() ? mj : 0.0f;
    float p[16];
#pragma unroll
    for (int r = 0; r < 16; ++r) p[r] = exp_tri(sv[r] - mj);
    float lj = p[0] + p[1];
#pragma unroll
    for (int r = 2; r < 16; ++r) lj = lj + p[r];
    lj = lj + other_half(lj);
    const float alpha = exp_tri(M - mj);
    L = __builtin_fmaf(L, alpha, lj);
    M = mj;
    v8s pf[2];
#pragma unroll
    for (int t = 0; t < 2; ++t) {
      v4u pk;
#pragma unroll
      for (int i = 0; i < 4; ++i) pk[i] = bf16_rne(p[8 * t + 2 * i]) | (bf16_rne(p[8 * t + 2 * i + 1]) << 16);
      pf[t] = __builtin_bit_cast(v8s, pk);
    }
#pragma unroll
    for (int dt = 0; dt < 2 / DA_PARTITIONS; ++dt) {
      acc[dt] = acc[dt] * alpha;
#if DA_TRANSPOSE_V
      const uint16_t* vcol = &sV[bi][(output_base + 16 * dt + m) * VKEYS];
#else
      const uint16_t* vcol = &sV[bi][output_base + 16 * dt + m];
#endif
#pragma unroll
      for (int c = 0; c < 2; ++c) {
#if DA_TRANSPOSE_V
        const int col = output_base + 16 * dt + m;
#if DA_TRANSPOSE_V == 3
        const v2u lo = *(const v2u*)(vcol + vkey(col, 16 * c + 8 * h));
        const v2u hi = *(const v2u*)(vcol + vkey(col, 16 * c + 8 * h + 4));
        const v4u pk = {lo[0], lo[1], hi[0], hi[1]};
#else
        const v4u pk = *(const v4u*)(vcol + vkey(col, 16 * c + 8 * h));
#endif
#else
        v4u pk;
#pragma unroll
        for (int i = 0; i < 4; ++i)
          pk[i] = vcol[(16 * c + 8 * h + 2 * i) * VROW] | ((uint32_t)vcol[(16 * c + 8 * h + 2 * i + 1) * VROW] << 16);
#endif
        acc[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(__builtin_bit_cast(v8s, pk), pf[c], acc[dt]);
      }
    }
  };

  // Prefetch ring: slot u holds tile j + u. Every issue is unconditional (tiles past the end are clamped to the
  // last one and never staged) so the compiler can count loads in order and wait only for the oldest slot
  // (s_wait_loadcnt 4 * (DA_DEPTH - 1)) instead of draining the ring. Double-buffered LDS: one barrier per tile.
  if (t0 < t1) {
    v4u rk[DA_DEPTH][2], rv[DA_DEPTH][2];
#pragma unroll
    for (int u = 0; u < DA_DEPTH; ++u) issue(min(t0 + u, t1 - 1), rk[u], rv[u]);
    int j = t0;
    for (; j + DA_DEPTH <= t1; j += DA_DEPTH) {
#pragma unroll
      for (int u = 0; u < DA_DEPTH; ++u) {
        stage(j + u, rk[u], rv[u], (j + u) & 1);
        issue(min(j + u + DA_DEPTH, t1 - 1), rk[u], rv[u]);
        lds_barrier();
        tile(j + u, (j + u) & 1);
      }
    }
#pragma unroll
    for (int u = 0; u < DA_DEPTH - 1; ++u)  // tail: < DA_DEPTH tiles, already in slots 0..
      if (j + u < t1) {
        stage(j + u, rk[u], rv[u], (j + u) & 1);
        lds_barrier();
        tile(j + u, (j + u) & 1);
      }
  }

  if (!rowok) return;
  uint16_t* orow = out + (long long)(q0 + qpos) * os0 + (long long)head * os1 + output_base + 8 * h;
#pragma unroll
  for (int dt = 0; dt < 2 / DA_PARTITIONS; ++dt) {
    uint32_t o[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
      o[i] = bf16_rne(acc[dt][2 * i] / L) | (bf16_rne(acc[dt][2 * i + 1] / L) << 16);
    *(uint4*)(orow + 16 * dt) = make_uint4(o[0], o[1], o[2], o[3]);
  }
}

// Decode check for the test: n16 chunks of 16 fp8 -> 16 bf16, same path as the kernel.
extern "C" __global__ void drafter_attn_decode_kernel(const v4u* in, v4u* out, int n16, const float* sc) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n16) return;
  v4u a, b;
  decode(in[i], *sc, a, b);
  out[2 * i] = a;
  out[2 * i + 1] = b;
}

// q [tokens, num_q_heads, 128] bf16; k_cache/v_cache fp8 views [blocks, block_size, num_kv_heads, 128] (byte
// strides; head dim contiguous); block_table [num_seqs, bt_stride] int32; seq_lens [num_seqs]; cu_seqlens_q
// [num_seqs + 1]; k_scale/v_scale -> one f32 each; out [tokens, num_q_heads, 128] bf16. window = SLIDING_WINDOW
// (= 1 + window_size[0], 2048 for the drafter), non-causal. Returns 0, or <0 for an unsupported call.
extern "C" int drafter_attn(void* out, const void* q, const void* k_cache, const void* v_cache,
                            const int* block_table, const int* seq_lens, const int* cu_seqlens_q,
                            const float* k_scale, const float* v_scale, float softmax_scale, int num_tokens,
                            int num_seqs, int num_q_heads, int num_kv_heads, int head_size, int block_size,
                            int window, long long bt_stride, long long q_s0, long long q_s1, long long o_s0,
                            long long o_s1, long long k_s0, long long k_s1, long long k_s2, long long v_s0,
                            long long v_s1, long long v_s2, hipStream_t stream) {
  // A program's tiles span <= 2 * window + 64 keys -> block-table entries must fit the LDS copy.
  if (head_size != HD || num_q_heads != QPK * num_kv_heads || window <= 0 || block_size <= 0 ||
      (2LL * window + 64) / block_size + 2 > MAXBT)
    return -1;
  const unsigned long long al = (uintptr_t)out | (uintptr_t)q | (uintptr_t)k_cache | (uintptr_t)v_cache;
  if ((al & 15) || ((q_s0 | q_s1 | o_s0 | o_s1) & 7) || ((k_s0 | k_s1 | k_s2 | v_s0 | v_s1 | v_s2) & 15)) return -2;
  hipLaunchKernelGGL(drafter_attn_kernel, dim3(num_tokens / BQ + num_seqs, num_kv_heads, DA_PARTITIONS), dim3(128), 0, stream,
                     (uint16_t*)out, (const uint16_t*)q, (const uint8_t*)k_cache, (const uint8_t*)v_cache,
                     block_table, seq_lens, cu_seqlens_q, k_scale, v_scale, softmax_scale, num_seqs, window,
                     block_size, bt_stride, q_s0, q_s1, o_s0, o_s1, k_s0, k_s1, k_s2, v_s0, v_s1, v_s2);
  return hipPeekAtLastError() == hipSuccess ? 0 : -3;
}

extern "C" int drafter_attn_decode(const void* in, void* out, int n16, const float* scale, hipStream_t stream) {
  hipLaunchKernelGGL(drafter_attn_decode_kernel, dim3((n16 + 255) / 256), dim3(256), 0, stream, (const v4u*)in,
                     (v4u*)out, n16, scale);
  return hipPeekAtLastError() == hipSuccess ? 0 : -3;
}

// Load the kernel's code object now, before any CUDA-graph capture (lazy first-launch loading inside a capture crashed
// the WSL runtime for the GDN brick, E002 2026-10-10). Returns a HIP error code.
extern "C" int drafter_attn_warmup()
{
  hipFuncAttributes attr;
  return (int)hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(&drafter_attn_kernel));
}
