// gdn_verify.hip -- T02-M02 mode E: one GDN layer's speculative verify step (causal conv + SiLU,
// q/k l2norm, gates, delta-rule recurrence over the candidates, a state snapshot per candidate) as
// ONE kernel on a full grid, with no grid barrier and no q/k/v round trip through HBM.
//
// WHAT IT REPLACES, per GDN layer per verify step (tracks/T02-kernels/milestones/M02-gdn-verify/SPEC.md):
//   * vLLM's core_attn_out.zero_() (FillFunctor, SPEC 2.3) -- this kernel writes every o row a
//     sequence covers (zeros where production would have skipped it) and zeroes the cudagraph pad rows;
//   * libr4d's r4d_gdn_fused_update_kernel<ABF16, STDT=f16> (60 us cold): same arguments, same
//     outputs, BYTE-IDENTICAL o, conv-state block and every snapshot slot.
// (The z copy, the other dead kernel, goes away in the caller by passing the strided z; SPEC 2.3.)
//
// WHY IT SHOULD BE FASTER. Production runs 32 workgroups (half the machine), work-loops 48 heads on
// them (two rounds), round-trips q/k/v through HBM across a grid barrier, and issues each token's
// loads inside its serial chain. Here: one workgroup per (sequence, v-head, row slice) -- 96 x 128
// threads at H=48, VS=2 -- every load issued at entry, conv/l2norm/gates computed into LDS, and the
// 8-token chain reads only LDS and registers. The floor is the 8 snapshots (12.6 MB/layer, 24.7 us);
// target <= 28 us cold.
//
// BIT-EXACTNESS. Each output value is produced by the same arithmetic in the same order as
// production; only WHO computes it and WHEN changes. Every such choice cites the libr4d line it
// mirrors. Citations: FU = repos/libr4d/r4d_gdn_fused_update_w4k128v128.h, ST = r4d_gdn_state.h,
// WM = r4d_gdn_wmma.h, CM = r4d_common.h, BS = build.sh (libr4d snapshot of 2026-10-10).
//   * thread (row, half) owns S[row][64*half .. +64) exactly as FU:175, so S*k, S*q and the state
//     update are the same 64-long sequential fp32 chains joined by one xor-1 shuffle (FU:222-232);
//   * the q/k sum of squares is the same per-half 64-long chain + xor-1 join (FU:208-213); the
//     normalised value bf2f(q)*qs is computed once into LDS instead of per thread (same product);
//   * conv: acc = bias (0.0f when absent) then 4 sequential mul+add, f2bf(silu) (FU:123-126);
//   * snapshots: fp32 state rounded RTNE to f16 by the same two-convert + perm (ST:51-55, 93-98);
//   * gates: computed per token from uniform values, so -- as in production, where tok and hv are
//     uniform -- they compile to the scalar unit (s_mul_f32, s_fmamk_f32, v_s_exp_f32,
//     v_s_log_f32), not to the per-lane v_exp_f32/v_log_f32 path whose bits are not proven equal;
//   * no FMA contraction anywhere: production builds with -ffp-contract=off (BS:35). The pragma
//     below makes this file independent of that flag; build with the flag too.
//
// CONV-STATE WRITE WITHOUT A GRID BARRIER. The q/k conv channels of k-head hk are read by every
// workgroup of the (n, hk) group (H/Hg heads x VS slices = 6), and the conv state is rolled in
// place, so the write has to wait for all six reads. Each workgroup arrives on a per-group counter
// after its reads (the FU:159-169 idiom); the LAST to arrive writes the q/k channels, and its
// wrapping atomicInc leaves the counter at zero for the next launch. Nobody waits, so there is
// no residency requirement and no deadlock at any N. v channels have one reader (their owner),
// which writes them itself.
//
// NULL SLOTS (SPEC 2.2), same rules as production:
//   conv slot (cache_idx) <= 0  -> no conv read/write, no recurrence;
//   read slot sidx[n][naccept-1] <= 0 -> no recurrence, no snapshot; o rows written as ZERO
//     (production leaves them alone; vLLM's fill had zeroed them, so fill + production == this);
//   write slot sidx[n][t] <= 0  -> that snapshot is skipped.
// One deliberate divergence, in a case production leaves undefined: conv slot <= 0 with a valid
// read slot. Production then runs the recurrence on unwritten q/k/v scratch; this kernel writes
// zeros to o and touches no state.
#include <hip/hip_runtime.h>

#pragma clang fp contract(off)

#define GV_K      128   // head_k (FU_K)
#define GV_V      128   // head_v (FU_V)
#define GV_W        4   // conv width (FU_W)
#define GV_HK      64   // k elements per thread (FU_HK)
#define GV_MAXT     8   // max candidates per sequence (1 + num_spec): sizes LDS
#define GV_PAD     68   // LDS pitch of one 64-float k half: +4 floats puts the two halves' ds_read_b128
                        // (the only two addresses a wave reads per instruction) in different banks
#ifndef GV_VS
#define GV_VS       2   // workgroups per v-head (row slices): 1, 2 or 4
#endif
#ifndef GV_NT_STORE
#define GV_NT_STORE 1   // snapshots as non-temporal stores (SPEC 4.2 P3); 0 = plain, as production
#endif
#ifndef GV_LAZY_TRACE
#define GV_LAZY_TRACE 0 // ADR-0006: exact deferred state prototype; normal ABI remains default
#endif
#ifndef GV_LOCAL_ARRIVAL
#define GV_LOCAL_ARRIVAL 0 // L-073: read-completion handshake, experimental until measured
#endif
#ifndef GV_COALESCED_STORES
#define GV_COALESCED_STORES 0 // L160: exact register transpose of snapshot stores.
#endif
#ifndef GV_VECTOR_STORES
#define GV_VECTOR_STORES 0 // L161: use VALU permutations and permuted store addresses.
#endif

// First-class vector types, not HIP's struct vectors: a struct copy from a const __restrict__
// source is a memcpy that LLVM forwards, i.e. it re-reads global memory at the point of use and
// silently undoes the load ordering below.
typedef unsigned       u32x4 __attribute__((ext_vector_type(4)));
typedef unsigned short u16x4 __attribute__((ext_vector_type(4)));

// ---- arithmetic helpers: copies of the production definitions, so this file stands alone -------
__device__ __forceinline__ float gv_bf2f(unsigned short h) {          // WM:39-41
  return __builtin_bit_cast(float, (unsigned)h << 16);
}
__device__ __forceinline__ unsigned short gv_f2bf(float f) {          // WM:16-18, round half away
  return (unsigned short)((__builtin_bit_cast(unsigned, f) + 0x8000u) >> 16);
}
__device__ __forceinline__ unsigned gv_f2h2(float a, float b) {       // ST:51-55, RTNE (never pkrtz)
  union { __fp16 h; unsigned short u; } qa, qb;
  qa.h = (__fp16)a; qb.h = (__fp16)b;
  return __builtin_amdgcn_perm((unsigned)qb.u, (unsigned)qa.u, 0x05040100u);
}
__device__ __forceinline__ float gv_h2f(unsigned short h) {           // ST:56-58
  union { unsigned short u; __fp16 h; } q; q.u = h; return (float)q.h;
}
__device__ __forceinline__ float gv_silu(float x) { return x / (1.0f + __expf(-x)); }   // FU:40
__device__ __forceinline__ float gv_softplus(float x, float thr) {                       // FU:41-44
  if (x > thr) return x;
  return (x > 0.0f) ? (x + __logf(1.0f + __expf(-x))) : __logf(1.0f + __expf(x));
}
// FU:45-48 split in two: the raw load (issued in P0) and the conversion (done where it is used).
// A bf16 input is read as the aligned dword holding it: with a uniform address that is one
// s_load_b32 (kmcnt), where a 16-bit load becomes a vector load + v_readfirstlane, i.e. a
// loadcnt wait inside the load block. Needs a 4-byte aligned base (checked by the launcher); the
// dword stays inside the in_proj row, whose a/b columns start at an even column.
template <int ABF16>
__device__ __forceinline__ unsigned gv_ab_raw(const void* p, size_t i) {
  const unsigned* w = (const unsigned*)__builtin_assume_aligned(p, 4);
  if (!ABF16) return w[i];
  const unsigned d = w[i >> 1];
  return (i & 1) ? d >> 16 : d & 0xffffu;
}
template <int ABF16>
__device__ __forceinline__ float gv_ab_f(unsigned raw) {
  return ABF16 ? gv_bf2f((unsigned short)raw) : __builtin_bit_cast(float, raw);
}
// Pin the order of global loads: the empty asm stops IR passes moving memory ops across it, the
// sched_barrier (CM:63) stops the machine scheduler. Neither emits an instruction or a wait.
__device__ __forceinline__ void gv_sched_barrier() {
  asm volatile("" ::: "memory");
  __builtin_amdgcn_sched_barrier(0);
}
// Workgroup barrier that orders LDS only (CM:65-74): a full __syncthreads() would also wait for
// the outstanding global loads/stores, i.e. stall on the state prefetch or the snapshot drain.
__device__ __forceinline__ void gv_lds_barrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// Grid: N * H * VS workgroups of 256/VS threads. Workgroup (n, hv, s) owns v rows
// [RV*s, RV*s + RV) of head hv; thread tid owns row RV*s + tid/2, k half tid&1 (FU:175).
template <int ABF16, int VS>
__global__ __launch_bounds__(256 / VS)
void gdn_verify_kernel(
    const unsigned short* __restrict__ x, long xpitch,
    const unsigned short* __restrict__ wgt, const unsigned short* __restrict__ bias,
    unsigned short* __restrict__ cstate, long cs_seq, long cs_dim, long cs_tok,
    const int* __restrict__ cache_idx, long ci_stride,
    const int* __restrict__ naccept, const int* __restrict__ cu,
    int N, int H, int Hg, int maxq,
    const void* __restrict__ av, const void* __restrict__ bv, long ab_stride,
    const float* __restrict__ A_log, const float* __restrict__ dt_bias,
    unsigned short* __restrict__ state, long st_slot, long st_head,
    unsigned short* __restrict__ o,
    const int* __restrict__ sidx, long sidx_stride,
    float scale, float sp_thr, int* __restrict__ cnt, int o_rows, float* __restrict__ trace)
{
  constexpr int NT  = 256 / VS;              // threads: two per v row
  constexpr int RV  = GV_V / VS;             // v rows this workgroup owns
  constexpr int NCH = 2 * GV_K + RV;         // conv channels it needs: q, k of head hk + its v rows
  constexpr int CI  = (NCH + NT - 1) / NT;   // conv channel slots per thread (VS=2: q, k, v|junk)
  constexpr int VC  = 4 * GV_PAD;            // column of v row 0 in a conv-output row
  constexpr int CW  = (VC + RV + 1 + 3) & ~3; // conv-output row: q lo|hi, k lo|hi, v, junk; float4-aligned
  constexpr int GW  = GV_W - 2;              // conv-state positions carried over from history (FU:140)
  static_assert(4 * GV_MAXT == 32, "l2norm maps (token, q|k, half) onto exactly one wave32");
  static_assert(NT >= 32 && GV_MAXT <= NT, "a workgroup must cover wave 0 and the gate threads");

  // cv_s[t][col]: col (2w + half)*GV_PAD + j holds q (w=0) / k (w=1) channel 64*half + j -- conv
  // output, then normalised in place to q-hat / k-hat; col VC + r holds v row RV*s + r. Lanes with
  // nothing to compute (channel slot >= NCH) write one shared junk column
  // instead of branching, so P0/P1 stay one basic block (see P0). ~17.5 KB at VS=2: 3 per 64 KB.
  __shared__ __attribute__((aligned(16))) float cv_s[GV_MAXT][CW];
  __shared__ float g_s[GV_MAXT][2];                    // [t]: decay exp(g), beta
  __shared__ float f_s[GV_MAXT][2];                    // [t]: l2norm factor of q (x scale), of k
  __shared__ unsigned short cs_s[GW + GV_MAXT][NCH + 1];   // conv-state bytes to write at the end
  __shared__ int last_s;

  const int tid  = threadIdx.x;
  __builtin_assume(tid < NT);
  const int s    = blockIdx.x % VS;
  const int hv   = (blockIdx.x / VS) % H;
  const int n    = blockIdx.x / (VS * H);
  const int hk   = hv / (H / Hg);                       // FU:180
  const int rl   = tid >> 1, half = tid & 1;            // FU:175
  const int row  = RV * s + rl;

  // channel slot c -> conv channel, q | k | v layout of the in_proj output (SPEC 1.1); c < NCH
  auto chan = [&](int c) {
    return c < GV_K ? hk * GV_K + c
         : c < 2 * GV_K ? (Hg + hk) * GV_K + (c - GV_K)
         : (2 * Hg + hv) * GV_V + RV * s + (c - 2 * GV_K);
  };
  auto col = [](int c) {
    return c < 2 * GV_K ? (c >> 6) * GV_PAD + (c & 63) : VC + (c < NCH ? c - 2 * GV_K : RV);
  };

  // Per-sequence scalars, all independent loads: the state load below is then two dependent hops
  // deep (naccept + sidx row -> state), not three (naccept -> sidx -> state).
  const int bos  = cu[n];
  const int T    = cu[n + 1] - bos;                                    // FU:74-75, 178-179
  const int slot = cache_idx[(size_t)n * ci_stride];                   // FU:76
  const int off  = naccept ? (naccept[n] - 1) : 0;     // FU:79 conv offset == FU:181 read column
  const int first = cu[N];                             // first cudagraph pad row of o (FU:252)
  int sl[GV_MAXT];
#pragma unroll
  for (int j = 0; j < GV_MAXT; ++j) sl[j] = sidx[(size_t)n * sidx_stride + (j < maxq ? j : maxq - 1)];
  int si = 0;
#pragma unroll
  for (int j = 0; j < GV_MAXT; ++j) si = (j == off) ? sl[j] : si;                 // FU:182

  if (T > 0 && slot > 0) {
    // ---- P0: every global load in ONE branch-free block, so all are in flight together --------
    // Anything conditional ends the block: the compiler waits for a load at the join, or sinks it
    // into the block that uses it, after the state load. So nothing here is conditional: channel
    // slots past NCH, tokens past T, a null read slot and an absent bias all load valid (if
    // useless) addresses, and what they produce lands in junk LDS or is never stored. The state
    // goes last: loadcnt retires in order, so the conv math can start while it is in flight.
    const int tl = T - 1;
    const unsigned short* bp = bias ? bias : wgt;
    u16x4 wraw[CI];
    unsigned braw[CI], hraw[CI][GV_W - 1], xraw[CI][GV_MAXT];
#pragma unroll
    for (int i = 0; i < CI; ++i) {
      const int c = tid + i * NT, d = chan(c < NCH ? c : NCH - 1);
      wraw[i] = *(const u16x4*)(wgt + (size_t)d * GV_W);                          // FU:86
      braw[i] = bp[d];                                                             // FU:90
      const unsigned short* sp = cstate + (size_t)slot * cs_seq + (size_t)d * cs_dim;
#pragma unroll
      for (int p = 0; p < GV_W - 1; ++p) hraw[i][p] = sp[(size_t)(off + p) * cs_tok];     // FU:97-103
#pragma unroll
      for (int t = 0; t < GV_MAXT; ++t)
        xraw[i][t] = x[(size_t)(bos + (t < tl ? t : tl)) * xpitch + d];            // FU:116
      gv_sched_barrier();       // keep the issue order = the use order
    }
    // Gate inputs of every token slot. The addresses are uniform, so these are scalar loads
    // (kmcnt, not loadcnt) and the gate math below runs on the scalar unit -- as it does in
    // production, where tok and hv are uniform too (FU:191, 217-220 compile to s_mul_f32,
    // v_s_exp_f32, v_s_log_f32...). Same instructions, so no question of v_exp vs v_s_exp bits.
    unsigned ga[GV_MAXT], gb[GV_MAXT];
#pragma unroll
    for (int t = 0; t < GV_MAXT; ++t) {
      const size_t ab = (size_t)(bos + (t < tl ? t : tl)) * ab_stride + hv;
      ga[t] = gv_ab_raw<ABF16>(av, ab);                                           // FU:218
      gb[t] = gv_ab_raw<ABF16>(bv, ab);                                           // FU:219
    }
    const float alr = A_log[hv], dtb = dt_bias[hv];                                // FU:191
    gv_sched_barrier();
    u32x4 sraw[GV_HK / 8];                                                         // ST:69-72
    {
      const u32x4* p = (const u32x4*)(state + (size_t)(si > 0 ? si : 0) * st_slot
                                      + (size_t)hv * st_head + (size_t)row * GV_K + half * GV_HK);
#pragma unroll
      for (int j = 0; j < GV_HK / 8; ++j) sraw[j] = p[j];                          // FU:187-189
    }
    gv_sched_barrier();

    // ---- P1: conv + SiLU for every (channel slot, token slot), into LDS -----------------------
    // All GV_MAXT token slots are computed; rows t >= T hold finite junk nobody reads.
#pragma unroll
    for (int i = 0; i < CI; ++i) {
      const int c = tid + i * NT, cs = c < NCH ? c : NCH;          // cs: conv-state stash column
      float wt[GV_W], win[GV_W];
#pragma unroll
      for (int p = 0; p < GV_W; ++p) wt[p] = gv_bf2f(wraw[i][p]);                  // FU:89
      const float bs = bias ? gv_bf2f(braw[i]) : 0.0f;                             // FU:90
#pragma unroll
      for (int p = 0; p < GV_W - 1; ++p) win[p] = gv_bf2f(hraw[i][p]);            // FU:102
#pragma unroll
      for (int t = 0; t < GV_MAXT; ++t) {
        win[GV_W - 1] = gv_bf2f(xraw[i][t]);                                       // FU:117-118
        float acc = bs;                                                            // FU:123
#pragma unroll
        for (int p = 0; p < GV_W; ++p) acc += wt[p] * win[p];                      // FU:125
        // FU:126 stores f2bf(silu) to the q/k/v scratch; phase 2 reads it back with bf2f
        // (FU:205, 227). Keep exactly that bf16 rounding, held as fp32.
        cv_s[t][col(c)] = gv_bf2f(gv_f2bf(gv_silu(acc)));
#pragma unroll
        for (int p = 0; p < GV_W - 1; ++p) win[p] = win[p + 1];                    // FU:134-137
      }
      // The rolled conv state of this channel: [hist1, hist2, x0 .. x(T-1)] -- FU:140-156 with
      // slen_eff = GW + T (the launcher enforces state_len_max == GW + maxq).
#pragma unroll
      for (int p = 0; p < GW; ++p) cs_s[p][cs] = hraw[i][p + 1];
#pragma unroll
      for (int t = 0; t < GV_MAXT; ++t) cs_s[GW + t][cs] = xraw[i][t];
    }
    {   // gates (FU:191, 217-220), uniform per token: every lane stores the same value, no branch
      const float alog = __expf(alr);
#pragma unroll
      for (int t = 0; t < GV_MAXT; ++t) {
        const float g  = -alog * gv_softplus(gv_ab_f<ABF16>(ga[t]) + dtb, sp_thr);
        const float bt = 1.0f / (1.0f + __expf(-gv_ab_f<ABF16>(gb[t])));
        g_s[t][0] = __expf(g);
        g_s[t][1] = bt;
      }
    }
    gv_lds_barrier();

    // q/k l2norm: lane = (token, q|k, half). Same 64-long chain per half and xor-1 join as FU:208-213.
    if (tid < 32) {
      const int t = tid >> 2, qk = tid & 3;                 // qk = 2w + half
      float ss = 0.0f;
#pragma unroll
      for (int j = 0; j < GV_HK; ++j) { const float e = cv_s[t][qk * GV_PAD + j]; ss += e * e; }
      ss += __shfl_xor(ss, 1, 32);
      const float r = __frsqrt_rn(ss + 1e-6f);
      if (qk == 0) f_s[t][0] = r * scale;                   // qs = rsqrt * scale (FU:213)
      if (qk == 2) f_s[t][1] = r;                           // ks = rsqrt
    }
    gv_lds_barrier();
    for (int e = tid; e < GV_MAXT * 4 * GV_HK; e += NT) {
      const int t = e >> 8, qk = (e >> 6) & 3, j = e & 63;
      cv_s[t][qk * GV_PAD + j] *= f_s[t][qk >> 1];                                 // FU:215
    }

    // ---- P2: arrive on the group counter. Every conv-state read of this workgroup is complete
    // (the fence waits for them) before thread 0 arrives -- FU:160-163, but nobody spins.
    // atomicInc wraps: the GS-th arrival reads GS-1 and leaves 0 behind, so the counter is zero
    // again for the next launch with no reset store (FU:164-165 needs an atomicExch). It also
    // keeps the AMDGPU atomic optimizer off (it rewrites atomicAdd into a per-wave form that
    // consumes the return value immediately, i.e. waits for the round trip here).
#if GV_LOCAL_ARRIVAL
    // P1 has consumed every shared conv-history read into conv outputs/history LDS, followed
    // by a workgroup barrier. No global stores need publication here. This barrier orders the
    // normalized q/k LDS writes; compiler barriers keep arrival after the completed reads.
    gv_lds_barrier();
    gv_sched_barrier();
#else
    __threadfence();
    __syncthreads();
#endif
    // Read only by thread 0, at the end. Deliberately not initialised in the other lanes: writing
    // them would touch the VGPR the atomic is still returning into and force a wait right here.
    const unsigned group = (unsigned)(H / Hg) * VS;
    unsigned arrived;
    if (tid == 0) arrived = atomicInc((unsigned*)cnt + (size_t)n * Hg + hk, group - 1);

#if GV_LAZY_TRACE
    // One owner per key head/decay. P2 has ordered all normalized LDS values.
    const size_t trace_cols = (size_t)Hg * GV_K + H + (size_t)H * GV_V * 2;
    if (s == 0 && hv % (H / Hg) == 0) {
      for (int e = tid; e < T * GV_K; e += NT) {
        const int t = e / GV_K, j = e % GV_K;
        trace[(size_t)(bos + t) * trace_cols + hk * GV_K + j] =
            cv_s[t][(2 + j / GV_HK) * GV_PAD + j % GV_HK];
      }
    }
    if (s == 0 && tid < T)
      trace[(size_t)(bos + tid) * trace_cols + Hg * GV_K + hv] = g_s[tid][0];
#endif

    // ---- P3: the recurrence over the candidates (FU:185-242) ---------------------------------
    if (si > 0) {
      float h[GV_HK];
#pragma unroll
      for (int j = 0; j < GV_HK / 8; ++j) {                                        // ST:73-76
        const unsigned short* sw = (const unsigned short*)&sraw[j];
#pragma unroll
        for (int e = 0; e < 8; ++e) h[8 * j + e] = gv_h2f(sw[e]);
      }
#pragma unroll 1
      for (int t = 0; t < T; ++t) {
        const float eg = g_s[t][0], bt = g_s[t][1];
        // q-hat and k-hat of this token, this thread's half, into registers up front (production
        // holds the same 2 x 64 values, FU:198-215): their LDS latency hides under the S*k chain.
        float kk[GV_HK], qq[GV_HK];
        const float4* k4 = (const float4*)&cv_s[t][(2 + half) * GV_PAD];
        const float4* q4 = (const float4*)&cv_s[t][half * GV_PAD];
#pragma unroll
        for (int j = 0; j < GV_HK; j += 4) {
          const float4 wk = k4[j >> 2], wq = q4[j >> 2];
          kk[j] = wk.x; kk[j + 1] = wk.y; kk[j + 2] = wk.z; kk[j + 3] = wk.w;
          qq[j] = wq.x; qq[j + 1] = wq.y; qq[j + 2] = wq.z; qq[j + 3] = wq.w;
        }
        float dot = 0.0f;
#pragma unroll
        for (int j = 0; j < GV_HK; ++j) { h[j] *= eg; dot += h[j] * kk[j]; }       // FU:222-224
        dot += __shfl_xor(dot, 1, 32);                                             // FU:225
        const float u = (cv_s[t][VC + rl] - dot) * bt;                             // FU:227
#if GV_LAZY_TRACE
        // Keep both halves: no assumption about NaN payloads or operand-order symmetry.
        trace[(size_t)(bos + t) * trace_cols + Hg * GV_K + H
              + ((size_t)hv * GV_V + row) * 2 + half] = u;
#endif
        float on = 0.0f;
#pragma unroll
        for (int j = 0; j < GV_HK; ++j) { h[j] += u * kk[j]; on += h[j] * qq[j]; }  // FU:229-231
        on += __shfl_xor(on, 1, 32);                                               // FU:232
        if (half == 0) o[((size_t)(bos + t) * H + hv) * GV_V + row] = gv_f2bf(on); // FU:234

#if !GV_LAZY_TRACE
        int so = sl[0];                                // = sidx[n][t], FU:236-241, from registers
#pragma unroll
        for (int j = 1; j < GV_MAXT; ++j) so = (t == j) ? sl[j] : so;
        if (so > 0) {
#if GV_COALESCED_STORES
          // Transform lane-strided 128-byte rows into eight contiguous 512-byte
          // wave stores. Conversion is unchanged; only ownership of packed bits
          // moves. Swap three register-index bits with lane bits 2..4, then
          // rotate the lane index. Store i/lane d consumes original lane
          // (4*i+d/8), vector d%8. No LDS or extra synchronization.
          u32x4 words[GV_HK / 8];
#pragma unroll
          for (int j = 0; j < GV_HK; j += 8)
            words[j / 8] = {gv_f2h2(h[j], h[j + 1]), gv_f2h2(h[j + 2], h[j + 3]),
                           gv_f2h2(h[j + 4], h[j + 5]), gv_f2h2(h[j + 6], h[j + 7])};
          const int lane = tid & 31;
#pragma unroll
          for (int bit = 0; bit < 3; ++bit) {
            const int reg_bit = 1 << bit, lane_bit = 1 << (bit + 2);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
              if (!(j & reg_bit)) {
                const u32x4 send = (lane & lane_bit) ? words[j] : words[j | reg_bit];
                u32x4 received;
#pragma unroll
                for (int c = 0; c < 4; ++c) {
#if GV_VECTOR_STORES
                  if (bit == 0)
                    received[c] = __builtin_amdgcn_permlane16(send[c], send[c], 0x32107654u, 0xba98fedcu, false, false);
                  else if (bit == 1)
                    received[c] = __builtin_amdgcn_permlane16(send[c], send[c], 0xfedcba98u, 0x76543210u, false, false);
                  else
                    received[c] = __builtin_amdgcn_permlanex16(send[c], send[c], 0x76543210u, 0xfedcba98u, false, false);
#else
                  received[c] = __shfl_xor(send[c], lane_bit, 32);
#endif
                }
                if (lane & lane_bit) words[j] = received;
                else words[j | reg_bit] = received;
              }
            }
          }
#if GV_VECTOR_STORES
          // L161: lane order within the contiguous wave region is irrelevant to
          // write coalescing. Address permutation removes 32 more lane shuffles.
          const int destination = ((lane & 3) << 3) | (lane >> 2);
#else
          const int from = ((lane & 7) << 2) | (lane >> 3);
          const int destination = lane;
#endif
          const int wave_row = RV * s + ((tid & ~31) >> 1);
          u32x4* contiguous = (u32x4*)(state + (size_t)so * st_slot + (size_t)hv * st_head
                                          + (size_t)wave_row * GV_K);
#pragma unroll
          for (int j = 0; j < 8; ++j) {
#if GV_VECTOR_STORES
            const u32x4 packed = words[j];
#else
            u32x4 packed;
#pragma unroll
            for (int c = 0; c < 4; ++c) packed[c] = __shfl(words[j][c], from, 32);
#endif
#if GV_NT_STORE
            __builtin_nontemporal_store(packed, contiguous + j * 32 + destination);
#else
            contiguous[j * 32 + destination] = packed;
#endif
          }
#else
          u32x4* p = (u32x4*)(state + (size_t)so * st_slot + (size_t)hv * st_head
                              + (size_t)row * GV_K + half * GV_HK);
#pragma unroll
          for (int j = 0; j < GV_HK; j += 8) {                                     // ST:93-98
            const u32x4 w = {gv_f2h2(h[j], h[j + 1]), gv_f2h2(h[j + 2], h[j + 3]),
                             gv_f2h2(h[j + 4], h[j + 5]), gv_f2h2(h[j + 6], h[j + 7])};
#if GV_NT_STORE
            __builtin_nontemporal_store(w, p + j / 8);
#else
            p[j / 8] = w;
#endif
          }
#endif
        }
#endif
      }
    } else if (half == 0) {
      for (int t = 0; t < T; ++t) o[((size_t)(bos + t) * H + hv) * GV_V + row] = 0;
    }

    // ---- conv-state write: own v channels always, the group's q/k channels if last to arrive ----
    // No acquire fence: nothing is READ after the arrival, and every write below is issued only
    // once the atomic has returned "last" (a control dependency), i.e. after the other arrivals,
    // each of which followed its reads' completion. An agent acquire fence would compile to
    // s_wait_storecnt 0 here and drain this wave's snapshots first.
    if (tid == 0) last_s = arrived == group - 1;
    gv_lds_barrier();
    const bool last = last_s;
#pragma unroll
    for (int i = 0; i < CI; ++i) {
      const int c = tid + i * NT;
      if (c < NCH && (c >= 2 * GV_K || last)) {
        unsigned short* dst = cstate + (size_t)slot * cs_seq + (size_t)chan(c) * cs_dim;
#pragma unroll
        for (int p = 0; p < GW + GV_MAXT; ++p)
          if (p < GW + T) dst[(size_t)p * cs_tok] = cs_s[p][c];                     // FU:154
      }
    }
  } else if (T > 0 && half == 0) {
    for (int t = 0; t < T; ++t) o[((size_t)(bos + t) * H + hv) * GV_V + row] = 0;
  }

  // ---- zero the cudagraph pad rows [cu[N], o_rows) of o (FU:245-258), strided over the grid ---
  {
    const int rowv4 = (H * GV_V) >> 2;
    for (int r = first + blockIdx.x; r < o_rows; r += gridDim.x) {
      ushort4* dst = (ushort4*)(o + (size_t)r * H * GV_V);
      for (int i = tid; i < rowv4; i += NT) dst[i] = make_ushort4(0, 0, 0, 0);
    }
  }
}

// Same argument list, order and meaning as r4d_gdn_fused_update_w4k128v128_bf16_f16state
// (FU:261-270), so a caller can swap the symbol. Differences:
//   * q, k, v (production's scratch) are unused and may be null;
//   * barrier_cnt must hold N * Hg zeroed int32 (one counter per (sequence, k-head) group, not one);
//     the kernel leaves them zero again, so a captured graph can replay it with no clearing pass;
//   * state_len_max must equal (width - 2) + max_query_len and max_query_len <= 8 -- the only
//     values for which production's conv-state write is defined (FU:144-153), checked here;
//   * N >= 1 (production would still zero the pad rows with no sequence; nothing calls it so).
//   * a and b must be 4-byte aligned (any torch tensor, and the in_proj slices production passes).
// Returns 0, or -1 geometry, -2 H % Hg, -3 no counter, -4 N/state_len/max_query_len, -5 a/b
// alignment, else a HIP error.
extern "C" int gdn_verify_w4k128v128_bf16_f16state(
    const void* x, long xpitch, const void* wgt, const void* bias, void* cstate,
    long cs_seq, long cs_dim, long cs_tok, int state_len_max, const void* cache_idx,
    long ci_stride, const void* num_accepted, const void* cu, int N, int H, int Hg,
    int K, int V, int width, int max_query_len, void* q, void* k, void* v,
    const void* a, const void* b, long ab_stride, int ab_is_bf16,
    const void* A_log, const void* dt_bias, void* state, long st_slot, long st_head,
    void* o, const void* sidx, long sidx_stride, float scale, float softplus_thr,
    void* barrier_cnt, int o_rows, void* stream)
{
  (void)q; (void)k; (void)v;
#if GV_LAZY_TRACE
  if (!q) return -6; // explicit FP32 update-trace workspace
#endif
  if (K != GV_K || V != GV_V || width != GV_W) return -1;
  if (Hg <= 0 || H % Hg) return -2;
  if (!barrier_cnt) return -3;
  if (N < 1 || max_query_len < 1 || max_query_len > GV_MAXT
      || state_len_max != GV_W - 2 + max_query_len)
    return -4;
  if (((size_t)a | (size_t)b) & 3) return -5;    // gv_ab_raw reads aligned dwords
  dim3 grid(N * H * GV_VS), blk(256 / GV_VS);
#define GV_LAUNCH(AB)                                                                          \
  hipLaunchKernelGGL((gdn_verify_kernel<AB, GV_VS>), grid, blk, 0, (hipStream_t)stream,        \
                     (const unsigned short*)x, xpitch, (const unsigned short*)wgt,             \
                     (const unsigned short*)bias, (unsigned short*)cstate, cs_seq, cs_dim,     \
                     cs_tok, (const int*)cache_idx, ci_stride, (const int*)num_accepted,       \
                     (const int*)cu, N, H, Hg, max_query_len, a, b, ab_stride,                 \
                     (const float*)A_log, (const float*)dt_bias, (unsigned short*)state,       \
                     st_slot, st_head, (unsigned short*)o, (const int*)sidx, sidx_stride,      \
                     scale, softplus_thr, (int*)barrier_cnt, o_rows, (float*)q)
  if (ab_is_bf16) GV_LAUNCH(1); else GV_LAUNCH(0);
#undef GV_LAUNCH
  return (int)hipGetLastError();
}

// Load this module's code objects now. HIP loads kernels lazily at their first launch; when that first launch falls
// inside a CUDA-graph capture (the served decode path is first exercised by the FULL-graph capture) the module load
// happens mid-capture, which crashed the WSL/librocdxg runtime (E002, 2026-10-10). Call once before any capture.
extern "C" int gdn_verify_warmup()
{
  hipFuncAttributes attr;
  hipError_t e0 = hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(&gdn_verify_kernel<0, GV_VS>));
  hipError_t e1 = hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(&gdn_verify_kernel<1, GV_VS>));
  return (int)(e0 != hipSuccess ? e0 : e1);
}
