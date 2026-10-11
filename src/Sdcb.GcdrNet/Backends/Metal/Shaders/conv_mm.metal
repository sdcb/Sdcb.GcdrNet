// conv_mm: simdgroup-MMA implicit-GEMM NHWC conv (Metal port of conv_cm.comp).
// A gathers input pixels through tap-major im2col addressing — no im2col
// materialization; W is backend-packed [oc][Kd] tap-major, Kd = K rounded up
// to 8 (zero-filled rows make pad k-slices contribute 0 for any gathered
// pixel). BM = 64 output pixels x BN output channels, 8 simdgroups
// (256 threads), fp16 storage + simdgroup fp16xfp32 MMA.
// cflags: bit0 bias, bit1 residual, bit2 reflect pad, bits4-6 act (1 relu, 2 gelu).
struct PcConvMm {
    uint mDim, nDim, kDim, flags;
    uint outW, inW, inH, cin, kW;
    uint sH, sW, pT, pL, kReal;
};

template <int BN, int SCK>
static inline void conv_mm_body(device const half* x,
                                device const half* w,
                                device const float* bias,
                                device const half* res,
                                device half* o,
                                constant PcConvMm& p,
                                uint2 wg, uint tid,
                                threadgroup half* tA, threadgroup half* tB,
                                threadgroup float* outst)
{
    const uint JN = BN / 16u;        // acc columns per simdgroup row-half
    uint sgid = tid >> 5;
    uint sm = (sgid & 3u) * 16u;     // 4 rows of 16 -> 64 M
    uint sn = (sgid >> 2) * (BN / 2u); // 2 cols of BN/2
    uint m0 = wg.x * 64u + sm;       // m tiles on gx (grid allows 2^31 in x)
    uint n0 = wg.y * BN + sn;

    simdgroup_float8x8 acc[2][8];
    for (uint i = 0u; i < 2u; i++) for (uint j = 0u; j < 8u; j++)
        acc[i][j] = simdgroup_float8x8(0.0f);

    bool reflect = (p.flags & 4u) != 0u;

    for (uint kb = 0u; kb < p.kDim; kb += 16u)
    {
        if (SCK == 0)
        {
            // vec4 gather: 64 rows x 16 k as one half4 per thread
            uint am = tid >> 2, ak = (tid & 3u) * 4u;
            uint gm = wg.x * 64u + am;
            uint gk = kb + ak;
            half4 av = half4(0.0h);
            if (gm < p.mDim && gk < p.kDim)
            {
                uint tap = gk / p.cin, ci = gk - tap * p.cin;
                uint dy = tap / p.kW, dx = tap - dy * p.kW;
                uint oy = gm / p.outW, ox = gm - oy * p.outW;
                int iy = int(oy * p.sH + dy) - int(p.pT);
                int ix = int(ox * p.sW + dx) - int(p.pL);
                if (reflect)
                {
                    if (uint(iy) >= p.inH) iy = mirror1(iy, int(p.inH));
                    if (uint(ix) >= p.inW) ix = mirror1(ix, int(p.inW));
                    av = *(device const half4*)(x + (((uint(iy) * p.inW + uint(ix)) * (p.cin >> 2)) + (ci >> 2)) * 4u);
                }
                else if (iy >= 0 && iy < int(p.inH) && ix >= 0 && ix < int(p.inW))
                    av = *(device const half4*)(x + (((uint(iy) * p.inW + uint(ix)) * (p.cin >> 2)) + (ci >> 2)) * 4u);
            }
            *(threadgroup half4*)(tA + am * 17 + ak) = av;
        }
        else
        {
            // scalar gather for cin % 4 != 0 (RGB stems): 64x16 one half per thread x4
            for (uint e = tid; e < 64u * 16u; e += 256u)
            {
                uint am = e >> 4, ak = e & 15u;
                uint gm = wg.x * 64u + am;
                uint gk = kb + ak;
                half av = 0.0h;
                if (gm < p.mDim && gk < p.kReal)
                {
                    uint tap = gk / p.cin, ci = gk - tap * p.cin;
                    uint dy = tap / p.kW, dx = tap - dy * p.kW;
                    uint oy = gm / p.outW, ox = gm - oy * p.outW;
                    int iy = int(oy * p.sH + dy) - int(p.pT);
                    int ix = int(ox * p.sW + dx) - int(p.pL);
                    if (reflect)
                    {
                        if (uint(iy) >= p.inH) iy = mirror1(iy, int(p.inH));
                        if (uint(ix) >= p.inW) ix = mirror1(ix, int(p.inW));
                        av = x[(uint(iy) * p.inW + uint(ix)) * p.cin + ci];
                    }
                    else if (iy >= 0 && iy < int(p.inH) && ix >= 0 && ix < int(p.inW))
                        av = x[(uint(iy) * p.inW + uint(ix)) * p.cin + ci];
                }
                tA[am * 17 + ak] = av;
            }
        }
        // B stage: [N][Kd] rows, one half4 per thread covers 64x16; BN>64 loops
        for (uint rr = tid >> 2; rr < uint(BN); rr += 64u)
        {
            uint bk = (tid & 3u) * 4u;
            uint gn = wg.y * BN + rr;
            uint gkB = kb + bk;
            half4 wv = half4(0.0h);
            if (gn < p.nDim && gkB < p.kDim)
                wv = *(device const half4*)(w + (ulong)gn * p.kDim + gkB);
            *(threadgroup half4*)(tB + rr * 17 + bk) = wv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0u; kk < 16u; kk += 8u)
        {
            simdgroup_half8x8 a0, a1;
            simdgroup_load(a0, tA + sm * 17 + kk, 17);
            simdgroup_load(a1, tA + (sm + 8u) * 17 + kk, 17);
            simdgroup_half8x8 bfr[8];
            for (uint j = 0u; j < JN; j++)
                simdgroup_load(bfr[j], tB + (sn + j * 8u) * 17 + kk, 17, ulong2(0, 0), true);
            for (uint j = 0u; j < JN; j++)
            {
                simdgroup_multiply_accumulate(acc[0][j], a0, bfr[j], acc[0][j]);
                simdgroup_multiply_accumulate(acc[1][j], a1, bfr[j], acc[1][j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint lane = tid & 31u;
    for (uint i = 0u; i < 2u; i++)
        for (uint j = 0u; j < JN; j++)
        {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_store(acc[i][j], outst + sgid * 64, 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint r = lane >> 2, c2 = (lane & 3u) * 2u;
            for (uint e = 0u; e < 2u; e++)
            {
                uint mm = m0 + i * 8u + r;
                uint nn = n0 + j * 8u + c2 + e;
                if (mm >= p.mDim || nn >= p.nDim) continue;
                float v = outst[sgid * 64 + r * 8 + c2 + e];
                if ((p.flags & 1u) != 0u) v += bias[nn];
                if ((p.flags & 2u) != 0u) v += float(res[mm * p.nDim + nn]);
                o[mm * p.nDim + nn] = half(cact(v, p.flags));
            }
        }
}

#define CONV_MM_INST(BN, SCK) \
kernel void conv_mm_##BN##_##SCK(device const half* x [[buffer(0)]], \
    device const half* w [[buffer(1)]], device const float* bias [[buffer(2)]], \
    device const half* res [[buffer(3)]], device half* o [[buffer(4)]], \
    constant PcConvMm& p [[buffer(5)]], uint2 wg [[threadgroup_position_in_grid]], \
    uint2 tp [[thread_position_in_threadgroup]]) \
{ \
    threadgroup half tA[64 * 17]; \
    threadgroup half tB[BN * 17]; \
    threadgroup float outst[8 * 64]; \
    conv_mm_body<BN, SCK>(x, w, bias, res, o, p, wg, tp.x, tA, tB, outst); \
}

CONV_MM_INST(128, 0)
CONV_MM_INST(64, 0)
CONV_MM_INST(32, 0)
CONV_MM_INST(128, 1)
CONV_MM_INST(64, 1)
CONV_MM_INST(32, 1)

// ---- gemm: C[M,N] = A[M,K]*B + bias[N]; B [K,N] or [N,K] (transB) ----
// tiled 64x64x16, 256 threads -> 4x4 each (fallback; MMA path uses conv_mm).
struct PcGemm { uint m, n, k, flags; };   // bit0 transB bit1 bias bit2 relu bit3 gelu
kernel void gemm(device const half* a [[buffer(0)]],
                 device const half* b [[buffer(1)]],
                 device const float* bs [[buffer(2)]],
                 device half* o [[buffer(3)]],
                 constant PcGemm& p [[buffer(4)]],
                 uint2 wg [[threadgroup_position_in_grid]],
                 uint2 tp [[thread_position_in_threadgroup]])
{
    const uint tid = tp.x;
    const uint BM = 64u, BN = 64u, BK = 16u;
    threadgroup float As[BK * BM];
    threadgroup float Bs[BK * BN];
    uint m0 = wg.y * BM;
    uint n0 = wg.x * BN;
    uint ty = tid >> 4, tx = tid & 15u;
    bool transB = (p.flags & 1u) != 0u;
    float acc[4][4];
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;
    for (uint k0 = 0u; k0 < p.k; k0 += BK)
    {
        for (uint e = tid; e < BK * BM; e += 256u)
        {
            uint kk = e >> 6, mm = e & 63u;
            uint gm = m0 + mm, gk = k0 + kk;
            As[e] = (gm < p.m && gk < p.k) ? float(a[gm * p.k + gk]) : 0.0f;
        }
        for (uint e = tid; e < BK * BN; e += 256u)
        {
            uint kk = e >> 6, nn = e & 63u;
            uint gn = n0 + nn, gk = k0 + kk;
            float v = 0.0f;
            if (gn < p.n && gk < p.k)
                v = float(transB ? b[gn * p.k + gk] : b[gk * p.n + gn]);
            Bs[e] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0u; kk < BK; kk++)
        {
            float av[4], bv[4];
            for (int i = 0; i < 4; i++) av[i] = As[kk * BM + ty * 4u + uint(i)];
            for (int j = 0; j < 4; j++) bv[j] = Bs[kk * BN + tx * 4u + uint(j)];
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    acc[i][j] += av[i] * bv[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    bool hasBias = (p.flags & 2u) != 0u;
    for (int i = 0; i < 4; i++)
    {
        uint gm = m0 + ty * 4u + uint(i);
        if (gm >= p.m) continue;
        for (int j = 0; j < 4; j++)
        {
            uint gn = n0 + tx * 4u + uint(j);
            if (gn >= p.n) continue;
            float v = acc[i][j];
            if (hasBias) v += bs[gn];
            if ((p.flags & 4u) != 0u) v = max(v, 0.0f);
            if ((p.flags & 8u) != 0u) v = 0.5f * v * (1.0f + ferf1(v * 0.7071067811865475f));
            o[gm * p.n + gn] = half(v);
        }
    }
}
