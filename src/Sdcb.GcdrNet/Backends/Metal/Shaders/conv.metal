// Conv kernels: conv_tile (implicit-GEMM tiled fallback, conv_gen/k33/k11
// equivalent), depthwise(4), convpool(_k3) fused stem+pool. fp16 storage, fp32 math.

struct PcConv {
    uint ic, ih, iw;
    uint oc, oh, ow;
    uint kh, kw, sh, sw;
    uint pt, pl;
    uint flags;     // bit0 bias, bit1 reflectPad, bit2 relu, bit3 hasAdd, bit4 gelu
    uint pad0;
};

// conv_tile: O[px,oc] = sum_k X[px,k]*W[k,oc], k = (ky*kw+kx)*ic+ci (coalesced).
// W backend-packed [K][OC] oc-fastest. 3D dispatch: gx=x-tile(16), gy=y-tile(8),
// gz=oc-tile(64); 256 threads, each 4oc x 8px. K-chunk 32 staged.
kernel void conv_tile(device const half* x [[buffer(0)]],
                      device const half* w [[buffer(1)]],
                      device const float* b [[buffer(2)]],
                      device const half* r [[buffer(3)]],
                      device half* o [[buffer(4)]],
                      constant PcConv& p [[buffer(5)]],
                      uint3 wg [[threadgroup_position_in_grid]],
                      uint3 tp [[thread_position_in_threadgroup]])
{
    const uint tid = tp.x;
    const uint BM = 64u, BH = 8u, BW = 16u, BN = BH * BW, BK = 32u, BKP = 36u;
    threadgroup float4 Asv[BK * BM / 4];
    threadgroup float4 Bsv[BN * BKP / 4];
    uint oc0 = wg.z * BM;
    uint oy0 = wg.y * BH;
    uint ox0 = wg.x * BW;
    uint K = p.kh * p.kw * p.ic;
    uint tm = tid & 15u, tn = tid >> 4;
    float4 acc[8] = { float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f),
                      float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f) };
    bool reflect = (p.flags & 2u) != 0u;
    uint bk = tid & 31u;

    for (uint k0 = 0u; k0 < K; k0 += BK)
    {
        for (uint e = tid; e < BK * BM; e += 256u)
        {
            uint k = e >> 6, m = e & 63u;
            uint gm = oc0 + m, gk = k0 + k;
            ((threadgroup float*)Asv)[e] = (gm < p.oc && gk < K) ? float(w[gk * p.oc + gm]) : 0.0f;
        }
        uint gk = k0 + bk;
        bool kok = gk < K;
        uint tap = kok ? gk / p.ic : 0u;
        uint ch = gk - tap * p.ic;
        uint ky = tap / p.kw, kx = tap - ky * p.kw;
        for (uint n = tid >> 5; n < BN; n += 8u)
        {
            float v = 0.0f;
            uint oy = oy0 + (n >> 4), ox = ox0 + (n & 15u);
            if (kok && oy < p.oh && ox < p.ow)
            {
                int sy = int(oy * p.sh + ky) - int(p.pt);
                int sx = int(ox * p.sw + kx) - int(p.pl);
                if (reflect)
                {
                    if (uint(sy) >= p.ih) sy = mirror1(sy, int(p.ih));
                    if (uint(sx) >= p.iw) sx = mirror1(sx, int(p.iw));
                    v = float(x[(uint(sy) * p.iw + uint(sx)) * p.ic + ch]);
                }
                else if (uint(sy) < p.ih && uint(sx) < p.iw)
                    v = float(x[(uint(sy) * p.iw + uint(sx)) * p.ic + ch]);
            }
            uint si = n * BKP + bk;
            ((threadgroup float*)Bsv)[si] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0u; k < BK; k += 4u)
        {
            float4 a0 = Asv[k * 16u + tm];
            float4 a1 = Asv[(k + 1u) * 16u + tm];
            float4 a2 = Asv[(k + 2u) * 16u + tm];
            float4 a3 = Asv[(k + 3u) * 16u + tm];
            for (int j = 0; j < 8; j++)
            {
                float4 bv = Bsv[(tn * 8u + uint(j)) * (BKP / 4u) + (k >> 2)];
                acc[j] += a0 * bv.x + a1 * bv.y + a2 * bv.z + a3 * bv.w;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    bool hasBias = (p.flags & 1u) != 0u;
    uint gm0 = oc0 + tm * 4u;
    float4 bias = float4(
        hasBias && gm0 < p.oc ? b[gm0] : 0.0f,
        hasBias && gm0 + 1u < p.oc ? b[gm0 + 1u] : 0.0f,
        hasBias && gm0 + 2u < p.oc ? b[gm0 + 2u] : 0.0f,
        hasBias && gm0 + 3u < p.oc ? b[gm0 + 3u] : 0.0f);
    for (int j = 0; j < 8; j++)
    {
        uint n = tn * 8u + uint(j);
        uint oy = oy0 + (n >> 4), ox = ox0 + (n & 15u);
        if (oy >= p.oh || ox >= p.ow || gm0 >= p.oc) continue;
        uint base = (oy * p.ow + ox) * p.oc + gm0;
        uint full = gm0 + 3u < p.oc ? 4u : p.oc - gm0;
        for (uint l = 0u; l < 4u && l < full; l++)
        {
            float v = acc[j][l] + bias[l];
            if ((p.flags & 8u) != 0u) v += float(r[base + l]);
            if ((p.flags & 4u) != 0u) v = max(v, 0.0f);
            if ((p.flags & 16u) != 0u) v = 0.5f * v * (1.0f + ferf1(v * 0.7071067811865475f));
            o[base + l] = half(v);
        }
    }
}

// ---- depthwise / depthwise4: out[oy,ox,c] = bias[c] + sum w[tap,c]*x[sy,sx,c] ----
struct PcDw {
    uint c, ih, iw, oh, ow;
    uint kh, kw, sh, sw, pt, pl;
    uint flags;         // bit0 hasBias, bit1 reflect, bit2 relu, bit4 gelu
    float padVal;
};
kernel void depthwise(device const half* x [[buffer(0)]],
                      device const half* w [[buffer(1)]],
                      device const float* b [[buffer(2)]],
                      device half* o [[buffer(3)]],
                      constant PcDw& p [[buffer(4)]],
                      uint gid [[thread_position_in_grid]])
{
    if (gid >= p.c * p.oh * p.ow) return;
    uint ch = gid % p.c, px = gid / p.c;
    uint ox = px % p.ow, oy = px / p.ow;
    bool refl = (p.flags & 2u) != 0u;
    float acc = ((p.flags & 1u) != 0u) ? b[ch] : 0.0f;
    for (uint ky = 0u; ky < p.kh; ky++)
    {
        int sy = int(oy * p.sh + ky) - int(p.pt);
        for (uint kx = 0u; kx < p.kw; kx++)
        {
            int sx = int(ox * p.sw + kx) - int(p.pl);
            float wv = float(w[(ky * p.kw + kx) * p.c + ch]);
            int yy = sy, xx = sx;
            if (uint(sy) >= p.ih || uint(sx) >= p.iw)
            {
                if (!refl) { acc += wv * p.padVal; continue; }
                if (uint(sy) >= p.ih) yy = mirror1(sy, int(p.ih));
                if (uint(sx) >= p.iw) xx = mirror1(sx, int(p.iw));
            }
            acc += wv * float(x[(uint(yy) * p.iw + uint(xx)) * p.c + ch]);
        }
    }
    if ((p.flags & 4u) != 0u) acc = max(acc, 0.0f);
    if ((p.flags & 16u) != 0u) acc = 0.5f * acc * (1.0f + ferf1(acc * 0.7071067811865475f));
    o[gid] = half(acc);
}
kernel void depthwise4(device const half4* x [[buffer(0)]],
                       device const half4* w [[buffer(1)]],
                       device const float4* b [[buffer(2)]],
                       device half4* o [[buffer(3)]],
                       constant PcDw& p [[buffer(4)]],
                       uint gid [[thread_position_in_grid]])
{
    uint c4 = p.c >> 2;
    if (gid >= c4 * p.oh * p.ow) return;
    uint ch = gid % c4, px = gid / c4;
    uint ox = px % p.ow, oy = px / p.ow;
    bool refl = (p.flags & 2u) != 0u;
    float4 acc = ((p.flags & 1u) != 0u) ? b[ch] : float4(0.0f);
    for (uint ky = 0u; ky < p.kh; ky++)
    {
        int sy = int(oy * p.sh + ky) - int(p.pt);
        for (uint kx = 0u; kx < p.kw; kx++)
        {
            int sx = int(ox * p.sw + kx) - int(p.pl);
            float4 wv = float4(w[(ky * p.kw + kx) * c4 + ch]);
            int yy = sy, xx = sx;
            if (uint(sy) >= p.ih || uint(sx) >= p.iw)
            {
                if (!refl) { acc += wv * p.padVal; continue; }
                if (uint(sy) >= p.ih) yy = mirror1(sy, int(p.ih));
                if (uint(sx) >= p.iw) xx = mirror1(sx, int(p.iw));
            }
            acc += wv * float4(x[(uint(yy) * p.iw + uint(xx)) * c4 + ch]);
        }
    }
    if ((p.flags & 4u) != 0u) acc = max(acc, float4(0.0f));
    if ((p.flags & 16u) != 0u)
        for (int l = 0; l < 4; l++) acc[l] = 0.5f * acc[l] * (1.0f + ferf1(acc[l] * 0.7071067811865475f));
    o[gid] = half4(acc);
}

// ---- convpool / convpool_k3: small-ic stem conv + fused 2x2/2 maxpool ----
struct PcConvPool {
    uint ic, ih, iw, oc, ph, pw, kh, kw, pt, pl, flags;
    float padv;
};
kernel void convpool(device const half* x [[buffer(0)]],
                     device const half* w [[buffer(1)]],
                     device const float* b [[buffer(2)]],
                     device half* o [[buffer(3)]],
                     constant PcConvPool& p [[buffer(4)]],
                     uint gid [[thread_position_in_grid]],
                     uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float4 ws[1024];
    uint K = p.kh * p.kw * p.ic, oc4 = p.oc >> 2;
    for (uint e = tid; e < K * p.oc; e += 256u) ((threadgroup float*)ws)[e] = float(w[e]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint groups = p.oc >> 3;
    if (gid >= p.ph * p.pw * groups) return;
    uint g = gid % groups, pp = gid / groups;
    uint pxx = pp % p.pw, pyy = pp / p.pw;
    bool reflect = (p.flags & 2u) != 0u, crelu = (p.flags & 4u) != 0u;
    uint n0 = g * 8u;
    float4 b0 = float4(0.0f), b1 = float4(0.0f);
    if ((p.flags & 1u) != 0u)
    {
        b0 = float4(b[n0], b[n0 + 1u], b[n0 + 2u], b[n0 + 3u]);
        b1 = float4(b[n0 + 4u], b[n0 + 5u], b[n0 + 6u], b[n0 + 7u]);
    }
    float4 m0 = float4(-3.4e38f), m1 = float4(-3.4e38f);
    for (uint d = 0u; d < 4u; d++)
    {
        uint oy = pyy * 2u + (d >> 1), ox = pxx * 2u + (d & 1u);
        float4 a0 = b0, a1 = b1;
        for (uint ky = 0u; ky < p.kh; ky++)
        {
            int iy = int(oy + ky) - int(p.pt);
            bool vy = iy >= 0 && iy < int(p.ih);
            if (!vy && reflect) { iy = mirror1(iy, int(p.ih)); vy = true; }
            for (uint kx = 0u; kx < p.kw; kx++)
            {
                int ix = int(ox + kx) - int(p.pl);
                bool vx = ix >= 0 && ix < int(p.iw);
                if (!vx && reflect) { ix = mirror1(ix, int(p.iw)); vx = true; }
                bool valid = vy && vx;
                uint base = valid ? (uint(iy) * p.iw + uint(ix)) * p.ic : 0u;
                uint kb = (ky * p.kw + kx) * p.ic;
                for (uint ci = 0u; ci < p.ic; ci++)
                {
                    float xv = valid ? float(x[base + ci]) : p.padv;
                    uint wr = (kb + ci) * oc4 + g * 2u;
                    a0 += xv * ws[wr];
                    a1 += xv * ws[wr + 1u];
                }
            }
        }
        if (crelu) { a0 = max(a0, float4(0.0f)); a1 = max(a1, float4(0.0f)); }
        m0 = max(m0, a0); m1 = max(m1, a1);
    }
    if ((p.flags & 8u) != 0u) { m0 = max(m0, float4(0.0f)); m1 = max(m1, float4(0.0f)); }
    uint ob = pp * p.oc + n0;
    for (uint l = 0u; l < 4u; l++) { o[ob + l] = half(m0[l]); o[ob + 4u + l] = half(m1[l]); }
}

kernel void convpool_k3(device const half* x [[buffer(0)]],
                        device const half* w [[buffer(1)]],
                        device const float* b [[buffer(2)]],
                        device half* o [[buffer(3)]],
                        constant PcConvPool& p [[buffer(4)]],
                        uint gid [[thread_position_in_grid]],
                        uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float4 ws[1024];
    uint K = 9u * p.ic, oc4 = p.oc >> 2;
    for (uint e = tid; e < K * p.oc; e += 256u) ((threadgroup float*)ws)[e] = float(w[e]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint groups = p.oc >> 3;
    if (gid >= p.ph * p.pw * groups) return;
    uint g = gid % groups, pp = gid / groups;
    uint pxx = pp % p.pw, pyy = pp / p.pw;
    bool reflect = (p.flags & 2u) != 0u, crelu = (p.flags & 4u) != 0u;
    int ro[4], co[4];
    bool rv[4], cv[4];
    for (uint r = 0u; r < 4u; r++)
    {
        int iy = int(pyy * 2u + r) - int(p.pt);
        bool vy = iy >= 0 && iy < int(p.ih);
        if (!vy && reflect) { iy = mirror1(iy, int(p.ih)); vy = true; }
        rv[r] = vy; ro[r] = vy ? iy * int(p.iw) : 0;
        int ix = int(pxx * 2u + r) - int(p.pl);
        bool vx = ix >= 0 && ix < int(p.iw);
        if (!vx && reflect) { ix = mirror1(ix, int(p.iw)); vx = true; }
        cv[r] = vx; co[r] = vx ? ix : 0;
    }
    uint n0 = g * 8u;
    float4 b0 = float4(0.0f), b1 = float4(0.0f);
    if ((p.flags & 1u) != 0u)
    {
        b0 = float4(b[n0], b[n0 + 1u], b[n0 + 2u], b[n0 + 3u]);
        b1 = float4(b[n0 + 4u], b[n0 + 5u], b[n0 + 6u], b[n0 + 7u]);
    }
    float4 a0[4], a1[4];
    for (uint d = 0u; d < 4u; d++) { a0[d] = b0; a1[d] = b1; }
    for (uint r = 0u; r < 4u; r++)
    {
        float xr[4][8];
        for (uint c = 0u; c < 4u; c++)
        {
            bool v = rv[r] && cv[c];
            uint base = uint(ro[r] + co[c]) * p.ic;
            for (uint ci = 0u; ci < 8u; ci++)
                xr[c][ci] = ci < p.ic ? (v ? float(x[base + ci]) : p.padv) : 0.0f;
        }
        for (uint dy = 0u; dy < 2u; dy++)
        {
            if (r < dy || r - dy > 2u) continue;
            uint ky = r - dy;
            for (uint kx = 0u; kx < 3u; kx++)
            {
                uint kb = (ky * 3u + kx) * p.ic;
                for (uint ci = 0u; ci < 8u; ci++)
                {
                    if (ci >= p.ic) break;
                    uint wr = (kb + ci) * oc4 + g * 2u;
                    float4 w0 = ws[wr], w1 = ws[wr + 1u];
                    for (uint dx = 0u; dx < 2u; dx++)
                    {
                        float xv = xr[kx + dx][ci];
                        a0[dy * 2u + dx] += xv * w0;
                        a1[dy * 2u + dx] += xv * w1;
                    }
                }
            }
        }
    }
    float4 m0 = float4(-3.4e38f), m1 = float4(-3.4e38f);
    for (uint d = 0u; d < 4u; d++)
    {
        if (crelu) { a0[d] = max(a0[d], float4(0.0f)); a1[d] = max(a1[d], float4(0.0f)); }
        m0 = max(m0, a0[d]); m1 = max(m1, a1[d]);
    }
    if ((p.flags & 8u) != 0u) { m0 = max(m0, float4(0.0f)); m1 = max(m1, float4(0.0f)); }
    uint ob = pp * p.oc + n0;
    for (uint l = 0u; l < 4u; l++) { o[ob + l] = half(m0[l]); o[ob + 4u + l] = half(m1[l]); }
}
