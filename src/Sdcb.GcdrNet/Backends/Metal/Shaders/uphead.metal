// uphead / uphead_sc: fused NHWC tail — one thread per output pixel:
//   s[c] = [relu](bilinear(x)[c]) + [relu](skip[c])          (never stored)
//   out[o] = act(b[o] + sum_c w[o][c] * s[c]),  o < oc <= 8
// uphead_sc recomputes the skip pixel from a small stride-1 'same' conv input.
// flags: bit0 relu(resize), bit1 relu(skip), bit2 has bias, bits3-4 act (1 relu, 2 sigmoid)
// cflags (_sc): bit0 conv bias, bit1 reflect pad, bit2 conv relu
struct PcUpHead {
    uint c, ih, iw, oh, ow, oc, flags;
    float shy, shx;
};
struct PcUpHeadSc {
    uint sic, kh, kw, pt, pl, cflags;
    float padv;
};
struct PcUpHeadAll {
    PcUpHead p;
    PcUpHeadSc sc;
};
constant uint MAXC = 128u;
constant uint MAXSC = 8u;

template <int SC>
static inline void uphead_body(device const half4* x,
                               device const half* s,
                               device const half* w,
                               device const float* b,
                               device half* o,
                               device const half* cw,
                               device const float* cb,
                               constant PcUpHead& p,
                               PcUpHeadSc sc,
                               uint gid, uint tid,
                               threadgroup float4* ws, threadgroup float4* cws)
{    for (uint e = tid; e < p.c * 8u; e += 256u)
    {
        uint ch = e >> 3, lane = e & 7u;
        ((threadgroup float*)ws)[e] = lane < p.oc ? float(w[lane * p.c + ch]) : 0.0f;
    }
    if (SC)
    {
        uint K = sc.kh * sc.kw * sc.sic;
        for (uint e = tid; e < K * p.c; e += 256u) ((threadgroup float*)cws)[e] = float(cw[e]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint px = gid;
    if (px >= p.oh * p.ow) return;
    uint ox = px % p.ow, oy = px / p.ow;

    float fy = (float(oy) + 0.5f) / p.shy - 0.5f;
    float fx = (float(ox) + 0.5f) / p.shx - 0.5f;
    int fy0 = int(floor(fy)), fx0 = int(floor(fx));
    float dy = fy - float(fy0), dx = fx - float(fx0);
    uint y0 = uint(clamp(fy0, 0, int(p.ih) - 1)), x0 = uint(clamp(fx0, 0, int(p.iw) - 1));
    uint y1 = uint(clamp(fy0 + 1, 0, int(p.ih) - 1)), x1 = uint(clamp(fx0 + 1, 0, int(p.iw) - 1));
    uint c4 = p.c >> 2;
    uint i00 = (y0 * p.iw + x0) * c4, i01 = (y0 * p.iw + x1) * c4;
    uint i10 = (y1 * p.iw + x0) * c4, i11 = (y1 * p.iw + x1) * c4;

    float4 sacc[MAXSC];
    uint is_ = px * c4;
    if (SC)
    {
        device const half* sx = (device const half*)s;
        bool cbias = (sc.cflags & 1u) != 0u, reflect = (sc.cflags & 2u) != 0u;
        for (uint q = 0u; q < MAXSC; q++)
            sacc[q] = cbias && q < c4 ? float4(cb[q * 4u], cb[q * 4u + 1u], cb[q * 4u + 2u], cb[q * 4u + 3u]) : float4(0.0f);
        for (uint ky = 0u; ky < sc.kh; ky++)
        {
            int iy = int(oy + ky) - int(sc.pt);
            bool vy = iy >= 0 && iy < int(p.oh);
            if (!vy && reflect) { iy = mirror1(iy, int(p.oh)); vy = true; }
            for (uint kx = 0u; kx < sc.kw; kx++)
            {
                int ix = int(ox + kx) - int(sc.pl);
                bool vx = ix >= 0 && ix < int(p.ow);
                if (!vx && reflect) { ix = mirror1(ix, int(p.ow)); vx = true; }
                bool valid = vy && vx;
                uint base = valid ? (uint(iy) * p.ow + uint(ix)) * sc.sic : 0u;
                uint kb = (ky * sc.kw + kx) * sc.sic;
                for (uint ci = 0u; ci < sc.sic; ci++)
                {
                    float xv = valid ? float(sx[base + ci]) : sc.padv;
                    uint wr = (kb + ci) * c4;
                    for (uint q = 0u; q < MAXSC; q++)
                        if (q < c4) sacc[q] += xv * cws[wr + q];
                }
            }
        }
        if ((sc.cflags & 4u) != 0u)
            for (uint q = 0u; q < MAXSC; q++) sacc[q] = max(sacc[q], float4(0.0f));
    }

    bool rr = (p.flags & 1u) != 0u, sr = (p.flags & 2u) != 0u;
    float4 a0 = float4(0.0f), a1 = float4(0.0f);
    if ((p.flags & 4u) != 0u)
    {
        a0 = float4(b[0], p.oc > 1u ? b[1] : 0.0f, p.oc > 2u ? b[2] : 0.0f, p.oc > 3u ? b[3] : 0.0f);
        if (p.oc > 4u)
            a1 = float4(b[4], p.oc > 5u ? b[5] : 0.0f, p.oc > 6u ? b[6] : 0.0f, p.oc > 7u ? b[7] : 0.0f);
    }
    for (uint q = 0u; q < c4; q++)
    {
        float4 sv;
        if (SC)
        {
            if (q >= MAXSC) break;
            sv = sacc[q];
        }
        else
            sv = float4(((device const half4*)s)[is_ + q]);
        float4 v = mix(mix(float4(x[i00 + q]), float4(x[i01 + q]), dx),
                       mix(float4(x[i10 + q]), float4(x[i11 + q]), dx), dy);
        if (rr) v = max(v, float4(0.0f));
        v += sr ? max(sv, float4(0.0f)) : sv;
        uint wb = q * 8u;
        a0 += v.x * ws[wb] + v.y * ws[wb + 2u] + v.z * ws[wb + 4u] + v.w * ws[wb + 6u];
        a1 += v.x * ws[wb + 1u] + v.y * ws[wb + 3u] + v.z * ws[wb + 5u] + v.w * ws[wb + 7u];
    }
    uint act = (p.flags >> 3) & 3u;
    if (act == 1u) { a0 = max(a0, float4(0.0f)); a1 = max(a1, float4(0.0f)); }
    else if (act == 2u) { a0 = 1.0f / (1.0f + exp(-a0)); a1 = 1.0f / (1.0f + exp(-a1)); }
    uint ob = px * p.oc;
    for (uint l = 0u; l < p.oc; l++)
        o[ob + l] = half(l < 4u ? a0[l] : a1[l - 4u]);
}

kernel void uphead(device const half4* x [[buffer(0)]],
                   device const half4* sk [[buffer(1)]],
                   device const half* w [[buffer(2)]],
                   device const float* b [[buffer(3)]],
                   device half* o [[buffer(4)]],
                   constant PcUpHead& p [[buffer(5)]],
                   uint gid [[thread_position_in_grid]],
                   uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float4 ws[MAXC * 2u];
    threadgroup float4 cws[1];
    PcUpHeadSc sc = { 0u, 0u, 0u, 0u, 0u, 0u, 0.0f };
    uphead_body<0>(x, (device const half*)sk, w, b, o, (device const half*)nullptr, (device const float*)nullptr, p, sc, gid, tid, ws, cws);
}

kernel void uphead_sc(device const half4* x [[buffer(0)]],
                      device const half* sx [[buffer(1)]],
                      device const half* w [[buffer(2)]],
                      device const float* b [[buffer(3)]],
                      device half* o [[buffer(4)]],
                      device const half* cw [[buffer(5)]],
                      device const float* cb [[buffer(6)]],
                      constant PcUpHeadAll& pa [[buffer(7)]],
                      uint gid [[thread_position_in_grid]],
                      uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float4 ws[MAXC * 2u];
    threadgroup float4 cws[1024];
    uphead_body<1>(x, sx, w, b, o, cw, cb, pa.p, pa.sc, gid, tid, ws, cws);
}
