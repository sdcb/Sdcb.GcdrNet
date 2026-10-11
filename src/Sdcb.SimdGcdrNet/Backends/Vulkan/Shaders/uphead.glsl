// Fused NHWC network tail (ExecOp.UpHead), one thread per output pixel:
//   s[c]   = [relu](bilinear(x)[c]) + [relu](skip[c])        (fp32, never stored)
//   out[o] = act(b[o] + sum_c w[o][c] * s[c]),  o < oc <= 8
// Replaces resize -> relu -> add -> 1x1 conv -> sigmoid, whose full-resolution
// intermediates would each round-trip device memory. Channels are read as
// 4-vectors: c % 4 == 0 (the compiler only fuses such tails; arena offsets are
// 64B-aligned).
// SKIPCONV (uphead_sc): binding 1 is the input of a small stride-1 'same' conv
// (sic < 8 channels, c <= 32 outputs) and each thread recomputes its skip pixel
// from it, so the full-resolution skip tensor is never stored either.
// flags: bit0 relu(resize), bit1 relu(skip), bit2 has bias, bits3-4 act (1 relu, 2 sigmoid)
// cflags (SKIPCONV): bit0 conv bias, bit1 reflect pad, bit2 conv relu
#extension GL_EXT_control_flow_attributes : require
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer X { ET4 x[]; };     // [ih][iw][c/4]
#ifdef SKIPCONV
layout(set = 0, binding = 1) readonly buffer S { ET sx[]; };     // [oh][ow][sic] conv input
#else
layout(set = 0, binding = 1) readonly buffer S { ET4 sk[]; };    // [oh][ow][c/4]
#endif
layout(set = 0, binding = 2) readonly buffer W { ET w[]; };      // [oc][c]
layout(set = 0, binding = 3) readonly buffer B { float b[]; };   // [oc] (dummy if none)
layout(set = 0, binding = 4) writeonly buffer O { ET o[]; };     // [oh][ow][oc]
#ifdef SKIPCONV
layout(set = 0, binding = 5) readonly buffer CW { ET cw[]; };    // [K][c], k = (ky*kw + kx)*sic + ci
layout(set = 0, binding = 6) readonly buffer CB { float cb[]; }; // [c] (dummy if none)
#endif

layout(push_constant) uniform PC {
    uint c, ih, iw, oh, ow, oc, flags;
    float shy, shx;
#ifdef SKIPCONV
    uint sic, kh, kw, pt, pl, cflags;
    float padv;
#endif
} p;

const uint MAXC = 128u;
shared vec4 ws[MAXC * 2u];        // [c][2]: oc lanes 0-3, 4-7 (zero-padded)
#ifdef SKIPCONV
const uint MAXSC = 8u;            // skip conv outputs / 4
shared vec4 cws[1024];            // [K][c/4], K * c <= 4096

int mirror(int i, int d) { if (d <= 1) return 0; while (i < 0 || i >= d) i = i < 0 ? -i : 2 * d - 2 - i; return i; }
#endif

#if defined(SKIPCONV) && defined(NP)
// NP horizontally adjacent pixels per invocation (wide-wave GPUs): every staged weight
// read from shared memory (skip conv and head) feeds NP pixels instead of one. Same
// per-pixel arithmetic and summation order as the one-pixel main below.
void main()
{
    uint tid = gl_LocalInvocationID.x;
    for (uint e = tid; e < p.c * 8u; e += 256u)
    {
        uint ch = e >> 3, lane = e & 7u;
        ws[e >> 2][e & 3u] = lane < p.oc ? float(w[lane * p.c + ch]) : 0.0;
    }
    uint K = p.kh * p.kw * p.sic;
    for (uint e = tid; e < K * p.c; e += 256u) cws[e >> 2][e & 3u] = float(cw[e]);
    barrier();

    uint gw = (p.ow + NP - 1u) / NP;
    uint pg = gl_GlobalInvocationID.x;
    if (pg >= p.oh * gw) return;
    uint oy = pg / gw, ox0 = (pg % gw) * NP;
    uint c4 = p.c >> 2;

    vec4 sacc[NP * MAXSC];
    bool cbias = (p.cflags & 1u) != 0u, reflect = (p.cflags & 2u) != 0u;
    [[unroll]] for (uint q = 0u; q < MAXSC; q++)
    {
        vec4 bq = cbias && q < c4 ? vec4(cb[q * 4u], cb[q * 4u + 1u], cb[q * 4u + 2u], cb[q * 4u + 3u]) : vec4(0.0);
        [[unroll]] for (uint u = 0u; u < NP; u++) sacc[u * MAXSC + q] = bq;
    }
    for (uint ky = 0u; ky < p.kh; ky++)
    {
        int iy = int(oy + ky) - int(p.pt);
        bool vy = iy >= 0 && iy < int(p.oh);
        if (!vy && reflect) { iy = mirror(iy, int(p.oh)); vy = true; }
        for (uint kx = 0u; kx < p.kw; kx++)
        {
            uint base[NP];
            bool valid[NP];
            [[unroll]] for (uint u = 0u; u < NP; u++)
            {
                int ix = int(ox0 + u + kx) - int(p.pl);
                bool vx = ix >= 0 && ix < int(p.ow);
                if (!vx && reflect) { ix = mirror(ix, int(p.ow)); vx = true; }
                valid[u] = vy && vx;
                base[u] = valid[u] ? (uint(iy) * p.ow + uint(ix)) * p.sic : 0u;
            }
            uint kb = (ky * p.kw + kx) * p.sic;
            for (uint ci = 0u; ci < p.sic; ci++)
            {
                float xv[NP];
                [[unroll]] for (uint u = 0u; u < NP; u++) xv[u] = valid[u] ? float(sx[base[u] + ci]) : p.padv;
                uint wr = (kb + ci) * c4;
                [[unroll]] for (uint q = 0u; q < MAXSC; q++)
                    if (q < c4)
                    {
                        vec4 wq = cws[wr + q];
                        [[unroll]] for (uint u = 0u; u < NP; u++) sacc[u * MAXSC + q] += xv[u] * wq;
                    }
            }
        }
    }
    if ((p.cflags & 4u) != 0u)
        [[unroll]] for (uint q = 0u; q < NP * MAXSC; q++) sacc[q] = max(sacc[q], vec4(0.0));

    bool rr = (p.flags & 1u) != 0u, sr = (p.flags & 2u) != 0u;
    vec4 bh0 = vec4(0.0), bh1 = vec4(0.0);
    if ((p.flags & 4u) != 0u)
    {
        bh0 = vec4(b[0], p.oc > 1u ? b[1] : 0.0, p.oc > 2u ? b[2] : 0.0, p.oc > 3u ? b[3] : 0.0);
        if (p.oc > 4u)
            bh1 = vec4(b[4], p.oc > 5u ? b[5] : 0.0, p.oc > 6u ? b[6] : 0.0, p.oc > 7u ? b[7] : 0.0);
    }
    float fy = (float(oy) + 0.5) / p.shy - 0.5;
    int fy0 = int(floor(fy));
    float dy = fy - float(fy0);
    uint y0 = uint(clamp(fy0, 0, int(p.ih) - 1)), y1 = uint(clamp(fy0 + 1, 0, int(p.ih) - 1));
    uint i00[NP], i01[NP], i10[NP], i11[NP];
    float dx[NP];
    vec4 a0[NP], a1[NP];
    [[unroll]] for (uint u = 0u; u < NP; u++)
    {
        float fx = (float(ox0 + u) + 0.5) / p.shx - 0.5;
        int fx0 = int(floor(fx));
        dx[u] = fx - float(fx0);
        uint x0 = uint(clamp(fx0, 0, int(p.iw) - 1)), x1 = uint(clamp(fx0 + 1, 0, int(p.iw) - 1));
        i00[u] = (y0 * p.iw + x0) * c4; i01[u] = (y0 * p.iw + x1) * c4;
        i10[u] = (y1 * p.iw + x0) * c4; i11[u] = (y1 * p.iw + x1) * c4;
        a0[u] = bh0; a1[u] = bh1;
    }
    [[unroll]] for (uint q = 0u; q < MAXSC; q++)
    {
        if (q < c4)
        {
            uint wb = q * 8u;   // 4 channels x 2 vec4
            vec4 h0 = ws[wb], h1 = ws[wb + 1u], h2 = ws[wb + 2u], h3 = ws[wb + 3u];
            vec4 h4 = ws[wb + 4u], h5 = ws[wb + 5u], h6 = ws[wb + 6u], h7 = ws[wb + 7u];
            [[unroll]] for (uint u = 0u; u < NP; u++)
            {
                vec4 s = sacc[u * MAXSC + q];
                vec4 v = mix(mix(vec4(x[i00[u] + q]), vec4(x[i01[u] + q]), dx[u]),
                             mix(vec4(x[i10[u] + q]), vec4(x[i11[u] + q]), dx[u]), dy);
                if (rr) v = max(v, vec4(0.0));
                v += sr ? max(s, vec4(0.0)) : s;
                a0[u] += v.x * h0 + v.y * h2 + v.z * h4 + v.w * h6;
                a1[u] += v.x * h1 + v.y * h3 + v.z * h5 + v.w * h7;
            }
        }
    }
    uint act = (p.flags >> 3) & 3u;
    [[unroll]] for (uint u = 0u; u < NP; u++)
    {
        uint ox = ox0 + u;
        if (ox < p.ow)
        {
            vec4 r0 = a0[u], r1 = a1[u];
            if (act == 1u) { r0 = max(r0, vec4(0.0)); r1 = max(r1, vec4(0.0)); }
            else if (act == 2u) { r0 = 1.0 / (1.0 + exp(-r0)); r1 = 1.0 / (1.0 + exp(-r1)); }
            uint ob = (oy * p.ow + ox) * p.oc;
            for (uint l = 0u; l < p.oc; l++)
                o[ob + l] = ET(l < 4u ? r0[l] : r1[l - 4u]);
        }
    }
}
#else
void main()
{
    uint tid = gl_LocalInvocationID.x;
    for (uint e = tid; e < p.c * 8u; e += 256u)
    {
        uint ch = e >> 3, lane = e & 7u;
        ws[e >> 2][e & 3u] = lane < p.oc ? float(w[lane * p.c + ch]) : 0.0;
    }
#ifdef SKIPCONV
    uint K = p.kh * p.kw * p.sic;
    for (uint e = tid; e < K * p.c; e += 256u) cws[e >> 2][e & 3u] = float(cw[e]);
#endif
    barrier();

    uint px = gl_GlobalInvocationID.x;
    if (px >= p.oh * p.ow) return;
    uint ox = px % p.ow, oy = px / p.ow;

    float fy = (float(oy) + 0.5) / p.shy - 0.5;
    float fx = (float(ox) + 0.5) / p.shx - 0.5;
    int fy0 = int(floor(fy)), fx0 = int(floor(fx));
    float dy = fy - float(fy0), dx = fx - float(fx0);
    uint y0 = uint(clamp(fy0, 0, int(p.ih) - 1)), x0 = uint(clamp(fx0, 0, int(p.iw) - 1));
    uint y1 = uint(clamp(fy0 + 1, 0, int(p.ih) - 1)), x1 = uint(clamp(fx0 + 1, 0, int(p.iw) - 1));
    uint c4 = p.c >> 2;
    uint i00 = (y0 * p.iw + x0) * c4, i01 = (y0 * p.iw + x1) * c4;
    uint i10 = (y1 * p.iw + x0) * c4, i11 = (y1 * p.iw + x1) * c4;

#ifdef SKIPCONV
    // skip pixel = conv(sx) at (oy, ox), all c channels in registers
    vec4 sacc[MAXSC];
    bool cbias = (p.cflags & 1u) != 0u, reflect = (p.cflags & 2u) != 0u;
    [[unroll]] for (uint q = 0u; q < MAXSC; q++)
        sacc[q] = cbias && q < c4 ? vec4(cb[q * 4u], cb[q * 4u + 1u], cb[q * 4u + 2u], cb[q * 4u + 3u]) : vec4(0.0);
    for (uint ky = 0u; ky < p.kh; ky++)
    {
        int iy = int(oy + ky) - int(p.pt);
        bool vy = iy >= 0 && iy < int(p.oh);
        if (!vy && reflect) { iy = mirror(iy, int(p.oh)); vy = true; }
        for (uint kx = 0u; kx < p.kw; kx++)
        {
            int ix = int(ox + kx) - int(p.pl);
            bool vx = ix >= 0 && ix < int(p.ow);
            if (!vx && reflect) { ix = mirror(ix, int(p.ow)); vx = true; }
            bool valid = vy && vx;
            uint base = valid ? (uint(iy) * p.ow + uint(ix)) * p.sic : 0u;
            uint kb = (ky * p.kw + kx) * p.sic;
            for (uint ci = 0u; ci < p.sic; ci++)
            {
                float xv = valid ? float(sx[base + ci]) : p.padv;
                uint wr = (kb + ci) * c4;
                [[unroll]] for (uint q = 0u; q < MAXSC; q++)
                    if (q < c4) sacc[q] += xv * cws[wr + q];
            }
        }
    }
    if ((p.cflags & 4u) != 0u)
        [[unroll]] for (uint q = 0u; q < MAXSC; q++) sacc[q] = max(sacc[q], vec4(0.0));
#else
    uint is_ = px * c4;
#endif

    bool rr = (p.flags & 1u) != 0u, sr = (p.flags & 2u) != 0u;
    vec4 a0 = vec4(0.0), a1 = vec4(0.0);
    if ((p.flags & 4u) != 0u)
    {
        a0 = vec4(b[0], p.oc > 1u ? b[1] : 0.0, p.oc > 2u ? b[2] : 0.0, p.oc > 3u ? b[3] : 0.0);
        if (p.oc > 4u)
            a1 = vec4(b[4], p.oc > 5u ? b[5] : 0.0, p.oc > 6u ? b[6] : 0.0, p.oc > 7u ? b[7] : 0.0);
    }
#ifdef SKIPCONV
    [[unroll]] for (uint q = 0u; q < MAXSC; q++)
    {
        if (q >= c4) break;
        vec4 s = sacc[q];
#else
    for (uint q = 0u; q < c4; q++)
    {
        vec4 s = vec4(sk[is_ + q]);
#endif
        vec4 v = mix(mix(vec4(x[i00 + q]), vec4(x[i01 + q]), dx),
                     mix(vec4(x[i10 + q]), vec4(x[i11 + q]), dx), dy);
        if (rr) v = max(v, vec4(0.0));
        v += sr ? max(s, vec4(0.0)) : s;
        uint wb = q * 8u;   // 4 channels x 2 vec4
        a0 += v.x * ws[wb] + v.y * ws[wb + 2u] + v.z * ws[wb + 4u] + v.w * ws[wb + 6u];
        a1 += v.x * ws[wb + 1u] + v.y * ws[wb + 3u] + v.z * ws[wb + 5u] + v.w * ws[wb + 7u];
    }
    uint act = (p.flags >> 3) & 3u;
    if (act == 1u) { a0 = max(a0, vec4(0.0)); a1 = max(a1, vec4(0.0)); }
    else if (act == 2u) { a0 = 1.0 / (1.0 + exp(-a0)); a1 = 1.0 / (1.0 + exp(-a1)); }
    uint ob = px * p.oc;
    for (uint l = 0u; l < p.oc; l++)
        o[ob + l] = ET(l < 4u ? a0[l] : a1[l - 4u]);
}
#endif
