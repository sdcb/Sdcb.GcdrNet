// Pipeline glue + elementwise/shape kernels. All activation storage is fp16
// (half/half4); math fp32 — see common.metal.

// ---- bgr2nhwc: BGR u8 HWC -> NHWC [h][w][3] /255, top/left replicate pad ----
struct PcPre { uint oh, ow, ih, iw, pt, pl, stride; };
kernel void bgr2nhwc(device const uint* bgr [[buffer(0)]],
                     device half* o [[buffer(1)]],
                     constant PcPre& p [[buffer(2)]],
                     uint2 gid [[thread_position_in_grid]])
{
    uint ox = gid.x;
    if (ox >= p.ow) return;
    uint oy = gid.y;
    uint sy = uint(clamp(int(oy) - int(p.pt), 0, int(p.ih) - 1));
    uint sx = uint(clamp(int(ox) - int(p.pl), 0, int(p.iw) - 1));
    uint s = sy * p.stride + sx * 3u, d = (oy * p.ow + ox) * 3u;
    for (uint c = 0u; c < 3u; c++)
    {
        uint bi = s + c;
        float v = float((bgr[bi >> 2] >> ((bi & 3u) * 8u)) & 0xFFu);
        o[d + c] = half(v / 255.0f);
    }
}

// ---- div6: dst[px,0..2] = I_s; dst[px,3..5] = clamp(I_s / max(I_sm,1e-6),0,1) ----
struct PcDiv6 { uint hw; };
kernel void div6(device const half* x [[buffer(0)]],
                 device const half* s [[buffer(1)]],
                 device half* o [[buffer(2)]],
                 constant PcDiv6& p [[buffer(3)]],
                 uint gid [[thread_position_in_grid]])
{
    if (gid >= p.hw) return;
    for (uint c = 0u; c < 3u; c++)
    {
        float v = float(x[gid * 3u + c]);
        o[gid * 6u + c] = half(v);
        o[gid * 6u + 3u + c] = half(clamp(v / max(float(s[gid * 3u + c]), 1e-6f), 0.0f, 1.0f));
    }
}

// ---- nhwc2bgr: NHWC [h][w][3] -> u8 BGR HWC, crop (pt,pl); one thread packs 4 bytes ----
struct PcPost { uint h, w, ih, iw, pt, pl, nwords; };
kernel void nhwc2bgr(device const half* x [[buffer(0)]],
                     device uint* o [[buffer(1)]],
                     constant PcPost& p [[buffer(2)]],
                     uint gid [[thread_position_in_grid]])
{
    if (gid >= p.nwords) return;
    uint byteIdx = gid * 4u;
    uint byteTotal = p.ih * p.iw * 3u;
    uint px = byteIdx / 3u, c = byteIdx - px * 3u;
    uint dy = px / p.iw, dx = px - dy * p.iw;
    uint packed = 0u;
    for (uint j = 0u; j < 4u; j++)
    {
        uint val = 0u;
        if (byteIdx < byteTotal)
        {
            float f = float(x[((dy + p.pt) * p.w + dx + p.pl) * 3u + c]) * 255.0f + 0.5f;
            val = uint(clamp(f, 0.0f, 255.0f));
        }
        packed |= val << (j * 8u);
        byteIdx++;
        if (++c == 3u) { c = 0u; if (++dx == p.iw) { dx = 0u; dy++; } }
    }
    o[gid] = packed;
}

// ---- unary: mode 0 relu, 1 sigmoid, 2 erf, 3 gelu, 4 copy ----
struct PcUnary { uint n, mode; };
kernel void unary(device const half* x [[buffer(0)]],
                  device half* o [[buffer(1)]],
                  constant PcUnary& p [[buffer(2)]],
                  uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) return;
    float v = float(x[gid]);
    float r;
    if (p.mode == 0u) r = max(v, 0.0f);
    else if (p.mode == 1u) r = 1.0f / (1.0f + fexp1(-v));
    else if (p.mode == 2u) r = ferf1(v);
    else if (p.mode == 3u) r = 0.5f * v * (1.0f + ferf1(v * 0.7071067811865475f));
    else r = v;
    o[gid] = half(r);
}

// ---- ew2: elementwise binary + act, rank<=4 broadcast ----
struct PcEw2 {
    uint n, s0, s1, s2;
    uint as0, as1, as2, as3;
    uint bs0, bs1, bs2, bs3;
    uint op, act;
};
kernel void ew2(device const half* a [[buffer(0)]],
                device const half* b [[buffer(1)]],
                device half* o [[buffer(2)]],
                constant PcEw2& p [[buffer(3)]],
                uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) return;
    uint d0 = gid / p.s0, r0 = gid % p.s0;
    uint d1 = r0 / p.s1, r1 = r0 % p.s1;
    uint d2 = r1 / p.s2, d3 = r1 % p.s2;
    float va = float(a[d0 * p.as0 + d1 * p.as1 + d2 * p.as2 + d3 * p.as3]);
    float vb = float(b[d0 * p.bs0 + d1 * p.bs1 + d2 * p.bs2 + d3 * p.bs3]);
    float v = p.op == 0u ? va + vb : p.op == 1u ? va - vb : p.op == 2u ? va * vb : va / vb;
    if (p.act == 1u) v = max(v, 0.0f);
    else if (p.act == 2u) v = 1.0f / (1.0f + exp(-v));
    o[gid] = half(v);
}

// ---- ew2v: same-shape vec4 fast path ----
struct PcEw2v { uint nv4, op, act; };
kernel void ew2v(device const half4* a [[buffer(0)]],
                 device const half4* b [[buffer(1)]],
                 device half4* o [[buffer(2)]],
                 constant PcEw2v& p [[buffer(3)]],
                 uint gid [[thread_position_in_grid]])
{
    if (gid >= p.nv4) return;
    float4 va = float4(a[gid]), vb = float4(b[gid]), v;
    if (p.op == 0u) v = va + vb;
    else if (p.op == 1u) v = va - vb;
    else if (p.op == 2u) v = va * vb;
    else v = va / vb;
    if (p.act == 1u) v = max(v, float4(0.0f));
    else if (p.act == 2u) v = 1.0f / (1.0f + exp(-v));
    o[gid] = half4(v);
}

// ---- transpose: rank<=4 permute ----
struct PcTranspose { uint n, os0, os1, os2, ps0, ps1, ps2, ps3; };
kernel void transpose(device const half* x [[buffer(0)]],
                      device half* o [[buffer(1)]],
                      constant PcTranspose& p [[buffer(2)]],
                      uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) return;
    uint d0 = gid / p.os0, r0 = gid % p.os0;
    uint d1 = r0 / p.os1, r1 = r0 % p.os1;
    uint d2 = r1 / p.os2, d3 = r1 % p.os2;
    o[gid] = x[d0 * p.ps0 + d1 * p.ps1 + d2 * p.ps2 + d3 * p.ps3];
}

// ---- slice: rank<=4 strided slice ----
struct PcSlice {
    uint n, os0, os1, os2;
    uint st0, st1, st2, st3;
    uint sp0, sp1, sp2, sp3;
    uint is0, is1, is2, is3;
};
kernel void slice(device const half* x [[buffer(0)]],
                  device half* o [[buffer(1)]],
                  constant PcSlice& p [[buffer(2)]],
                  uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) return;
    uint d0 = gid / p.os0, r0 = gid % p.os0;
    uint d1 = r0 / p.os1, r1 = r0 % p.os1;
    uint d2 = r1 / p.os2, d3 = r1 % p.os2;
    o[gid] = x[(p.st0 + d0 * p.sp0) * p.is0 + (p.st1 + d1 * p.sp1) * p.is1
             + (p.st2 + d2 * p.sp2) * p.is2 + (p.st3 + d3 * p.sp3) * p.is3];
}

// ---- concat_ax: copy one input into dst at axis offset ----
struct PcConcat { uint n, inner, srcAxis, dstAxis, axisOff; };
kernel void concat_ax(device const half* x [[buffer(0)]],
                      device half* o [[buffer(1)]],
                      constant PcConcat& p [[buffer(2)]],
                      uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) return;
    uint ii = gid % p.inner;
    uint a = (gid / p.inner) % p.srcAxis;
    uint oi = gid / (p.inner * p.srcAxis);
    o[(oi * p.dstAxis + a + p.axisOff) * p.inner + ii] = x[gid];
}

// ---- pad: rank-4 all-dims pad, constant or reflect ----
struct PcPad {
    uint id0, id1, id2, id3;
    uint od0, od1, od2, od3;
    uint pb0, pb1, pb2, pb3;
    uint mode;          // 0 const 1 reflect
    float value;
};
static inline int padcoord(uint oidx, uint pb, uint idim, uint mode, thread bool& miss)
{
    int s = int(oidx) - int(pb);
    if (uint(s) >= idim)
    {
        if (mode == 0u) { miss = true; return 0; }
        s = mirror1(s, int(idim));
    }
    miss = false; return s;
}
kernel void pad(device const half* x [[buffer(0)]],
                device half* o [[buffer(1)]],
                constant PcPad& p [[buffer(2)]],
                uint gid [[thread_position_in_grid]])
{
    uint n = p.od0 * p.od1 * p.od2 * p.od3;
    if (gid >= n) return;
    uint s2 = p.od3, s1 = p.od2 * s2, s0 = p.od1 * s1;
    uint d0 = gid / s0, r0 = gid % s0;
    uint d1 = r0 / s1, r1 = r0 % s1;
    uint d2 = r1 / s2, d3 = r1 % s2;
    bool m0, m1, m2, m3;
    int c0 = padcoord(d0, p.pb0, p.id0, p.mode, m0);
    int c1 = padcoord(d1, p.pb1, p.id1, p.mode, m1);
    int c2 = padcoord(d2, p.pb2, p.id2, p.mode, m2);
    int c3 = padcoord(d3, p.pb3, p.id3, p.mode, m3);
    if (m0 || m1 || m2 || m3) { o[gid] = half(p.value); return; }
    uint si2 = p.id3, si1 = p.id2 * si2, si0 = p.id1 * si1;
    o[gid] = x[uint(c0) * si0 + uint(c1) * si1 + uint(c2) * si2 + uint(c3)];
}

// ---- shift: NHWC channel-shift (fused shift-MLP) ----
struct PcShift {
    uint c, h, w, axis;
    uint ge0, ge1, ge2, ge3, ge4;
    int of0, of1, of2, of3, of4;
};
static inline int shiftoff(uint ch, constant PcShift& p)
{
    return ch < p.ge0 ? p.of0 : ch < p.ge1 ? p.of1 : ch < p.ge2 ? p.of2 : ch < p.ge3 ? p.of3 : ch < p.ge4 ? p.of4 : 0;
}
kernel void shift(device const half* x [[buffer(0)]],
                  device half* o [[buffer(1)]],
                  constant PcShift& p [[buffer(2)]],
                  uint gid [[thread_position_in_grid]])
{
    if (gid >= p.c * p.h * p.w) return;
    uint ch = gid % p.c, px = gid / p.c;
    uint ox = px % p.w, oy = px / p.w;
    int off = shiftoff(ch, p);
    int sy = int(oy), sx = int(ox);
    if (p.axis == 2u) sy += off; else sx += off;
    o[gid] = half((uint(sy) < p.h && uint(sx) < p.w) ? float(x[(uint(sy) * p.w + uint(sx)) * p.c + ch]) : 0.0f);
}
kernel void shift4(device const half4* x [[buffer(0)]],
                   device half4* o [[buffer(1)]],
                   constant PcShift& p [[buffer(2)]],
                   uint gid [[thread_position_in_grid]])
{
    uint c4 = p.c >> 2;
    if (gid >= c4 * p.h * p.w) return;
    uint q = gid % c4, px = gid / c4, ch = q * 4u;
    uint ox = px % p.w, oy = px / p.w;
    int off = shiftoff(ch, p);
    int sy = int(oy), sx = int(ox);
    if (p.axis == 2u) sy += off; else sx += off;
    o[gid] = (uint(sy) < p.h && uint(sx) < p.w) ? x[(uint(sy) * p.w + uint(sx)) * c4 + q] : half4(0.0h);
}

// ---- bilinear / bilinear4: NHWC half-pixel bilinear + fused add ----
struct PcBilinear { uint c, ih, iw, oh, ow, flags; float shy, shx; };
kernel void bilinear(device const half* x [[buffer(0)]],
                     device const half* a [[buffer(1)]],
                     device half* o [[buffer(2)]],
                     constant PcBilinear& p [[buffer(3)]],
                     uint gid [[thread_position_in_grid]])
{
    if (gid >= p.c * p.oh * p.ow) return;
    uint ch = gid % p.c, px = gid / p.c;
    uint ox = px % p.ow, oy = px / p.ow;
    float fy = (float(oy) + 0.5f) / p.shy - 0.5f;
    float fx = (float(ox) + 0.5f) / p.shx - 0.5f;
    int fy0 = int(floor(fy)), fx0 = int(floor(fx));
    float dy = fy - float(fy0), dx = fx - float(fx0);
    uint y0 = uint(clamp(fy0, 0, int(p.ih) - 1)), x0 = uint(clamp(fx0, 0, int(p.iw) - 1));
    uint y1 = uint(clamp(fy0 + 1, 0, int(p.ih) - 1)), x1 = uint(clamp(fx0 + 1, 0, int(p.iw) - 1));
    float v00 = float(x[(y0 * p.iw + x0) * p.c + ch]);
    float v01 = float(x[(y0 * p.iw + x1) * p.c + ch]);
    float v10 = float(x[(y1 * p.iw + x0) * p.c + ch]);
    float v11 = float(x[(y1 * p.iw + x1) * p.c + ch]);
    float v = mix(mix(v00, v01, dx), mix(v10, v11, dx), dy);
    if ((p.flags & 1u) != 0u) v = max(v, 0.0f);
    if ((p.flags & 2u) != 0u)
    {
        float s = float(a[gid]);
        v += (p.flags & 4u) != 0u ? max(s, 0.0f) : s;
    }
    o[gid] = half(v);
}
kernel void bilinear4(device const half4* x [[buffer(0)]],
                      device const half4* a [[buffer(1)]],
                      device half4* o [[buffer(2)]],
                      constant PcBilinear& p [[buffer(3)]],
                      uint gid [[thread_position_in_grid]])
{
    uint c4 = p.c >> 2;
    if (gid >= c4 * p.oh * p.ow) return;
    uint ch = gid % c4, px = gid / c4;
    uint ox = px % p.ow, oy = px / p.ow;
    float fy = (float(oy) + 0.5f) / p.shy - 0.5f;
    float fx = (float(ox) + 0.5f) / p.shx - 0.5f;
    int fy0 = int(floor(fy)), fx0 = int(floor(fx));
    float dy = fy - float(fy0), dx = fx - float(fx0);
    uint y0 = uint(clamp(fy0, 0, int(p.ih) - 1)), x0 = uint(clamp(fx0, 0, int(p.iw) - 1));
    uint y1 = uint(clamp(fy0 + 1, 0, int(p.ih) - 1)), x1 = uint(clamp(fx0 + 1, 0, int(p.iw) - 1));
    float4 v00 = float4(x[(y0 * p.iw + x0) * c4 + ch]);
    float4 v01 = float4(x[(y0 * p.iw + x1) * c4 + ch]);
    float4 v10 = float4(x[(y1 * p.iw + x0) * c4 + ch]);
    float4 v11 = float4(x[(y1 * p.iw + x1) * c4 + ch]);
    float4 v = mix(mix(v00, v01, dx), mix(v10, v11, dx), dy);
    if ((p.flags & 1u) != 0u) v = max(v, float4(0.0f));
    if ((p.flags & 2u) != 0u)
    {
        float4 s = float4(a[gid]);
        v += (p.flags & 4u) != 0u ? max(s, float4(0.0f)) : s;
    }
    o[gid] = half4(v);
}

// ---- maxpool / maxpool4: NHWC max pool + optional relu ----
struct PcPool {
    uint c, ih, iw, oh, ow;
    uint kh, kw, sh, sw, pt, pl, relu;
};
kernel void maxpool(device const half* x [[buffer(0)]],
                    device half* o [[buffer(1)]],
                    constant PcPool& p [[buffer(2)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.c * p.oh * p.ow) return;
    uint ch = gid % p.c, px = gid / p.c;
    uint ox = px % p.ow, oy = px / p.ow;
    float m = -3.4e38f;
    for (uint ky = 0u; ky < p.kh; ky++)
    {
        int sy = int(oy * p.sh + ky) - int(p.pt);
        if (uint(sy) >= p.ih) continue;
        for (uint kx = 0u; kx < p.kw; kx++)
        {
            int sx = int(ox * p.sw + kx) - int(p.pl);
            if (uint(sx) >= p.iw) continue;
            m = max(m, float(x[(uint(sy) * p.iw + uint(sx)) * p.c + ch]));
        }
    }
    if (m == -3.4e38f) m = 0.0f;
    if (p.relu != 0u && m < 0.0f) m = 0.0f;
    o[gid] = half(m);
}
kernel void maxpool4(device const half4* x [[buffer(0)]],
                     device half4* o [[buffer(1)]],
                     constant PcPool& p [[buffer(2)]],
                     uint gid [[thread_position_in_grid]])
{
    uint c4 = p.c >> 2;
    if (gid >= c4 * p.oh * p.ow) return;
    uint ch = gid % c4, px = gid / c4;
    uint ox = px % p.ow, oy = px / p.ow;
    float4 m = float4(-3.4e38f);
    for (uint ky = 0u; ky < p.kh; ky++)
    {
        int sy = int(oy * p.sh + ky) - int(p.pt);
        if (uint(sy) >= p.ih) continue;
        for (uint kx = 0u; kx < p.kw; kx++)
        {
            int sx = int(ox * p.sw + kx) - int(p.pl);
            if (uint(sx) >= p.iw) continue;
            m = max(m, float4(x[(uint(sy) * p.iw + uint(sx)) * c4 + ch]));
        }
    }
    if (m.x == -3.4e38f) m = float4(0.0f);
    if (p.relu != 0u) m = select(m, float4(0.0f), m < float4(0.0f));
    o[gid] = half4(m);
}

// ---- layernorm: one workgroup (256 threads) per row ----
struct PcLn { uint len; uint rows; float eps; };
kernel void layernorm(device const half* x [[buffer(0)]],
                      device const float* w [[buffer(1)]],
                      device const float* b [[buffer(2)]],
                      device half* o [[buffer(3)]],
                      constant PcLn& p [[buffer(4)]],
                      uint wg [[threadgroup_position_in_grid]],
                      uint tid [[thread_position_in_threadgroup]])
{
    // one simdgroup (32 lanes) per row, 8 rows per threadgroup: lane-strided
    // accumulate + simd_sum only, no threadgroup barrier at all.
    uint lane = tid & 31u;
    uint row = wg * 8u + (tid >> 5);
    if (row >= p.rows) return;
    uint base = row * p.len;
    float s = 0.0f, s2 = 0.0f;
    for (uint i = lane; i < p.len; i += 32u)
    {
        float v = float(x[base + i]);
        s += v; s2 += v * v;
    }
    s = simd_sum(s); s2 = simd_sum(s2);
    float mean = s / float(p.len);
    float inv = rsqrt(s2 / float(p.len) - mean * mean + p.eps);
    for (uint i = lane; i < p.len; i += 32u)
        o[base + i] = half((float(x[base + i]) - mean) * inv * w[i] + b[i]);
}
