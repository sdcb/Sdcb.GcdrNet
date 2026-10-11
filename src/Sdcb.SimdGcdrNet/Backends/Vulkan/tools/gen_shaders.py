"""Dev-time generator: rewrites the layout-sensitive shaders to NHWC and templates
every shader's storage type through et.glsl (fp32 / fp16 storage builds).
Run once; the .comp outputs are committed. Kept for reference/regeneration."""
import os, sys

D = sys.argv[1]
H = '#version 450\n#extension GL_GOOGLE_include_directive : require\n#include "et.glsl"\n'
ERF = '''float ferf(float v)
{
    float s = v < 0.0 ? -1.0 : 1.0;
    float ax = abs(v);
    float t = 1.0 / (1.0 + 0.3275911 * ax);
    float poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
    return (1.0 - poly * exp(-ax * ax)) * s;
}
'''
MIR = 'int mirror(int i, int d) { if (d <= 1) return 0; while (i < 0 || i >= d) i = i < 0 ? -i : 2 * d - 2 - i; return i; }\n'


def w(name, body):
    with open(os.path.join(D, name + '.comp'), 'w', encoding='utf-8', newline='\n') as f:
        f.write(H + body.replace('@ERF@', ERF).replace('@MIR@', MIR))


w('conv_tile', r'''// NHWC implicit-GEMM conv: O[px, oc] = sum_k X[px, k] * W[k, oc] with
// k = (ky*KW + kx)*IC + ic, so consecutive k are consecutive channels of one
// input pixel (coalesced loads). W is backend-packed [K][OC] (oc fastest).
// 3D dispatch: gx = x-tile(16), gy = y-tile(8), gz = oc-tile(64); 256 threads,
// each 4 oc x 8 pixels (vec4 accumulators over oc). K-chunk 32 staged in
// shared: As[k][m] (vec4 over m), Bs[n][k] (row stride 36 floats, vec4 over k).
// Accumulation is always fp32. Compile-time kernel/stride: -DKH= -DKW= -DSH= -DSW=.
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer X  { ET x[]; };
layout(set = 0, binding = 1) readonly buffer W  { ET w[]; };
layout(set = 0, binding = 2) readonly buffer B  { float b[]; };   // per-OC bias (dummy if none)
layout(set = 0, binding = 3) readonly buffer R  { ET r[]; };      // same-shape residual (dummy if none)
layout(set = 0, binding = 4) writeonly buffer O { ET o[]; };

layout(push_constant) uniform PC {
    uint ic, ih, iw;
    uint oc, oh, ow;
    uint kh, kw, sh, sw;
    uint pt, pl;
    uint flags;     // bit0 hasBias, bit1 reflectPad, bit2 relu, bit3 hasAdd, bit4 gelu
    uint pad0;
} p;

#ifndef KH
#define KH (p.kh)
#endif
#ifndef KW
#define KW (p.kw)
#endif
#ifndef SH
#define SH (p.sh)
#endif
#ifndef SW
#define SW (p.sw)
#endif

const uint BM = 64u;            // oc tile
const uint BH = 8u, BW = 16u;   // spatial tile
const uint BN = BH * BW;        // 128 pixels per wg
const uint BK = 32u;
const uint BKP = 36u;           // Bs row stride in floats (vec4 aligned)
shared vec4 Asv[BK * BM / 4];
shared vec4 Bsv[BN * BKP / 4];

@MIR@
@ERF@
void main()
{
    uint oc0 = gl_WorkGroupID.z * BM;
    uint oy0 = gl_WorkGroupID.y * BH;
    uint ox0 = gl_WorkGroupID.x * BW;
    uint K = KH * KW * p.ic;
    uint tid = gl_LocalInvocationID.x;
    uint tm = tid & 15u, tn = tid >> 4;    // tm: 4 oc each, tn: 8 pixels each

    vec4 acc[8];
    for (int j = 0; j < 8; j++) acc[j] = vec4(0.0);

    bool reflect = (p.flags & 2u) != 0u;
    uint bk = tid & 31u;                   // fixed k lane for Bs loads (256 % 32 == 0)

    for (uint k0 = 0u; k0 < K; k0 += BK)
    {
        // As[k][m] = W[k0+k][oc0+m]
        for (uint e = tid; e < BK * BM; e += 256u)
        {
            uint k = e >> 6, m = e & 63u;
            uint gm = oc0 + m, gk = k0 + k;
            Asv[e >> 2][e & 3u] = (gm < p.oc && gk < K) ? float(w[gk * p.oc + gm]) : 0.0;
        }
        // Bs[n][k]: a warp reads 32 consecutive k = channels of one pixel tap
        uint gk = k0 + bk;
        bool kok = gk < K;
        uint tap = kok ? gk / p.ic : 0u;
        uint ch = gk - tap * p.ic;
        uint ky = tap / KW, kx = tap - ky * KW;
        for (uint n = tid >> 5; n < BN; n += 8u)
        {
            float v = 0.0;
            uint oy = oy0 + (n >> 4), ox = ox0 + (n & 15u);
            if (kok && oy < p.oh && ox < p.ow)
            {
                int sy = int(oy * SH + ky) - int(p.pt);
                int sx = int(ox * SW + kx) - int(p.pl);
                if (reflect)
                {
                    if (uint(sy) >= p.ih) sy = mirror(sy, int(p.ih));
                    if (uint(sx) >= p.iw) sx = mirror(sx, int(p.iw));
                    v = float(x[(uint(sy) * p.iw + uint(sx)) * p.ic + ch]);
                }
                else if (uint(sy) < p.ih && uint(sx) < p.iw)
                    v = float(x[(uint(sy) * p.iw + uint(sx)) * p.ic + ch]);
            }
            uint si = n * BKP + bk;
            Bsv[si >> 2][si & 3u] = v;
        }
        barrier();

        for (uint k = 0u; k < BK; k += 4u)
        {
            vec4 a0 = Asv[k * 16u + tm];
            vec4 a1 = Asv[(k + 1u) * 16u + tm];
            vec4 a2 = Asv[(k + 2u) * 16u + tm];
            vec4 a3 = Asv[(k + 3u) * 16u + tm];
            for (uint j = 0u; j < 8u; j++)
            {
                vec4 bv = Bsv[(tn * 8u + j) * (BKP / 4u) + (k >> 2)];
                acc[j] += a0 * bv.x + a1 * bv.y + a2 * bv.z + a3 * bv.w;
            }
        }
        barrier();
    }

    bool hasBias = (p.flags & 1u) != 0u;
    bool relu = (p.flags & 4u) != 0u;
    bool hasAdd = (p.flags & 8u) != 0u;
    bool gelu = (p.flags & 16u) != 0u;
    uint gm0 = oc0 + tm * 4u;
    vec4 bias = vec4(0.0);
    for (uint i = 0u; i < 4u; i++)
        if (hasBias && gm0 + i < p.oc) bias[i] = b[gm0 + i];
    for (uint j = 0u; j < 8u; j++)
    {
        uint n = tn * 8u + j;
        uint oy = oy0 + (n >> 4), ox = ox0 + (n & 15u);
        if (oy >= p.oh || ox >= p.ow) continue;
        uint base = (oy * p.ow + ox) * p.oc;
        for (uint i = 0u; i < 4u; i++)
        {
            uint gm = gm0 + i;
            if (gm >= p.oc) break;
            float v = acc[j][i] + bias[i];
            if (hasAdd) v += float(r[base + gm]);
            if (relu) v = max(v, 0.0);
            if (gelu) v = 0.5 * v * (1.0 + ferf(v * 0.7071067811865475));
            o[base + gm] = ET(v);
        }
    }
}
''')

w('depthwise', r'''// NHWC depthwise conv: out[oy,ox,c] = bias[c] + sum w[tap,c] * x[sy,sx,c]
// (w backend-packed [tap][C]); channel-fastest flat index -> coalesced.
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer X  { ET x[]; };
layout(set = 0, binding = 1) readonly buffer W  { ET w[]; };
layout(set = 0, binding = 2) readonly buffer B  { float b[]; };
layout(set = 0, binding = 3) writeonly buffer O { ET o[]; };

layout(push_constant) uniform PC {
    uint c, ih, iw, oh, ow;
    uint kh, kw, sh, sw, pt, pl;
    uint flags;         // bit0 hasBias, bit1 reflect, bit2 relu, bit4 gelu
    float padVal;
} p;

@MIR@
@ERF@
void main()
{
    uint id = gl_GlobalInvocationID.x;
    if (id >= p.c * p.oh * p.ow) return;
    uint ch = id % p.c, px = id / p.c;
    uint ox = px % p.ow, oy = px / p.ow;
    bool refl = (p.flags & 2u) != 0u;
    float acc = ((p.flags & 1u) != 0u) ? b[ch] : 0.0;
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
                if (uint(sy) >= p.ih) yy = mirror(sy, int(p.ih));
                if (uint(sx) >= p.iw) xx = mirror(sx, int(p.iw));
            }
            acc += wv * float(x[(uint(yy) * p.iw + uint(xx)) * p.c + ch]);
        }
    }
    if ((p.flags & 4u) != 0u) acc = max(acc, 0.0);
    if ((p.flags & 16u) != 0u) acc = 0.5 * acc * (1.0 + ferf(acc * 0.7071067811865475));
    o[id] = ET(acc);
}
''')

w('maxpool', r'''// NHWC maxpool + optional relu (channel-fastest flat index).
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer X { ET x[]; };
layout(set = 0, binding = 1) writeonly buffer O { ET o[]; };

layout(push_constant) uniform PC {
    uint c, ih, iw, oh, ow;
    uint kh, kw, sh, sw, pt, pl;
    uint relu;
} p;

void main()
{
    uint id = gl_GlobalInvocationID.x;
    if (id >= p.c * p.oh * p.ow) return;
    uint ch = id % p.c, px = id / p.c;
    uint ox = px % p.ow, oy = px / p.ow;
    float m = -3.4e38;
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
    if (m == -3.4e38) m = 0.0;
    if (p.relu != 0u && m < 0.0) m = 0.0;
    o[id] = ET(m);
}
''')

w('bilinear', r'''// NHWC bilinear resize, half-pixel centers, optional relu.
// src_coord = (dst + 0.5) / scale - 0.5
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer X { ET x[]; };
layout(set = 0, binding = 1) writeonly buffer O { ET o[]; };

layout(push_constant) uniform PC {
    uint c, ih, iw, oh, ow, relu;
    float shy, shx;     // scales
} p;

void main()
{
    uint id = gl_GlobalInvocationID.x;
    if (id >= p.c * p.oh * p.ow) return;
    uint ch = id % p.c, px = id / p.c;
    uint ox = px % p.ow, oy = px / p.ow;

    float fy = (float(oy) + 0.5) / p.shy - 0.5;
    float fx = (float(ox) + 0.5) / p.shx - 0.5;
    int fy0 = int(floor(fy)), fx0 = int(floor(fx));
    float dy = fy - float(fy0), dx = fx - float(fx0);
    uint y0 = uint(clamp(fy0, 0, int(p.ih) - 1)), x0 = uint(clamp(fx0, 0, int(p.iw) - 1));
    uint y1 = uint(clamp(fy0 + 1, 0, int(p.ih) - 1)), x1 = uint(clamp(fx0 + 1, 0, int(p.iw) - 1));

    float v00 = float(x[(y0 * p.iw + x0) * p.c + ch]);
    float v01 = float(x[(y0 * p.iw + x1) * p.c + ch]);
    float v10 = float(x[(y1 * p.iw + x0) * p.c + ch]);
    float v11 = float(x[(y1 * p.iw + x1) * p.c + ch]);
    float v = mix(mix(v00, v01, dx), mix(v10, v11, dx), dy);
    if (p.relu != 0u) v = max(v, 0.0);
    o[id] = ET(v);
}
''')

w('shift', r'''// NHWC ChannelShift (fused shift-MLP): per channel group g, out[y,x,c] =
// in[y+off,x,c] (axis=2) or in[y,x+off,c] (axis=3), zero-fill. Groups [gs..ge).
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer X { ET x[]; };
layout(set = 0, binding = 1) writeonly buffer O { ET o[]; };

layout(push_constant) uniform PC {
    uint c, h, w, axis;
    uint ge0, ge1, ge2, ge3, ge4;     // group end channel indices
    int  of0, of1, of2, of3, of4;     // per-group offsets
} p;

void main()
{
    uint i = gl_GlobalInvocationID.x;
    if (i >= p.c * p.h * p.w) return;
    uint ch = i % p.c, px = i / p.c;
    uint ox = px % p.w, oy = px / p.w;
    int off = ch < p.ge0 ? p.of0 : ch < p.ge1 ? p.of1 : ch < p.ge2 ? p.of2 : ch < p.ge3 ? p.of3 : ch < p.ge4 ? p.of4 : 0;
    int sy = int(oy), sx = int(ox);
    if (p.axis == 2u) sy += off; else sx += off;
    o[i] = ET((uint(sy) < p.h && uint(sx) < p.w) ? float(x[(uint(sy) * p.w + uint(sx)) * p.c + ch]) : 0.0);
}
''')

w('bgr2nhwc', r'''// BGR u8 HWC -> NHWC [h][w][3] /255 with top/left replicate padding.
// 2D grid: x = ox, y = oy; one thread writes the 3 channels of one pixel.
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer BGR { uint bgr[]; };  // packed u8
layout(set = 0, binding = 1) writeonly buffer O { ET o[]; };

layout(push_constant) uniform PC {
    uint oh, ow;        // padded out dims
    uint ih, iw;        // src image dims
    uint pt, pl;        // pad top/left
    uint stride;        // src row stride in bytes
} p;

float fetchByte(uint byteIdx) { return float((bgr[byteIdx >> 2] >> ((byteIdx & 3u) * 8u)) & 0xFFu); }

void main()
{
    uint ox = gl_GlobalInvocationID.x;
    if (ox >= p.ow) return;
    uint oy = gl_GlobalInvocationID.y;
    uint sy = uint(clamp(int(oy) - int(p.pt), 0, int(p.ih) - 1));
    uint sx = uint(clamp(int(ox) - int(p.pl), 0, int(p.iw) - 1));
    uint s = sy * p.stride + sx * 3u, d = (oy * p.ow + ox) * 3u;
    for (uint c = 0u; c < 3u; c++) o[d + c] = ET(fetchByte(s + c) / 255.0);
}
''')

w('div6', r'''// NHWC DR-Net input: dst[px,0..2] = I_s, dst[px,3..5] = clamp(I_s / max(I_sm,1e-6), 0, 1).
// Computed in fp32 (I_sm is a divisor: its storage error passes straight through).
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer X { ET x[]; };
layout(set = 0, binding = 1) readonly buffer S { ET s[]; };
layout(set = 0, binding = 2) writeonly buffer O { ET o[]; };

layout(push_constant) uniform PC { uint hw; } p;

void main()
{
    uint px = gl_GlobalInvocationID.x;
    if (px >= p.hw) return;
    for (uint c = 0u; c < 3u; c++)
    {
        float v = float(x[px * 3u + c]);
        o[px * 6u + c] = ET(v);
        o[px * 6u + 3u + c] = ET(clamp(v / max(float(s[px * 3u + c]), 1e-6), 0.0, 1.0));
    }
}
''')

w('nhwc2bgr', r'''// NHWC [h][w][3] -> u8 BGR HWC, cropping (pt,pl) margins. One thread packs 4
// output bytes (may straddle pixels) into one aligned uint write.
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer X { ET x[]; };
layout(set = 0, binding = 1) writeonly buffer O { uint o[]; };     // packed u8

layout(push_constant) uniform PC {
    uint h, w;          // padded src dims
    uint ih, iw;        // dst image dims
    uint pt, pl;        // crop top/left
    uint nwords;        // ceil(ih*iw*3 / 4)
} p;

void main()
{
    uint w4 = gl_GlobalInvocationID.x;
    if (w4 >= p.nwords) return;
    uint byteIdx = w4 * 4u;
    uint byteTotal = p.ih * p.iw * 3u;
    uint px = byteIdx / 3u, c = byteIdx - px * 3u;
    uint dy = px / p.iw, dx = px - dy * p.iw;
    uint packed = 0u;
    for (uint j = 0u; j < 4u; j++)
    {
        uint val = 0u;
        if (byteIdx < byteTotal)
        {
            float f = float(x[((dy + p.pt) * p.w + dx + p.pl) * 3u + c]) * 255.0 + 0.5;
            val = uint(clamp(f, 0.0, 255.0));
        }
        packed |= val << (j * 8u);
        byteIdx++;
        if (++c == 3u) { c = 0u; if (++dx == p.iw) { dx = 0u; dy++; } }
    }
    o[w4] = packed;
}
''')


def fix(name, subs):
    path = os.path.join(D, name + '.comp')
    s = open(path, encoding='utf-8').read()
    if 'et.glsl' in s:
        return
    s = s.replace('#version 450\n', H, 1)
    for a, b in subs:
        assert a in s, (name, a)
        s = s.replace(a, b)
    open(path, 'w', encoding='utf-8', newline='\n').write(s)


F = ('{ float x[]; }', '{ ET x[]; }'), ('{ float o[]; }', '{ ET o[]; }')
fix('ew2', [('readonly buffer A { float a[]; }', 'readonly buffer A { ET a[]; }'),
            ('readonly buffer B { float b[]; }', 'readonly buffer B { ET b[]; }'),
            F[1], ('float va = a[', 'float va = float(a['), ('float vb = b[', 'float vb = float(b['),
            ('p.as3];', 'p.as3]);'), ('p.bs3];', 'p.bs3]);'), ('o[i] = v;', 'o[i] = ET(v);')])
fix('ew2v', [('{ vec4 a[]; }', '{ ET4 a[]; }'), ('{ vec4 b[]; }', '{ ET4 b[]; }'), ('{ vec4 o[]; }', '{ ET4 o[]; }'),
             ('vec4 va = a[i], vb = b[i], v;', 'vec4 va = vec4(a[i]), vb = vec4(b[i]), v;'), ('o[i] = v;', 'o[i] = ET4(v);')])
fix('unary', [*F, ('float v = x[i];', 'float v = float(x[i]);'), ('o[i] = r;', 'o[i] = ET(r);')])
fix('slice', [*F])
fix('concat_ax', [*F])
fix('transpose', [*F])
fix('pad', [*F, ('o[i] = p.value; return;', 'o[i] = ET(p.value); return;')])
fix('layernorm', [*F, ('float v = x[base + i];', 'float v = float(x[base + i]);'),
                  ('o[base + i] = (x[base + i] - mean) * inv * w[i] + b[i];',
                   'o[base + i] = ET((float(x[base + i]) - mean) * inv * w[i] + b[i]);')])
fix('gemm', [('{ float a[]; }', '{ ET a[]; }'), ('readonly buffer B  { float b[]; }', 'readonly buffer B  { ET b[]; }'), F[1],
             ('? a[gm * p.k + gk] : 0.0;', '? float(a[gm * p.k + gk]) : 0.0;'),
             ('v = transB ? b[gn * p.k + gk] : b[gk * p.n + gn];', 'v = float(transB ? b[gn * p.k + gk] : b[gk * p.n + gn]);'),
             ('o[gm * p.n + gn] = v;', 'o[gm * p.n + gn] = ET(v);')])
print('ok')
