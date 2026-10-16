// GCDR pipeline glue kernels. All activation storage is fp16
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

