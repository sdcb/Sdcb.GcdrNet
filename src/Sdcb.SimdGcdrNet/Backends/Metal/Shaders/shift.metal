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

