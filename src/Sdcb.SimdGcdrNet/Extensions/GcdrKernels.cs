using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.Wasm;
using System.Runtime.Intrinsics.X86;
using Sdcb.SimdOnnx.Engine;
using Sdcb.SimdOnnx.Kernels;
using static Sdcb.SimdOnnx.Kernels.Nhwc;

using F32x8 = System.Runtime.Intrinsics.Vector256<float>;
using F32x16 = System.Runtime.Intrinsics.Vector512<float>;
using V128 = System.Runtime.Intrinsics.Vector128<float>;
using V4 = System.Runtime.Intrinsics.Vector128<float>;
using V256 = System.Runtime.Intrinsics.Vector256;

namespace Sdcb.SimdGcdrNet.Extensions;

/// <summary>Packed UpHead weights: [c][8] oc lanes (zero-padded) + bias[8].</summary>
internal sealed class UpHeadPlan
{
    public required float[] W;
    public required float[] Bias;
    public int Oc, C;
}


/// <summary>
/// CPU kernels of the GCDR extension ops (<see cref="UpHeadOp"/>, <see cref="ChannelShiftOp"/>),
/// built on the engine's NHWC helpers (<see cref="Nhwc"/>).
/// </summary>
internal static unsafe partial class GcdrKernels
{
    private static readonly ParallelOptions s_po = new() { MaxDegreeOfParallelism = Avx2Kernels.Threads };

    /// <summary>Packs UpHead weights [oc, c, 1, 1] as [c][8] (oc lanes, zero-padded) + bias[8].</summary>
    public static UpHeadPlan PackUpHead(ReadOnlySpan<float> w, int oc, int c, ReadOnlySpan<float> bias)
    {
        if (oc > 8) throw new NotSupportedException($"UpHead supports oc <= 8 (got {oc}).");
        float[] pw = new float[c * 8], pb = new float[8];
        for (int o = 0; o < oc; o++)
            for (int ch = 0; ch < c; ch++) pw[ch * 8 + o] = w[o * c + ch];
        if (!bias.IsEmpty) bias.CopyTo(pb);
        return new UpHeadPlan { W = pw, Bias = pb, Oc = oc, C = c };
    }

    /// <summary>
    /// out[y,x,o] = act(bias[o] + Σ_c w[o,c]·s[c]), s = [relu](resize(src)) + [relu](skip): the
    /// full-resolution resize/add tensors live only in a per-pixel stack buffer. Per-element
    /// math follows the unfused nodes (bias-initialized, channel-ascending accumulation).
    /// </summary>
    public static void UpHead(ReadOnlySpan<float> src, int ih, int iw, int c, ReadOnlySpan<float> skip,
        Span<float> dst, int oh, int ow, float sy, float sx, UpHeadP p, UpHeadPlan kp,
        int skipH = 0, int skipW = 0, int skipC = 0)
    {
        // SkipConv: `skip` is the conv's input [skipH, skipW, skipC]; each task recomputes
        // its skip row (same direct-conv kernel → same values) instead of reading a stored one
        ConvP? sc = p.SkipConv;
        var skp = sc?.KernelPlan as NhwcConvPlan;
        if (kp.C != c || dst.Length != (long)oh * ow * kp.Oc
            || (sc is null ? skip.Length != (long)oh * ow * c
                : skp is null || skp.Depthwise || skp.WinoU is not null || sc.AddValue >= 0 || skip.Length != (long)skipH * skipW * skipC))
            throw new NotSupportedException("UpHead shape mismatch.");
        int[] y0 = new int[oh], y1 = new int[oh], x0 = new int[ow], x1 = new int[ow];
        float[] wy = new float[oh], wx = new float[ow];
        ResizeAxis(oh, ih, sy, y0, y1, wy);
        ResizeAxis(ow, iw, sx, x0, x1, wx);
        bool relu = p.ResizeRelu, skipRelu = p.SkipRelu, avx = UseAvx;
        int oc = kp.Oc;
        bool neonUp = UseNeon && c % 4 == 0 && oc <= 4;
        bool pkUp = UsePacked && c % 4 == 0 && oc <= 4;
        ActivationKind act = p.Act;
        int[] rm = [], cm = [];
        float[] padVec = [];
        if (sc is not null)
        {
            rm = Map(oh, sc.Kernel[0], sc.Strides[0], sc.Pads[0], skipH, sc.PadMode);
            cm = Map(ow, sc.Kernel[1], sc.Strides[1], sc.Pads[1], skipW, sc.PadMode);
            padVec = new float[skipC * sc.Kernel[1] + Ob];   // one constant-pad kernel row (Dense512Row)
            Array.Fill(padVec, sc.PadValue);
        }
        fixed (float* sp0 = src, dp0 = dst, kp0 = skip, wp0 = kp.W, bp0 = kp.Bias)
        fixed (float* cw = skp?.W, cb = skp?.Bias, pv = padVec)
        fixed (int* rmp = rm, cmp = cm)
        {
            float* sp = sp0, dp = dp0, kpp = kp0, wp = wp0, bp = bp0;
            var ca = sc is null ? default
                : new DenseArgs(kp0, cw, cb, pv, rmp, cmp, skipW, skipC, ow, c, sc.Kernel[0], sc.Kernel[1], skp!.Blocks, sc.Act, avx);
            if (UseAvx512)
            {
                UpHead512(sp, iw, c, kpp, sc is not null, ca, dp, oh, ow, y0, y1, x0, x1, wy, wx, relu, skipRelu, wp, bp, oc, act);
                return;
            }
            For(oh, (long)oh * ow * c * (oc + 4), y =>
            {
                float* s = stackalloc float[c + 8];
                float* acc = stackalloc float[8];
                float* r0 = sp + (long)y0[y] * iw * c, r1 = sp + (long)y1[y] * iw * c;
                var vY = new Vector<float>(wy[y]);
                float wY = wy[y];
                float* orow = dp + (long)y * ow * oc;
                float[]? band = sc is null ? null : BandBuf((long)ow * c);
                fixed (float* sb = band)
                {
                    float* krow = kpp + (long)y * ow * c;
                    if (sc is not null)
                    {
                        ConvRow(ca, y, sb, null);
                        krow = sb;
                    }
                    if (neonUp)
                    {
                        fixed (int* xa = x0, xb = x1)
                        fixed (float* wxp = wx)
                            UpHeadRowNeon(r0, r1, xa, xb, wxp, wY, krow, c, ow, wp, bp, orow, oc, relu, skipRelu, act);
                        return;
                    }
                    if (pkUp)
                    {
                        fixed (int* xa = x0, xb = x1)
                        fixed (float* wxp = wx)
                            UpHeadRowPk(r0, r1, xa, xb, wxp, wY, krow, c, ow, wp, bp, orow, oc, relu, skipRelu, act);
                        return;
                    }
                    for (int x = 0; x < ow; x++)
                    {
                        ResizePixel(r0 + (long)x0[x] * c, r0 + (long)x1[x] * c, r1 + (long)x0[x] * c, r1 + (long)x1[x] * c,
                            wx[x], wY, new Vector<float>(wx[x]), vY, c, relu, krow + (long)x * c, skipRelu, s);
                        float* o = orow + (long)x * oc;
                        if (avx)
                        {
                            F32x8 a = Avx.LoadVector256(bp);
                            float* w = wp;
                            for (int ch = 0; ch < c; ch++, w += 8)
                                a = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(s + ch), Avx.LoadVector256(w), a);
                            Avx.Store(acc, a);
                        }
                        else if (UseNeon)
                        {
                            V128 a = AdvSimd.LoadVector128(bp), b = AdvSimd.LoadVector128(bp + 4);
                            float* w = wp;
                            for (int ch = 0; ch < c; ch++, w += 8)
                            {
                                V128 v = AdvSimd.LoadAndReplicateToVector128(s + ch);
                                a = AdvSimd.FusedMultiplyAdd(a, AdvSimd.LoadVector128(w), v);
                                b = AdvSimd.FusedMultiplyAdd(b, AdvSimd.LoadVector128(w + 4), v);
                            }
                            AdvSimd.Store(acc, a); AdvSimd.Store(acc + 4, b);
                        }
                        else
                        {
                            for (int l = 0; l < 8; l++) acc[l] = bp[l];
                            float* w = wp;
                            for (int ch = 0; ch < c; ch++, w += 8)
                            {
                                float v = s[ch];
                                for (int l = 0; l < oc; l++) acc[l] += v * w[l];
                            }
                        }
                        for (int l = 0; l < oc; l++) o[l] = acc[l];
                    }
                    if (act != ActivationKind.None) ActRow(orow, ow * oc, act, avx);
                }
            });
        }
    }


    // out[y,x,c] = in[y+off_g, x, c] (logical axis 2) or in[y, x+off_g, c] (axis 3), zero-fill
    public static void ChannelShift(ReadOnlySpan<float> src, int h, int w, int c,
        Span<float> dst, ShiftP p)
    {
        int[] ends = p.GroupEnds, offs = p.Offsets;
        bool alongH = p.Axis == 2;
        fixed (float* sp0 = src, dp0 = dst)
        {
            float* sp = sp0, dp = dp0;
            For(h, (long)h * w * c, y =>
            {
                for (int x = 0; x < w; x++)
                {
                    float* o = dp + ((long)y * w + x) * c;
                    int g0 = 0;
                    for (int g = 0; g < offs.Length; g++)
                    {
                        int g1 = ends[g];
                        int sy = alongH ? y + offs[g] : y, sx = alongH ? x : x + offs[g];
                        if ((uint)sy < (uint)h && (uint)sx < (uint)w)
                            Copy(sp + ((long)sy * w + sx) * c + g0, o + g0, g1 - g0);
                        else
                            Fill(o + g0, g1 - g0, 0f);
                        g0 = g1;
                    }
                    if (g0 < c) Copy(sp + ((long)y * w + x) * c + g0, o + g0, c - g0);
                }
            });
        }
    }
    // out[c,y,x] = in[c, y+dy_g] (axis=2) or in[c,y,x+dx_g] (axis=3), zero-fill
    public static void ChannelShiftNchw(ReadOnlySpan<float> src, int[] inShape, Span<float> dst,
        int[] outShape, ShiftP p)
    {
        int c = inShape[1], h = inShape[2], w = inShape[3];
        if (Avx2.IsSupported && Fma.IsSupported && c * h * w >= 8192)
        {
            ChannelShiftNchwAvx2(src, c, h, w, dst, p.Axis, p.Offsets, p.GroupEnds);
            return;
        }
        int g0 = 0;
        for (int g = 0; g < p.Offsets.Length; g++)
        {
            int g1 = p.GroupEnds[g];
            int off = p.Offsets[g];
            for (int ch = g0; ch < g1; ch++)
            {
                ReadOnlySpan<float> s = src.Slice(ch * h * w, h * w);
                Span<float> d = dst.Slice(ch * h * w, h * w);
                if (p.Axis == 2)
                {
                    for (int y = 0; y < h; y++)
                    {
                        int sy = y + off;
                        if ((uint)sy < (uint)h) s.Slice(sy * w, w).CopyTo(d.Slice(y * w, w));
                        else d.Slice(y * w, w).Clear();
                    }
                }
                else if (Math.Abs(off) >= w)
                {
                    d.Clear();   // shifted fully out of range -> all zeros
                }
                else
                {
                    // out[c,y,x] = in[c,y,x+off], zero-fill — same convention as axis 2
                    if (off >= 0)
                        for (int y = 0; y < h; y++)
                        {
                            ReadOnlySpan<float> sr = s.Slice(y * w, w);
                            Span<float> dr = d.Slice(y * w, w);
                            sr.Slice(off).CopyTo(dr.Slice(0, w - off));
                            dr.Slice(w - off).Clear();
                        }
                    else
                        for (int y = 0; y < h; y++)
                        {
                            ReadOnlySpan<float> sr = s.Slice(y * w, w);
                            Span<float> dr = d.Slice(y * w, w);
                            sr.Slice(0, w + off).CopyTo(dr.Slice(-off));
                            dr.Slice(0, -off).Clear();
                        }
                }
            }
            g0 = g1;
        }
    }
    private static void ChannelShiftNchwAvx2(ReadOnlySpan<float> src, int c, int h, int w,
        Span<float> dst, int axis, int[] offsets, int[] groupEnds)
    {
        fixed (float* sp0 = src, dp0 = dst)
        fixed (int* offp = offsets, endp = groupEnds)
        {
            float* sp = sp0, dp = dp0;
            int* offa = offp, enda = endp;
            int groups = offsets.Length;
            Parallel.For(0, c * h, s_po, t =>
            {
                int ch = t / h, y = t % h;
                int off = 0;
                for (int g = 0, g0 = 0; g < groups; g++)
                {
                    int g1 = enda[g];
                    if (ch >= g0 && ch < g1) { off = offa[g]; break; }
                    g0 = g1;
                }
                float* sr = sp + (ch * h + y) * w;
                float* dr = dp + (ch * h + y) * w;
                if (axis == 2)
                {
                    int sy = y + off;
                    if ((uint)sy < (uint)h)
                        Buffer.MemoryCopy(sp + (ch * h + sy) * w, dr, w * 4L, w * 4L);
                    else
                    {
                        F32x8 z = F32x8.Zero;
                        int x = 0;
                        for (; x + 8 <= w; x += 8) Avx.Store(dr + x, z);
                        for (; x < w; x++) dr[x] = 0f;
                    }
                }
                else if (Math.Abs(off) >= w)
                {
                    int x = 0;
                    for (; x + 8 <= w; x += 8) Avx.Store(dr + x, F32x8.Zero);
                    for (; x < w; x++) dr[x] = 0f;
                }
                else if (off >= 0)
                {
                    // dr[x] = sr[x+off]; tail zero
                    Buffer.MemoryCopy(sr + off, dr, (w - off) * 4L, (w - off) * 4L);
                    int x = w - off;
                    for (; x < w; x++) dr[x] = 0f;
                }
                else
                {
                    // dr[x] = sr[x+off] with off<0: head zero, then copy
                    int x = 0;
                    for (; x + 8 <= -off; x += 8) Avx.Store(dr + x, F32x8.Zero);
                    for (; x < -off; x++) dr[x] = 0f;
                    Buffer.MemoryCopy(sr, dr - off, (w + off) * 4L, (w + off) * 4L);
                }
            });
        }
    }}
