using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Sdcb.GcdrNet.Engine;

using F32x8 = System.Runtime.Intrinsics.Vector256<float>;
using V256 = System.Runtime.Intrinsics.Vector256;

namespace Sdcb.GcdrNet.Kernels;

/// <summary>
/// AVX2+FMA kernels: multi-threaded, register-tiled NCHW conv (OC4 x YB2),
/// deinterleaved stride-2 conv, depthwise, 1x1, maxpool, GEMM, LayerNorm.
/// Inputs are pinned for the duration of each call so Parallel.For lambdas
/// can capture raw pointers (spans are not capturable). Edge pixels go
/// through a scalar spill path so every pad mode stays exact.
/// </summary>
internal static unsafe class Avx2Kernels
{
    internal static readonly int Threads = InitThreads();
    private static int InitThreads()
    {
        string? s = Environment.GetEnvironmentVariable("GCDRNET_THREADS");
        return int.TryParse(s, out int t) && t > 0 ? t : Environment.ProcessorCount;
    }
    private static readonly ParallelOptions PO = new() { MaxDegreeOfParallelism = Threads };

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static F32x8 Fma(F32x8 a, F32x8 b, F32x8 c) => V256.FusedMultiplyAdd(a, b, c);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static F32x8 Ld(float* p) => Avx.LoadVector256(p);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void St(float* p, F32x8 v) => Avx.Store(p, v);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static F32x8 Bc(float* p) => Avx2.BroadcastScalarToVector256(p);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int Reflect(int i, int dim)
    {
        if (dim <= 1) return 0;
        while (i < 0 || i >= dim) i = i < 0 ? -i : 2 * dim - 2 - i;
        return i;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ApplyAct(ref F32x8 v, ActivationKind act)
    {
        if (act == ActivationKind.Relu) v = V256.Max(v, F32x8.Zero);
        else if (act == ActivationKind.Sigmoid) v = SimdMath.SigmoidCore(v);
    }

    // lane-wise fetch with pad resolution (edge chunks only)
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static F32x8 Spill8(float* sp, int ih, int iw, int sy, int sx,
        PadMode mode, float padVal, float* spill)
    {
        for (int l = 0; l < 8; l++)
        {
            int px = sx + l;
            spill[l] = (uint)px < (uint)iw ? sp[sy * iw + px]
                : mode == PadMode.Reflect ? sp[sy * iw + Reflect(px, iw)]
                : padVal;
        }
        return Ld(spill);
    }

    private static void StoreRow(float* dp, int oStride,
        int xw, ActivationKind act, F32x8 a0, F32x8 a1, F32x8 a2, F32x8 a3)
    {
        ApplyAct(ref a0, act); ApplyAct(ref a1, act);
        ApplyAct(ref a2, act); ApplyAct(ref a3, act);
        if (xw == 8)
        {
            St(dp, a0);
            St(dp + oStride, a1);
            St(dp + 2 * oStride, a2);
            St(dp + 3 * oStride, a3);
            return;
        }
        float* tmp = stackalloc float[32];
        St(tmp, a0); St(tmp + 8, a1); St(tmp + 16, a2); St(tmp + 24, a3);
        for (int l = 0; l < xw; l++)
            for (int j = 0; j < 4; j++)
                dp[j * oStride + l] = tmp[j * 8 + l];
    }

    // ======================== dense conv 3x3 s1 ========================
    public static void Conv3x3S1(ReadOnlySpan<float> src, int ic, int ih, int iw,
        ReadOnlySpan<float> w, ReadOnlySpan<float> bias, Span<float> dst,
        int oc, int oh, int ow, int pt, int pl,
        PadMode mode, float padVal, ActivationKind act)
    {
        int yb = (oh + 1) / 2, ob = oc / 4;
        bool hasBias = !bias.IsEmpty;
        fixed (float* sp0 = src, wp0 = w, bp0 = bias, dp0 = dst)
        {
            float* sp = sp0, wp = wp0, bp = bp0, dp = dp0;
            Parallel.For(0, ob * yb, PO, tile =>
            {
                Conv3x3Tile(sp, ic, ih, iw, wp, hasBias ? bp : null, dp, oc, oh, ow,
                    pt, pl, mode, padVal, act, tile / yb * 4, tile % yb * 2);
            });
        }
    }

    private static void Conv3x3Tile(float* sp, int ic, int ih, int iw,
        float* wp, float* bp, float* dp,
        int oc, int oh, int ow, int pt, int pl,
        PadMode mode, float padVal, ActivationKind act, int o0, int y0)
    {
        bool two = y0 + 1 < oh;
        int y1 = two ? y0 + 1 : y0;
        int wStride = ic * 9;
        int ostride = oh * ow;
        float* spill = stackalloc float[8];

        for (int x0 = 0; x0 < ow; x0 += 8)
        {
            int xw = Math.Min(8, ow - x0);
            F32x8 a0, a1, a2, a3, b0, b1, b2, b3;
            if (bp is null)
            {
                a0 = a1 = a2 = a3 = b0 = b1 = b2 = b3 = F32x8.Zero;
            }
            else
            {
                a0 = b0 = Bc(bp + o0);
                a1 = b1 = Bc(bp + o0 + 1);
                a2 = b2 = Bc(bp + o0 + 2);
                a3 = b3 = Bc(bp + o0 + 3);
            }
            for (int c = 0; c < ic; c++)
            {
                float* sbase = sp + c * ih * iw;
                float* wb = wp + (o0 * ic + c) * 9;
                for (int ky = 0; ky < 3; ky++)
                {
                    int syA = y0 + ky - pt, syB = y1 + ky - pt;
                    int ryA = (uint)syA < (uint)ih ? syA
                        : mode == PadMode.Reflect ? Reflect(syA, ih) : -1;
                    int ryB = !two ? -1
                        : (uint)syB < (uint)ih ? syB
                        : mode == PadMode.Reflect ? Reflect(syB, ih) : -1;
                    for (int kx = 0; kx < 3; kx++)
                    {
                        int sx = x0 + kx - pl;
                        int wi = ky * 3 + kx;
                        F32x8 wv0 = Bc(wb + wi);
                        F32x8 wv1 = Bc(wb + wi + wStride);
                        F32x8 wv2 = Bc(wb + wi + 2 * wStride);
                        F32x8 wv3 = Bc(wb + wi + 3 * wStride);
                        bool interior = (uint)sx < (uint)iw && sx + 8 <= iw;
                        if (ryA >= 0)
                        {
                            F32x8 v = interior
                                ? Ld(sbase + ryA * iw + sx)
                                : Spill8(sbase, ih, iw, ryA, sx, mode, padVal, spill);
                            a0 = Fma(v, wv0, a0); a1 = Fma(v, wv1, a1); a2 = Fma(v, wv2, a2); a3 = Fma(v, wv3, a3);
                        }
                        else if (padVal != 0f)
                        {
                            F32x8 pv = V256.Create(padVal);
                            a0 = Fma(pv, wv0, a0); a1 = Fma(pv, wv1, a1);
                            a2 = Fma(pv, wv2, a2); a3 = Fma(pv, wv3, a3);
                        }
                        if (ryB >= 0)
                        {
                            F32x8 v = interior
                                ? Ld(sbase + ryB * iw + sx)
                                : Spill8(sbase, ih, iw, ryB, sx, mode, padVal, spill);
                            b0 = Fma(v, wv0, b0); b1 = Fma(v, wv1, b1); b2 = Fma(v, wv2, b2); b3 = Fma(v, wv3, b3);
                        }
                        else if (ryB < 0 && padVal != 0f)
                        {
                            F32x8 pv = V256.Create(padVal);
                            b0 = Fma(pv, wv0, b0); b1 = Fma(pv, wv1, b1);
                            b2 = Fma(pv, wv2, b2); b3 = Fma(pv, wv3, b3);
                        }
                    }
                }
            }
            float* db = dp + (o0 * oh + y0) * ow + x0;
            StoreRow(db, ostride, xw, act, a0, a1, a2, a3);
            if (two) StoreRow(db + ow, ostride, xw, act, b0, b1, b2, b3);
        }
    }

    // ======================== conv 1x1 ========================
    public static void Conv1x1(ReadOnlySpan<float> src, int ic, int hw,
        ReadOnlySpan<float> w, ReadOnlySpan<float> bias, Span<float> dst,
        int oc, ActivationKind act)
    {
        int stripes = Math.Clamp(hw / 32768, 1, 8);
        int stride = hw / stripes;
        bool hasBias = !bias.IsEmpty;
        fixed (float* sp0 = src, wp0 = w, bp0 = bias, dp0 = dst)
        {
            float* sp = sp0, wp = wp0, bp = bp0, dp = dp0;
            Parallel.For(0, ((oc + 3) / 4) * stripes, PO, tile =>
            {
                int o0 = tile / stripes * 4;
                int hw0 = tile % stripes * stride;
                int hw1 = tile % stripes == stripes - 1 ? hw : hw0 + stride;
                Conv1x1Strip(sp, ic, wp, hasBias ? bp : null, dp,
                    o0, Math.Min(4, oc - o0), hw0, hw1, hw, act);
            });
        }
    }

    private static void Conv1x1Strip(float* sp, int ic,
        float* wp, float* bp, float* dp,
        int o0, int ocLim, int hw0, int hw1, int hw, ActivationKind act)
    {
        int i = hw0;
        for (; i + 8 <= hw1; i += 8)
        {
            F32x8 a0 = V256.Create(bp is null ? 0f : bp[o0]);
            F32x8 a1 = V256.Create(ocLim > 1 && bp is not null ? bp[o0 + 1] : 0f);
            F32x8 a2 = V256.Create(ocLim > 2 && bp is not null ? bp[o0 + 2] : 0f);
            F32x8 a3 = V256.Create(ocLim > 3 && bp is not null ? bp[o0 + 3] : 0f);
            for (int c = 0; c < ic; c++)
            {
                F32x8 v = Ld(sp + c * hw + i);
                a0 = Fma(v, Bc(wp + o0 * ic + c), a0);
                if (ocLim > 1) a1 = Fma(v, Bc(wp + (o0 + 1) * ic + c), a1);
                if (ocLim > 2) a2 = Fma(v, Bc(wp + (o0 + 2) * ic + c), a2);
                if (ocLim > 3) a3 = Fma(v, Bc(wp + (o0 + 3) * ic + c), a3);
            }
            float* db = dp + o0 * hw + i;
            ApplyAct(ref a0, act);
            St(db, a0);
            if (ocLim > 1) { ApplyAct(ref a1, act); St(db + hw, a1); }
            if (ocLim > 2) { ApplyAct(ref a2, act); St(db + 2 * hw, a2); }
            if (ocLim > 3) { ApplyAct(ref a3, act); St(db + 3 * hw, a3); }
        }
        for (; i < hw1; i++)
            for (int j = 0; j < ocLim; j++)
            {
                float acc = bp is null ? 0f : bp[o0 + j];
                for (int c = 0; c < ic; c++)
                    acc += sp[c * hw + i] * wp[(o0 + j) * ic + c];
                float r = act == ActivationKind.Relu ? MathF.Max(acc, 0f)
                    : act == ActivationKind.Sigmoid ? 1f / (1f + MathF.Exp(-acc)) : acc;
                dp[(o0 + j) * hw + i] = r;
            }
    }

    // ======================== conv 3x3 s2 (patch embed) ========================
    private static readonly Vector256<int> Evens = Vector256.Create(0, 2, 4, 6, 0, 0, 0, 0);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static F32x8 LoadStride2(float* p)
    {
        F32x8 a = Ld(p);
        F32x8 b = Ld(p + 8);
        var ea = Avx2.PermuteVar8x32(a.AsInt32(), Evens);
        var eb = Avx2.PermuteVar8x32(b.AsInt32(), Evens);
        return V256.Create(ea.GetLower().AsSingle(), eb.GetLower().AsSingle());
    }

    public static void Conv3x3S2(ReadOnlySpan<float> src, int ic, int ih, int iw,
        ReadOnlySpan<float> w, ReadOnlySpan<float> bias, Span<float> dst,
        int oc, int oh, int ow, int pt, int pl,
        PadMode mode, float padVal, ActivationKind act)
    {
        int yb = (oh + 1) / 2, ob = oc / 4;
        bool hasBias = !bias.IsEmpty;
        fixed (float* sp0 = src, wp0 = w, bp0 = bias, dp0 = dst)
        {
            float* sp = sp0, wp = wp0, bp = bp0, dp = dp0;
            Parallel.For(0, ob * yb, PO, tile =>
            {
                ConvS2Tile(sp, ic, ih, iw, wp, hasBias ? bp : null, dp, oc, oh, ow,
                    pt, pl, mode, padVal, act, tile / yb * 4, tile % yb * 2);
            });
        }
    }

    private static void ConvS2Tile(float* sp, int ic, int ih, int iw,
        float* wp, float* bp, float* dp,
        int oc, int oh, int ow, int pt, int pl,
        PadMode mode, float padVal, ActivationKind act, int o0, int y0)
    {
        bool two = y0 + 1 < oh;
        int y1 = two ? y0 + 1 : y0;
        int wStride = ic * 9;
        int ostride = oh * ow;
        float* spill = stackalloc float[16];

        for (int x0 = 0; x0 < ow; x0 += 8)
        {
            int xw = Math.Min(8, ow - x0);
            F32x8 a0, a1, a2, a3, b0, b1, b2, b3;
            if (bp is null)
            {
                a0 = a1 = a2 = a3 = b0 = b1 = b2 = b3 = F32x8.Zero;
            }
            else
            {
                a0 = b0 = Bc(bp + o0);
                a1 = b1 = Bc(bp + o0 + 1);
                a2 = b2 = Bc(bp + o0 + 2);
                a3 = b3 = Bc(bp + o0 + 3);
            }
            for (int c = 0; c < ic; c++)
            {
                float* sbase = sp + c * ih * iw;
                float* wb = wp + (o0 * ic + c) * 9;
                for (int ky = 0; ky < 3; ky++)
                {
                    int syA = y0 * 2 + ky - pt, syB = y1 * 2 + ky - pt;
                    int ryA = (uint)syA < (uint)ih ? syA
                        : mode == PadMode.Reflect ? Reflect(syA, ih) : -1;
                    int ryB = !two ? -1
                        : (uint)syB < (uint)ih ? syB
                        : mode == PadMode.Reflect ? Reflect(syB, ih) : -1;
                    for (int kx = 0; kx < 3; kx++)
                    {
                        int sx = x0 * 2 + kx - pl;
                        int wi = ky * 3 + kx;
                        F32x8 wv0 = Bc(wb + wi);
                        F32x8 wv1 = Bc(wb + wi + wStride);
                        F32x8 wv2 = Bc(wb + wi + 2 * wStride);
                        F32x8 wv3 = Bc(wb + wi + 3 * wStride);
                        bool interior = sx >= 0 && sx + 15 < iw;
                        if (ryA >= 0)
                        {
                            F32x8 v;
                            if (interior)
                            {
                                v = LoadStride2(sbase + ryA * iw + sx);
                            }
                            else
                            {
                                for (int l = 0; l < 8; l++)
                                {
                                    int px = sx + 2 * l;
                                    spill[l] = x0 + l < ow && (uint)px < (uint)iw
                                        ? sbase[ryA * iw + px]
                                        : x0 + l < ow && mode == PadMode.Reflect
                                            ? sbase[ryA * iw + Reflect(px, iw)]
                                            : padVal;
                                }
                                v = Ld(spill);
                            }
                            a0 = Fma(v, wv0, a0); a1 = Fma(v, wv1, a1); a2 = Fma(v, wv2, a2); a3 = Fma(v, wv3, a3);
                        }
                        else if (padVal != 0f)
                        {
                            F32x8 pv = V256.Create(padVal);
                            a0 = Fma(pv, wv0, a0); a1 = Fma(pv, wv1, a1);
                            a2 = Fma(pv, wv2, a2); a3 = Fma(pv, wv3, a3);
                        }
                        if (ryB >= 0)
                        {
                            F32x8 v;
                            if (interior)
                            {
                                v = LoadStride2(sbase + ryB * iw + sx);
                            }
                            else
                            {
                                for (int l = 0; l < 8; l++)
                                {
                                    int px = sx + 2 * l;
                                    spill[l] = x0 + l < ow && (uint)px < (uint)iw
                                        ? sbase[ryB * iw + px]
                                        : x0 + l < ow && mode == PadMode.Reflect
                                            ? sbase[ryB * iw + Reflect(px, iw)]
                                            : padVal;
                                }
                                v = Ld(spill);
                            }
                            b0 = Fma(v, wv0, b0); b1 = Fma(v, wv1, b1); b2 = Fma(v, wv2, b2); b3 = Fma(v, wv3, b3);
                        }
                        else if (ryB < 0 && padVal != 0f)
                        {
                            F32x8 pv = V256.Create(padVal);
                            b0 = Fma(pv, wv0, b0); b1 = Fma(pv, wv1, b1);
                            b2 = Fma(pv, wv2, b2); b3 = Fma(pv, wv3, b3);
                        }
                    }
                }
            }
            float* db = dp + (o0 * oh + y0) * ow + x0;
            StoreRow(db, ostride, xw, act, a0, a1, a2, a3);
            if (two) StoreRow(db + ow, ostride, xw, act, b0, b1, b2, b3);
        }
    }

    // ======================== depthwise conv ========================
    public static void Depthwise(ReadOnlySpan<float> src, int ic, int ih, int iw,
        ReadOnlySpan<float> w, ReadOnlySpan<float> bias, Span<float> dst,
        int oh, int ow, int kh, int kw, int pt, int pl,
        PadMode mode, float padVal, ActivationKind act)
    {
        bool hasBias = !bias.IsEmpty;
        fixed (float* sp0 = src, wp0 = w, bp0 = bias, dp0 = dst)
        {
            float* sp = sp0, wp = wp0, bp = bp0, dp = dp0;
            Parallel.For(0, ic * oh, PO, t =>
        {
            int c = t / oh, y = t % oh;
            float* sbase = sp + c * ih * iw;
            float* drow = dp + (c * oh + y) * ow;
            float* wk = wp + c * kh * kw;
            float b = bp is null ? 0f : bp[c];
            float* spill = stackalloc float[16];
            for (int x0 = 0; x0 < ow; x0 += 8)
            {
                int xw = Math.Min(8, ow - x0);
                F32x8 acc = V256.Create(b);
                for (int ky = 0; ky < kh; ky++)
                {
                    int sy = y + ky - pt;
                    int ry = (uint)sy < (uint)ih ? sy
                        : mode == PadMode.Reflect ? Reflect(sy, ih) : -1;
                    for (int kx = 0; kx < kw; kx++)
                    {
                        int sx = x0 + kx - pl;
                        float wv = wk[ky * kw + kx];
                        F32x8 v;
                        if (ry >= 0 && sx >= 0 && sx + 8 <= iw)
                        {
                            v = Ld(sbase + ry * iw + sx);
                        }
                        else
                        {
                            for (int l = 0; l < 8; l++)
                            {
                                int px = sx + l;
                                spill[l] = x0 + l < ow && ry >= 0 && (uint)px < (uint)iw
                                    ? sbase[ry * iw + px]
                                    : x0 + l < ow && ry >= 0 && mode == PadMode.Reflect
                                        ? sbase[ry * iw + Reflect(px, iw)]
                                        : padVal;
                            }
                            v = Ld(spill);
                        }
                        acc = Fma(v, V256.Create(wv), acc);
                    }
                }
                ApplyAct(ref acc, act);
                if (xw == 8)
                {
                    St(drow + x0, acc);
                }
                else
                {
                    St(spill, acc);
                    for (int l = 0; l < xw; l++) drow[x0 + l] = spill[l];
                }
            }
            });
        }
    }

    // ======================== maxpool 2x2 s2 ========================
    public static void MaxPool2x2(ReadOnlySpan<float> src, int c, int ih, int iw,
        Span<float> dst, int oh, int ow, bool fuseRelu)
    {
        fixed (float* sp0 = src, dp0 = dst)
        {
            float* sp = sp0, dp = dp0;
            Parallel.For(0, c * oh, PO, t =>
        {
            int ch = t / oh, y = t % oh;
            float* sbase = sp + ch * ih * iw;
            float* drow = dp + (ch * oh + y) * ow;
            int r0 = Math.Min(y * 2, ih - 1) * iw, r1 = Math.Min(y * 2 + 1, ih - 1) * iw;
            float* spill = stackalloc float[16];
            for (int x0 = 0; x0 < ow; x0 += 8)
            {
                int xw = Math.Min(8, ow - x0);
                int sx = x0 * 2;
                F32x8 m;
                if (sx + 15 < iw)
                {
                    m = V256.Max(LoadStride2(sbase + r0 + sx), LoadStride2(sbase + r0 + sx + 1));
                    m = V256.Max(m, V256.Max(LoadStride2(sbase + r1 + sx), LoadStride2(sbase + r1 + sx + 1)));
                }
                else
                {
                    for (int l = 0; l < 8; l++)
                    {
                        if (x0 + l >= ow) { spill[l] = float.NegativeInfinity; continue; }
                        int xx = Math.Min(sx + 2 * l, iw - 2);
                        float mx = MathF.Max(sbase[r0 + xx], sbase[r0 + xx + 1]);
                        spill[l] = MathF.Max(mx, MathF.Max(sbase[r1 + xx], sbase[r1 + xx + 1]));
                    }
                    m = Ld(spill);
                }
                if (fuseRelu) m = V256.Max(m, F32x8.Zero);
                if (xw == 8)
                {
                    St(drow + x0, m);
                }
                else
                {
                    St(spill, m);
                    for (int l = 0; l < xw; l++) drow[x0 + l] = spill[l];
                }
            }
            });
        }
    }

    // ======================== GEMM (B row-major [K,N]) ========================
    public static void GemmKN(ReadOnlySpan<float> a, int M, int K,
        ReadOnlySpan<float> b, int N, ReadOnlySpan<float> bias, Span<float> dst)
    {
        bool hasBias = !bias.IsEmpty;
        fixed (float* ap0 = a, bp0 = b, cp0 = bias, dp0 = dst)
        {
            float* ap = ap0, bp = bp0, cp = cp0, dp = dp0;
            Parallel.For(0, (M + 3) / 4, PO, t =>
            {
                int m0 = t * 4;
                GemmTile(ap, m0, Math.Min(4, M - m0), K, bp, N, hasBias ? cp : null, dp);
            });
        }
    }

    private static void GemmTile(float* ap, int m0, int mc, int K,
        float* bp, int N, float* cp, float* dp)
    {
        int nv = N - N % 16;
        for (int n0 = 0; n0 < nv; n0 += 16)
        {
            F32x8 c00, c01;
            if (cp is null)
            {
                c00 = c01 = F32x8.Zero;
            }
            else
            {
                c00 = Ld(cp + n0);
                c01 = Ld(cp + n0 + 8);
            }
            F32x8 c10 = c00, c11 = c01, c20 = c00, c21 = c01, c30 = c00, c31 = c01;
            for (int k = 0; k < K; k++)
            {
                F32x8 bv0 = Ld(bp + k * N + n0);
                F32x8 bv1 = Ld(bp + k * N + n0 + 8);
                F32x8 av = Bc(ap + m0 * K + k);
                c00 = Fma(av, bv0, c00); c01 = Fma(av, bv1, c01);
                if (mc > 1) { av = Bc(ap + (m0 + 1) * K + k); c10 = Fma(av, bv0, c10); c11 = Fma(av, bv1, c11); }
                if (mc > 2) { av = Bc(ap + (m0 + 2) * K + k); c20 = Fma(av, bv0, c20); c21 = Fma(av, bv1, c21); }
                if (mc > 3) { av = Bc(ap + (m0 + 3) * K + k); c30 = Fma(av, bv0, c30); c31 = Fma(av, bv1, c31); }
            }
            float* db = dp + m0 * N + n0;
            St(db, c00); St(db + 8, c01);
            if (mc > 1) { St(db + N, c10); St(db + N + 8, c11); }
            if (mc > 2) { St(db + 2 * N, c20); St(db + 2 * N + 8, c21); }
            if (mc > 3) { St(db + 3 * N, c30); St(db + 3 * N + 8, c31); }
        }
        for (int n0 = nv; n0 < N; n0++)
            for (int m = 0; m < mc; m++)
            {
                float s = cp is null ? 0f : cp[n0];
                for (int k = 0; k < K; k++)
                    s += ap[(m0 + m) * K + k] * bp[k * N + n0];
                dp[(m0 + m) * N + n0] = s;
            }
    }

    // ======================== LayerNorm ========================
    public static void LayerNorm(ReadOnlySpan<float> x, int rows, int c,
        ReadOnlySpan<float> w, ReadOnlySpan<float> b, Span<float> dst, float eps)
    {
        fixed (float* xp0 = x, wp0 = w, bp0 = b, dp0 = dst)
        {
            float* xp = xp0, wp = wp0, bp = bp0, dp = dp0;
            Parallel.For(0, rows, PO, r =>
        {
            float* xr = xp + r * c;
            float* dr = dp + r * c;
            F32x8 sum = F32x8.Zero, sq = F32x8.Zero;
            int i = 0;
            for (; i + 8 <= c; i += 8)
            {
                F32x8 v = Ld(xr + i);
                sum = V256.Add(sum, v);
                sq = Fma(v, v, sq);
            }
            float s = 0, ss = 0;
            for (int l = 0; l < 8; l++) { s += sum[l]; ss += sq[l]; }
            for (; i < c; i++) { float v = xr[i]; s += v; ss += v * v; }
            float mean = s / c;
            float inv = 1f / MathF.Sqrt(MathF.Max(ss / c - mean * mean, 0f) + eps);
            F32x8 vm = V256.Create(mean), vi = V256.Create(inv);
            for (i = 0; i + 8 <= c; i += 8)
            {
                F32x8 v = Ld(xr + i);
                F32x8 wv = Ld(wp + i);
                F32x8 bv = Ld(bp + i);
                F32x8 t = V256.Subtract(v, vm);
                St(dr + i, Fma(t, V256.Multiply(vi, wv), bv));
            }
            for (; i < c; i++) dr[i] = (xr[i] - mean) * inv * wp[i] + bp[i];
            });
        }
    }

    // ======================== parallel unary ========================
    // op: 0 relu, 1 sigmoid, 2 gelu, 3 erf
    public static void Unary(ReadOnlySpan<float> src, Span<float> dst, int n, int op)
    {
        int chunks = Math.Clamp(n / 8192, 1, Threads * 4);
        int stride = (n + chunks - 1) / chunks;
        fixed (float* sp0 = src, dp0 = dst)
        {
            float* sp = sp0, dp = dp0;
            Parallel.For(0, chunks, PO, t =>
            {
                int i0 = t * stride, i1 = Math.Min(n, i0 + stride);
                int i = i0;
                for (; i + 8 <= i1; i += 8)
                {
                    F32x8 v = Ld(sp + i);
                    ApplyUnary(ref v, op);
                    St(dp + i, v);
                }
                for (; i < i1; i++) dp[i] = UnaryScalar(sp[i], op);
            });
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ApplyUnary(ref F32x8 v, int op)
    {
        if (op == 0) v = V256.Max(v, F32x8.Zero);
        else if (op == 1) v = SimdMath.SigmoidCore(v);
        else if (op == 2) v = SimdMath.GeluCore(v);
        else v = SimdMath.ErfCore(v);
    }

    private static float UnaryScalar(float x, int op) => op switch
    {
        0 => MathF.Max(x, 0f),
        1 => 1f / (1f + MathF.Exp(-x)),
        2 => 0.5f * x * (1f + SimdMath.ErfScalar(x * 0.70710678118654752f)),
        _ => SimdMath.ErfScalar(x),
    };

    // ======================== parallel binary elementwise ========================
    // op: 0 add, 1 sub, 2 mul, 3 div
    public static void Ew2(ReadOnlySpan<float> a, ReadOnlySpan<float> b,
        Span<float> dst, int n, int op, ActivationKind act = ActivationKind.None)
    {
        int chunks = Math.Clamp(n / 8192, 1, Threads * 4);
        int stride = (n + chunks - 1) / chunks;
        fixed (float* ap0 = a, bp0 = b, dp0 = dst)
        {
            float* ap = ap0, bp = bp0, dp = dp0;
            Parallel.For(0, chunks, PO, t =>
            {
                int i0 = t * stride, i1 = Math.Min(n, i0 + stride);
                int i = i0;
                for (; i + 8 <= i1; i += 8)
                {
                    F32x8 v = EwCore(Ld(ap + i), Ld(bp + i), op);
                    ApplyAct(ref v, act);
                    St(dp + i, v);
                }
                for (; i < i1; i++)
                {
                    float v = EwScalar(ap[i], bp[i], op);
                    dp[i] = act == ActivationKind.Relu ? MathF.Max(v, 0f) : v;
                }
            });
        }
    }

    // scalar broadcast: dst[i] = src[i] op s  (rev: dst[i] = s op src[i])
    public static void Ew1(ReadOnlySpan<float> src, Span<float> dst,
        int n, float s, int op, bool rev = false, ActivationKind act = ActivationKind.None)
    {
        int chunks = Math.Clamp(n / 8192, 1, Threads * 4);
        int stride = (n + chunks - 1) / chunks;
        fixed (float* ap0 = src, dp0 = dst)
        {
            float* ap = ap0, dp = dp0;
            F32x8 vs = V256.Create(s);
            Parallel.For(0, chunks, PO, t =>
            {
                int i0 = t * stride, i1 = Math.Min(n, i0 + stride);
                int i = i0;
                for (; i + 8 <= i1; i += 8)
                {
                    F32x8 v = Ld(ap + i);
                    v = rev ? EwCore(vs, v, op) : EwCore(v, vs, op);
                    ApplyAct(ref v, act);
                    St(dp + i, v);
                }
                for (; i < i1; i++)
                {
                    float v = rev ? EwScalar(s, ap[i], op) : EwScalar(ap[i], s, op);
                    dp[i] = act == ActivationKind.Relu ? MathF.Max(v, 0f) : v;
                }
            });
        }
    }

    // broadcast over a contiguous inner block: b[bo] constant per outer row
    public static void EwInner(ReadOnlySpan<float> a, ReadOnlySpan<float> b,
        int[] bStr, int[] outShape, int[] outStr, int rank, int inner,
        Span<float> dst, int n, int op, ActivationKind act = ActivationKind.None)
    {
        int outer = n / inner;
        fixed (float* ap0 = a, bp0 = b, dp0 = dst)
        {
            float* ap = ap0, bp = bp0, dp = dp0;
            Parallel.For(0, outer, PO, e =>
            {
                int rem = e, bo = 0;
                for (int d = 0; d < rank - 1; d++)
                {
                    int cc = rem / (outStr[d] / inner);
                    rem %= outStr[d] / inner;
                    bo += cc * bStr[d];
                }
                float s = bp[bo];
                F32x8 vs = V256.Create(s);
                float* ar = ap + e * inner, dr = dp + e * inner;
                int i = 0;
                for (; i + 8 <= inner; i += 8)
                {
                    F32x8 v = EwCore(Ld(ar + i), vs, op);
                    ApplyAct(ref v, act);
                    St(dr + i, v);
                }
                for (; i < inner; i++)
                {
                    float v = EwScalar(ar[i], s, op);
                    dr[i] = act == ActivationKind.Relu ? MathF.Max(v, 0f) : v;
                }
            });
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static F32x8 EwCore(F32x8 a, F32x8 b, int op) => op switch
    {
        0 => V256.Add(a, b),
        1 => V256.Subtract(a, b),
        2 => V256.Multiply(a, b),
        _ => V256.Divide(a, b),
    };

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float EwScalar(float a, float b, int op) => op switch
    {
        0 => a + b, 1 => a - b, 2 => a * b, _ => a / b,
    };

    // ======================== bilinear resize (half-pixel) ========================
    public static void Bilinear(ReadOnlySpan<float> src, int c, int ih, int iw,
        Span<float> dst, int oh, int ow,
        int[] y0, int[] y1, float[] wy, int[] x0, int[] x1, float[] wx,
        ActivationKind act = ActivationKind.None)
    {
        fixed (float* sp0 = src, dp0 = dst)
        fixed (int* y0p = y0, y1p = y1, x0p = x0, x1p = x1)
        fixed (float* wyp = wy, wxp = wx)
        {
            float* sp = sp0, dp = dp0;
            int* y0a = y0p, y1a = y1p, x0a = x0p, x1a = x1p;
            float* wya = wyp, wxa = wxp;
            Parallel.For(0, c * oh, PO, t =>
            {
                int ch = t / oh, y = t % oh;
                float* r0 = sp + (ch * ih + y0a[y]) * iw;
                float* r1 = sp + (ch * ih + y1a[y]) * iw;
                float* dr = dp + (ch * oh + y) * ow;
                F32x8 wyv = V256.Create(wya[y]);
                int x = 0;
                for (; x + 8 <= ow; x += 8)
                {
                    var i0 = Avx.LoadVector256(x0a + x);
                    var i1 = Avx.LoadVector256(x1a + x);
                    F32x8 wxv = Ld(wxa + x);
                    F32x8 t0 = Avx2.GatherVector256(r0, i0, 4);
                    F32x8 t1 = Avx2.GatherVector256(r0, i1, 4);
                    F32x8 b0 = Avx2.GatherVector256(r1, i0, 4);
                    F32x8 b1 = Avx2.GatherVector256(r1, i1, 4);
                    F32x8 top = Fma(V256.Subtract(t1, t0), wxv, t0);
                    F32x8 bot = Fma(V256.Subtract(b1, b0), wxv, b0);
                    F32x8 v = Fma(V256.Subtract(bot, top), wyv, top);
                    ApplyAct(ref v, act);
                    St(dr + x, v);
                }
                for (; x < ow; x++)
                {
                    float t0 = r0[x0a[x]], t1v = r0[x1a[x]];
                    float b0 = r1[x0a[x]], b1v = r1[x1a[x]];
                    float top = t0 + (t1v - t0) * wxa[x];
                    float bot = b0 + (b1v - b0) * wxa[x];
                    float v = top + (bot - top) * wya[y];
                    dr[x] = act == ActivationKind.Relu ? MathF.Max(v, 0f) : v;
                }
            });
        }
    }

    // ======================== tiled transpose [a,b] -> [b,a] ========================
    public static void Transpose2D(ReadOnlySpan<float> src, int a, int b,
        Span<float> dst, int copies = 1, int srcStride = 0, int dstStride = 0)
    {
        if (srcStride == 0) srcStride = a * b;
        if (dstStride == 0) dstStride = a * b;
        int ya = (a + 7) / 8, xb = (b + 7) / 8;
        fixed (float* sp0 = src, dp0 = dst)
        {
            float* sp = sp0, dp = dp0;
            Parallel.For(0, copies * ya, PO, t =>
            {
                int o = t / ya, ty = t % ya;
                float* sb = sp + o * srcStride, db = dp + o * dstStride;
                for (int tx = 0; tx < xb; tx++)
                {
                    int y0 = ty * 8, x0 = tx * 8;
                    if (y0 + 8 <= a && x0 + 8 <= b)
                    {
                        T8(sb + y0 * b + x0, b, db + x0 * a + y0, a);
                    }
                    else
                    {
                        int ye = Math.Min(y0 + 8, a), xe = Math.Min(x0 + 8, b);
                        for (int y = y0; y < ye; y++)
                            for (int x = x0; x < xe; x++)
                                db[x * a + y] = sb[y * b + x];
                    }
                }
            });
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void T8(float* s, int ls, float* d, int ld)
    {
        F32x8 r0 = Ld(s), r1 = Ld(s + ls), r2 = Ld(s + 2 * ls), r3 = Ld(s + 3 * ls);
        F32x8 r4 = Ld(s + 4 * ls), r5 = Ld(s + 5 * ls), r6 = Ld(s + 6 * ls), r7 = Ld(s + 7 * ls);
        F32x8 t0 = Avx.UnpackLow(r0, r1), t1 = Avx.UnpackHigh(r0, r1);
        F32x8 t2 = Avx.UnpackLow(r2, r3), t3 = Avx.UnpackHigh(r2, r3);
        F32x8 t4 = Avx.UnpackLow(r4, r5), t5 = Avx.UnpackHigh(r4, r5);
        F32x8 t6 = Avx.UnpackLow(r6, r7), t7 = Avx.UnpackHigh(r6, r7);
        const byte LO = 0x44, HI = 0xEE;
        F32x8 u0 = Avx.Shuffle(t0, t2, LO), u1 = Avx.Shuffle(t0, t2, HI);
        F32x8 u2 = Avx.Shuffle(t1, t3, LO), u3 = Avx.Shuffle(t1, t3, HI);
        F32x8 u4 = Avx.Shuffle(t4, t6, LO), u5 = Avx.Shuffle(t4, t6, HI);
        F32x8 u6 = Avx.Shuffle(t5, t7, LO), u7 = Avx.Shuffle(t5, t7, HI);
        const byte L = 0x20, H = 0x31;
        St(d, Avx.Permute2x128(u0, u4, L));
        St(d + ld, Avx.Permute2x128(u1, u5, L));
        St(d + 2 * ld, Avx.Permute2x128(u2, u6, L));
        St(d + 3 * ld, Avx.Permute2x128(u3, u7, L));
        St(d + 4 * ld, Avx.Permute2x128(u0, u4, H));
        St(d + 5 * ld, Avx.Permute2x128(u1, u5, H));
        St(d + 6 * ld, Avx.Permute2x128(u2, u6, H));
        St(d + 7 * ld, Avx.Permute2x128(u3, u7, H));
    }

    // ======================== pad ========================
    public static void Pad4(ReadOnlySpan<float> src, int nc, int ih, int iw,
        Span<float> dst, int oh, int ow, int[] mapY, int[] mapX, float padVal)
    {
        // contiguous interior run of mapX (identity-mapped region)
        int xl = 0; while (xl < ow && mapX[xl] < 0) xl++;
        int xr = ow; while (xr > xl && mapX[xr - 1] < 0) xr--;
        bool interior = true;
        for (int x = xl; x < xr; x++) if (mapX[x] != x - xl + mapX[xl]) { interior = false; break; }
        int srcX0 = xl < xr ? mapX[xl] : 0;
        fixed (float* sp0 = src, dp0 = dst)
        fixed (int* myp = mapY, mxp = mapX)
        {
            float* sp = sp0, dp = dp0;
            int* mya = myp, mxa = mxp;
            Parallel.For(0, nc * oh, PO, t =>
            {
                int ch = t / oh, y = t % oh;
                float* dr = dp + (ch * oh + y) * ow;
                int my = mya[y];
                if (my < 0)
                {
                    F32x8 pv = V256.Create(padVal);
                    int x = 0;
                    for (; x + 8 <= ow; x += 8) St(dr + x, pv);
                    for (; x < ow; x++) dr[x] = padVal;
                    return;
                }
                float* sr = sp + (ch * ih + my) * iw;
                for (int x = 0; x < xl; x++) dr[x] = mxa[x] < 0 ? padVal : sr[mxa[x]];
                for (int x = xr; x < ow; x++) dr[x] = mxa[x] < 0 ? padVal : sr[mxa[x]];
                if (interior)
                {
                    float* s = sr + srcX0;
                    int i = 0, len = xr - xl;
                    for (; i + 8 <= len; i += 8) St(dr + xl + i, Ld(s + i));
                    for (; i < len; i++) dr[xl + i] = s[i];
                }
                else
                {
                    for (int x = xl; x < xr; x++) dr[x] = sr[mxa[x]];
                }
            });
        }
    }

    // ======================== channel shift ========================
    public static void ChannelShift(ReadOnlySpan<float> src, int c, int h, int w,
        Span<float> dst, int axis, int[] offsets, int[] groupEnds)
    {
        fixed (float* sp0 = src, dp0 = dst)
        fixed (int* offp = offsets, endp = groupEnds)
        {
            float* sp = sp0, dp = dp0;
            int* offa = offp, enda = endp;
            int groups = offsets.Length;
            Parallel.For(0, c * h, PO, t =>
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
                        for (; x + 8 <= w; x += 8) St(dr + x, z);
                        for (; x < w; x++) dr[x] = 0f;
                    }
                }
                else if (Math.Abs(off) >= w)
                {
                    int x = 0;
                    for (; x + 8 <= w; x += 8) St(dr + x, F32x8.Zero);
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
                    for (; x + 8 <= -off; x += 8) St(dr + x, F32x8.Zero);
                    for (; x < -off; x++) dr[x] = 0f;
                    Buffer.MemoryCopy(sr, dr - off, (w + off) * 4L, (w + off) * 4L);
                }
            });
        }
    }
}
