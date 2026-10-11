using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using Sdcb.SimdGcdrNet.Engine;

using V4 = System.Runtime.Intrinsics.Vector128<float>;

namespace Sdcb.SimdGcdrNet.Kernels;

/// <summary>
/// AArch64 AdvSimd (NEON) tier: the 6 × 16 conv / Gemm micro-kernel as fmla-by-element on
/// 4-channel input loads, 24 register accumulators (of 32 vector registers), used wherever the
/// x86 tiers take <see cref="TileAvxK"/>; direct convs feed it kernel-row runs (kw·ic contiguous
/// NHWC floats per tap row) so the 3- / 6-channel stems run the same loop.
/// </summary>
internal static unsafe partial class Nhwc
{
    /// <summary>NEON tier switch: <c>AdvSimd.Arm64.IsSupported</c> (DOTNET_EnableAdvSimd=0 drops to the
    /// scalar kernels). Settable only for the in-process tier comparison tests.</summary>
    internal static bool UseNeon = AdvSimd.Arm64.IsSupported;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static V4 Ld4(float* p) => AdvSimd.LoadVector128(p);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static V4 Fma4(V4 acc, V4 a, V4 b) => AdvSimd.FusedMultiplyAdd(acc, a, b);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static V4 Act4(V4 v, ActivationKind act) => act switch
    {
        ActivationKind.Relu => AdvSimd.Max(v, V4.Zero),
        ActivationKind.Sigmoid => SimdMath.SigmoidCore(v),
        ActivationKind.Gelu => SimdMath.GeluCore(v),
        _ => v,
    };

    /// <summary>6 × 16 tile: <see cref="TileNeonK"/> into a scratch tile, then the residual /
    /// activation epilogue per row.</summary>
    private static void TileNeon(float** ptrs, int rows, int taps, int ic, float* wb, float* bb,
        float* outp, float* addp, int oc, int lanes, ActivationKind act)
    {
        float* part = stackalloc float[Tile * Ob];
        TileNeonK(ptrs, taps, ic, wb, bb, part, Ob);
        TileNeonStore(part, rows, outp, addp, oc, lanes, act);
    }

    /// <summary>Residual / activation epilogue of a finished 6 × 16 scratch tile.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void TileNeonStore(float* part, int rows, float* outp, float* addp, int oc, int lanes, ActivationKind act)
    {
        for (int r = 0; r < rows; r++, part += Ob)
        {
            float* o = outp + (long)r * oc;
            float* ad = addp == null ? null : addp + (long)r * oc;
            if (lanes == Ob)
            {
                for (int l = 0; l < Ob; l += 4)
                {
                    V4 v = Ld4(part + l);
                    if (ad != null) v = AdvSimd.Add(v, Ld4(ad + l));
                    AdvSimd.Store(o + l, Act4(v, act));
                }
                continue;
            }
            for (int l = 0; l < lanes; l++)
                o[l] = ActS(ad != null ? part[l] + ad[l] : part[l], act);
        }
    }

    /// <summary>6 × 16 k loop over all ic channels of every tap (weights [tap][ic][16]), bias-initialized.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void TileNeonK(float** ptrs, int taps, int ic, float* wb, float* bb, float* dst, int ds) =>
        TileNeonK(ptrs, taps, 0, ic, (long)ic * Ob, wb, bb, dst, ds);

    /// <summary>
    /// The 6 × 16 k loop: dst[r*ds + 0..15] = init + Σ x·w for all <see cref="Tile"/> rows, where
    /// ptrs[tap*Tile + r] + coff = row r's ic inputs of tap and wb + tap·wTap its [ic][16] weights;
    /// init = bias <paramref name="bb"/>, or (bb == null) the partial sums already in dst. Four
    /// channels per step (one 128-bit load per pixel, fmla by element; the next weight vector is
    /// loaded one group ahead for the in-order A53), ic % 4 leftovers by broadcast. Never calls
    /// out, so the 24 accumulators stay in registers.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void TileNeonK(float** ptrs, int taps, int coff, int ic, long wTap, float* wb, float* bb, float* dst, int ds,
        bool relu = false)
    {
        V4 a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5, c0, c1, c2, c3, c4, c5, d0, d1, d2, d3, d4, d5;
        if (bb != null)
        {
            a0 = Ld4(bb); b0 = Ld4(bb + 4); c0 = Ld4(bb + 8); d0 = Ld4(bb + 12);
            a1 = a2 = a3 = a4 = a5 = a0;
            b1 = b2 = b3 = b4 = b5 = b0;
            c1 = c2 = c3 = c4 = c5 = c0;
            d1 = d2 = d3 = d4 = d5 = d0;
        }
        else
        {
            float* s = dst;
            a0 = Ld4(s); b0 = Ld4(s + 4); c0 = Ld4(s + 8); d0 = Ld4(s + 12); s += ds;
            a1 = Ld4(s); b1 = Ld4(s + 4); c1 = Ld4(s + 8); d1 = Ld4(s + 12); s += ds;
            a2 = Ld4(s); b2 = Ld4(s + 4); c2 = Ld4(s + 8); d2 = Ld4(s + 12); s += ds;
            a3 = Ld4(s); b3 = Ld4(s + 4); c3 = Ld4(s + 8); d3 = Ld4(s + 12); s += ds;
            a4 = Ld4(s); b4 = Ld4(s + 4); c4 = Ld4(s + 8); d4 = Ld4(s + 12); s += ds;
            a5 = Ld4(s); b5 = Ld4(s + 4); c5 = Ld4(s + 8); d5 = Ld4(s + 12);
        }
        int ic4 = ic & ~3;
        for (int tap = 0; tap < taps; tap++)
        {
            float** pp = ptrs + tap * Tile;
            float* p0 = pp[0] + coff, p1 = pp[1] + coff, p2 = pp[2] + coff, p3 = pp[3] + coff, p4 = pp[4] + coff, p5 = pp[5] + coff;
            float* w = wb + tap * wTap;
            int c = 0;
            for (; c < ic4; c += 4, w += 4 * Ob)
            {
                V4 x0 = Ld4(p0 + c), x1 = Ld4(p1 + c), x2 = Ld4(p2 + c);
                V4 x3 = Ld4(p3 + c), x4 = Ld4(p4 + c), x5 = Ld4(p5 + c);
                V4 wv0 = Ld4(w), wv1;
                wv1 = Ld4(w + 4);
                a0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a0, wv0, x0, 0);
                a1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a1, wv0, x1, 0);
                a2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a2, wv0, x2, 0);
                a3 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a3, wv0, x3, 0);
                a4 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a4, wv0, x4, 0);
                a5 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a5, wv0, x5, 0);
                wv0 = Ld4(w + 8);
                b0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b0, wv1, x0, 0);
                b1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b1, wv1, x1, 0);
                b2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b2, wv1, x2, 0);
                b3 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b3, wv1, x3, 0);
                b4 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b4, wv1, x4, 0);
                b5 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b5, wv1, x5, 0);
                wv1 = Ld4(w + 12);
                c0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c0, wv0, x0, 0);
                c1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c1, wv0, x1, 0);
                c2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c2, wv0, x2, 0);
                c3 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c3, wv0, x3, 0);
                c4 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c4, wv0, x4, 0);
                c5 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c5, wv0, x5, 0);
                wv0 = Ld4(w + 16);
                d0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(d0, wv1, x0, 0);
                d1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(d1, wv1, x1, 0);
                d2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(d2, wv1, x2, 0);
                d3 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(d3, wv1, x3, 0);
                d4 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(d4, wv1, x4, 0);
                d5 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(d5, wv1, x5, 0);
                wv1 = Ld4(w + 20);
                a0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a0, wv0, x0, 1);
                a1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a1, wv0, x1, 1);
                a2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a2, wv0, x2, 1);
                a3 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a3, wv0, x3, 1);
                a4 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a4, wv0, x4, 1);
                a5 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a5, wv0, x5, 1);
                wv0 = Ld4(w + 24);
                b0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b0, wv1, x0, 1);
                b1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b1, wv1, x1, 1);
                b2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b2, wv1, x2, 1);
                b3 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b3, wv1, x3, 1);
                b4 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b4, wv1, x4, 1);
                b5 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b5, wv1, x5, 1);
                wv1 = Ld4(w + 28);
                c0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c0, wv0, x0, 1);
                c1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c1, wv0, x1, 1);
                c2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c2, wv0, x2, 1);
                c3 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c3, wv0, x3, 1);
                c4 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c4, wv0, x4, 1);
                c5 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c5, wv0, x5, 1);
                wv0 = Ld4(w + 32);
                d0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(d0, wv1, x0, 1);
                d1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(d1, wv1, x1, 1);
                d2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(d2, wv1, x2, 1);
                d3 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(d3, wv1, x3, 1);
                d4 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(d4, wv1, x4, 1);
                d5 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(d5, wv1, x5, 1);
                wv1 = Ld4(w + 36);
                a0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a0, wv0, x0, 2);
                a1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a1, wv0, x1, 2);
                a2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a2, wv0, x2, 2);
                a3 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a3, wv0, x3, 2);
                a4 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a4, wv0, x4, 2);
                a5 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a5, wv0, x5, 2);
                wv0 = Ld4(w + 40);
                b0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b0, wv1, x0, 2);
                b1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b1, wv1, x1, 2);
                b2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b2, wv1, x2, 2);
                b3 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b3, wv1, x3, 2);
                b4 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b4, wv1, x4, 2);
                b5 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b5, wv1, x5, 2);
                wv1 = Ld4(w + 44);
                c0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c0, wv0, x0, 2);
                c1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c1, wv0, x1, 2);
                c2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c2, wv0, x2, 2);
                c3 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c3, wv0, x3, 2);
                c4 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c4, wv0, x4, 2);
                c5 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c5, wv0, x5, 2);
                wv0 = Ld4(w + 48);
                d0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(d0, wv1, x0, 2);
                d1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(d1, wv1, x1, 2);
                d2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(d2, wv1, x2, 2);
                d3 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(d3, wv1, x3, 2);
                d4 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(d4, wv1, x4, 2);
                d5 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(d5, wv1, x5, 2);
                wv1 = Ld4(w + 52);
                a0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a0, wv0, x0, 3);
                a1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a1, wv0, x1, 3);
                a2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a2, wv0, x2, 3);
                a3 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a3, wv0, x3, 3);
                a4 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a4, wv0, x4, 3);
                a5 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a5, wv0, x5, 3);
                wv0 = Ld4(w + 56);
                b0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b0, wv1, x0, 3);
                b1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b1, wv1, x1, 3);
                b2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b2, wv1, x2, 3);
                b3 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b3, wv1, x3, 3);
                b4 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b4, wv1, x4, 3);
                b5 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b5, wv1, x5, 3);
                wv1 = Ld4(w + 60);
                c0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c0, wv0, x0, 3);
                c1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c1, wv0, x1, 3);
                c2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c2, wv0, x2, 3);
                c3 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c3, wv0, x3, 3);
                c4 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c4, wv0, x4, 3);
                c5 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c5, wv0, x5, 3);
                d0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(d0, wv1, x0, 3);
                d1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(d1, wv1, x1, 3);
                d2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(d2, wv1, x2, 3);
                d3 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(d3, wv1, x3, 3);
                d4 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(d4, wv1, x4, 3);
                d5 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(d5, wv1, x5, 3);
            }
            for (; c < ic; c++, w += Ob)
            {
                V4 w0 = Ld4(w), w1 = Ld4(w + 4), w2 = Ld4(w + 8), w3 = Ld4(w + 12);
                V4 x = AdvSimd.LoadAndReplicateToVector128(p0 + c);
                a0 = Fma4(a0, w0, x); b0 = Fma4(b0, w1, x); c0 = Fma4(c0, w2, x); d0 = Fma4(d0, w3, x);
                x = AdvSimd.LoadAndReplicateToVector128(p1 + c);
                a1 = Fma4(a1, w0, x); b1 = Fma4(b1, w1, x); c1 = Fma4(c1, w2, x); d1 = Fma4(d1, w3, x);
                x = AdvSimd.LoadAndReplicateToVector128(p2 + c);
                a2 = Fma4(a2, w0, x); b2 = Fma4(b2, w1, x); c2 = Fma4(c2, w2, x); d2 = Fma4(d2, w3, x);
                x = AdvSimd.LoadAndReplicateToVector128(p3 + c);
                a3 = Fma4(a3, w0, x); b3 = Fma4(b3, w1, x); c3 = Fma4(c3, w2, x); d3 = Fma4(d3, w3, x);
                x = AdvSimd.LoadAndReplicateToVector128(p4 + c);
                a4 = Fma4(a4, w0, x); b4 = Fma4(b4, w1, x); c4 = Fma4(c4, w2, x); d4 = Fma4(d4, w3, x);
                x = AdvSimd.LoadAndReplicateToVector128(p5 + c);
                a5 = Fma4(a5, w0, x); b5 = Fma4(b5, w1, x); c5 = Fma4(c5, w2, x); d5 = Fma4(d5, w3, x);
            }
        }
        if (relu)
        {
            V4 z = V4.Zero;
            a0 = AdvSimd.Max(a0, z); b0 = AdvSimd.Max(b0, z); c0 = AdvSimd.Max(c0, z); d0 = AdvSimd.Max(d0, z);
            a1 = AdvSimd.Max(a1, z); b1 = AdvSimd.Max(b1, z); c1 = AdvSimd.Max(c1, z); d1 = AdvSimd.Max(d1, z);
            a2 = AdvSimd.Max(a2, z); b2 = AdvSimd.Max(b2, z); c2 = AdvSimd.Max(c2, z); d2 = AdvSimd.Max(d2, z);
            a3 = AdvSimd.Max(a3, z); b3 = AdvSimd.Max(b3, z); c3 = AdvSimd.Max(c3, z); d3 = AdvSimd.Max(d3, z);
            a4 = AdvSimd.Max(a4, z); b4 = AdvSimd.Max(b4, z); c4 = AdvSimd.Max(c4, z); d4 = AdvSimd.Max(d4, z);
            a5 = AdvSimd.Max(a5, z); b5 = AdvSimd.Max(b5, z); c5 = AdvSimd.Max(c5, z); d5 = AdvSimd.Max(d5, z);
        }
        StoreRow(dst, a0, b0, c0, d0); dst += ds;
        StoreRow(dst, a1, b1, c1, d1); dst += ds;
        StoreRow(dst, a2, b2, c2, d2); dst += ds;
        StoreRow(dst, a3, b3, c3, d3); dst += ds;
        StoreRow(dst, a4, b4, c4, d4); dst += ds;
        StoreRow(dst, a5, b5, c5, d5);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void StoreRow(float* d, V4 a, V4 b, V4 c, V4 e)
    {
        AdvSimd.Store(d, a); AdvSimd.Store(d + 4, b); AdvSimd.Store(d + 8, c); AdvSimd.Store(d + 12, e);
    }

    [ThreadStatic] private static float[]? t_runs;

    /// <summary>
    /// One <see cref="UpHead"/> output row for oc &lt;= 4, c % 4 == 0: per 4 channels the bilinear
    /// lerps (fma), [relu], + [relu](skip), then fmla by element into one 4-lane accumulator against
    /// the packed [c][8] weights — the resized pixel never leaves registers.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void UpHeadRowNeon(float* r0, float* r1, int* x0, int* x1, float* wx, float wy, float* krow,
        int c, int ow, float* wp, float* bp, float* orow, int oc, bool relu, bool skipRelu, ActivationKind act)
    {
        V4 vY = Vector128.Create(wy), bias = Ld4(bp);
        for (int x = 0; x < ow; x++)
        {
            float* a = r0 + (long)x0[x] * c, b = r0 + (long)x1[x] * c, e = r1 + (long)x0[x] * c, f = r1 + (long)x1[x] * c;
            float* k = krow + (long)x * c;
            V4 vX = Vector128.Create(wx[x]), acc = bias;
            float* w = wp;
            for (int ch = 0; ch < c; ch += 4, w += 32)
            {
                V4 va = Ld4(a + ch), ve = Ld4(e + ch);
                V4 top = Fma4(va, AdvSimd.Subtract(Ld4(b + ch), va), vX);
                V4 bot = Fma4(ve, AdvSimd.Subtract(Ld4(f + ch), ve), vX);
                V4 v = Fma4(top, AdvSimd.Subtract(bot, top), vY);
                if (relu) v = AdvSimd.Max(v, V4.Zero);
                V4 s = Ld4(k + ch);
                v = AdvSimd.Add(v, skipRelu ? AdvSimd.Max(s, V4.Zero) : s);
                acc = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(acc, Ld4(w), v, 0);
                acc = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(acc, Ld4(w + 8), v, 1);
                acc = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(acc, Ld4(w + 16), v, 2);
                acc = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(acc, Ld4(w + 24), v, 3);
            }
            acc = Act4(acc, act);
            float* o = orow + (long)x * oc;
            o[0] = acc.ToScalar();
            if (oc > 1) o[1] = acc.GetElement(1);
            if (oc > 2) o[2] = acc.GetElement(2);
            if (oc > 3) o[3] = acc.GetElement(3);
        }
    }

    /// <summary>Thread-local pinned scratch (the pointer outlives any fixed scope).</summary>
    private static T* Pinned<T>(ref T[]? slot, int n) where T : unmanaged
    {
        if (slot is not { } b || b.Length < n) slot = b = GC.AllocateUninitializedArray<T>(n, pinned: true);
        return (T*)Unsafe.AsPointer(ref System.Runtime.InteropServices.MemoryMarshal.GetArrayDataReference(b));
    }

    /// <summary>
    /// <see cref="DenseRow"/> on NEON: a k×k conv as kh taps over kernel-row runs. For each tap row
    /// the kw·ic inputs of a pixel are contiguous in NHWC (weights [ky][kx][ic][16] are already
    /// [ky][kx·ic][16]), so interior pixels point straight into the source; pixels whose run
    /// crosses a padded / reflected border are gathered into a thread-local buffer, constant-pad
    /// rows read <see cref="DenseArgs.PadVec"/>.
    /// </summary>
    private static void DenseRowNeon(in DenseArgs a, int y, int blk, float* orow, float* arow)
    {
        int kh = a.Kh, kw = a.Kw, ic = a.Ic, ow = a.Ow, oc = a.Oc, run = kw * ic;
        float** ptrs = stackalloc float*[Tile * kh];
        float* part = stackalloc float[Tile * Ob];
        float* wb = a.W + (long)blk * kh * run * Ob;
        float* bb = a.Bias + blk * Ob;
        int lanes = Math.Min(Ob, oc - blk * Ob);
        bool direct = lanes == Ob && arow == null && a.Act is ActivationKind.None or ActivationKind.Relu;
        float* gather = null;
        for (int x0 = 0; x0 < ow; x0 += Tile)
        {
            int rows = Math.Min(Tile, ow - x0);
            for (int ky = 0; ky < kh; ky++)
            {
                int r = a.Rm[y * kh + ky];
                for (int i = 0; i < Tile; i++)
                {
                    int* cmx = a.Cm + (x0 + Math.Min(i, rows - 1)) * kw;
                    int cx = cmx[0];
                    float* p;
                    if (r < 0) p = a.PadVec;
                    else if (cx >= 0 && cmx[kw - 1] == cx + kw - 1) p = a.Src + ((long)r * a.Iw + cx) * ic;
                    else
                    {
                        if (gather == null) gather = Pinned(ref t_runs, Tile * kh * run);
                        p = gather + (ky * Tile + i) * run;
                        for (int kx = 0; kx < kw; kx++)
                        {
                            int c = cmx[kx];
                            Buffer.MemoryCopy(c < 0 ? a.PadVec : a.Src + ((long)r * a.Iw + c) * ic, p + kx * ic, ic * 4L, ic * 4L);
                        }
                    }
                    ptrs[ky * Tile + i] = p;
                }
            }
            long o = (long)x0 * oc + blk * Ob;
            if (direct && rows == Tile)
            {
                // full tile, no residual / transcendental epilogue: store straight from the accumulators
                TileNeonK(ptrs, kh, 0, run, (long)run * Ob, wb, bb, orow + o, oc, a.Act == ActivationKind.Relu);
                continue;
            }
            TileNeonK(ptrs, kh, run, wb, bb, part, Ob);
            TileNeonStore(part, rows, orow + o, arow != null ? arow + o : null, oc, lanes, a.Act);
        }
    }
}
