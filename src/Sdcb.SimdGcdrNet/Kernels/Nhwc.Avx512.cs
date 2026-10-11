using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Sdcb.SimdGcdrNet.Engine;

using F32x8 = System.Runtime.Intrinsics.Vector256<float>;
using F32x16 = System.Runtime.Intrinsics.Vector512<float>;

namespace Sdcb.SimdGcdrNet.Kernels;

/// <summary>
/// AVX-512 tier of the NHWC conv / Gemm micro-kernel: <see cref="Tile5"/> pixels × 32 oc
/// (two packed 16-oc blocks, one zmm each) per pass, so every broadcast feeds two FMAs and
/// the 32 zmm registers hold the whole accumulator tile. Per output element the reduction
/// is the AVX2 one (bias-init, taps then channels ascending, one FMA each) and the
/// epilogue reuses <see cref="Store16"/>'s math, so results are bit-identical to AVX2.
/// The k loops store to a scratch tile and never call out (calls make RyuJIT spill the
/// accumulators); the residual/activation epilogue runs afterwards.
/// </summary>
internal static unsafe partial class Nhwc
{
    private const int Tile5 = 10;

    /// <summary>AVX-512 tier switch: <c>Avx512F.IsSupported</c>, off with GCDRNET_AVX512=0 or the
    /// .NET 8/9-style DOTNET_EnableAVX512F=0 (the .NET 10 JIT only honours DOTNET_EnableAVX512).
    /// Mutable for in-process A/B tests.</summary>
    internal static bool UseAvx512 = Avx512F.IsSupported
        && Environment.GetEnvironmentVariable("GCDRNET_AVX512") != "0"
        && Environment.GetEnvironmentVariable("DOTNET_EnableAVX512F") != "0";

    /// <summary>Tile5 rows × 32 oc: dst[r*ds + 0..15] ← block wb, dst[r*ds + 16..31] ← block wb + wStep.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Tile512x32(float** ptrs, int taps, int ic, float* wb, long wStep, float* bb, float* dst, int ds)
    {
        F32x16 a0 = Avx512F.LoadVector512(bb), b0 = Avx512F.LoadVector512(bb + Ob);
        F32x16 a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0, a8 = a0, a9 = a0;
        F32x16 b1 = b0, b2 = b0, b3 = b0, b4 = b0, b5 = b0, b6 = b0, b7 = b0, b8 = b0, b9 = b0;
        for (int tap = 0; tap < taps; tap++)
        {
            float** pp = ptrs + tap * Tile5;
            float* p0 = pp[0], p1 = pp[1], p2 = pp[2], p3 = pp[3], p4 = pp[4], p5 = pp[5], p6 = pp[6], p7 = pp[7], p8 = pp[8], p9 = pp[9];
            float* w = wb + (long)tap * ic * Ob;
            float* x = w + wStep;
            for (int c = 0; c < ic; c++, w += Ob, x += Ob)
            {
                F32x16 wa = Avx512F.LoadVector512(w), wx = Avx512F.LoadVector512(x);
                F32x16 v = Vector512.Create(p0[c]);
                a0 = Avx512F.FusedMultiplyAdd(v, wa, a0); b0 = Avx512F.FusedMultiplyAdd(v, wx, b0);
                v = Vector512.Create(p1[c]);
                a1 = Avx512F.FusedMultiplyAdd(v, wa, a1); b1 = Avx512F.FusedMultiplyAdd(v, wx, b1);
                v = Vector512.Create(p2[c]);
                a2 = Avx512F.FusedMultiplyAdd(v, wa, a2); b2 = Avx512F.FusedMultiplyAdd(v, wx, b2);
                v = Vector512.Create(p3[c]);
                a3 = Avx512F.FusedMultiplyAdd(v, wa, a3); b3 = Avx512F.FusedMultiplyAdd(v, wx, b3);
                v = Vector512.Create(p4[c]);
                a4 = Avx512F.FusedMultiplyAdd(v, wa, a4); b4 = Avx512F.FusedMultiplyAdd(v, wx, b4);
                v = Vector512.Create(p5[c]);
                a5 = Avx512F.FusedMultiplyAdd(v, wa, a5); b5 = Avx512F.FusedMultiplyAdd(v, wx, b5);
                v = Vector512.Create(p6[c]);
                a6 = Avx512F.FusedMultiplyAdd(v, wa, a6); b6 = Avx512F.FusedMultiplyAdd(v, wx, b6);
                v = Vector512.Create(p7[c]);
                a7 = Avx512F.FusedMultiplyAdd(v, wa, a7); b7 = Avx512F.FusedMultiplyAdd(v, wx, b7);
                v = Vector512.Create(p8[c]);
                a8 = Avx512F.FusedMultiplyAdd(v, wa, a8); b8 = Avx512F.FusedMultiplyAdd(v, wx, b8);
                v = Vector512.Create(p9[c]);
                a9 = Avx512F.FusedMultiplyAdd(v, wa, a9); b9 = Avx512F.FusedMultiplyAdd(v, wx, b9);
            }
        }
        Avx512F.Store(dst, a0); Avx512F.Store(dst + Ob, b0); dst += ds;
        Avx512F.Store(dst, a1); Avx512F.Store(dst + Ob, b1); dst += ds;
        Avx512F.Store(dst, a2); Avx512F.Store(dst + Ob, b2); dst += ds;
        Avx512F.Store(dst, a3); Avx512F.Store(dst + Ob, b3); dst += ds;
        Avx512F.Store(dst, a4); Avx512F.Store(dst + Ob, b4); dst += ds;
        Avx512F.Store(dst, a5); Avx512F.Store(dst + Ob, b5); dst += ds;
        Avx512F.Store(dst, a6); Avx512F.Store(dst + Ob, b6); dst += ds;
        Avx512F.Store(dst, a7); Avx512F.Store(dst + Ob, b7); dst += ds;
        Avx512F.Store(dst, a8); Avx512F.Store(dst + Ob, b8); dst += ds;
        Avx512F.Store(dst, a9); Avx512F.Store(dst + Ob, b9);
    }

    /// <summary>Tile5 rows × 16 oc (odd block count tail): dst[r*ds + 0..15].</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Tile512x16(float** ptrs, int taps, int ic, float* wb, float* bb, float* dst, int ds)
    {
        F32x16 a0 = Avx512F.LoadVector512(bb);
        F32x16 a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0, a8 = a0, a9 = a0;
        for (int tap = 0; tap < taps; tap++)
        {
            float** pp = ptrs + tap * Tile5;
            float* p0 = pp[0], p1 = pp[1], p2 = pp[2], p3 = pp[3], p4 = pp[4], p5 = pp[5], p6 = pp[6], p7 = pp[7], p8 = pp[8], p9 = pp[9];
            float* w = wb + (long)tap * ic * Ob;
            for (int c = 0; c < ic; c++, w += Ob)
            {
                F32x16 wa = Avx512F.LoadVector512(w);
                a0 = Avx512F.FusedMultiplyAdd(Vector512.Create(p0[c]), wa, a0);
                a1 = Avx512F.FusedMultiplyAdd(Vector512.Create(p1[c]), wa, a1);
                a2 = Avx512F.FusedMultiplyAdd(Vector512.Create(p2[c]), wa, a2);
                a3 = Avx512F.FusedMultiplyAdd(Vector512.Create(p3[c]), wa, a3);
                a4 = Avx512F.FusedMultiplyAdd(Vector512.Create(p4[c]), wa, a4);
                a5 = Avx512F.FusedMultiplyAdd(Vector512.Create(p5[c]), wa, a5);
                a6 = Avx512F.FusedMultiplyAdd(Vector512.Create(p6[c]), wa, a6);
                a7 = Avx512F.FusedMultiplyAdd(Vector512.Create(p7[c]), wa, a7);
                a8 = Avx512F.FusedMultiplyAdd(Vector512.Create(p8[c]), wa, a8);
                a9 = Avx512F.FusedMultiplyAdd(Vector512.Create(p9[c]), wa, a9);
            }
        }
        Avx512F.Store(dst, a0); dst += ds;
        Avx512F.Store(dst, a1); dst += ds;
        Avx512F.Store(dst, a2); dst += ds;
        Avx512F.Store(dst, a3); dst += ds;
        Avx512F.Store(dst, a4); dst += ds;
        Avx512F.Store(dst, a5); dst += ds;
        Avx512F.Store(dst, a6); dst += ds;
        Avx512F.Store(dst, a7); dst += ds;
        Avx512F.Store(dst, a8); dst += ds;
        Avx512F.Store(dst, a9);
    }

    /// <summary>Runs the 32- or 16-wide kernel for one tile into <paramref name="part"/> ([Tile5][32]).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Tile512(float** ptrs, int nblk, int taps, int ic, float* wb, long wStep, float* bb, float* part)
    {
        if (nblk == 2) Tile512x32(ptrs, taps, ic, wb, wStep, bb, part, 2 * Ob);
        else Tile512x16(ptrs, taps, ic, wb, bb, part, 2 * Ob);
    }

    /// <summary>Epilogue of one tile: part ([Tile5][32]) → rows × (lanes0 + lanes1) outputs, + residual, act.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void Epilogue512(float* part, int rows, int nblk, float* outp, float* addp, int oc,
        int lanes0, int lanes1, ActivationKind act)
    {
        bool simple = act is ActivationKind.None or ActivationKind.Relu;
        for (int r = 0; r < rows; r++, part += 2 * Ob)
        {
            float* o = outp + (long)r * oc;
            float* ad = addp == null ? null : addp + (long)r * oc;
            for (int h = 0; h < nblk; h++)
            {
                int lanes = h == 0 ? lanes0 : lanes1;
                float* s = part + h * Ob, oh = o + h * Ob, ah = ad == null ? null : ad + h * Ob;
                if (lanes == Ob && simple)
                {
                    F32x16 v = Avx512F.LoadVector512(s);
                    if (ah != null) v = Avx512F.Add(v, Avx512F.LoadVector512(ah));
                    if (act == ActivationKind.Relu) v = Avx512F.Max(v, F32x16.Zero);
                    Avx512F.Store(oh, v);
                }
                else
                    Store16(oh, ah, Avx.LoadVector256(s), Avx.LoadVector256(s + 8), lanes, act);
            }
        }
    }

    private static void Dense512(ReadOnlySpan<float> src, int iw, int ic, Span<float> dst,
        int oh, int ow, int oc, int kh, int kw, int[] rm, int[] cm, float[] padVec,
        NhwcConvPlan kp, ReadOnlySpan<float> add, ActivationKind act)
    {
        int taps = kh * kw, nb = kp.Blocks, np = (nb + 1) / 2;
        bool hasAdd = !add.IsEmpty;
        fixed (float* sp0 = src, dp0 = dst, ap0 = add, wp0 = kp.W, bp0 = kp.Bias, pv0 = padVec)
        fixed (int* rm0 = rm, cm0 = cm)
        {
            var a = new DenseArgs(sp0, wp0, bp0, pv0, rm0, cm0, iw, ic, ow, oc, kh, kw, nb, act, true);
            float* dp = dp0, ap = ap0;
            For(oh * np, (long)oh * ow * oc * taps * ic / 64, t =>
            {
                int y = t / np, blk = t % np * 2;
                long o = (long)y * ow * oc;
                Dense512Row(a, y, blk, dp + o, hasAdd ? ap + o : null);
            });
        }
    }

    /// <summary>All oc blocks of direct-conv output row <paramref name="y"/> (the row-level entry of
    /// the fused paths: conv+pool bands, UpHead skip recompute). AVX-512 tier: block pairs through
    /// <see cref="Dense512Row"/> (needs a padVec of ic*kw + 16 floats); else <see cref="DenseRow"/>.</summary>
    private static void ConvRow(in DenseArgs a, int y, float* orow, float* arow)
    {
        if (UseAvx512)
            for (int blk = 0; blk < a.Nb; blk += 2) Dense512Row(a, y, blk, orow, arow);
        else
            for (int blk = 0; blk < a.Nb; blk++) DenseRow(a, y, blk, orow, arow);
    }

    /// <summary>Output row <paramref name="y"/>, oc blocks blk and blk+1 (if present) of a direct conv
    /// into <paramref name="orow"/> (the row's first pixel; addend row likewise).</summary>
    private static void Dense512Row(in DenseArgs a, int y, int blk, float* orow, float* arow)
    {
        int kh = a.Kh, kw = a.Kw, taps = kh * kw, ic = a.Ic, ow = a.Ow, oc = a.Oc, nblk = Math.Min(2, a.Nb - blk);
        long wStep = (long)taps * ic * Ob;
        float** ptrs = stackalloc float*[Tile5 * taps];
        float* part = stackalloc float[Tile5 * 2 * Ob];
        float* wb = a.W + blk * wStep;
        float* bb = a.Bias + blk * Ob;
        int lanes0 = Math.Min(Ob, oc - blk * Ob), lanes1 = nblk == 2 ? Math.Min(Ob, oc - (blk + 1) * Ob) : 0;
        int* rm = a.Rm, cm = a.Cm;
        for (int x0 = 0; x0 < ow; x0 += Tile5)
        {
            int rows = Math.Min(Tile5, ow - x0);
            // every column of the tile reads kw adjacent input pixels (no column pad/reflect): one
            // kernel row is then a single kw*ic float run in the order of the packed [tap][ic]
            // weights, so the tile walks kh "taps" of kw*ic channels (fewer pointers, longer k loops)
            bool merged = kw > 1;
            for (int i = 0; i < rows && merged; i++)
            {
                int* cx = cm + (x0 + i) * kw;
                merged = cx[0] >= 0;
                for (int kx = 1; kx < kw; kx++) merged &= cx[kx] == cx[0] + kx;
            }
            if (merged)
            {
                for (int ky = 0; ky < kh; ky++)
                {
                    int r = rm[y * kh + ky];
                    for (int i = 0; i < Tile5; i++)
                    {
                        int c = cm[(x0 + Math.Min(i, rows - 1)) * kw];
                        ptrs[ky * Tile5 + i] = r < 0 ? a.PadVec : a.Src + ((long)r * a.Iw + c) * ic;
                    }
                }
                Tile512(ptrs, nblk, kh, kw * ic, wb, wStep, bb, part);
            }
            else
            {
                for (int tap = 0; tap < taps; tap++)
                {
                    int r = rm[y * kh + tap / kw];
                    for (int i = 0; i < Tile5; i++)
                    {
                        int c = cm[(x0 + Math.Min(i, rows - 1)) * kw + tap % kw];
                        ptrs[tap * Tile5 + i] = r < 0 || c < 0 ? a.PadVec : a.Src + ((long)r * a.Iw + c) * ic;
                    }
                }
                Tile512(ptrs, nblk, taps, ic, wb, wStep, bb, part);
            }
            long o = (long)x0 * oc + blk * Ob;
            Epilogue512(part, rows, nblk, orow + o, arow != null ? arow + o : null, oc, lanes0, lanes1, a.Act);
        }
    }

    private static void Gemm512(ReadOnlySpan<float> a, int m, int k, Span<float> dst, int n,
        NhwcConvPlan kp, ActivationKind act)
    {
        const int G = 8;   // row tiles per task sharing one weight panel walk
        int nb = kp.Blocks, np = (nb + 1) / 2, tiles = (m + Tile5 - 1) / Tile5, groups = (tiles + G - 1) / G;
        long wStep = (long)k * Ob;
        fixed (float* ap0 = a, dp0 = dst, wp0 = kp.W, bp0 = kp.Bias)
        {
            float* ap = ap0, dp = dp0, wp = wp0, bp = bp0;
            For(groups * np, (long)m * n * k / 64, t =>
            {
                int g = t / np, blk = t % np * 2, nblk = Math.Min(2, nb - blk);
                float** ptrs = stackalloc float*[Tile5];
                float* part = stackalloc float[Tile5 * 2 * Ob];
                float* wb = wp + blk * wStep, bb = bp + blk * Ob;
                int lanes0 = Math.Min(Ob, n - blk * Ob), lanes1 = nblk == 2 ? Math.Min(Ob, n - (blk + 1) * Ob) : 0;
                for (int tile = g * G, te = Math.Min(tiles, tile + G); tile < te; tile++)
                {
                    int m0 = tile * Tile5, rows = Math.Min(Tile5, m - m0);
                    for (int i = 0; i < Tile5; i++) ptrs[i] = ap + (long)(m0 + Math.Min(i, rows - 1)) * k;
                    Tile512(ptrs, nblk, 1, k, wb, wStep, bb, part);
                    Epilogue512(part, rows, nblk, dp + (long)m0 * n + blk * Ob, null, n, lanes0, lanes1, act);
                }
            });
        }
    }

    // ======================== memory-bound NHWC ops (c % 16 == 0) ========================

    /// <summary><see cref="ResizePixel"/> with 16-lane vectors: same per-lane ops (and IEEE Max) → same bits.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ResizePixel512(float* a, float* b, float* e, float* f, F32x16 vX, F32x16 vY,
        int c, bool relu, float* add, bool addRelu, float* o)
    {
        for (int ch = 0; ch < c; ch += 16)
        {
            var va = Vector512.Load(a + ch); var ve = Vector512.Load(e + ch);
            var top = va + (Vector512.Load(b + ch) - va) * vX;
            var bot = ve + (Vector512.Load(f + ch) - ve) * vX;
            var v = top + (bot - top) * vY;
            if (relu) v = Vector512.Max(v, F32x16.Zero);
            if (add != null)
            {
                var sk = Vector512.Load(add + ch);
                v += addRelu ? Vector512.Max(sk, F32x16.Zero) : sk;
            }
            v.Store(o + ch);
        }
    }

    /// <summary>MaxPool, one output pixel per step: max over the valid taps in registers, one store
    /// (the generic path copies the first tap and read-modify-writes the output per tap).</summary>
    private static void MaxPool512(float* sp, int iw, int c, float* dp, int oh, int ow, int kh, int kw,
        int[] rm, int[] cm, bool relu)
    {
        For(oh, (long)oh * ow * c * kh * kw, y =>
        {
            float** taps = stackalloc float*[kh * kw];
            for (int x = 0; x < ow; x++)
            {
                int n = 0;
                for (int ky = 0; ky < kh; ky++)
                {
                    int r = rm[y * kh + ky];
                    if (r < 0) continue;
                    for (int kx = 0; kx < kw; kx++)
                    {
                        int cc = cm[x * kw + kx];
                        if (cc >= 0) taps[n++] = sp + ((long)r * iw + cc) * c;
                    }
                }
                float* o = dp + ((long)y * ow + x) * c;
                if (n == 0) { Fill(o, c, 0f); continue; }
                for (int ch = 0; ch < c; ch += 16)
                {
                    var v = Vector512.Load(taps[0] + ch);
                    for (int k = 1; k < n; k++) v = Vector512.Max(v, Vector512.Load(taps[k] + ch));
                    if (relu) v = Vector512.Max(v, F32x16.Zero);
                    v.Store(o + ch);
                }
            }
        });
    }

    /// <summary>
    /// <see cref="UpHead"/> for the AVX-512 tier: 16-lane resize, and the c-long oc dot runs as four
    /// interleaved per-pixel FMA chains (one chain per pixel was FMA-latency bound). Each pixel keeps
    /// its bias-initialized, channel-ascending chain, so outputs match the AVX2 path bit for bit.
    /// </summary>
    private static void UpHead512(float* sp, int iw, int c, float* kpp, bool skipConv, DenseArgs ca, float* dp, int oh, int ow,
        int[] y0, int[] y1, int[] x0, int[] x1, float[] wy, float[] wx, bool relu, bool skipRelu,
        float* wp, float* bp, int oc, ActivationKind act)
    {
        bool v512 = c % 16 == 0;
        For(oh, (long)oh * ow * c * (oc + 4), y =>
        {
            float* s = stackalloc float[4 * c + 8];
            float* acc = stackalloc float[4 * 8];
            float* r0 = sp + (long)y0[y] * iw * c, r1 = sp + (long)y1[y] * iw * c;
            var vY = new System.Numerics.Vector<float>(wy[y]);
            var vY16 = Vector512.Create(wy[y]);
            float wY = wy[y];
            float* orow = dp + (long)y * ow * oc;
            float* krow = kpp + (long)y * ow * c;
            float[]? band = skipConv ? BandBuf((long)ow * c) : null;
            fixed (float* sb = band)
            {
                if (skipConv)
                {
                    ConvRow(ca, y, sb, null);   // the skip row, recomputed by the same conv kernel
                    krow = sb;
                }
                for (int x = 0; x < ow; x += 4)
                {
                    int n = Math.Min(4, ow - x);
                    for (int i = 0; i < n; i++)
                    {
                        int xx = x + i;
                        float* a = r0 + (long)x0[xx] * c, b = r0 + (long)x1[xx] * c, e = r1 + (long)x0[xx] * c, f = r1 + (long)x1[xx] * c;
                        if (v512) ResizePixel512(a, b, e, f, Vector512.Create(wx[xx]), vY16, c, relu, krow + (long)xx * c, skipRelu, s + i * c);
                        else ResizePixel(a, b, e, f, wx[xx], wY, new System.Numerics.Vector<float>(wx[xx]), vY, c, relu, krow + (long)xx * c, skipRelu, s + i * c);
                    }
                    UpHeadDot4(s, c, wp, bp, acc, n);
                    for (int i = 0; i < n; i++)
                    {
                        float* o = orow + (long)(x + i) * oc;
                        for (int l = 0; l < oc; l++) o[l] = acc[i * 8 + l];
                    }
                }
                if (act != ActivationKind.None) ActRow(orow, ow * oc, act, true);
            }
        });
    }

    /// <summary>acc[i*8 + l] = bias[l] + Σ_ch s_i[ch]·w[ch][l] for 4 pixels (s_i = s + i*c), chains interleaved.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void UpHeadDot4(float* s, int c, float* wp, float* bp, float* acc, int n)
    {
        float* s0 = s, s1 = n > 1 ? s + c : s, s2 = n > 2 ? s + 2 * c : s, s3 = n > 3 ? s + 3 * c : s;
        F32x8 a0 = Avx.LoadVector256(bp), a1 = a0, a2 = a0, a3 = a0;
        float* w = wp;
        for (int ch = 0; ch < c; ch++, w += 8)
        {
            F32x8 wv = Avx.LoadVector256(w);
            a0 = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(s0 + ch), wv, a0);
            a1 = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(s1 + ch), wv, a1);
            a2 = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(s2 + ch), wv, a2);
            a3 = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(s3 + ch), wv, a3);
        }
        Avx.Store(acc, a0); Avx.Store(acc + 8, a1); Avx.Store(acc + 16, a2); Avx.Store(acc + 24, a3);
    }
}
