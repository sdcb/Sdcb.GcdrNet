using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Sdcb.GcdrNet.Engine;

using F32x8 = System.Runtime.Intrinsics.Vector256<float>;
using V256 = System.Runtime.Intrinsics.Vector256;

namespace Sdcb.GcdrNet.Kernels;

/// <summary>Backend-private packed conv weights for the NHWC CPU kernels (lives in <see cref="ConvP.KernelPlan"/>).</summary>
internal sealed class NhwcConvPlan
{
    public required float[] W;      // dense: [oc/16][tap][ic][16] (oc zero-padded); depthwise: [tap][c]
    public required float[] Bias;   // dense: padded to blocks*16; depthwise: [c]
    public bool Depthwise;
    public int Blocks;
    public float[]? WinoU;  // Winograd F(4x4,3x3) weights [36][oc/16][ic][16], null = direct conv
}

/// <summary>Packed <see cref="ExecOp.UpHead"/> weights: [c][8] oc lanes (zero-padded) + bias[8].</summary>
internal sealed class UpHeadPlan
{
    public required float[] W;
    public required float[] Bias;
    public int Oc, C;
}

/// <summary>
/// Channels-last (NHWC, N==1) kernels. Shapes are passed in storage order
/// ([h,w,c]). Every pad mode goes through precomputed source row/col maps
/// (-1 = constant pad → a pad-value pixel vector), so the inner loops never
/// branch on borders. Dense/pointwise conv: register-tiled 6 pixels × 16 oc
/// AVX2 micro-kernel (after SimdPaddleOCR's Nhwc.Conv) with bias-init and a
/// fused residual + activation epilogue; other tiers use Vector&lt;float&gt;.
/// </summary>
internal static unsafe partial class Nhwc
{
    private const int Ob = 16, Tile = 6;
    private static readonly ParallelOptions PO = new() { MaxDegreeOfParallelism = Avx2Kernels.Threads };
    private static bool UseAvx => Avx2.IsSupported && Fma.IsSupported;

    private static void For(int n, long work, Action<int> body)
    {
        if (work < 32768 || n == 1) { for (int i = 0; i < n; i++) body(i); return; }
        Parallel.For(0, n, PO, body);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int Reflect(int i, int dim)
    {
        if (dim <= 1) return 0;
        while (i < 0 || i >= dim) i = i < 0 ? -i : 2 * dim - 2 - i;
        return i;
    }

    /// <summary>map[o*k+kk] = source index of output o, tap kk, or -1 for constant pad.</summary>
    private static int[] Map(int outLen, int k, int stride, int pad, int inLen, PadMode mode)
    {
        int[] m = new int[outLen * k];
        for (int o = 0; o < outLen; o++)
            for (int kk = 0; kk < k; kk++)
            {
                int s = o * stride + kk - pad;
                m[o * k + kk] = (uint)s < (uint)inLen ? s : mode == PadMode.Reflect ? Reflect(s, inLen) : -1;
            }
        return m;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float ActS(float v, ActivationKind act) =>
        act == ActivationKind.Relu ? MathF.Max(v, 0f) :
        act == ActivationKind.Sigmoid ? 1f / (1f + MathF.Exp(-v)) :
        act == ActivationKind.Gelu ? 0.5f * v * (1f + SimdMath.ErfScalar(v * 0.70710678f)) : v;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static F32x8 Act8(F32x8 v, ActivationKind act) => act switch
    {
        ActivationKind.Relu => V256.Max(v, F32x8.Zero),
        ActivationKind.Sigmoid => SimdMath.SigmoidCore(v),
        ActivationKind.Gelu => SimdMath.GeluCore(v),
        _ => v,
    };

    // ======================== weight packing ========================
    public static NhwcConvPlan Pack(ReadOnlySpan<float> w, int[] wShape, ReadOnlySpan<float> bias, int group, int ic)
    {
        int oc = wShape[0], icg = wShape[1], taps = wShape[2] * wShape[3];
        if (group > 1)
        {
            if (group != ic || icg != 1 || oc != ic)
                throw new NotSupportedException($"NHWC grouped conv supports depthwise only (g={group}, ic={ic}, oc={oc}).");
            float[] pw = new float[taps * oc];
            for (int c = 0; c < oc; c++)
                for (int t = 0; t < taps; t++) pw[t * oc + c] = w[c * taps + t];
            float[] pb = new float[oc];
            if (!bias.IsEmpty) bias.CopyTo(pb);
            return new NhwcConvPlan { W = pw, Bias = pb, Depthwise = true };
        }
        int nb = (oc + Ob - 1) / Ob;
        float[] p = new float[checked(nb * taps * ic * Ob)];
        for (int o = 0; o < oc; o++)
            for (int c = 0; c < ic; c++)
                for (int t = 0; t < taps; t++)
                    p[(((o / Ob) * taps + t) * ic + c) * Ob + o % Ob] = w[(o * ic + c) * taps + t];
        float[] b = new float[nb * Ob];
        if (!bias.IsEmpty) bias.CopyTo(b);
        return new NhwcConvPlan { W = p, Bias = b, Blocks = nb };
    }

    /// <summary>Packs Gemm B ([K,N], or [N,K] when transB) as [N/16][K][16] — a 1x1 conv panel.</summary>
    public static NhwcConvPlan PackGemm(ReadOnlySpan<float> b, int k, int n, bool transB, ReadOnlySpan<float> bias)
    {
        int nb = (n + Ob - 1) / Ob;
        float[] p = new float[checked(nb * k * Ob)];
        for (int o = 0; o < n; o++)
            for (int c = 0; c < k; c++)
                p[((o / Ob) * k + c) * Ob + o % Ob] = transB ? b[o * k + c] : b[c * n + o];
        float[] pb = new float[nb * Ob];
        if (!bias.IsEmpty) bias.CopyTo(pb);
        return new NhwcConvPlan { W = p, Bias = pb, Blocks = nb };
    }

    /// <summary>dst[M,N] = act(a[M,K]·B + bias) through the 6x16 conv micro-kernel.</summary>
    public static void Gemm(ReadOnlySpan<float> a, int m, int k, Span<float> dst, int n,
        NhwcConvPlan kp, ActivationKind act)
    {
        const int G = 8;   // row tiles per task sharing one weight panel walk
        int nb = kp.Blocks, tiles = (m + Tile - 1) / Tile, groups = (tiles + G - 1) / G;
        bool avx = UseAvx;
        fixed (float* ap0 = a, dp0 = dst, wp0 = kp.W, bp0 = kp.Bias)
        {
            float* ap = ap0, dp = dp0, wp = wp0, bp = bp0;
            For(groups * nb, (long)m * n * k / 64, t =>
            {
                int g = t / nb, blk = t % nb;
                float** ptrs = stackalloc float*[Tile];
                float* wb = wp + (long)blk * k * Ob, bb = bp + blk * Ob;
                int lanes = Math.Min(Ob, n - blk * Ob);
                for (int tile = g * G, te = Math.Min(tiles, tile + G); tile < te; tile++)
                {
                    int m0 = tile * Tile, rows = Math.Min(Tile, m - m0);
                    for (int i = 0; i < Tile; i++) ptrs[i] = ap + (long)(m0 + Math.Min(i, rows - 1)) * k;
                    float* outp = dp + (long)m0 * n + blk * Ob;
                    if (avx) TileAvx(ptrs, rows, 1, k, wb, bb, outp, null, n, lanes, act);
                    else TileVec(ptrs, rows, 1, k, wb, bb, outp, null, n, lanes, act);
                }
            });
        }
    }

    // ======================== conv ========================
    public static void Conv(ReadOnlySpan<float> src, int ih, int iw, int ic,
        Span<float> dst, int oh, int ow, int oc, ConvP p, NhwcConvPlan kp, ReadOnlySpan<float> add)
    {
        if (!add.IsEmpty && add.Length != dst.Length)
            throw new NotSupportedException($"conv residual addend has {add.Length} elems, expected {dst.Length}.");
        if (kp.WinoU is not null)
        {
            WinogradConv(src, ih, iw, ic, dst, oh, ow, oc, p, kp, add);
            return;
        }
        int kh = p.Kernel[0], kw = p.Kernel[1];
        int[] rm = Map(oh, kh, p.Strides[0], p.Pads[0], ih, p.PadMode);
        int[] cm = Map(ow, kw, p.Strides[1], p.Pads[1], iw, p.PadMode);
        float[] padVec = new float[ic + Ob];
        Array.Fill(padVec, p.PadValue);
        if (kp.Depthwise)
            Depthwise(src, iw, ic, dst, oh, ow, kh, kw, rm, cm, padVec, kp, add, p.Act);
        else
            Dense(src, iw, ic, dst, oh, ow, oc, kh, kw, rm, cm, padVec, kp, add, p.Act);
    }

    private static void Dense(ReadOnlySpan<float> src, int iw, int ic, Span<float> dst,
        int oh, int ow, int oc, int kh, int kw, int[] rm, int[] cm, float[] padVec,
        NhwcConvPlan kp, ReadOnlySpan<float> add, ActivationKind act)
    {
        int taps = kh * kw, nb = kp.Blocks;
        bool hasAdd = !add.IsEmpty, avx = UseAvx;
        fixed (float* sp0 = src, dp0 = dst, ap0 = add, wp0 = kp.W, bp0 = kp.Bias, pv0 = padVec)
        fixed (int* rm0 = rm, cm0 = cm)
        {
            float* sp = sp0, dp = dp0, ap = ap0, wp = wp0, bp = bp0, pv = pv0;
            int* rmp = rm0, cmp = cm0;
            For(oh * nb, (long)oh * ow * oc * taps * ic / 64, t =>
            {
                int y = t / nb, blk = t % nb;
                float** ptrs = stackalloc float*[Tile * taps];
                float* wb = wp + (long)blk * taps * ic * Ob;
                float* bb = bp + blk * Ob;
                int lanes = Math.Min(Ob, oc - blk * Ob);
                for (int x0 = 0; x0 < ow; x0 += Tile)
                {
                    int rows = Math.Min(Tile, ow - x0);
                    for (int tap = 0; tap < taps; tap++)
                    {
                        int r = rmp[y * kh + tap / kw];
                        for (int i = 0; i < Tile; i++)
                        {
                            int c = cmp[(x0 + Math.Min(i, rows - 1)) * kw + tap % kw];
                            ptrs[tap * Tile + i] = r < 0 || c < 0 ? pv : sp + ((long)r * iw + c) * ic;
                        }
                    }
                    long o = ((long)y * ow + x0) * oc + blk * Ob;
                    float* outp = dp + o;
                    float* addp = hasAdd ? ap + o : null;
                    if (avx) TileAvx(ptrs, rows, taps, ic, wb, bb, outp, addp, oc, lanes, act);
                    else TileVec(ptrs, rows, taps, ic, wb, bb, outp, addp, oc, lanes, act);
                }
            });
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void TileAvx(float** ptrs, int rows, int taps, int ic, float* wb, float* bb,
        float* outp, float* addp, int oc, int lanes, ActivationKind act)
    {
        F32x8 a0 = Avx.LoadVector256(bb), b0 = Avx.LoadVector256(bb + 8);
        F32x8 a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0;
        F32x8 b1 = b0, b2 = b0, b3 = b0, b4 = b0, b5 = b0;
        for (int tap = 0; tap < taps; tap++)
        {
            float** pp = ptrs + tap * Tile;
            float* p0 = pp[0], p1 = pp[1], p2 = pp[2], p3 = pp[3], p4 = pp[4], p5 = pp[5];
            float* w = wb + (long)tap * ic * Ob;
            for (int c = 0; c < ic; c++, w += Ob)
            {
                F32x8 w0 = Avx.LoadVector256(w), w1 = Avx.LoadVector256(w + 8);
                F32x8 v0 = Avx.BroadcastScalarToVector256(p0 + c), v1 = Avx.BroadcastScalarToVector256(p1 + c);
                a0 = Fma.MultiplyAdd(v0, w0, a0); b0 = Fma.MultiplyAdd(v0, w1, b0);
                a1 = Fma.MultiplyAdd(v1, w0, a1); b1 = Fma.MultiplyAdd(v1, w1, b1);
                F32x8 v2 = Avx.BroadcastScalarToVector256(p2 + c), v3 = Avx.BroadcastScalarToVector256(p3 + c);
                a2 = Fma.MultiplyAdd(v2, w0, a2); b2 = Fma.MultiplyAdd(v2, w1, b2);
                a3 = Fma.MultiplyAdd(v3, w0, a3); b3 = Fma.MultiplyAdd(v3, w1, b3);
                F32x8 v4 = Avx.BroadcastScalarToVector256(p4 + c), v5 = Avx.BroadcastScalarToVector256(p5 + c);
                a4 = Fma.MultiplyAdd(v4, w0, a4); b4 = Fma.MultiplyAdd(v4, w1, b4);
                a5 = Fma.MultiplyAdd(v5, w0, a5); b5 = Fma.MultiplyAdd(v5, w1, b5);
            }
        }
        Store16(outp, addp, a0, b0, lanes, act);
        if (rows > 1) Store16(outp + oc, addp == null ? null : addp + oc, a1, b1, lanes, act);
        if (rows > 2) Store16(outp + 2 * oc, addp == null ? null : addp + 2 * oc, a2, b2, lanes, act);
        if (rows > 3) Store16(outp + 3 * oc, addp == null ? null : addp + 3 * oc, a3, b3, lanes, act);
        if (rows > 4) Store16(outp + 4 * oc, addp == null ? null : addp + 4 * oc, a4, b4, lanes, act);
        if (rows > 5) Store16(outp + 5 * oc, addp == null ? null : addp + 5 * oc, a5, b5, lanes, act);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Store16(float* o, float* add, F32x8 a, F32x8 b, int lanes, ActivationKind act)
    {
        if (lanes == Ob)
        {
            if (add != null) { a = Avx.Add(a, Avx.LoadVector256(add)); b = Avx.Add(b, Avx.LoadVector256(add + 8)); }
            Avx.Store(o, Act8(a, act)); Avx.Store(o + 8, Act8(b, act));
            return;
        }
        // channel tail (e.g. the final oc=3 conv): only the valid lanes touch memory
        float* t = stackalloc float[Ob];
        Avx.Store(t, a); Avx.Store(t + 8, b);
        for (int l = 0; l < lanes; l++)
            o[l] = ActS(add != null ? t[l] + add[l] : t[l], act);
    }

    private static void TileVec(float** ptrs, int rows, int taps, int ic, float* wb, float* bb,
        float* outp, float* addp, int oc, int lanes, ActivationKind act)
    {
        int VL = Vector<float>.Count;
        // software Vector<T> (DOTNET_EnableHWIntrinsic=0) is far slower than plain scalar
        bool vec = Vector.IsHardwareAccelerated && Ob % VL == 0;
        float* acc = stackalloc float[Ob];
        for (int i = 0; i < rows; i++)
        {
            for (int l = 0; l < Ob; l++) acc[l] = bb[l];
            for (int tap = 0; tap < taps; tap++)
            {
                float* p = ptrs[tap * Tile + i];
                float* w = wb + (long)tap * ic * Ob;
                for (int c = 0; c < ic; c++, w += Ob)
                {
                    float v = p[c];
                    if (vec)
                    {
                        var vv = new Vector<float>(v);
                        for (int l = 0; l < Ob; l += VL)
                            Simd.VectorStore(acc + l, Simd.VectorLoad(acc + l) + Simd.VectorLoad(w + l) * vv);
                    }
                    else
                        for (int l = 0; l < Ob; l++) acc[l] += v * w[l];
                }
            }
            float* o = outp + (long)i * oc;
            float* ad = addp == null ? null : addp + (long)i * oc;
            for (int l = 0; l < lanes; l++)
                o[l] = ActS(ad != null ? acc[l] + ad[l] : acc[l], act);
        }
    }

    private static void Depthwise(ReadOnlySpan<float> src, int iw, int c, Span<float> dst,
        int oh, int ow, int kh, int kw, int[] rm, int[] cm, float[] padVec,
        NhwcConvPlan kp, ReadOnlySpan<float> add, ActivationKind act)
    {
        int taps = kh * kw;
        bool hasAdd = !add.IsEmpty, avx = UseAvx;
        int VL = Vector<float>.Count;
        fixed (float* sp0 = src, dp0 = dst, ap0 = add, wp0 = kp.W, bp0 = kp.Bias, pv0 = padVec)
        fixed (int* rm0 = rm, cm0 = cm)
        {
            float* sp = sp0, dp = dp0, ap = ap0, wp = wp0, bp = bp0, pv = pv0;
            int* rmp = rm0, cmp = cm0;
            For(oh, (long)oh * ow * c * taps, y =>
            {
                float** ptrs = stackalloc float*[taps];
                for (int x = 0; x < ow; x++)
                {
                    for (int tap = 0; tap < taps; tap++)
                    {
                        int r = rmp[y * kh + tap / kw], cc = cmp[x * kw + tap % kw];
                        ptrs[tap] = r < 0 || cc < 0 ? pv : sp + ((long)r * iw + cc) * c;
                    }
                    long ob = ((long)y * ow + x) * c;
                    float* o = dp + ob;
                    float* ad = hasAdd ? ap + ob : null;
                    int ch = 0;
                    if (avx)
                    {
                        for (; ch + 8 <= c; ch += 8)
                        {
                            F32x8 acc = Avx.LoadVector256(bp + ch);
                            for (int tap = 0; tap < taps; tap++)
                                acc = Fma.MultiplyAdd(Avx.LoadVector256(ptrs[tap] + ch), Avx.LoadVector256(wp + tap * c + ch), acc);
                            if (ad != null) acc = Avx.Add(acc, Avx.LoadVector256(ad + ch));
                            Avx.Store(o + ch, Act8(acc, act));
                        }
                    }
                    else if (Vector.IsHardwareAccelerated && act is ActivationKind.None or ActivationKind.Relu)
                    {
                        for (; ch + VL <= c; ch += VL)
                        {
                            var acc = Simd.VectorLoad(bp + ch);
                            for (int tap = 0; tap < taps; tap++)
                                acc += Simd.VectorLoad(ptrs[tap] + ch) * Simd.VectorLoad(wp + tap * c + ch);
                            if (ad != null) acc += Simd.VectorLoad(ad + ch);
                            if (act == ActivationKind.Relu) acc = Vector.Max(acc, Vector<float>.Zero);
                            Simd.VectorStore(o + ch, acc);
                        }
                    }
                    for (; ch < c; ch++)
                    {
                        float acc = bp[ch];
                        for (int tap = 0; tap < taps; tap++) acc += ptrs[tap][ch] * wp[tap * c + ch];
                        if (ad != null) acc += ad[ch];
                        o[ch] = ActS(acc, act);
                    }
                }
            });
        }
    }

    // ======================== pixel-vector helpers ========================
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Copy(float* s, float* d, int n) =>
        Buffer.MemoryCopy(s, d, (long)n * 4, (long)n * 4);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Fill(float* d, int n, float v) => new Span<float>(d, n).Fill(v);

    // ======================== maxpool ========================
    public static void MaxPool(ReadOnlySpan<float> src, int ih, int iw, int c,
        Span<float> dst, int oh, int ow, PoolP p)
    {
        int kh = p.Kernel[0], kw = p.Kernel[1];
        int[] rm = Map(oh, kh, p.Strides[0], p.Pads[0], ih, PadMode.Constant);
        int[] cm = Map(ow, kw, p.Strides[1], p.Pads[1], iw, PadMode.Constant);
        bool relu = p.FuseRelu;
        int VL = Vector<float>.Count;
        fixed (float* sp0 = src, dp0 = dst)
        {
            float* sp = sp0, dp = dp0;
            For(oh, (long)oh * ow * c * kh * kw, y =>
            {
                for (int x = 0; x < ow; x++)
                {
                    float* o = dp + ((long)y * ow + x) * c;
                    bool any = false;
                    for (int ky = 0; ky < kh; ky++)
                    {
                        int r = rm[y * kh + ky];
                        if (r < 0) continue;
                        for (int kx = 0; kx < kw; kx++)
                        {
                            int cc = cm[x * kw + kx];
                            if (cc < 0) continue;
                            float* s = sp + ((long)r * iw + cc) * c;
                            if (!any) { Copy(s, o, c); any = true; continue; }
                            int ch = 0;
                            for (; ch + VL <= c; ch += VL)
                                Simd.VectorStore(o + ch, Vector.Max(Simd.VectorLoad(o + ch), Simd.VectorLoad(s + ch)));
                            for (; ch < c; ch++) o[ch] = MathF.Max(o[ch], s[ch]);
                        }
                    }
                    if (!any) Fill(o, c, 0f);
                    else if (relu)
                        for (int ch = 0; ch < c; ch++) o[ch] = MathF.Max(o[ch], 0f);
                }
            });
        }
    }

    // ======================== bilinear resize (half-pixel) ========================
    private static void ResizeAxis(int outLen, int inLen, float scale, int[] i0, int[] i1, float[] wt)
    {
        for (int o = 0; o < outLen; o++)
        {
            float f = (o + 0.5f) / scale - 0.5f;
            float fl = MathF.Floor(f);
            i0[o] = Math.Clamp((int)fl, 0, inLen - 1);
            i1[o] = Math.Clamp((int)fl + 1, 0, inLen - 1);
            wt[o] = f - fl;
        }
    }

    /// <summary>One output pixel of the bilinear resize into <paramref name="o"/> (c channels):
    /// v = [relu](lerp) + [relu](add) — the same per-element ops as separate Resize/Relu/Add nodes.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ResizePixel(float* a, float* b, float* e, float* f, float wX, float wY,
        Vector<float> vX, Vector<float> vY, int c, bool relu, float* add, bool addRelu, float* o)
    {
        int VL = Vector<float>.Count, ch = 0;
        for (; ch + VL <= c; ch += VL)
        {
            var va = Simd.VectorLoad(a + ch); var ve = Simd.VectorLoad(e + ch);
            var top = va + (Simd.VectorLoad(b + ch) - va) * vX;
            var bot = ve + (Simd.VectorLoad(f + ch) - ve) * vX;
            var v = top + (bot - top) * vY;
            if (relu) v = Vector.Max(v, Vector<float>.Zero);
            if (add != null)
            {
                var s = Simd.VectorLoad(add + ch);
                v += addRelu ? Vector.Max(s, Vector<float>.Zero) : s;
            }
            Simd.VectorStore(o + ch, v);
        }
        for (; ch < c; ch++)
        {
            float top = a[ch] + (b[ch] - a[ch]) * wX;
            float bot = e[ch] + (f[ch] - e[ch]) * wX;
            float v = top + (bot - top) * wY;
            if (relu) v = MathF.Max(v, 0f);
            if (add != null) v += addRelu ? MathF.Max(add[ch], 0f) : add[ch];
            o[ch] = v;
        }
    }

    /// <summary>NHWC bilinear resize; optional same-shape addend (out = [relu](resize) + [relu](add)).</summary>
    public static void ResizeBilinear(ReadOnlySpan<float> src, int ih, int iw, int c,
        Span<float> dst, int oh, int ow, float sy, float sx, ActivationKind act,
        ReadOnlySpan<float> add = default, bool addRelu = false)
    {
        if (!add.IsEmpty && add.Length != dst.Length)
            throw new NotSupportedException($"resize addend has {add.Length} elems, expected {dst.Length}.");
        int[] y0 = new int[oh], y1 = new int[oh], x0 = new int[ow], x1 = new int[ow];
        float[] wy = new float[oh], wx = new float[ow];
        ResizeAxis(oh, ih, sy, y0, y1, wy);
        ResizeAxis(ow, iw, sx, x0, x1, wx);
        bool relu = act == ActivationKind.Relu, hasAdd = !add.IsEmpty;
        fixed (float* sp0 = src, dp0 = dst, ap0 = add)
        {
            float* sp = sp0, dp = dp0, ap = ap0;
            For(oh, (long)oh * ow * c, y =>
            {
                float* r0 = sp + (long)y0[y] * iw * c, r1 = sp + (long)y1[y] * iw * c;
                var vY = new Vector<float>(wy[y]);
                float wY = wy[y];
                for (int x = 0; x < ow; x++)
                {
                    long ob = ((long)y * ow + x) * c;
                    ResizePixel(r0 + (long)x0[x] * c, r0 + (long)x1[x] * c, r1 + (long)x0[x] * c, r1 + (long)x1[x] * c,
                        wx[x], wY, new Vector<float>(wx[x]), vY, c, relu, hasAdd ? ap + ob : null, addRelu, dp + ob);
                }
            });
        }
    }

    // ======================== fused upsample tail ========================
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
        Span<float> dst, int oh, int ow, float sy, float sx, UpHeadP p, UpHeadPlan kp)
    {
        if (kp.C != c || skip.Length != (long)oh * ow * c || dst.Length != (long)oh * ow * kp.Oc)
            throw new NotSupportedException("UpHead shape mismatch.");
        int[] y0 = new int[oh], y1 = new int[oh], x0 = new int[ow], x1 = new int[ow];
        float[] wy = new float[oh], wx = new float[ow];
        ResizeAxis(oh, ih, sy, y0, y1, wy);
        ResizeAxis(ow, iw, sx, x0, x1, wx);
        bool relu = p.ResizeRelu, skipRelu = p.SkipRelu, avx = UseAvx;
        int oc = kp.Oc;
        ActivationKind act = p.Act;
        fixed (float* sp0 = src, dp0 = dst, kp0 = skip, wp0 = kp.W, bp0 = kp.Bias)
        {
            float* sp = sp0, dp = dp0, kpp = kp0, wp = wp0, bp = bp0;
            For(oh, (long)oh * ow * c * (oc + 4), y =>
            {
                float* s = stackalloc float[c + 8];
                float* acc = stackalloc float[8];
                float* r0 = sp + (long)y0[y] * iw * c, r1 = sp + (long)y1[y] * iw * c;
                var vY = new Vector<float>(wy[y]);
                float wY = wy[y];
                float* orow = dp + (long)y * ow * oc;
                for (int x = 0; x < ow; x++)
                {
                    ResizePixel(r0 + (long)x0[x] * c, r0 + (long)x1[x] * c, r1 + (long)x0[x] * c, r1 + (long)x1[x] * c,
                        wx[x], wY, new Vector<float>(wx[x]), vY, c, relu, kpp + ((long)y * ow + x) * c, skipRelu, s);
                    float* o = orow + (long)x * oc;
                    if (avx)
                    {
                        F32x8 a = Avx.LoadVector256(bp);
                        float* w = wp;
                        for (int ch = 0; ch < c; ch++, w += 8)
                            a = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(s + ch), Avx.LoadVector256(w), a);
                        Avx.Store(acc, a);
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
            });
        }
    }

    private static void ActRow(float* p, int n, ActivationKind act, bool avx)
    {
        int i = 0;
        if (avx)
            for (; i + 8 <= n; i += 8) Avx.Store(p + i, Act8(Avx.LoadVector256(p + i), act));
        else if (act == ActivationKind.Sigmoid)
        {
            var row = new Span<float>(p + i, n - i);
            SimdMath.Sigmoid(row, row);
            return;
        }
        for (; i < n; i++) p[i] = ActS(p[i], act);
    }

    // ======================== pad (spatial only) ========================
    public static void Pad(ReadOnlySpan<float> src, int ih, int iw, int c,
        Span<float> dst, int oh, int ow, PadP p)
    {
        int rank = p.Pads.Length / 2;
        if (rank != 4) throw new NotSupportedException("NHWC Pad supports rank-4 only.");
        int[] rm = Map(oh, 1, 1, p.Pads[2], ih, p.Mode);
        int[] cm = Map(ow, 1, 1, p.Pads[3], iw, p.Mode);
        float val = p.Value;
        fixed (float* sp0 = src, dp0 = dst)
        {
            float* sp = sp0, dp = dp0;
            For(oh, (long)oh * ow * c, y =>
            {
                int r = rm[y];
                for (int x = 0; x < ow; x++)
                {
                    float* o = dp + ((long)y * ow + x) * c;
                    int cc = cm[x];
                    if (r < 0 || cc < 0) Fill(o, c, val);
                    else Copy(sp + ((long)r * iw + cc) * c, o, c);
                }
            });
        }
    }

    // ======================== channel shift ========================
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
}
