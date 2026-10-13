using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.Wasm;
using System.Runtime.Intrinsics.X86;
using Sdcb.SimdGcdrNet.Engine;
using F32x8 = System.Runtime.Intrinsics.Vector256<float>;

namespace Sdcb.SimdGcdrNet.Kernels;

/// <summary>
/// Winograd F(4x4, 3x3) for dense 3x3 stride-1 NHWC convs: 36 multiplies per
/// 4x4 output tile and (ic, oc) pair instead of 144. Tiles are processed in
/// cache-sized blocks per task — input transform into a block-local
/// V[36][T][ic], then per 16-oc block 36 small GEMMs through the 6x16 conv
/// micro-kernel into M[36][T][16], then the output transform with the usual
/// bias / residual / activation epilogue — so the 2.25x-expanded transform
/// tensors never leave L2.
/// </summary>
internal static unsafe partial class Nhwc
{
    private static readonly bool s_winograd = Environment.GetEnvironmentVariable("GCDRNET_WINOGRAD") != "0";

    [ThreadStatic] private static float[]? t_wino;

    /// <summary>Dense 3x3 s1 convs with enough channels for the transforms to amortize.</summary>
    internal static bool WinogradEligible(ConvP p, int[] wShape, int ic) =>
        s_winograd && Vector.IsHardwareAccelerated && p.Group == 1
        && wShape[2] == 3 && wShape[3] == 3 && p.Strides[0] == 1 && p.Strides[1] == 1
        && p.Dilations[0] == 1 && p.Dilations[1] == 1
        && ic >= 16 && ic % Vector<float>.Count == 0 && ic % 8 == 0;

    // G (6x3) of F(4,3): U = G g G^T
    private static readonly double[,] s_g =
    {
        { 1.0 / 4, 0, 0 },
        { -1.0 / 6, -1.0 / 6, -1.0 / 6 },
        { -1.0 / 6, 1.0 / 6, -1.0 / 6 },
        { 1.0 / 24, 1.0 / 12, 1.0 / 6 },
        { 1.0 / 24, -1.0 / 12, 1.0 / 6 },
        { 0, 0, 1 },
    };

    /// <summary>Transformed weights [36][oc/16][ic][16] (oc zero-padded), from ONNX [oc][ic][3][3].</summary>
    internal static float[] PackWinograd(ReadOnlySpan<float> w, int oc, int ic)
    {
        int nb = (oc + Ob - 1) / Ob;
        float[] u = new float[checked(36 * nb * ic * Ob)];
        double* gg = stackalloc double[18];
        for (int o = 0; o < oc; o++)
            for (int c = 0; c < ic; c++)
            {
                ReadOnlySpan<float> k = w.Slice((o * ic + c) * 9, 9);
                // gg = G·k (6x3)
                for (int i = 0; i < 6; i++)
                    for (int j = 0; j < 3; j++)
                        gg[i * 3 + j] = s_g[i, 0] * k[j] + s_g[i, 1] * k[3 + j] + s_g[i, 2] * k[6 + j];
                // U = gg·G^T (6x6)
                for (int i = 0; i < 6; i++)
                    for (int j = 0; j < 6; j++)
                    {
                        double v = gg[i * 3] * s_g[j, 0] + gg[i * 3 + 1] * s_g[j, 1] + gg[i * 3 + 2] * s_g[j, 2];
                        u[(((i * 6 + j) * nb + o / Ob) * ic + c) * Ob + o % Ob] = (float)v;
                    }
            }
        return u;
    }

    /// <summary>Per-call Winograd constants shared by every tile block.</summary>
    private readonly unsafe struct WinoArgs
    {
        public readonly float* Src, U, Bias, PadVec;
        public readonly int Ih, Iw, Ic, Oh, Ow, Oc, Tw, T, Nb, PadT, PadL;
        public readonly bool Reflect, Avx, Avx512, Pk;
        public readonly ActivationKind Act;

        public WinoArgs(float* src, float* u, float* bias, float* padVec, int ih, int iw, int ic,
            int oh, int ow, int oc, int nb, ConvP p)
        {
            Src = src; U = u; Bias = bias; PadVec = padVec;
            Ih = ih; Iw = iw; Ic = ic; Oh = oh; Ow = ow; Oc = oc; Nb = nb;
            Tw = (ow + 3) / 4;
            Avx512 = UseAvx512;
            Avx = UseAvx;
            Pk = UsePacked;
            // V block ~256 KB (AVX2); the AVX-512 tier (Tile5-row x 32-oc GEMM tiles) measured
            // best at ~64 KB on Zen 5, NEON at ~16 KB (one Tile row for ic >= 32: four RK3399 A53s share 512 KB L2)
            int tr = Avx512 ? Tile5 : Pk ? PkTile : Tile, vBudget = Avx512 ? 16384 : UseNeon ? 4096 : 65536;
            T = Math.Clamp(vBudget / (36 * ic) / tr * tr, tr, 60 / tr * tr);
            PadT = p.Pads[0]; PadL = p.Pads[1];
            Reflect = p.PadMode == PadMode.Reflect;
            Act = p.Act;
        }
    }

    private static void WinogradConv(ReadOnlySpan<float> src, int ih, int iw, int ic,
        Span<float> dst, int oh, int ow, int oc, ConvP p, NhwcConvPlan kp, ReadOnlySpan<float> add)
    {
        float[] padVec = new float[ic];
        Array.Fill(padVec, p.PadValue);
        bool hasAdd = !add.IsEmpty;
        fixed (float* sp = src, dp0 = dst, ap0 = add, up = kp.WinoU, bp = kp.Bias, pv = padVec)
        {
            var a = new WinoArgs(sp, up, bp, pv, ih, iw, ic, oh, ow, oc, kp.Blocks, p);
            int nt = (oh + 3) / 4 * a.Tw, chunks = (nt + a.T - 1) / a.T;
            float* dp = dp0, ap = hasAdd ? ap0 : null;
            if (UseNeon && 36L * a.Nb * ic * Ob > NeonWinoUFloats)
            {
                WinogradGrouped(a, dp, ap);
                return;
            }
            For(chunks, (long)oh * ow * oc * ic, chunk =>
            {
                int t0 = chunk * a.T;
                WinoTiles(a, t0, Math.Min(a.T, nt - t0), dp, ap, 0);
            });
        }
    }

    /// <summary>
    /// Tiles [t0, t0+tn) (raster order, tn &lt;= T) of one Winograd conv, written to
    /// <paramref name="dp"/> whose row 0 is output row <paramref name="rowBase"/> (rows past
    /// Oh are skipped). Every tile's math is independent of how tiles are grouped.
    /// </summary>
    private static void WinoTiles(in WinoArgs a, int t0, int tn, float* dp, float* ap, int rowBase)
    {
        int ic = a.Ic, T = a.T, tw = a.Tw, nb = a.Nb, oc = a.Oc, ow = a.Ow, oh = a.Oh;
        bool avx = a.Avx, avx512 = a.Avx512, pk = a.Pk;
        // AVX-512: Tile5-row x 32-oc GEMM tiles over block pairs; M rows hold both blocks
        int tr = avx512 ? Tile5 : pk ? PkTile : Tile, step = avx512 ? 2 : 1, mRow = step * Ob;
        int vSize = 36 * T * ic, mSize = 36 * T * mRow;
        int need = vSize + mSize + 16 * Ob + 2 * Ob;
        float[] buf = t_wino is { } tb && tb.Length >= need ? tb : (t_wino = new float[need]);
        fixed (float* b0 = buf)
        {
            float* V = b0, M = b0 + vSize, tmp = M + mSize, zero = tmp + 16 * Ob;
            new Span<float>(zero, 2 * Ob).Clear();
            float** dp6 = stackalloc float*[36];
            for (int t = 0; t < tn; t++)
            {
                int ty = (t0 + t) / tw, tx = (t0 + t) % tw;
                int iy0 = ty * 4 - a.PadT, ix0 = tx * 4 - a.PadL;
                for (int i = 0; i < 6; i++)
                {
                    int r = SrcIdx(iy0 + i, a.Ih, a.Reflect);
                    for (int j = 0; j < 6; j++)
                    {
                        int c = SrcIdx(ix0 + j, a.Iw, a.Reflect);
                        dp6[i * 6 + j] = r < 0 || c < 0 ? a.PadVec : a.Src + ((long)r * a.Iw + c) * ic;
                    }
                }
                if (avx512 && ic % 16 == 0) InputTransform512(dp6, ic, V + (long)t * ic, (long)T * ic);
                else InputTransform(dp6, ic, V + (long)t * ic, (long)T * ic);
            }

            float** ptrs = stackalloc float*[Math.Max(Tile, Tile5)];
            for (int blk0 = 0; blk0 < nb; blk0 += step)
            {
                int nblk = Math.Min(step, nb - blk0);
                for (int xi = 0; xi < 36; xi++)
                {
                    float* wb = a.U + ((long)xi * nb + blk0) * ic * Ob;
                    float* vx = V + (long)xi * T * ic, mx = M + (long)xi * T * mRow;
                    for (int r0 = 0; r0 < tn; r0 += tr)
                    {
                        int rows = Math.Min(tr, tn - r0);
                        for (int i = 0; i < tr; i++) ptrs[i] = vx + (long)(r0 + Math.Min(i, rows - 1)) * ic;
                        // AVX2 / NEON / PackedSimd write all tr rows into M (T is a multiple of tr); rows past tn are never read
                        if (avx512) Tile512(ptrs, nblk, 1, ic, wb, (long)ic * Ob, zero, mx + (long)r0 * mRow);
                        else if (avx) TileAvxK(ptrs, 1, ic, wb, zero, mx + (long)r0 * Ob, Ob);
                        else if (UseNeon) TileNeonK(ptrs, 1, ic, wb, zero, mx + (long)r0 * Ob, Ob);
                        else if (pk) TilePkK(ptrs, PkTile, 1, ic, (long)ic * Ob, wb, zero, mx + (long)r0 * Ob, Ob);
                        else TileVec(ptrs, rows, 1, ic, wb, zero, mx + (long)r0 * Ob, null, Ob, Ob, ActivationKind.None);
                    }
                }
                for (int h = 0; h < nblk; h++)
                {
                    int blk = blk0 + h;
                    WinoOutput(M + h * Ob, (long)T * mRow, mRow, tn, t0, tw, oh, ow, oc, blk * Ob, a.Bias + blk * Ob,
                        dp, ap, rowBase, a.Act, tmp, avx, avx512);
                }
            }
        }
    }

    /// <summary>Output transform + bias/residual/activation epilogue of one 16-oc block for tn tiles;
    /// tile t's 36 products are m[xi*mStride + t*mRow + 0..15].</summary>
    private static void WinoOutput(float* m, long mStride, int mRow, int tn, int t0, int tw, int oh, int ow, int oc,
        int ocOff, float* bb, float* dp, float* ap, int rowBase, ActivationKind act, float* tmp, bool avx, bool avx512)
    {
        int lanes = Math.Min(Ob, oc - ocOff);
        bool fast512 = avx512 && lanes == Ob && act is ActivationKind.None or ActivationKind.Relu;
        bool fastNeon = UseNeon && lanes == Ob;
        bool fastPk = UsePacked && lanes == Ob;
        for (int t = 0; t < tn; t++)
        {
            int ty = (t0 + t) / tw, tx = (t0 + t) % tw;
            if (avx512) OutputTransform512(m + (long)t * mRow, mStride, tmp);
            else OutputTransform(m + (long)t * mRow, mStride, tmp);
            for (int py = 0; py < 4; py++)
            {
                int oy = ty * 4 + py;
                if (oy >= oh) break;
                for (int px = 0; px < 4; px++)
                {
                    int ox = tx * 4 + px;
                    if (ox >= ow) break;
                    long o = ((long)(oy - rowBase) * ow + ox) * oc + ocOff;
                    float* tp = tmp + (py * 4 + px) * Ob;
                    if (fast512)
                    {
                        var v = Avx512F.Add(Avx512F.LoadVector512(tp), Avx512F.LoadVector512(bb));
                        if (ap != null) v = Avx512F.Add(v, Avx512F.LoadVector512(ap + o));
                        if (act == ActivationKind.Relu) v = Avx512F.Max(v, Vector512<float>.Zero);
                        Avx512F.Store(dp + o, v);
                    }
                    else if (fastNeon)
                        for (int l = 0; l < Ob; l += 4)
                        {
                            var v = AdvSimd.Add(AdvSimd.LoadVector128(tp + l), AdvSimd.LoadVector128(bb + l));
                            if (ap != null) v = AdvSimd.Add(v, AdvSimd.LoadVector128(ap + o + l));
                            AdvSimd.Store(dp + o + l, Act4(v, act));
                        }
                    else if (fastPk)
                        for (int l = 0; l < Ob; l += 4)
                        {
                            var v = PackedSimd.Add(LdP(tp + l), LdP(bb + l));
                            if (ap != null) v = PackedSimd.Add(v, LdP(ap + o + l));
                            PackedSimd.Store(dp + o + l, ActP(v, act));
                        }
                    else if (avx)
                        Store16(dp + o, ap != null ? ap + o : null,
                            Avx.Add(Avx.LoadVector256(tp), Avx.LoadVector256(bb)),
                            Avx.Add(Avx.LoadVector256(tp + 8), Avx.LoadVector256(bb + 8)), lanes, act);
                    else
                        for (int l = 0; l < lanes; l++)
                        {
                            float v = tp[l] + bb[l];
                            if (ap != null) v += ap[o + l];
                            dp[o + l] = ActS(v, act);
                        }
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int SrcIdx(int i, int n, bool reflect) =>
        (uint)i < (uint)n ? i : reflect ? Reflect(i, n) : -1;

    /// <summary>V[xi] = (B^T d B)[xi] for one tile, all channels; V row stride per xi = vStride.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void InputTransform(float** d, int ic, float* v, long vStride)
    {
        int VL = Vector<float>.Count;
        var t = stackalloc Vector<float>[36];
        var four = new Vector<float>(4f);
        var five = new Vector<float>(5f);
        var two = new Vector<float>(2f);
        for (int c = 0; c < ic; c += VL)
        {
            // columns: t = B^T d
            for (int j = 0; j < 6; j++)
            {
                var d0 = Simd.VectorLoad(d[j] + c); var d1 = Simd.VectorLoad(d[6 + j] + c);
                var d2 = Simd.VectorLoad(d[12 + j] + c); var d3 = Simd.VectorLoad(d[18 + j] + c);
                var d4 = Simd.VectorLoad(d[24 + j] + c); var d5 = Simd.VectorLoad(d[30 + j] + c);
                t[j] = four * d0 - five * d2 + d4;
                t[6 + j] = d3 + d4 - four * (d1 + d2);
                t[12 + j] = d4 - d3 + four * (d1 - d2);
                t[18 + j] = d4 - d2 + two * (d3 - d1);
                t[24 + j] = d4 - d2 - two * (d3 - d1);
                t[30 + j] = four * d1 - five * d3 + d5;
            }
            // rows: V = t B
            for (int i = 0; i < 6; i++)
            {
                var r0 = t[i * 6]; var r1 = t[i * 6 + 1]; var r2 = t[i * 6 + 2];
                var r3 = t[i * 6 + 3]; var r4 = t[i * 6 + 4]; var r5 = t[i * 6 + 5];
                float* o = v + c;
                long b = (long)(i * 6) * vStride;
                Simd.VectorStore(o + b, four * r0 - five * r2 + r4);
                Simd.VectorStore(o + b + vStride, r3 + r4 - four * (r1 + r2));
                Simd.VectorStore(o + b + 2 * vStride, r4 - r3 + four * (r1 - r2));
                Simd.VectorStore(o + b + 3 * vStride, r4 - r2 + two * (r3 - r1));
                Simd.VectorStore(o + b + 4 * vStride, r4 - r2 - two * (r3 - r1));
                Simd.VectorStore(o + b + 5 * vStride, four * r1 - five * r3 + r5);
            }
        }
    }

    /// <summary>tmp[(py*4+px)*16 + l] = (A^T m A)[py,px] for one tile's 16 oc lanes; m row stride per xi = mStride.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void OutputTransform(float* m, long mStride, float* tmp)
    {
        int VL = Vector<float>.Count;
        var s = stackalloc Vector<float>[24];
        var two = new Vector<float>(2f);
        var four = new Vector<float>(4f);
        var eight = new Vector<float>(8f);
        for (int l = 0; l < Ob; l += VL)
        {
            // columns: s = A^T m (4x6)
            for (int j = 0; j < 6; j++)
            {
                var m0 = Simd.VectorLoad(m + j * mStride + l); var m1 = Simd.VectorLoad(m + (6 + j) * mStride + l);
                var m2 = Simd.VectorLoad(m + (12 + j) * mStride + l); var m3 = Simd.VectorLoad(m + (18 + j) * mStride + l);
                var m4 = Simd.VectorLoad(m + (24 + j) * mStride + l); var m5 = Simd.VectorLoad(m + (30 + j) * mStride + l);
                var a = m1 + m2; var b = m1 - m2; var c = m3 + m4; var e = m3 - m4;
                s[j] = m0 + a + c;
                s[6 + j] = b + two * e;
                s[12 + j] = a + four * c;
                s[18 + j] = b + eight * e + m5;
            }
            // rows: y = s A (4x4)
            for (int i = 0; i < 4; i++)
            {
                var r0 = s[i * 6]; var r1 = s[i * 6 + 1]; var r2 = s[i * 6 + 2];
                var r3 = s[i * 6 + 3]; var r4 = s[i * 6 + 4]; var r5 = s[i * 6 + 5];
                var a = r1 + r2; var b = r1 - r2; var c = r3 + r4; var e = r3 - r4;
                float* o = tmp + i * 4 * Ob + l;
                Simd.VectorStore(o, r0 + a + c);
                Simd.VectorStore(o + Ob, b + two * e);
                Simd.VectorStore(o + 2 * Ob, a + four * c);
                Simd.VectorStore(o + 3 * Ob, b + eight * e + r5);
            }
        }
    }

    /// <summary><see cref="InputTransform"/> with 16 lanes (AVX-512; ic % 16 == 0). Same per-lane ops → same bits.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void InputTransform512(float** d, int ic, float* v, long vStride)
    {
        var t = stackalloc Vector512<float>[36];
        var four = Vector512.Create(4f);
        var five = Vector512.Create(5f);
        var two = Vector512.Create(2f);
        for (int c = 0; c < ic; c += 16)
        {
            for (int j = 0; j < 6; j++)
            {
                var d0 = Vector512.Load(d[j] + c); var d1 = Vector512.Load(d[6 + j] + c);
                var d2 = Vector512.Load(d[12 + j] + c); var d3 = Vector512.Load(d[18 + j] + c);
                var d4 = Vector512.Load(d[24 + j] + c); var d5 = Vector512.Load(d[30 + j] + c);
                t[j] = four * d0 - five * d2 + d4;
                t[6 + j] = d3 + d4 - four * (d1 + d2);
                t[12 + j] = d4 - d3 + four * (d1 - d2);
                t[18 + j] = d4 - d2 + two * (d3 - d1);
                t[24 + j] = d4 - d2 - two * (d3 - d1);
                t[30 + j] = four * d1 - five * d3 + d5;
            }
            for (int i = 0; i < 6; i++)
            {
                var r0 = t[i * 6]; var r1 = t[i * 6 + 1]; var r2 = t[i * 6 + 2];
                var r3 = t[i * 6 + 3]; var r4 = t[i * 6 + 4]; var r5 = t[i * 6 + 5];
                float* o = v + c + (long)(i * 6) * vStride;
                (four * r0 - five * r2 + r4).Store(o);
                (r3 + r4 - four * (r1 + r2)).Store(o + vStride);
                (r4 - r3 + four * (r1 - r2)).Store(o + 2 * vStride);
                (r4 - r2 + two * (r3 - r1)).Store(o + 3 * vStride);
                (r4 - r2 - two * (r3 - r1)).Store(o + 4 * vStride);
                (four * r1 - five * r3 + r5).Store(o + 5 * vStride);
            }
        }
    }

    /// <summary><see cref="OutputTransform"/> with one 16-lane vector per element (AVX-512).</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void OutputTransform512(float* m, long mStride, float* tmp)
    {
        var s = stackalloc Vector512<float>[24];
        var two = Vector512.Create(2f);
        var four = Vector512.Create(4f);
        var eight = Vector512.Create(8f);
        for (int j = 0; j < 6; j++)
        {
            var m0 = Vector512.Load(m + j * mStride); var m1 = Vector512.Load(m + (6 + j) * mStride);
            var m2 = Vector512.Load(m + (12 + j) * mStride); var m3 = Vector512.Load(m + (18 + j) * mStride);
            var m4 = Vector512.Load(m + (24 + j) * mStride); var m5 = Vector512.Load(m + (30 + j) * mStride);
            var a = m1 + m2; var b = m1 - m2; var c = m3 + m4; var e = m3 - m4;
            s[j] = m0 + a + c;
            s[6 + j] = b + two * e;
            s[12 + j] = a + four * c;
            s[18 + j] = b + eight * e + m5;
        }
        for (int i = 0; i < 4; i++)
        {
            var r0 = s[i * 6]; var r1 = s[i * 6 + 1]; var r2 = s[i * 6 + 2];
            var r3 = s[i * 6 + 3]; var r4 = s[i * 6 + 4]; var r5 = s[i * 6 + 5];
            var a = r1 + r2; var b = r1 - r2; var c = r3 + r4; var e = r3 - r4;
            float* o = tmp + i * 4 * Ob;
            (r0 + a + c).Store(o);
            (b + two * e).Store(o + Ob);
            (a + four * c).Store(o + 2 * Ob);
            (b + eight * e + r5).Store(o + 3 * Ob);
        }
    }
}
