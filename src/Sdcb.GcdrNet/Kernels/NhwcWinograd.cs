using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;
using Sdcb.GcdrNet.Engine;
using F32x8 = System.Runtime.Intrinsics.Vector256<float>;

namespace Sdcb.GcdrNet.Kernels;

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
        public readonly bool Reflect, Avx;
        public readonly ActivationKind Act;

        public WinoArgs(float* src, float* u, float* bias, float* padVec, int ih, int iw, int ic,
            int oh, int ow, int oc, int nb, ConvP p)
        {
            Src = src; U = u; Bias = bias; PadVec = padVec;
            Ih = ih; Iw = iw; Ic = ic; Oh = oh; Ow = ow; Oc = oc; Nb = nb;
            Tw = (ow + 3) / 4;
            T = Math.Clamp(65536 / (36 * ic) / Tile * Tile, Tile, 60);   // V block ~256 KB
            PadT = p.Pads[0]; PadL = p.Pads[1];
            Reflect = p.PadMode == PadMode.Reflect;
            Avx = UseAvx;
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
        bool avx = a.Avx, hasAdd = ap != null;
        int vSize = 36 * T * ic, mSize = 36 * T * Ob;
        int need = vSize + mSize + 16 * Ob + Ob;
        float[] buf = t_wino is { } tb && tb.Length >= need ? tb : (t_wino = new float[need]);
        fixed (float* b0 = buf)
        {
            float* V = b0, M = b0 + vSize, tmp = M + mSize, zero = tmp + 16 * Ob;
            new Span<float>(zero, Ob).Clear();
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
                InputTransform(dp6, ic, V + (long)t * ic, (long)T * ic);
            }

            float** ptrs = stackalloc float*[Tile];
            for (int blk = 0; blk < nb; blk++)
            {
                int lanes = Math.Min(Ob, oc - blk * Ob);
                for (int xi = 0; xi < 36; xi++)
                {
                    float* wb = a.U + ((long)xi * nb + blk) * ic * Ob;
                    float* vx = V + (long)xi * T * ic, mx = M + (long)xi * T * Ob;
                    for (int r0 = 0; r0 < tn; r0 += Tile)
                    {
                        int rows = Math.Min(Tile, tn - r0);
                        for (int i = 0; i < Tile; i++) ptrs[i] = vx + (long)(r0 + Math.Min(i, rows - 1)) * ic;
                        if (avx) TileAvx(ptrs, rows, 1, ic, wb, zero, mx + (long)r0 * Ob, null, Ob, Ob, ActivationKind.None);
                        else TileVec(ptrs, rows, 1, ic, wb, zero, mx + (long)r0 * Ob, null, Ob, Ob, ActivationKind.None);
                    }
                }
                float* bb = a.Bias + blk * Ob;
                for (int t = 0; t < tn; t++)
                {
                    int ty = (t0 + t) / tw, tx = (t0 + t) % tw;
                    OutputTransform(M + (long)t * Ob, (long)T * Ob, tmp);
                    for (int py = 0; py < 4; py++)
                    {
                        int oy = ty * 4 + py;
                        if (oy >= oh) break;
                        for (int px = 0; px < 4; px++)
                        {
                            int ox = tx * 4 + px;
                            if (ox >= ow) break;
                            long o = ((long)(oy - rowBase) * ow + ox) * oc + blk * Ob;
                            float* tp = tmp + (py * 4 + px) * Ob;
                            if (avx)
                                Store16(dp + o, hasAdd ? ap + o : null,
                                    Avx.Add(Avx.LoadVector256(tp), Avx.LoadVector256(bb)),
                                    Avx.Add(Avx.LoadVector256(tp + 8), Avx.LoadVector256(bb + 8)), lanes, a.Act);
                            else
                                for (int l = 0; l < lanes; l++)
                                {
                                    float v = tp[l] + bb[l];
                                    if (hasAdd) v += ap[o + l];
                                    dp[o + l] = ActS(v, a.Act);
                                }
                        }
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
}
