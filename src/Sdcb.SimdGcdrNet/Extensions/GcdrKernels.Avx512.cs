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

internal static unsafe partial class GcdrKernels
{
    /// <summary>
    /// <see cref="UpHead"/> for the AVX-512 tier: 16-lane resize, and the c-long oc dot runs as four
    /// interleaved per-pixel FMA chains (one chain per pixel was FMA-latency bound). Each pixel keeps
    /// its bias-initialized, channel-ascending chain, so outputs match the AVX2 path bit for bit.
    /// </summary>
    internal static void UpHead512(float* sp, int iw, int c, float* kpp, bool skipConv, DenseArgs ca, float* dp, int oh, int ow,
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
    internal static void UpHeadDot4(float* s, int c, float* wp, float* bp, float* acc, int n)
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
