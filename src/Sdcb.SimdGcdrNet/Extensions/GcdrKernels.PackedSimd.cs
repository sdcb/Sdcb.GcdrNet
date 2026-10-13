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
    /// One <see cref="UpHead"/> output row for oc &lt;= 4, c % 4 == 0 (see <see cref="UpHeadRowNeon"/>):
    /// per 4 channels the bilinear lerps, [relu], + [relu](skip), then each channel is splatted back
    /// from a 16-byte spill (WASM SIMD has no by-element multiply) into one 4-lane accumulator.
    /// Same per-element ops as the unfused Vector&lt;float&gt; path.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void UpHeadRowPk(float* r0, float* r1, int* x0, int* x1, float* wx, float wy, float* krow,
        int c, int ow, float* wp, float* bp, float* orow, int oc, bool relu, bool skipRelu, ActivationKind act)
    {
        V4 vY = PackedSimd.Splat(wy), bias = LdP(bp);
        float* s = stackalloc float[4];
        for (int x = 0; x < ow; x++)
        {
            float* a = r0 + (long)x0[x] * c, b = r0 + (long)x1[x] * c, e = r1 + (long)x0[x] * c, f = r1 + (long)x1[x] * c;
            float* k = krow + (long)x * c;
            V4 vX = PackedSimd.Splat(wx[x]), acc = bias;
            float* w = wp;
            for (int ch = 0; ch < c; ch += 4, w += 32)
            {
                V4 va = LdP(a + ch), ve = LdP(e + ch);
                V4 top = MadP(va, PackedSimd.Subtract(LdP(b + ch), va), vX);
                V4 bot = MadP(ve, PackedSimd.Subtract(LdP(f + ch), ve), vX);
                V4 v = MadP(top, PackedSimd.Subtract(bot, top), vY);
                if (relu) v = PackedSimd.PseudoMax(V4.Zero, v);
                V4 sk = LdP(k + ch);
                v = PackedSimd.Add(v, skipRelu ? PackedSimd.PseudoMax(V4.Zero, sk) : sk);
                PackedSimd.Store(s, v);
                acc = MadP(acc, PackedSimd.LoadScalarAndSplatVector128(s), LdP(w));
                acc = MadP(acc, PackedSimd.LoadScalarAndSplatVector128(s + 1), LdP(w + 8));
                acc = MadP(acc, PackedSimd.LoadScalarAndSplatVector128(s + 2), LdP(w + 16));
                acc = MadP(acc, PackedSimd.LoadScalarAndSplatVector128(s + 3), LdP(w + 24));
            }
            PackedSimd.Store(s, ActP(acc, act));
            float* o = orow + (long)x * oc;
            for (int l = 0; l < oc; l++) o[l] = s[l];
        }
    }
}
