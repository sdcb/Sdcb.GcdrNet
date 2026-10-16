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
    /// One <see cref="UpHead"/> output row for oc &lt;= 4, c % 4 == 0: per 4 channels the bilinear
    /// lerps (fma), [relu], + [relu](skip), then fmla by element into one 4-lane accumulator against
    /// the packed [c][8] weights — the resized pixel never leaves registers.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void UpHeadRowNeon(float* r0, float* r1, int* x0, int* x1, float* wx, float wy, float* krow,
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
}
