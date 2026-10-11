using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Sdcb.SimdGcdrNet.Kernels;

/// <summary>
/// Vectorized transcendental math: exp / sigmoid / erf / gelu.
/// Precision targets ~1e-6 absolute so fused activations stay well inside the
/// 2e-3 validation budget.
/// </summary>
internal static class SimdMath
{
    // ---------- exp(x): cephes-style 2^n * poly(r) ----------
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<float> Exp(Vector<float> x)
    {
        int VL = Vector<float>.Count;
        // clamp to avoid overflow in scale
        x = Vector.Max(x, new Vector<float>(-87.3f));
        x = Vector.Min(x, new Vector<float>(88.7f));
        float invLn2 = 1.4426950408889634f;
        var fx = x * new Vector<float>(invLn2) + new Vector<float>(0.5f);
        // floor(fx)
        var n = Vector.ConvertToInt32(fx);
        var nf = Vector.ConvertToSingle(n);
        // nf > fx (i.e. fx negative half case) fix
        var mask = Vector.GreaterThan(nf, fx);
        nf -= Vector.BitwiseAnd(Vector<float>.One, Vector.AsVectorSingle(mask));
        // r = x - n*ln2 (hi/lo parts for precision)
        const float ln2hi = 0.693359375f, ln2lo = -0.00021219444005469058f;
        var r = x - nf * new Vector<float>(ln2hi) - nf * new Vector<float>(ln2lo);
        // exp(r) poly, degree 5 (cephes)
        var p = new Vector<float>(1.9875691500e-4f);
        p = p * r + new Vector<float>(1.3981999507e-3f);
        p = p * r + new Vector<float>(8.3334519073e-3f);
        p = p * r + new Vector<float>(4.1665795894e-2f);
        p = p * r + new Vector<float>(1.6666665459e-1f);
        p = p * r + new Vector<float>(5.0000001201e-1f);
        var y = p * (r * r) + r + Vector<float>.One;
        // scale by 2^n
        var ni = Vector.ConvertToInt32(nf) + new Vector<int>(127);
        var pow2n = Vector.AsVectorSingle(ni << 23);
        return y * pow2n;
    }

    public static void Exp(ReadOnlySpan<float> src, Span<float> dst)
    {
        int n = src.Length, i = 0, VL = Vector<float>.Count;
        for (; i + VL <= n; i += VL)
            Simd.VectorStore(dst, i, Exp(Simd.VectorLoad(src, i)));
        for (; i < n; i++) dst[i] = MathF.Exp(src[i]);
    }

    // ---------- sigmoid = 1/(1+e^-x) ----------
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<float> SigmoidV(Vector<float> x) =>
        Vector<float>.One / (Vector<float>.One + Exp(-x));

    public static void Sigmoid(ReadOnlySpan<float> src, Span<float> dst)
    {
        int n = src.Length, i = 0, VL = Vector<float>.Count;
        for (; i + VL <= n; i += VL)
            Simd.VectorStore(dst, i, SigmoidV(Simd.VectorLoad(src, i)));
        for (; i < n; i++) dst[i] = 1f / (1f + MathF.Exp(-src[i]));
    }

    // ---------- erf(x): Abramowitz–Stegun 7.1.26 (max ~1.5e-7) ----------
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<float> ErfV(Vector<float> x)
    {
        var sign = Vector.ConditionalSelect(Vector.LessThan(x, Vector<float>.Zero),
            -Vector<float>.One, Vector<float>.One);
        var ax = Vector.Abs(x);
        var t = Vector<float>.One / (Vector<float>.One + new Vector<float>(0.3275911f) * ax);
        var p = new Vector<float>(1.061405429f);
        p = p * t + new Vector<float>(-1.453152027f);
        p = p * t + new Vector<float>(1.421413741f);
        p = p * t + new Vector<float>(-0.284496736f);
        p = p * t + new Vector<float>(0.254829592f);
        var y = Vector<float>.One - p * t * Exp(-ax * ax);
        // apply sign (sign(0)=0 → y*0 = 0 correct)
        return y * sign;
    }

    public static void Erf(ReadOnlySpan<float> src, Span<float> dst)
    {
        int n = src.Length, i = 0, VL = Vector<float>.Count;
        for (; i + VL <= n; i += VL)
            Simd.VectorStore(dst, i, ErfV(Simd.VectorLoad(src, i)));
        for (; i < n; i++) dst[i] = ErfScalar(src[i]);
    }

    // ---------- gelu(x) = 0.5*x*(1+erf(x/√2)) ----------
    public static void Gelu(ReadOnlySpan<float> src, Span<float> dst)
    {
        int n = src.Length, i = 0, VL = Vector<float>.Count;
        var invSqrt2 = new Vector<float>(0.70710678118654752f);
        var half = new Vector<float>(0.5f);
        for (; i + VL <= n; i += VL)
        {
            var x = Simd.VectorLoad(src, i);
            var g = half * x * (Vector<float>.One + ErfV(x * invSqrt2));
            Simd.VectorStore(dst, i, g);
        }
        for (; i < n; i++) dst[i] = 0.5f * src[i] * (1f + ErfScalar(src[i] * 0.70710678118654752f));
    }

    // ---------- AVX2 cores (used by Avx2Kernels fused epilogues) ----------
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector256<float> ExpCore(Vector256<float> x)
    {
        x = Vector256.Max(x, Vector256.Create(-87.3f));
        x = Vector256.Min(x, Vector256.Create(88.7f));
        var fx = Vector256.FusedMultiplyAdd(x, Vector256.Create(1.4426950408889634f), Vector256.Create(0.5f));
        var nf = Vector256.Floor(fx);
        var r = Vector256.FusedMultiplyAdd(nf, Vector256.Create(-0.693359375f), x);
        r = Vector256.FusedMultiplyAdd(nf, Vector256.Create(0.00021219444005469058f), r);
        var p = Vector256.Create(1.9875691500e-4f);
        p = Vector256.FusedMultiplyAdd(p, r, Vector256.Create(1.3981999507e-3f));
        p = Vector256.FusedMultiplyAdd(p, r, Vector256.Create(8.3334519073e-3f));
        p = Vector256.FusedMultiplyAdd(p, r, Vector256.Create(4.1665795894e-2f));
        p = Vector256.FusedMultiplyAdd(p, r, Vector256.Create(1.6666665459e-1f));
        p = Vector256.FusedMultiplyAdd(p, r, Vector256.Create(5.0000001201e-1f));
        var y = Vector256.Add(Vector256.FusedMultiplyAdd(p, Vector256.Multiply(r, r), r), Vector256<float>.One);
        var ni = Avx2.Add(Avx2.ConvertToVector256Int32(nf), Vector256.Create(127));
        var pow2n = Avx2.ShiftLeftLogical(ni, 23).AsSingle();
        return Vector256.Multiply(y, pow2n);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector256<float> SigmoidCore(Vector256<float> x) =>
        Vector256.Divide(Vector256<float>.One,
            Vector256.Add(Vector256<float>.One, ExpCore(Vector256.Negate(x))));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector256<float> ErfCore(Vector256<float> x)
    {
        var sign = Avx.Or(Avx.And(x, Vector256.Create(-0f)),
            Vector256<float>.One);
        var ax = Avx.AndNot(Vector256.Create(-0f), x);
        var t = Vector256.Divide(Vector256<float>.One,
            Vector256.Add(Vector256<float>.One, Vector256.Multiply(Vector256.Create(0.3275911f), ax)));
        var p = Vector256.Create(1.061405429f);
        p = Vector256.FusedMultiplyAdd(p, t, Vector256.Create(-1.453152027f));
        p = Vector256.FusedMultiplyAdd(p, t, Vector256.Create(1.421413741f));
        p = Vector256.FusedMultiplyAdd(p, t, Vector256.Create(-0.284496736f));
        p = Vector256.FusedMultiplyAdd(p, t, Vector256.Create(0.254829592f));
        var y = Vector256.Subtract(Vector256<float>.One,
            Vector256.Multiply(Vector256.Multiply(p, t), ExpCore(Vector256.Negate(Vector256.Multiply(ax, ax)))));
        return Vector256.Multiply(y, sign);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector256<float> GeluCore(Vector256<float> x) =>
        Vector256.Multiply(Vector256.Multiply(Vector256.Create(0.5f), x),
            Vector256.Add(Vector256<float>.One,
                ErfCore(Vector256.Multiply(x, Vector256.Create(0.70710678118654752f)))));

    // ---------- 128-bit cores (NEON tier fused epilogues; same formulas as the AVX2 cores) ----------
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector128<float> ExpCore(Vector128<float> x)
    {
        x = Vector128.Max(x, Vector128.Create(-87.3f));
        x = Vector128.Min(x, Vector128.Create(88.7f));
        var fx = Vector128.FusedMultiplyAdd(x, Vector128.Create(1.4426950408889634f), Vector128.Create(0.5f));
        var nf = Vector128.Floor(fx);
        var r = Vector128.FusedMultiplyAdd(nf, Vector128.Create(-0.693359375f), x);
        r = Vector128.FusedMultiplyAdd(nf, Vector128.Create(0.00021219444005469058f), r);
        var p = Vector128.Create(1.9875691500e-4f);
        p = Vector128.FusedMultiplyAdd(p, r, Vector128.Create(1.3981999507e-3f));
        p = Vector128.FusedMultiplyAdd(p, r, Vector128.Create(8.3334519073e-3f));
        p = Vector128.FusedMultiplyAdd(p, r, Vector128.Create(4.1665795894e-2f));
        p = Vector128.FusedMultiplyAdd(p, r, Vector128.Create(1.6666665459e-1f));
        p = Vector128.FusedMultiplyAdd(p, r, Vector128.Create(5.0000001201e-1f));
        var y = Vector128.FusedMultiplyAdd(p, r * r, r) + Vector128<float>.One;
        var ni = Vector128.ConvertToInt32(nf) + Vector128.Create(127);
        return y * Vector128.ShiftLeft(ni, 23).AsSingle();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector128<float> SigmoidCore(Vector128<float> x) =>
        Vector128<float>.One / (Vector128<float>.One + ExpCore(-x));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector128<float> ErfCore(Vector128<float> x)
    {
        var signBit = Vector128.Create(-0f);
        var sign = (x & signBit) | Vector128<float>.One;
        var ax = Vector128.AndNot(x, signBit);
        var t = Vector128<float>.One / (Vector128<float>.One + Vector128.Create(0.3275911f) * ax);
        var p = Vector128.Create(1.061405429f);
        p = Vector128.FusedMultiplyAdd(p, t, Vector128.Create(-1.453152027f));
        p = Vector128.FusedMultiplyAdd(p, t, Vector128.Create(1.421413741f));
        p = Vector128.FusedMultiplyAdd(p, t, Vector128.Create(-0.284496736f));
        p = Vector128.FusedMultiplyAdd(p, t, Vector128.Create(0.254829592f));
        var y = Vector128<float>.One - p * t * ExpCore(-(ax * ax));
        return y * sign;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector128<float> GeluCore(Vector128<float> x) =>
        Vector128.Create(0.5f) * x * (Vector128<float>.One + ErfCore(x * Vector128.Create(0.70710678118654752f)));

    internal static float ErfScalar(float x)
    {
        float ax = MathF.Abs(x);
        float t = 1f / (1f + 0.3275911f * ax);
        float p = ((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
        float y = 1f - p * MathF.Exp(-ax * ax);
        return x < 0 ? -y : y;
    }
}
