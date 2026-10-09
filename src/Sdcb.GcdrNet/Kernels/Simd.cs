using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Sdcb.GcdrNet.Kernels;

/// <summary>Shared Vector&lt;float&gt; helpers (width-agnostic SIMD).</summary>
internal static class Simd
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> VectorLoad(ReadOnlySpan<float> source, int offset) =>
        Vector.LoadUnsafe(ref Unsafe.Add(ref MemoryMarshal.GetReference(source), offset));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void VectorStore(Span<float> destination, int offset, Vector<float> value) =>
        value.StoreUnsafe(ref Unsafe.Add(ref MemoryMarshal.GetReference(destination), offset));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe Vector<float> VectorLoad(float* source) =>
        Vector.LoadUnsafe(ref Unsafe.AsRef<float>(source));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe void VectorStore(float* destination, Vector<float> value) =>
        value.StoreUnsafe(ref Unsafe.AsRef<float>(destination));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> VectorAddMul(Vector<float> acc, Vector<float> value, float weight) =>
        acc + value * new Vector<float>(weight);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> VectorAddMul(Vector<float> acc, Vector<float> value, Vector<float> weight) =>
        acc + value * weight;
}
