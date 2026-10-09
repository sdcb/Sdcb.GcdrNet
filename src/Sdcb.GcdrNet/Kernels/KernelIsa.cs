using System.Numerics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics.Arm;

namespace Sdcb.GcdrNet.Kernels;

/// <summary>ISA tiers for kernel dispatch, highest first.</summary>
internal enum KernelIsa
{
    Scalar = 0,     // plain C# loops — the reference implementation
    Vector = 1,     // System.Numerics.Vector<float> (128/256/512-bit generic)
    Avx2 = 2,       // x86 AVX2+FMA
    Avx512 = 3,     // x86 AVX-512F
    AdvSimd = 4,    // ARM64 NEON (best-effort port)
}

internal static class KernelDispatch
{
    private static KernelIsa? _override;
    private static readonly KernelIsa _detected = Detect();

    /// <summary>Active ISA. Settable via GCDRNET_ISA env var or the internal
    /// test override; defaults to best supported.</summary>
    public static KernelIsa Current => _override ?? _detected;

    /// <summary>Test hook: force the ISA for this process (null restores auto).</summary>
    internal static void Force(KernelIsa? isa) => _override = isa;

    private static KernelIsa Detect()
    {
        string? env = Environment.GetEnvironmentVariable("GCDRNET_ISA");
        if (env != null)
        {
            return env.Trim().ToLowerInvariant() switch
            {
                "scalar" => KernelIsa.Scalar,
                "vector" => KernelIsa.Vector,
                "avx2" => KernelIsa.Avx2,
                "avx512" or "avx512f" => KernelIsa.Avx512,
                "advsimd" or "neon" => KernelIsa.AdvSimd,
                _ => throw new ArgumentException($"Unknown GCDRNET_ISA '{env}'"),
            };
        }
        if (Avx512F.IsSupported) return KernelIsa.Avx512;
        if (Avx2.IsSupported && Fma.IsSupported) return KernelIsa.Avx2;
        if (AdvSimd.Arm64.IsSupported) return KernelIsa.AdvSimd;
        if (Vector<float>.IsSupported) return KernelIsa.Vector;
        return KernelIsa.Scalar;
    }
}
