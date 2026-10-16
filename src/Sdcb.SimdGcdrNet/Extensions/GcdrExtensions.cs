using System.Reflection;
using Sdcb.SimdOnnx.Engine;

namespace Sdcb.SimdGcdrNet.Extensions;

/// <summary>The GCDRNet-specific fusions (shift-MLP, upsample tail) registered with the engine.</summary>
internal static class GcdrExtensions
{
    /// <summary>Assembly embedding the extension shaders.</summary>
    public static Assembly Owner { get; } = typeof(GcdrExtensions).Assembly;

    public static EngineExtensions Instance { get; } = new EngineExtensions()
        .Add(new FuseShiftPass())
        .Add(new FuseUpHeadPass())
        .Add(new UpHeadSkipRecomputePass());
}
