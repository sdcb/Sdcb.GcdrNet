using Sdcb.SimdGcdrNet.Onnx;

namespace Sdcb.SimdGcdrNet.Engine;

/// <summary>
/// An <see cref="OnnxModel"/> specialized to a concrete input shape: all
/// dynamic-shape plumbing folded, ops lowered to the exec set, buffers
/// planned. Immutable; share across sessions freely.
/// </summary>
public sealed class CompiledModel
{
    internal ExecPlan Plan { get; }

    private CompiledModel(ExecPlan plan) => Plan = plan;

    /// <summary>Compiles <paramref name="model"/> for a fixed input shape.</summary>
    /// <param name="model">The ONNX model to compile.</param>
    /// <param name="inputShape">Concrete input shape, e.g. <c>[1, 3, H, W]</c>.</param>
    public static CompiledModel Compile(OnnxModel model, ReadOnlySpan<int> inputShape) =>
        new(GraphCompiler.Compile(model, inputShape));

    /// <summary>Creates a new CPU <see cref="InferenceSession"/> with its own arena.</summary>
    public InferenceSession CreateSession() => new(Plan);

    /// <summary>Number of executable nodes after folding and fusion.</summary>
    public int NodeCount => Plan.Nodes.Length;

    /// <summary>Size of the activation arena, in floats, a session needs.</summary>
    public int ArenaFloats => Plan.ArenaFloats;
}
