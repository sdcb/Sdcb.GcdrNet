using Sdcb.GcdrNet.Onnx;

namespace Sdcb.GcdrNet.Engine;

/// <summary>
/// An <see cref="OnnxModel"/> specialized to a concrete input shape: all
/// dynamic-shape plumbing folded, ops lowered to the exec set, buffers
/// planned. Immutable; share across sessions freely.
/// </summary>
public sealed class CompiledModel
{
    internal ExecPlan Plan { get; }

    private CompiledModel(ExecPlan plan) => Plan = plan;

    public static CompiledModel Compile(OnnxModel model, ReadOnlySpan<int> inputShape) =>
        new(GraphCompiler.Compile(model, inputShape));

    public InferenceSession CreateSession() => new(Plan);

    public int NodeCount => Plan.Nodes.Length;
    public int ArenaFloats => Plan.ArenaFloats;
}
