using System.Reflection;

namespace Sdcb.SimdGcdrNet.Backends.Vulkan;

/// <summary>Loads SPIR-V blobs embedded next to their .comp sources.</summary>
internal static class VkShaders
{
    private static readonly Assembly s_asm = typeof(VkShaders).Assembly;
    private const string Prefix = "Sdcb.SimdGcdrNet.Backends.Vulkan.Shaders.";

    public static byte[] Load(string name)
    {
        using Stream? s = s_asm.GetManifestResourceStream(Prefix + name + ".spv");
        if (s == null)
            throw new InvalidOperationException(
                $"Embedded shader '{name}' not found. Available: {string.Join(", ", s_asm.GetManifestResourceNames().Where(n => n.StartsWith(Prefix)))}");
        byte[] b = new byte[s.Length];
        s.ReadExactly(b);
        return b;
    }
}
