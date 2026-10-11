using System.Resources;

namespace Sdcb.SimdGcdrNet.Models;

/// <summary>
/// The GCDRNet models (simplified GC-Net + DR-Net ONNX, ~21 MB) embedded in this assembly.
/// This package has no dependencies, so it is versioned independently of the engine:
/// <code>
/// using GcdrEnhancer enh = GcdrEnhancer.FromBytes(GcdrModels.GcNet, GcdrModels.DrNet, GcdrBackend.Auto);
/// </code>
/// </summary>
public static class GcdrModels
{
    private static readonly Resource s_gcNet = new("gcnet.onnx"), s_drNet = new("drnet.onnx");

    /// <summary>The GC-Net (shading estimation) ONNX model. Zero-copy view of the embedded
    /// resource inside the loaded assembly image.</summary>
    public static ReadOnlySpan<byte> GcNet => s_gcNet.Span;

    /// <summary>The DR-Net (detail restoration) ONNX model. Zero-copy view of the embedded
    /// resource inside the loaded assembly image.</summary>
    public static ReadOnlySpan<byte> DrNet => s_drNet.Span;

    /// <summary>Lazily resolved embedded resource. Normally the runtime returns an
    /// <see cref="UnmanagedMemoryStream"/> over the mapped assembly image, valid for the life of
    /// the (non-collectible) assembly; any other stream is read once into an array.</summary>
    private sealed unsafe class Resource(string name)
    {
        private byte* _ptr;
        private int _length = -1;
        private byte[]? _copy;
        private readonly Lock _lock = new();

        public ReadOnlySpan<byte> Span
        {
            get
            {
                int len = Volatile.Read(ref _length);   // published last, after _ptr / _copy
                if (len < 0) { Load(); len = _length; }
                return _copy ?? new ReadOnlySpan<byte>(_ptr, len);
            }
        }

        private void Load()
        {
            lock (_lock)
            {
                if (_length >= 0) return;
                using Stream s = typeof(GcdrModels).Assembly.GetManifestResourceStream(name)
                    ?? throw new MissingManifestResourceException(
                        $"Embedded model '{name}' not found in {typeof(GcdrModels).Assembly.GetName().Name}.");
                if (s is UnmanagedMemoryStream ums)
                {
                    _ptr = ums.PositionPointer;
                    Volatile.Write(ref _length, checked((int)ums.Length));
                }
                else
                {
                    using var ms = new MemoryStream(checked((int)s.Length));
                    s.CopyTo(ms);
                    _copy = ms.ToArray();
                    Volatile.Write(ref _length, _copy.Length);
                }
            }
        }
    }
}
