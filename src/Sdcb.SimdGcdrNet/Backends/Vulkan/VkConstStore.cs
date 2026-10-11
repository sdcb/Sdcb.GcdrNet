namespace Sdcb.SimdGcdrNet.Backends.Vulkan;

/// <summary>
/// Device-side const tensors (backend-packed weights) plus a 64B zero buffer,
/// shared by every <see cref="VkGraphModel"/> of one model: per-size plans copy
/// the same initializers, so instead of re-packing and re-uploading ~20 MB per
/// input size, a new graph whose const list is element-for-element identical
/// (same consumer packing, same contents) reuses the existing buffer. A graph
/// whose consts differ gets a private, unshared store. Ref-counted; the
/// buffers are freed when the last graph releases.
/// </summary>
internal sealed class VkConstStore
{
    private static readonly Dictionary<(VkDevice dev, object key, bool f16), VkConstStore> s_shared = new();

    private readonly (VkDevice, object, bool)? _key;
    private readonly float[][] _src;
    private readonly int[] _kinds;
    private int _refs = 1;
    internal readonly VkBuffer Consts, Dummy;
    internal readonly ulong[] Offsets;
    internal readonly ulong Bytes;

    private VkConstStore(VkDevice dev, (VkDevice, object, bool)? key, float[][] src, int[] kinds,
        ulong[] offsets, byte[] host)
    {
        _key = key;
        _src = src;
        _kinds = kinds;
        Offsets = offsets;
        Bytes = (ulong)host.LongLength;
        Consts = dev.NewStorageBuffer(Bytes, hostVisible: dev.CoherentDeviceLocal);
        Dummy = dev.NewStorageBuffer(64, hostVisible: dev.CoherentDeviceLocal);
        unsafe
        {
            fixed (byte* hp = host) dev.Upload(Consts, hp, Bytes);
            byte* z = stackalloc byte[64];
            new Span<byte>(z, 64).Clear();
            dev.Upload(Dummy, z, 64);
        }
    }

    /// <summary>
    /// Returns a store holding <paramref name="src"/> (kinds = consumer packing ids),
    /// shared under <paramref name="key"/> when an identical list was uploaded before.
    /// <paramref name="pack"/> (offsets, host bytes) only runs when a new store is built.
    /// </summary>
    internal static VkConstStore Acquire(VkDevice dev, object? key, bool f16, float[][] src, int[] kinds,
        Func<(ulong[] offsets, byte[] host)> pack)
    {
        if (key is null)
        {
            var (o, h) = pack();
            return new VkConstStore(dev, null, src, kinds, o, h);
        }
        var k = (dev, key, f16);
        lock (s_shared)
        {
            if (s_shared.TryGetValue(k, out VkConstStore? s) && s.Matches(src, kinds))
            {
                s._refs++;
                return s;
            }
            var (offs, host) = pack();
            bool share = s is null;   // keep the first registered store on a mismatch
            var ns = new VkConstStore(dev, share ? k : null, src, kinds, offs, host);
            if (share) s_shared[k] = ns;
            return ns;
        }
    }

    private bool Matches(float[][] src, int[] kinds)
    {
        if (src.Length != _src.Length || !kinds.AsSpan().SequenceEqual(_kinds)) return false;
        for (int i = 0; i < src.Length; i++)
            if (!ReferenceEquals(src[i], _src[i]) && !src[i].AsSpan().SequenceEqual(_src[i])) return false;
        return true;
    }

    internal void Release()
    {
        lock (s_shared)
        {
            if (--_refs > 0) return;
            if (_key is { } k && s_shared.TryGetValue(k, out VkConstStore? cur) && cur == this)
                s_shared.Remove(k);
        }
        Consts.Free();
        Dummy.Free();
    }
}
