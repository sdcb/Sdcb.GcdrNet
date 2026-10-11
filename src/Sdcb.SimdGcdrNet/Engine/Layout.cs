namespace Sdcb.SimdGcdrNet.Engine;

/// <summary>
/// Physical-layout algebra. A layout <c>L</c> lists, for each physical
/// (storage) dim i, the logical dim it holds: storage is row-major over
/// <c>[shape[L[0]], shape[L[1]], ...]</c>. Logical shapes never change —
/// fusion passes and perm reasoning stay in logical space; kernels address by
/// <see cref="Phys"/>. Size-1 dims carry no stride, so layouts that order the
/// non-1 dims identically are interchangeable (<see cref="Equivalent"/>).
/// </summary>
internal static class Layout
{
    /// <summary>Channels-last for rank-4 [n,c,h,w]: storage [n,h,w,c].</summary>
    public static readonly int[] Nhwc = [0, 2, 3, 1];

    public static int[] Identity(int rank)
    {
        int[] l = new int[rank];
        for (int i = 0; i < rank; i++) l[i] = i;
        return l;
    }

    public static bool IsIdentity(int[] l)
    {
        for (int i = 0; i < l.Length; i++) if (l[i] != i) return false;
        return true;
    }

    public static int[] Phys(int[] shape, int[] l)
    {
        int[] p = new int[l.Length];
        for (int i = 0; i < l.Length; i++) p[i] = shape[l[i]];
        return p;
    }

    public static int[] Inverse(int[] p)
    {
        int[] inv = new int[p.Length];
        for (int i = 0; i < p.Length; i++) inv[p[i]] = i;
        return inv;
    }

    /// <summary>
    /// Layout of y = Transpose(x, perm) as a view of x's storage. y logical k
    /// is x logical perm[k], so x logical d is y logical perm⁻¹[d]:
    /// L'[i] = perm⁻¹[L[i]]. Always representable — every Transpose dissolves.
    /// </summary>
    public static int[] Transposed(int[] l, int[] perm)
    {
        int[] inv = Inverse(perm);
        int[] r = new int[l.Length];
        for (int i = 0; i < l.Length; i++) r[i] = inv[l[i]];
        return r;
    }

    /// <summary>Same storage order for <paramref name="shape"/> (size-1 dims ignored).</summary>
    public static bool Equivalent(int[] shape, int[] a, int[] b) =>
        a.Where(d => shape[d] != 1).SequenceEqual(b.Where(d => shape[d] != 1));

    /// <summary>Physical perm relaying storage from layout <paramref name="from"/> to
    /// <paramref name="to"/>: out phys dim i = in phys dim q[i].</summary>
    public static int[] PhysPerm(int[] from, int[] to) =>
        to.Select(d => Array.IndexOf(from, d)).ToArray();

    /// <summary>
    /// Layout of a zero-copy reshape <paramref name="src"/>→<paramref name="dst"/>
    /// of storage laid out as <paramref name="l"/>, or null when the reshape
    /// needs data movement. The logical reshape factors into groups (a run of
    /// source dims ↔ a run of target dims with equal product); it is a view iff
    /// each group's non-1 source dims sit consecutively, in logical order, in
    /// storage. Target dims are then ordered by their group's storage position.
    /// </summary>
    public static int[]? Reshaped(int[] src, int[] l, int[] dst)
    {
        int[] sd = Enumerable.Range(0, src.Length).Where(d => src[d] != 1).ToArray();
        int[] td = Enumerable.Range(0, dst.Length).Where(d => dst[d] != 1).ToArray();
        // storage rank of each non-1 source dim
        int[] rank = new int[src.Length];
        int r = 0;
        foreach (int d in l) if (src[d] != 1) rank[d] = r++;

        double[] key = new double[dst.Length];
        int i = 0, j = 0;
        while (i < sd.Length || j < td.Length)
        {
            if (i >= sd.Length || j >= td.Length) return null;   // element count mismatch
            var gs = new List<int> { sd[i] }; long ps = src[sd[i++]];
            var gt = new List<int> { td[j] }; long pt = dst[td[j++]];
            while (ps != pt)
            {
                if (ps < pt) { if (i >= sd.Length) return null; gs.Add(sd[i]); ps *= src[sd[i++]]; }
                else { if (j >= td.Length) return null; gt.Add(td[j]); pt *= dst[td[j++]]; }
            }
            for (int k = 1; k < gs.Count; k++)
                if (rank[gs[k]] != rank[gs[k - 1]] + 1) return null;
            for (int k = 0; k < gt.Count; k++)
                key[gt[k]] = rank[gs[0]] + k / (double)(gt.Count + 1);
        }
        // size-1 target dims ride right behind their logical predecessor (or lead),
        // so an identity source yields an identity target
        double prev = -1;
        for (int e = 0; e < dst.Length; e++)
        {
            if (dst[e] != 1) prev = key[e];
            else key[e] = prev + 1e-3 * (e + 1) / (dst.Length + 1);
        }
        return Enumerable.Range(0, dst.Length).OrderBy(e => key[e]).ThenBy(e => e).ToArray();
    }

    public static string Str(int[]? l) => l is null ? "-" : "[" + string.Join(',', l) + "]";
}
