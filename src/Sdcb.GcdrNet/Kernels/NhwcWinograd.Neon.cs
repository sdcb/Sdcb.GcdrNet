namespace Sdcb.GcdrNet.Kernels;

/// <summary>
/// NEON Winograd schedule for layers whose transformed weights U (36·oc·ic floats) outgrow the L2
/// caches (the deep, low-resolution layers): the per-task tile blocks of <see cref="WinoTiles"/> stream
/// every U panel once per block of tiles — DRAM-bound on the RK3399 (512 KB A53 / 1 MB A72 L2).
/// Instead tiles are taken in large groups and each group runs three parallel phases: input
/// transforms into V[36][Tg][ic], one GEMM task per (xi, oc block) that loads its U panel once for
/// the whole group, then the output transforms with the usual epilogue. Same per-element math as
/// <see cref="WinoTiles"/> (same transforms, same k loop), so the outputs are bit-identical.
/// </summary>
internal static unsafe partial class Nhwc
{
    /// <summary>U sizes (floats) above this take <see cref="WinogradGrouped"/> on NEON (512 KB).</summary>
    private const long NeonWinoUFloats = 128 * 1024;

    [ThreadStatic] private static float[]? t_wgV, t_wgM;

    private static void WinogradGrouped(in WinoArgs a, float* dp, float* ap)
    {
        int ic = a.Ic, nb = a.Nb, tw = a.Tw, oh = a.Oh, ow = a.Ow, oc = a.Oc;
        int nt = (oh + 3) / 4 * tw;
        // V group ~2 MB (spread over the L2s, re-read nb times), whole Tile-row GEMM tiles
        int tg = Math.Clamp(524288 / (36 * ic) / Tile * Tile, Tile, (nt + Tile - 1) / Tile * Tile);
        float* V = Pinned(ref t_wgV, 36 * tg * ic), M = Pinned(ref t_wgM, 36 * nb * tg * Ob);
        WinoArgs ca = a;
        for (int g0 = 0; g0 < nt; g0 += tg)
        {
            int tn = Math.Min(tg, nt - g0);
            Parallel.For(0, (tn + 7) / 8, PO, k =>
            {
                float** dp6 = stackalloc float*[36];
                for (int t = k * 8, te = Math.Min(tn, t + 8); t < te; t++)
                {
                    int ty = (g0 + t) / tw, tx = (g0 + t) % tw;
                    int iy0 = ty * 4 - ca.PadT, ix0 = tx * 4 - ca.PadL;
                    for (int i = 0; i < 6; i++)
                    {
                        int r = SrcIdx(iy0 + i, ca.Ih, ca.Reflect);
                        for (int j = 0; j < 6; j++)
                        {
                            int c = SrcIdx(ix0 + j, ca.Iw, ca.Reflect);
                            dp6[i * 6 + j] = r < 0 || c < 0 ? ca.PadVec : ca.Src + ((long)r * ca.Iw + c) * ic;
                        }
                    }
                    InputTransform(dp6, ic, V + (long)t * ic, (long)tg * ic);
                }
            });
            Parallel.For(0, 36 * nb, PO, j =>
            {
                int xi = j / nb, blk = j % nb;
                float** ptrs = stackalloc float*[Tile];
                float* zero = stackalloc float[Ob];
                new Span<float>(zero, Ob).Clear();
                float* wb = ca.U + ((long)xi * nb + blk) * ic * Ob;
                float* vx = V + (long)xi * tg * ic, mx = M + ((long)xi * nb + blk) * tg * Ob;
                // all Tile rows land in M (tg is a multiple of Tile); rows past tn are never read
                for (int r0 = 0; r0 < tn; r0 += Tile)
                {
                    int rows = Math.Min(Tile, tn - r0);
                    for (int i = 0; i < Tile; i++) ptrs[i] = vx + (long)(r0 + Math.Min(i, rows - 1)) * ic;
                    TileNeonK(ptrs, 1, ic, wb, zero, mx + (long)r0 * Ob, Ob);
                }
            });
            int chunks = (tn + 15) / 16;
            Parallel.For(0, chunks * nb, PO, j =>
            {
                int blk = j % nb, t0 = j / nb * 16;
                float* tmp = stackalloc float[16 * Ob];
                WinoOutput(M + ((long)blk * tg + t0) * Ob, (long)nb * tg * Ob, Ob, Math.Min(16, tn - t0), g0 + t0,
                    tw, oh, ow, oc, blk * Ob, ca.Bias + blk * Ob, dp, ap, 0, ca.Act, tmp, false, false);
            });
        }
    }
}
