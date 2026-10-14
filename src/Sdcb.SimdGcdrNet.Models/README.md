# Sdcb.SimdGcdrNet.Models

The [GCDRNet](https://github.com/ZZZHANG-jx/GCDRNet) models (simplified GC-Net + DR-Net ONNX, ~21 MB) as
embedded resources, for [Sdcb.SimdGcdrNet](https://www.nuget.org/packages/Sdcb.SimdGcdrNet) — pure C#
document-image shadow removal / appearance enhancement (CPU SIMD, Vulkan, Metal; no onnxruntime).

No dependencies, versioned independently of the engine. The models are exposed as zero-copy
`ReadOnlySpan<byte>` views of the loaded assembly image, so there are no model files to deploy.

```csharp
using Sdcb.SimdGcdrNet;
using Sdcb.SimdGcdrNet.Models;

using GcdrEnhancer enh = GcdrEnhancer.FromBytes(GcdrModels.GcNet, GcdrModels.DrNet, GcdrBackend.Auto);
byte[] dst = enh.Enhance(bgr, width, height);   // tightly packed BGR24
```

See the [project README](https://github.com/sdcb/SimdGcdrNet#readme) for the full API, backends and benchmarks.

## License

The model weights come from [GCDRNet](https://github.com/ZZZHANG-jx/GCDRNet) (MIT License, Copyright (c) 2024
Jiaxin Zhang); the full notice is in `THIRD-PARTY-NOTICES.txt` in this package. If you use them, please cite:

```text
@article{zhang2023appearance,
  title={Appearance Enhancement for Camera-captured Document Images in the Wild},
  author={Zhang, Jiaxin and Liang, Lingyu and Ding, Kai and Guo, Fengjun and Jin, Lianwen},
  journal={IEEE Transactions on Artificial Intelligence},
  year={2023}
}
```
