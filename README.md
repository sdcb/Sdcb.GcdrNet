# Sdcb.SimdGcdrNet

[English](#english) | [中文](#中文)

---

## English

**Sdcb.SimdGcdrNet** is a pure C# implementation of [GCDRNet](https://github.com/ZZZHANG-jx/GCDRNet)
(*Appearance Enhancement for Camera-captured Document Images in the Wild*, IEEE TAI 2023) —
document-image shadow removal and appearance enhancement **without onnxruntime or any other
third-party inference engine**. It ships its own compiled-model inference engine, inspired by
`Sdcb.SimdPaddleOCR`.

The model is a two-stage pipeline:

1. **GC-Net** predicts a per-pixel multiplicative shading/illumination map `I_sm`.
2. **Division correction**: `I_GC = clamp(I_s / I_sm, 0, 1)`.
3. **DR-Net** consumes the 6-channel input `[I_s; I_GC]` and produces the enhanced image.

### Features

- **Zero inference-engine dependency** — self-contained ONNX proto reader, graph compiler
  (shape specialization, constant folding, operator fusion, buffer-arena planning).
- **CPU backend** with ISA tiered dispatch: AVX-512 → AVX2 (+FMA, multithreaded) → `Vector<float>` → Scalar on x64;
  NEON (AArch64 AdvSimd) → Scalar on arm64.
- **Vulkan compute backend** (Windows/Linux/Android `libvulkan`): push-descriptors, ReBAR-aware
  staging, single-recorded-command-buffer graph execution, fused `bgr2nchw → GC → div6 → DR →
  nchw2bgr` pipeline with arbitrary source/destination strides.
- **Metal compute backend** (macOS, Apple Silicon): pure C# ObjC P/Invoke (no Xamarin/Metal.NET),
  fp16 storage + fp32 math, simdgroup-MMA implicit-GEMM convolutions — see
  [docs/apple-m4-metal.md](docs/apple-m4-metal.md) (~7.5× CPU on M4).
- A NEON kernel slot is reserved in the ISA hierarchy.

### Performance (RTX 3080 Ti, 1600×1152)

| Path | Time | vs PyTorch baseline |
|---|---|---|
| Vulkan e2e (incl. upload/download + pre/post) | **~9.6 ms** | PyTorch pure inference 30.5 ms; with pre/post 57.2 ms |
| Vulkan corpus (49 RealDAE, seq order) | **~28 ms/img** | methodology: [docs/corpus.md](docs/corpus.md), current numbers: [docs/perf.md](docs/perf.md) |
| CPU AVX2 corpus (same set) | ~722 ms/img | peak working set ~0.8 GB (Vulkan: 0.33 GB + 0.42 GB VRAM) |
| CPU AVX2 (16 threads) gc+dr inference | ~365 ms | — |

Ryzen AI 9 H 365 (AVX-512) + Radeon 880M (RDNA 3.5 iGPU): see [docs/amd-h365-880m.md](docs/amd-h365-880m.md)
(Vulkan ~100 ms/img, CPU AVX-512 ~285 ms/img on the same corpus).

Ryzen 9 5950X (AVX2, 16C/32T) + Intel Arc B580 (Xe2, XMX via 8x16x16 coopmat): see [docs/amd-5950x-b580.md](docs/amd-5950x-b580.md)
(Vulkan ~36 ms/img, CPU AVX2 ~414 ms/img on the same corpus).

Rockchip RK3399 (2× A72 + 4× A53, NEON, passively cooled): see [docs/rockchip-rk3399.md](docs/rockchip-rk3399.md)
(CPU NEON ~9.0 s/img on the same corpus, 2.6× the generic `Vector<float>` path at a locked 1 GHz).

Snapdragon 8 Gen 3 (realme GT5 Pro, Android 16; Adreno 750 without coopmat, 64/128-lane waves): see [docs/snapdragon-8gen3.md](docs/snapdragon-8gen3.md)
(Vulkan ~486 ms/img with register-tiled conv/GEMM kernels, 2.6× the shared-memory path; CPU NEON ~1.03 s/img on the same corpus).

### Quick start

| Package | Contents |
|---|---|
| `Sdcb.SimdGcdrNet` | the engine (CPU / Vulkan / Metal); ships LINQPad samples (System.Drawing, ImageSharp, OpenCvSharp, SkiaSharp + SimdPaddleOCR) |
| `Sdcb.SimdGcdrNet.Models` | the GC-Net + DR-Net ONNX models (~21 MB) as embedded resources |

```csharp
using Sdcb.SimdGcdrNet;
using Sdcb.SimdGcdrNet.Models;

// Auto: Metal on macOS Apple Silicon, else Vulkan, else CPU. Also: Cpu (default) / Vulkan / Metal.
using GcdrEnhancer enh = GcdrModels.CreateEnhancer(GcdrBackend.Auto);
// without the models package: GcdrEnhancer.FromFiles("gcdrnet_gcnet_sim.onnx", "gcdrnet_drnet_sim.onnx", ...)
// (models in tests/testdata/models), or FromBytes / FromStreams

byte[] dst = enh.Enhance(bgr, width: 1600, height: 1152);   // tightly packed BGR24, HWC
// strided, any interleaved 8-bit layout, output in the same layout (alpha copied through):
enh.Enhance(srcSpan, w, h, srcStride, dstSpan, dstStride, GcdrPixelFormat.Rgba32);
```

Pixel formats: `Bgr24` (OpenCV `Mat`, GDI+ 24bpp), `Rgb24`, `Bgra32` (SkiaSharp, GDI+ 32bpp),
`Rgba32` (ImageSharp, browser canvas). `dst` may be the same buffer as `src` (same stride) for in-place enhancement.

Input is padded to a multiple of 32 with top/left **replicate** padding and cropped back —
matching the reference Python implementation.

### Layout

- `src/Sdcb.SimdGcdrNet/Onnx` — minimal protobuf reader + ONNX model/graph objects
- `src/Sdcb.SimdGcdrNet/Engine` — `GraphCompiler` (specialize → fold → fuse → arena-plan),
  `InferenceSession`
- `src/Sdcb.SimdGcdrNet/Kernels` — Scalar / `Vector<float>` / AVX2 / AVX-512 kernels (`Ops`, `Avx2Kernels`, `Nhwc`,
  `PixelOps`, `SimdMath`)
- `src/Sdcb.SimdGcdrNet/Backends/Vulkan` — `Vk` P/Invoke layer, `VkDevice`, `VkShaders` (embedded
  SPIR-V), `VkGraph` dispatch lowering, `VkSession`, `VkPipeline2` (fused pipeline), `VkBackend`
- `src/Sdcb.SimdGcdrNet/Backends/Vulkan/Shaders` — GLSL sources; `tools/build-shaders.bat`
  recompiles `.comp` → `.spv` (committed; dev-time only, not part of `dotnet build`)
- `src/Sdcb.SimdGcdrNet/Backends/Metal` — `ObjC`/`Metal` P/Invoke layer, `MtlGraph` dispatch
  lowering, `MtlSession`, `MtlPipeline` (fused pipeline), `MetalBackend`
- `src/Sdcb.SimdGcdrNet/Backends/Metal/Shaders` — MSL sources (`.metal`, embedded resources,
  compiled to PSOs at run time — no pre-build step)
- `bench/Sdcb.SimdGcdrNet.Benchmark` — `--check`, `--bench` (CPU), `--benchvk` (Vulkan e2e),
  `--benchmtl` / `--checkgpu` / `--mtlprof` / `--mtrace` (Metal e2e, diff check, per-shader
  profile, CPU-vs-Metal per-node trace), `--vkprof` (per-shader/per-dispatch GPU timing), `--vkcheck` / `--vknodes`
  (Vulkan vs CPU, whole graph / per node)
- `bench/Sdcb.SimdGcdrNet.AndroidBench` — the same modes inside an Android APK (`run.ps1`: build → adb install → push testdata → run → pull logs)
- `src/Sdcb.SimdGcdrNet.Models` — `GcdrModels` over the embedded ONNX files (taken from `tests/testdata/models`)
- `src/Sdcb.SimdGcdrNet/linqpad-samples` — LINQPad samples packed into `Sdcb.SimdGcdrNet` (tag `linqpad-samples`)
- `build/` — LINQPad scripts to pack (`01-build-packages`) and publish (`02-publish-proget-nuget`) the
  packages; versions live in `00-common.linq` and `Directory.Build.props`

### Tests

```bash
dotnet test tests/Sdcb.SimdGcdrNet.Tests -c Release
```

Golden outputs come from onnxruntime runs of the simplified models (CPU ≤ 1e-4 /
Vulkan ≤ 5e-4); regenerate with `dotnet run --project tools/GenGolden` (pure C#:
OnnxRuntime oracle + SkiaSharp decode). `tests/testdata/realdae/` holds a 50-image
RealDAE corpus (`NN_<color|shadow|bleed>.webp` + paired `NN_gt.webp` ground truths);
`CorpusTests` smoke-tests decode→enhance, and the benchmark's `--corpus [cpu|vulkan]`
mode reports per-image ms, MP/s, memory and PSNR/MAE vs the `_gt` pairs.
Kernel dispatch follows the runtime's
`IsSupported` flags, so ISA tiers are exercised via the standard runtime knobs —
`test-all.ps1` runs the default tier (AVX-512 where supported), `DOTNET_EnableAVX512=0` and
`DOTNET_EnableAVX512F=0` (AVX2), `DOTNET_EnableAVX2=0` (128-bit `Vector<float>`),
`DOTNET_EnableAVX=0` (SSE, no FMA) and `DOTNET_EnableHWIntrinsic=0` (scalar).

### Notes

- Vulkan path requires a compute-capable device; `GcdrBackend.Auto` falls back to CPU.
- Metal path requires macOS on Apple Silicon with Apple7+ (simdgroup-MMA, i.e. M1+; an x64 process
  under Rosetta also qualifies); `Auto` checks `SupportsApple7` before choosing it,
  `GcdrBackend.Metal` selects it unconditionally. Set `GCDRNET_MTL_VERBOSE=1` to print the chosen device.
- Thread-safety: `GcdrEnhancer` is thread-safe (CPU: per-size session pools; Vulkan: size-agnostic worker pool); individual
  `InferenceSession`/`VkInferenceSession` objects are not.
- Not covered: batched inference (N>1), fp16/int8.
- Native AOT works (`IsAotCompatible`, no trim/AOT warnings; CPU and Vulkan output verified identical to JIT). On **x64** the executable
  project must set the instruction set, otherwise ILC compiles for the SSE baseline, folds
  `Avx2.IsSupported` / `Avx512F.IsSupported` to `false` and the CPU backend silently runs the 128-bit
  `Vector<float>` tier (~1.3× slower):

  ```xml
  <IlcInstructionSet>x86-64-v3</IlcInstructionSet>   <!-- AVX2+FMA; x86-64-v4 for AVX-512 -->
  ```

  The binary then refuses to start on CPUs without those extensions. arm64 needs nothing (NEON is in
  the baseline); GPU backends are unaffected.

---

## 中文

**Sdcb.SimdGcdrNet** 是 [GCDRNet](https://github.com/ZZZHANG-jx/GCDRNet)（相机拍摄文档去阴影/外观增强，
IEEE TAI 2023）的**纯 C# 推理库**——不依赖 onnxruntime 等第三方推理引擎，像 `Sdcb.SimdPaddleOCR`
一样自带 CompiledModel 推理引擎。

模型为两级串联：**GC-Net** 估计逐像素光照/阴影图 `I_sm` → `I_GC = clamp(I_s/I_sm)` 除法校正 →
**DR-Net** 消费 6 通道 `[I_s; I_GC]` 输出增强图。

### 特性

- **零推理引擎依赖**：自带 ONNX proto 读取 + 图编译（shape 特化、常量折叠、算子融合、arena 规划）
- **CPU 后端**：x64 上 AVX-512 → AVX2+FMA 多线程 → `Vector<float>` → Scalar 四档 ISA 自动分发；arm64 上 NEON（AdvSimd）→ Scalar
- **Vulkan 后端**：push-descriptor、ReBAR 感知 staging、单命令缓冲图执行、
  融合流水线（bgr2nchw→GC→div6→DR→nchw2bgr），支持任意 src/dst stride
- 预留 Metal 后端接口位

### 性能（RTX 3080 Ti，1600×1152）

| 路径 | 耗时 | 对比 PyTorch |
|---|---|---|
| Vulkan 端到端（含上下行+前后处理） | **~9.6 ms** | PyTorch 纯推理 30.5 ms / 含前后处理 57.2 ms |
| Vulkan corpus（49 张 RealDAE，序号序） | **~28 ms/张** | 口径见 [docs/corpus.md](docs/corpus.md)，最新数据见 [docs/perf.md](docs/perf.md) |
| CPU AVX2 corpus（同集） | ~722 ms/张 | 峰值 working set ~0.8 GB（Vulkan：0.33 GB + 显存 0.42 GB） |
| CPU AVX2（16 线程）gc+dr 推理 | ~365 ms | — |

锐龙 AI 9 H 365（AVX-512）+ Radeon 880M（RDNA 3.5 核显）见 [docs/amd-h365-880m.md](docs/amd-h365-880m.md)
（同一语料 Vulkan ~100 ms/张，CPU AVX-512 ~285 ms/张）。

锐龙 9 5950X（AVX2，16C/32T）+ Intel Arc B580（Xe2，8x16x16 coopmat 走 XMX）见 [docs/amd-5950x-b580.md](docs/amd-5950x-b580.md)
（同一语料 Vulkan ~36 ms/张，CPU AVX2 ~414 ms/张）。

瑞芯微 RK3399（2× A72 + 4× A53，NEON，被动散热）见 [docs/rockchip-rk3399.md](docs/rockchip-rk3399.md)
（同一语料 CPU NEON ~9.0 s/张；锁 1 GHz 时是通用 `Vector<float>` 路径的 2.6 倍）。

骁龙 8 Gen 3（真我 GT5 Pro，Android 16；Adreno 750 无 coopmat，64/128 lane wave）见 [docs/snapdragon-8gen3.md](docs/snapdragon-8gen3.md)
（同一语料 Vulkan ~486 ms/张，寄存器分块卷积/GEMM 内核比共享内存路径快 2.6 倍；CPU NEON ~1.03 s/张）。

### 快速上手

| 包 | 内容 |
|---|---|
| `Sdcb.SimdGcdrNet` | 推理引擎（CPU / Vulkan / Metal）；附带 LINQPad 示例（System.Drawing、ImageSharp、OpenCvSharp、SkiaSharp + SimdPaddleOCR） |
| `Sdcb.SimdGcdrNet.Models` | GC-Net + DR-Net 两个 ONNX 模型（约 21 MB），以嵌入资源形式提供 |

```csharp
using Sdcb.SimdGcdrNet;
using Sdcb.SimdGcdrNet.Models;

// Auto：macOS Apple Silicon 优先 Metal，否则 Vulkan，最后 CPU；也可显式 Cpu（默认）/ Vulkan / Metal
using GcdrEnhancer enh = GcdrModels.CreateEnhancer(GcdrBackend.Auto);
// 不用模型包：GcdrEnhancer.FromFiles("gcdrnet_gcnet_sim.onnx", "gcdrnet_drnet_sim.onnx", ...)
// （模型见 tests/testdata/models），或 FromBytes / FromStreams

byte[] dst = enh.Enhance(bgrBytes, width: 1600, height: 1152);   // 紧密排列的 BGR24，HWC
// 带 stride、任意 8 位交错格式，输出与输入同格式（alpha 原样拷回）：
enh.Enhance(src, w, h, srcStride, dst, dstStride, GcdrPixelFormat.Rgba32);
```

像素格式：`Bgr24`（OpenCV `Mat`、GDI+ 24bpp）、`Rgb24`、`Bgra32`（SkiaSharp、GDI+ 32bpp）、
`Rgba32`（ImageSharp、浏览器 canvas）。`dst` 可以和 `src` 是同一块内存（stride 相同），即原地增强。

输入补齐到 32 倍数（上/左 replicate padding），输出自动裁回，与参考 Python 实现一致。

### 构建 shader

`src/Sdcb.SimdGcdrNet/Backends/Vulkan/Shaders/*.spv` 已随源码提交（嵌入资源）；
改动 `.comp` 后运行 `tools/build-shaders.bat`（需要 glslc / Vulkan SDK）重新生成。

### 测试

```bash
dotnet test tests/Sdcb.SimdGcdrNet.Tests -c Release
```

golden 由 onnxruntime 对简化模型跑出（CPU 阈值 ≤1e-4 / Vulkan ≤5e-4）；
重新生成用 `dotnet run --project tools/GenGolden`（纯 C#，OnnxRuntime oracle +
SkiaSharp 解码）。`tests/testdata/realdae/` 存 50 张 RealDAE 语料
（`NN_<color|shadow|bleed>.webp` 及配对 `NN_gt.webp` 基准图）；`CorpusTests`
做 decode→enhance 冒烟，benchmark `--corpus [cpu|vulkan]` 按图报耗时、MP/s、
内存及对照 `_gt` 的 PSNR/MAE。
内核分发直接跟随 runtime 的 `IsSupported` 标志，
改 CPU 内核请用 `test-all.ps1` 跑 ISA 矩阵：
`DOTNET_EnableAVX2=0`（128 位 Vector）、`DOTNET_EnableAVX=0`（SSE 无 FMA）、
`DOTNET_EnableHWIntrinsic=0`（纯标量），以及 `DOTNET_EnableAVX512=0` / `DOTNET_EnableAVX512F=0`（AVX2）。

### Native AOT

支持 Native AOT（`IsAotCompatible`，裁剪 / AOT 无警告；CPU、Vulkan 输出与 JIT 逐字节一致）。**x64** 上可执行项目必须指定指令集，
否则 ILC 按 SSE 基线编译，`Avx2.IsSupported` / `Avx512F.IsSupported` 被折成 `false`，CPU 后端会静默退到
128 位 `Vector<float>` 档（慢约 1.3 倍）：

```xml
<IlcInstructionSet>x86-64-v3</IlcInstructionSet>   <!-- AVX2+FMA；x86-64-v4 才有 AVX-512 -->
```

这样编出的程序在不支持这些扩展的 CPU 上无法启动。arm64 不用设（NEON 在基线里）；GPU 后端不受影响。
