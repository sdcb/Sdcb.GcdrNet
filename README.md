# Sdcb.GcdrNet

[English](#english) | [中文](#中文)

---

## English

**Sdcb.GcdrNet** is a pure C# implementation of [GCDRNet](https://github.com/ZZZHANG-jx/GCDRNet)
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
- Backend abstraction (`GcdrBackend`) leaves room for a future Metal backend.

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

### Quick start

```csharp
using Sdcb.GcdrNet;

// gcnet/drnet simplified ONNX (see tests/testdata/models)
using var enh = GcdrEnhancer.FromFiles("gcdrnet_gcnet_sim.onnx", "gcdrnet_drnet_sim.onnx",
    GcdrBackend.Auto);   // Auto: Vulkan if available, else CPU. Also: Cpu / Vulkan.

byte[] bgr = File.ReadAllBytes("input.bgr");          // tightly packed BGR, HWC
byte[] dst = enh.Enhance(bgr, width: 1600, height: 1152);
// or strided: enh.Enhance(srcSpan, w, h, srcStride, dstSpan, dstStride);
```

Input is padded to a multiple of 32 with top/left **replicate** padding and cropped back —
matching the reference Python implementation.

### Layout

- `src/Sdcb.GcdrNet/Onnx` — minimal protobuf reader + ONNX model/graph objects
- `src/Sdcb.GcdrNet/Engine` — `GraphCompiler` (specialize → fold → fuse → arena-plan),
  `InferenceSession`
- `src/Sdcb.GcdrNet/Kernels` — Scalar / `Vector<float>` / AVX2 / AVX-512 kernels (`Ops`, `Avx2Kernels`, `Nhwc`,
  `PixelOps`, `SimdMath`)
- `src/Sdcb.GcdrNet/Backends/Vulkan` — `Vk` P/Invoke layer, `VkDevice`, `VkShaders` (embedded
  SPIR-V), `VkGraph` dispatch lowering, `VkSession`, `VkPipeline2` (fused pipeline), `VkBackend`
- `src/Sdcb.GcdrNet/Backends/Vulkan/Shaders` — GLSL sources; `tools/build-shaders.bat`
  recompiles `.comp` → `.spv` (committed; dev-time only, not part of `dotnet build`)
- `bench/Sdcb.GcdrNet.Benchmark` — `--check`, `--bench` (CPU), `--benchvk` (Vulkan e2e),
  `--vkprof` (per-shader/per-dispatch GPU timing)

### Tests

```bash
dotnet test tests/Sdcb.GcdrNet.Tests -c Release
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
- Thread-safety: `GcdrEnhancer` is thread-safe (CPU: per-size session pools; Vulkan: size-agnostic worker pool); individual
  `InferenceSession`/`VkInferenceSession` objects are not.
- Not covered: batched inference (N>1), fp16/int8, Metal (interface-ready).

---

## 中文

**Sdcb.GcdrNet** 是 [GCDRNet](https://github.com/ZZZHANG-jx/GCDRNet)（相机拍摄文档去阴影/外观增强，
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

### 快速上手

```csharp
using Sdcb.GcdrNet;

using var enh = GcdrEnhancer.FromFiles("gcdrnet_gcnet_sim.onnx", "gcdrnet_drnet_sim.onnx",
    GcdrBackend.Auto);   // Auto：有 Vulkan 用 Vulkan，否则 CPU；也可显式 Cpu / Vulkan

byte[] dst = enh.Enhance(bgrBytes, width: 1600, height: 1152);
// 非紧密 stride：enh.Enhance(src, w, h, srcStride, dst, dstStride);
```

输入补齐到 32 倍数（上/左 replicate padding），输出自动裁回，与参考 Python 实现一致。

### 构建 shader

`src/Sdcb.GcdrNet/Backends/Vulkan/Shaders/*.spv` 已随源码提交（嵌入资源）；
改动 `.comp` 后运行 `tools/build-shaders.bat`（需要 glslc / Vulkan SDK）重新生成。

### 测试

```bash
dotnet test tests/Sdcb.GcdrNet.Tests -c Release
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
