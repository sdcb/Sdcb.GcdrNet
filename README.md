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
- **CPU backend** with ISA tiered dispatch: AVX2 (+FMA, multithreaded) → `Vector<float>` → Scalar.
- **Vulkan compute backend** (Windows/Linux/Android `libvulkan`): push-descriptors, ReBAR-aware
  staging, single-recorded-command-buffer graph execution, fused `bgr2nchw → GC → div6 → DR →
  nchw2bgr` pipeline with arbitrary source/destination strides.
- Backend abstraction (`GcdrBackend`) leaves room for a future Metal backend;
  AVX-512 / NEON kernel slots reserved in the ISA hierarchy.

### Performance (RTX 3080 Ti, 1600×1152)

| Path | Time | vs PyTorch baseline |
|---|---|---|
| Vulkan e2e (incl. upload/download + pre/post) | **~30.8 ms** | PyTorch pure inference 30.5 ms; with pre/post 57.2 ms |
| CPU AVX2 (16 threads) e2e | ~758 ms | — |

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
- `src/Sdcb.GcdrNet/Kernels` — Scalar / `Vector<float>` / AVX2 kernels (`Ops`, `Avx2Kernels`,
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

Golden outputs come from onnxruntime runs of the simplified models (max diff ≤ 2e-3;
Vulkan node-by-node vs CPU within ~1e-4). CPU kernel changes should be re-checked under
`HYMT2SHARP_FORCE_PORTABLE=1` (Vector tier), `DOTNET_EnableAVX=0`, `DOTNET_EnableAVX512=0`.

### Notes

- Vulkan path requires a compute-capable device; `GcdrBackend.Auto` falls back to CPU.
- Thread-safety: `GcdrEnhancer` is thread-safe (per-size session pooling); individual
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
- **CPU 后端**：AVX2+FMA 多线程 → `Vector<float>` → Scalar 三档 ISA 自动分发
- **Vulkan 后端**：push-descriptor、ReBAR 感知 staging、单命令缓冲图执行、
  融合流水线（bgr2nchw→GC→div6→DR→nchw2bgr），支持任意 src/dst stride
- 预留 Metal 后端接口位与 AVX-512/NEON 内核位

### 性能（RTX 3080 Ti，1600×1152）

| 路径 | 耗时 | 对比 PyTorch |
|---|---|---|
| Vulkan 端到端（含上下行+前后处理） | **~30.8 ms** | PyTorch 纯推理 30.5 ms / 含前后处理 57.2 ms |
| CPU AVX2（16 线程）端到端 | ~758 ms | — |

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

golden 由 onnxruntime 对简化模型跑出（阈值 ≤2e-3）；改 CPU 内核请在
`HYMT2SHARP_FORCE_PORTABLE=1`、`DOTNET_EnableAVX=0`、`DOTNET_EnableAVX512=0` 下复测。
