// Storage element type. Built twice: name.spv (fp32) and name_h.spv (-DF16).
// F16 = 16-bit *storage* only (VK 1.1 storageBuffer16BitAccess): every load is
// widened with float()/vec4() and all arithmetic, including reductions, runs
// in fp32 — no shaderFloat16 ALU, no fp16 accumulation.
#ifdef F16
#extension GL_EXT_shader_16bit_storage : require
#define ET float16_t
#define ET4 f16vec4
#else
#define ET float
#define ET4 vec4
#endif
