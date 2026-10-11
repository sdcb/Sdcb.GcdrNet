// Common helpers for the GCDRNet Metal kernels (fp16 storage, fp32 math —
// same contract as the Vulkan *_h shaders: half is storage-only, every load
// widens to float and all arithmetic runs in fp32).
#include <metal_stdlib>
using namespace metal;

static inline float ferf1(float v)
{
    float s = v < 0.0f ? -1.0f : 1.0f;
    float ax = fabs(v);
    float t = 1.0f / (1.0f + 0.3275911f * ax);
    float poly = ((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t
        - 0.284496736f) * t + 0.254829592f) * t;
    return (1.0f - poly * exp(-ax * ax)) * s;
}

static inline float fexp1(float v)
{
    v = clamp(v, -87.3f, 88.7f);
    float fx = v * 1.4426950408889634f + 0.5f;
    float nf = floor(fx);
    float r = v - nf * 0.693359375f - nf * -0.00021219444005469058f;
    float poly = (((((1.9875691500e-4f * r + 1.3981999507e-3f) * r + 8.3334519073e-3f) * r
        + 4.1665795894e-2f) * r + 1.6666665459e-1f) * r + 5.0000001201e-1f) * r * r + r + 1.0f;
    return poly * as_type<float>((uint(int(nf)) + 127u) << 23);
}

// Conv epilogue activation: bits 4-6 (relu=1, gelu=2, sigmoid=3).
static inline float cact(float v, uint flags)
{
    uint a = (flags >> 4) & 7u;
    if (a == 1u) return max(v, 0.0f);
    if (a == 2u) return 0.5f * v * (1.0f + ferf1(v * 0.7071067811865475f));
    if (a == 3u) return 1.0f / (1.0f + fexp1(-v));
    return v;
}

static inline int mirror1(int i, int d)
{
    if (d <= 1) return 0;
    while (i < 0 || i >= d) i = i < 0 ? -i : 2 * d - 2 - i;
    return i;
}
