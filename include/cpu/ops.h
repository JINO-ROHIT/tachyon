#include "tensor.h"

void gelu(const Tensor<1> &in, Tensor<1> &out);
float silu(float &in);

float sdot(const float* a, const float* b, int n);

void precompute_rope_params(int rotary_dim, float theta, int ctx_length, float* cos_cache, float* sin_cache);
void apply_rope(float* vec, int rotary_dim, int pos, const float* cos_cache, const float* sin_cache);
void layernorm(const Tensor<1> &in, Tensor<1>&out, Tensor<1> weight, Tensor<1> bias, float eps);