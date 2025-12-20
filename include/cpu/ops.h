#pragma once

#include "include/tensor.h"

float sdot(const float* a, const float* b, int n);

void gelu(const Tensor<1> &in, Tensor<1> &out);
float silu(float &in);

void precompute_rope_params(int rotary_dim, float theta, int ctx_length, float* cos_cache, float* sin_cache);
void apply_rope(float* vec, int rotary_dim, int pos, const float* cos_cache, const float* sin_cache);
void layernorm(const Tensor<1> &in, Tensor<1> &out, Tensor<1> weight, Tensor<1> bias, float eps);
void rmsnorm(const Tensor<1> &in, Tensor<1> &out, float eps);
void mlp(const Tensor<1> &in, const Tensor<2> &mlp_weight, Tensor<1> &out, float bias);
void simple_mha(
    const Tensor<1>& input,
    const Tensor<2>& query_weight,
    const Tensor<2>& key_seq,
    const Tensor<2>& value_seq,
    Tensor<1>& out,
    int num_heads,
    int head_dim);