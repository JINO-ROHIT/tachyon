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
void mha(const Tensor<1> &input, Tensor<2> &kv_cache, const Tensor<2> &query_weight, const Tensor<2> &key_weight, const Tensor<2> &value_weight, Tensor<2> &out, const int num_heads, const int head_dim, const int emb_dim, const int pos);