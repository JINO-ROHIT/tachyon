#include "include/tensor.h"
#include "include/cpu/ops.h"

#include <cmath>

//TO-DO later use the config in metadata json instead of hardcoded config params

//activation functions

void gelu(const Tensor<1> &in, Tensor<1> &out){
    float* data = (float*) in.data;
    float* o = (float*) out.data;

    for(int i = 0; i < in.shape[0]; i++){
        float x = data[i];
        o[i] = 0.5f * x * (1.0f + tanhf(0.797885f * (x + 0.044715f * x * x * x)));
    }
}

float silu(float &in){
    return in / (1.0f + expf(-in)); 
}

// math operations

float sdot(const float* a, const float* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}


// layer specific operations

void softmax(const float* input, float* output, int n) {
    float max_val = input[0];
    for (int i = 1; i < n; ++i) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }

    for (int i = 0; i < n; ++i) {
        output[i] /= sum;
    }
}

// check if we need freq config? 
void precompute_rope_params(int rotary_dim, float theta, int ctx_length, float* cos_cache, float* sin_cache){ // ctx_length * (rotary_dim / 2)
    int n_pairs = rotary_dim / 2;

    for (int p = 0; p < ctx_length; ++p) {
        for (int k = 0; k < n_pairs; ++k) {
            float inv_freq = powf(theta, -k / n_pairs); // theta raised to (-pos / total_pairs)
            float angle = p * inv_freq;

            cos_cache[p * n_pairs + k] = cosf(angle);
            sin_cache[p * n_pairs + k] = sinf(angle);
        }
    }
}

//TO-DO switch to tensor later
// qi = qi * cos - qi+1 * sin
// qi+1 = qi * sin + qi+1 * cos;
void apply_rope(float* vec, int rotary_dim, int pos, const float* cos_cache, const float* sin_cache){
    int n_pairs = rotary_dim / 2;

    for (int k = 0; k < n_pairs; ++k) {
        int i = 2 * k;
        float c = cos_cache[pos * n_pairs + k];
        float s = sin_cache[pos * n_pairs + k];

        float v0 = vec[i];
        float v1 = vec[i + 1];
        vec[i]     = v0 * c - v1 * s;
        vec[i + 1] = v0 * s + v1 * c;
    }
}

// https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.normalization.RMSNorm.html
void rmsnorm(const Tensor<1> &in, Tensor<1> &out, float eps){
    float* data = (float*) in.data;
    float* o = (float*) out.data;

    float sum_sq = 0.0f;
    int n = in.shape[0];

    for(int i = 0; i < n; i++){
        sum_sq += data[i] * data[i];
    }
    const float inv_rms = 1.0f / sqrtf(eps + sum_sq / n); 

    for(int i = 0; i < n; i++){
        o[i] = data[i] * inv_rms;
    }
}

// https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
void layernorm(const Tensor<1> &in, Tensor<1>&out, Tensor<1> weight, Tensor<1> bias, float eps){
    float* data = (float*) in.data;
    float* o = (float*) out.data;

    float sum_ele = 0.0f;
    float sum_sq = 0.0f;
    int n = in.shape[0];


    for(int i = 0; i < n; i++){
        sum_ele += data[i];
        sum_sq += data[i] * data[i];
    }

    const float mean = sum_ele / n;
    const float var = sum_sq / n - mean * mean;
    const float invstddev = 1.0 / sqrtf(var + eps);

    float *w = (float*) weight.data;
    float *b = (float*) bias.data;

    for (int j = 0; j < n; j++) {
        o[j] = (data[j] - mean) * invstddev * w[j] + b[j];
    }

}

// grouped query attention