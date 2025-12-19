#include "include/tensor.h"
#include "include/cpu/ops.h"

#include <cmath>
#include <vector>

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

void mlp(const Tensor<1> &in, const Tensor<2> &mlp_weight, Tensor<1> &out, float bias){
    const int hidden_size = in.shape[0];
    const int output_size = out.shape[0];

    float* x = (float*)in.data;
    float* weight = (float*) mlp_weight.data;
    float* y = (float*) out.data;

    for(int i = 0; i < output_size; i++){
        const float* weight_row = &weight[i * hidden_size];
        y[i] = sdot(x, weight_row, hidden_size) + bias;
    }
}


// softmax(q * kT / sqrt(dim)) * V (TO-DO verify correctness)
void mha(const Tensor<1> &input, Tensor<2> &kv_cache, const Tensor<2> &query_weight, const Tensor<2> &key_weight, const Tensor<2> &value_weight, Tensor<2> &out, const int num_heads, const int head_dim, const int emb_dim, const int pos){

    float* data = (float*) input.data;
    float* q_weight_ptr = (float*) query_weight.data; float* k_weight_ptr = (float*) key_weight.data; float* v_weight_ptr = (float*) value_weight.data;
    float* o = (float*) out.data;

    Tensor<1> q_proj(emb_dim); Tensor<1> k_proj(emb_dim); Tensor<1> v_proj(emb_dim);

    for(int d = 0; d < emb_dim; d++){
        q_proj[d] = sdot(data, q_weight_ptr, emb_dim);
        k_proj[d] = sdot(data, k_weight_ptr, emb_dim);
        v_proj[d] = sdot(data, v_weight_ptr, emb_dim);
        q_weight_ptr += emb_dim; k_weight_ptr += emb_dim; v_weight_ptr += emb_dim;
    }

    float* kv_cache_ptr = (float*)kv_cache.data + pos * kv_cache.shape[1];
    float* k_proj_data = (float*)k_proj.data; float* v_proj_data = (float*)v_proj.data;

    // copy k
    for(int i = 0; i < emb_dim; i++){
        kv_cache_ptr[i] = k_proj_data[i];
    }
    // copy v
    for(int i = 0; i < emb_dim; i++){
        kv_cache_ptr[emb_dim + i] = v_proj_data[i];
    }

    Tensor<1> attn_out_buf(emb_dim);
    std::fill((float*)attn_out_buf.data, (float*)attn_out_buf.data + emb_dim, 0.0f);

    const float attn_scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    std::vector<float> attention_scores(pos + 1);
    std::vector<float> attention_weights(pos + 1);

    for(int head = 0; head < num_heads; head++){
        int head_offset = head * head_dim;

        float* query_head = (float*)q_proj.data + head_offset;
        float* output_head = (float*)attn_out_buf.data + head_offset;

        for(int prev_pos = 0; prev_pos <= pos; prev_pos++){
            const float* key_head = (const float*)kv_cache.data + prev_pos * kv_cache.shape[1] + head_offset;
            float score = sdot(query_head, key_head, head_dim) * attn_scale;
            attention_scores[prev_pos] = score;
        }

        softmax(attention_scores.data(), attention_weights.data(), pos + 1);

        for(int prev_pos = 0; prev_pos <= pos; prev_pos++){
            float attention_weight = attention_weights[prev_pos];
            const float* value_head = (const float*)kv_cache.data + prev_pos * kv_cache.shape[1] + emb_dim + head_offset;

            for(int d = 0; d < head_dim; d++){
                output_head[d] += attention_weight * value_head[d];
            }
        }
    }

    float* attn_buf = (float*)attn_out_buf.data;
    for(int d = 0; d < emb_dim; d++){
        o[d] = attn_buf[d];
    }
}

// grouped query attention (for llama 3?)