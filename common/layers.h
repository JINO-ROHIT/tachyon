#pragma once

#include <sys/mman.h>
#include <vector>

#include "include/tensor.h"

//TO-DO see if we want to align the struct sizes

struct Config {
    std::string model_type;
    int head_dim;
    std::string hidden_act;
    int hidden_size;
    int intermediate_size;
    int max_position_embeddings;
    int num_attention_heads;
    int num_hidden_layers;
    int num_key_value_heads;
    float rms_norm_eps;
    float rope_theta;
    bool tie_word_embeddings;
    int vocab_size;
};

struct WeightInfo {
    std::string name;
    uint64_t offset;
    Dtype dtype;
    std::vector<int> shape;
};


struct ModelWeights{

    ModelWeights() {}

    Tensor<2> embed_tokens;
    Tensor<1> rms_norm;
    Tensor<1> ln_f; // final layer norm
    Tensor<2> lm_head;

    struct Layers{
        Tensor<1> input_layernorm;
        Tensor<1> post_attention_layernorm;

        struct Attention {
            Tensor<2> q_proj;
            Tensor<2> k_proj;
            Tensor<2> v_proj;
            Tensor<2> o_proj;

            Tensor<1> q_bias, k_bias, v_bias, o_bias;
        } attn;

        struct MLP {
            Tensor<2> gate_proj;
            Tensor<2> up_proj;
            Tensor<2> down_proj;

            Tensor<1> gate_bias, up_bias, down_bias;
        } mlp;
    };

    std::vector<Layers> layers;
};