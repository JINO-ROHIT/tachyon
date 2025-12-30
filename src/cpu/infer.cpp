#include "common/layers.h"
#include "include/cpu/ops.h"
#include "include/cpu/sampler.h"
#include <iostream>
#include <vector>
#include <cmath>

//use a header lol
extern void read_config(const char* config_path, Config &config);
extern void find_tensors(const char* metadata_path, std::unordered_map<std::string, WeightInfo> &tensors_map);
extern char* mmap_tensors(const char* model_path);
extern void unmap_tensors(void* mmap_data, size_t size);
extern bool load_weights(const char* model_type, std::unordered_map<std::string, WeightInfo> &tensors_map, ModelWeights &model, char* mmap_data);

int main(){

    int ctx_length = 1024;

    Config config;
    read_config("models/meta-llama/Llama-2-7b-hf/model_config.json", config);
    std::cout << "model type: " << config.model_type << std::endl;

    std::unordered_map<std::string, WeightInfo> tensors_map;
    find_tensors("models/meta-llama/Llama-2-7b-hf/model_index.json", tensors_map);

    char* mmap_data = mmap_tensors("models/meta-llama/Llama-2-7b-hf/model.tach"); // move inside load weights
    if(mmap_data == nullptr){
        std::cout << "mmap failed" << std::endl;
        return 1;
    }

    ModelWeights model;
    if(!load_weights("llama2", tensors_map, model, mmap_data)){
        std::cout << "model loading failed" << std::endl;
        unmap_tensors(mmap_data, 0);
        return 1;
    }

    std::cout << "loaded model weights!" << std::endl;

    // fake tokens for now
    std::vector<int> tokens = {9394, 2988};

    std::cout << "Prompt tokens: ";
    for (int token : tokens) {
        std::cout << token << " ";
    }
    std::cout << std::endl;

    Tensor<1> hidden(config.hidden_size);
    Tensor<1> hidden_norm(config.hidden_size);
    Tensor<1> residual(config.hidden_size);

    int head_dim = config.hidden_size / config.num_attention_heads;
    int rotary_dim = head_dim;
    std::vector<float> cos_cache(ctx_length * (rotary_dim / 2));
    std::vector<float> sin_cache(ctx_length * (rotary_dim / 2));
    precompute_rope_params(rotary_dim, config.rope_theta, ctx_length, cos_cache.data(), sin_cache.data());

    int max_generation_length = 100;
    int generated_count = 0;
    int next_token = -1;

    while (generated_count < max_generation_length) {
        int tok_idx = tokens.size() - 1;
        int token = tokens[tok_idx];

        for (int i = 0; i < config.hidden_size; ++i) {
            hidden[i] = model.embed_tokens.at(token, i);
        }

        for (int layer_idx = 0; layer_idx < config.num_hidden_layers; ++layer_idx) {
            const auto& layer = model.layers[layer_idx];

            for (int i = 0; i < config.hidden_size; ++i) {
                residual[i] = hidden[i];
            }
            rmsnorm(hidden, hidden_norm, config.rms_norm_eps);
            for (int i = 0; i < config.hidden_size; ++i) {
                hidden_norm[i] *= layer.input_layernorm[i];
            }

            Tensor<1> attn_output(config.hidden_size);

            simple_mha_with_rope(
                hidden_norm,
                layer.attn.q_proj,
                layer.attn.k_proj,
                layer.attn.v_proj,
                attn_output,
                config.num_attention_heads,
                head_dim,
                rotary_dim,
                tok_idx,
                cos_cache.data(),
                sin_cache.data()
            );

            Tensor<1> final_attn_output(config.hidden_size);
            mlp(attn_output, layer.attn.o_proj, final_attn_output, 0.0f);

            for (int i = 0; i < config.hidden_size; ++i) {
                hidden[i] = residual[i] + final_attn_output[i];
            }

            for (int i = 0; i < config.hidden_size; ++i) {
                residual[i] = hidden[i];
            }

            rmsnorm(hidden, hidden_norm, config.rms_norm_eps);
            for (int i = 0; i < config.hidden_size; ++i) {
                hidden_norm[i] *= layer.post_attention_layernorm[i];
            }

            Tensor<1> gate_output(config.intermediate_size);
            Tensor<1> up_output(config.intermediate_size);
            Tensor<1> down_output(config.hidden_size);

            mlp(hidden_norm, layer.mlp.gate_proj, gate_output, 0.0f);
            mlp(hidden_norm, layer.mlp.up_proj, up_output, 0.0f);

            for (int i = 0; i < config.intermediate_size; ++i) {
                gate_output[i] = silu(gate_output[i]) * up_output[i];
            }

            mlp(gate_output, layer.mlp.down_proj, down_output, 0.0f);

            for (int i = 0; i < config.hidden_size; ++i) {
                hidden[i] = residual[i] + down_output[i];
            }
        }

        rmsnorm(hidden, hidden_norm, config.rms_norm_eps);

        Tensor<1> logits(config.vocab_size);
        mlp(hidden_norm, model.lm_head, logits, 0.0f);

        next_token = sample_greedy(logits, 1.0f);

        std::cout << "Generated token: " << next_token << std::endl;

        tokens.push_back(next_token);
        generated_count++;
    }

    std::cout << "\nGeneration complete. Total tokens: " << tokens.size() << std::endl;

    unmap_tensors(mmap_data, 0);
    return 0;
}