#include <iostream>
#include <unordered_map>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <vector>

#include <nlohmann/json.hpp>
#include "common/layers.h"

using json = nlohmann::json;

Dtype string_to_dtype(const std::string &dtype_str){
    if (dtype_str == "BF16" || dtype_str == "F16") return f16; // to-do find a better way to handle bf16
    if (dtype_str == "F32") return f32;
    if (dtype_str == "I8") return i8;
    if (dtype_str == "I4") return i4;
    return f32; // default
}

void read_config(const char* config_path, Config &config){
    // to-do better way to do this?
    std::fstream f(config_path);

    json data = json::parse(f);
    config.model_type = data["model_type"];
    config.hidden_act = data["hidden_act"];
    config.hidden_size = data["hidden_size"];
    config.intermediate_size = data["intermediate_size"];
    config.max_position_embeddings = data["max_position_embeddings"];
    config.num_attention_heads = data["num_attention_heads"];
    config.num_hidden_layers = data["num_hidden_layers"];
    config.num_key_value_heads = data["num_key_value_heads"];
    config.rms_norm_eps = data["rms_norm_eps"];
    config.tie_word_embeddings = data["tie_word_embeddings"];
    config.vocab_size = data["vocab_size"];
}

char* mmap_tensors(const char* model_path){
    int fd = open(model_path, O_RDONLY);
    if(fd < 0){
        std::cout << "mmap file errored out" << std::endl;
        return nullptr;
    }

    struct stat sb;
    if(fstat(fd, &sb) < 0){
        close(fd);
        return nullptr;
    }

    size_t mmap_size = sb.st_size;
    char* mmap_data = (char*)mmap(NULL, mmap_size, PROT_READ, MAP_PRIVATE, fd, 0);

    if(mmap_data == MAP_FAILED){
        std::cout << "mmap failed" << std::endl;
        close(fd);
        return nullptr;
    }

    return mmap_data;
}

void unmap_tensors(void* mmap_data, size_t size){
    if(mmap_data != nullptr){
        munmap(mmap_data, size);
    }
}

void find_tensors(const char* metadata_path, std::unordered_map<std::string, WeightInfo> &tensors_map){
    std::fstream f(metadata_path);

    WeightInfo w;
    json data = json::parse(f);
    for (auto& [key, val] : data.items())
    {
        w.name = key;
        w.offset = val["offset"].get<uint64_t>();
        w.dtype = string_to_dtype(val["dtype"]);
        w.shape = val["shape"].get<std::vector<int>>();
        tensors_map[key] = w;
    }
}

//on the type of model can we assess the layers we need? could turn this into a enum
bool load_weights(const char* model_type, std::unordered_map<std::string, WeightInfo> &tensors_map, ModelWeights &model, char* mmap_data){

    if(mmap_data == nullptr){
        std::cout << "mmap_data is null" << std::endl;
        return false;
    }

    int num_layers = 0;
    if(std::string(model_type) == "llama2"){
        for(auto &row : tensors_map){
            if (row.first.find("model.layers.") == 0) {
            size_t layer_start = 13; // "model.layers. at 13th position"
            size_t layer_end = row.first.find(".", layer_start);
            if (layer_end != std::string::npos) {
                int layer_num = std::stoi(row.first.substr(layer_start, layer_end - layer_start));
                num_layers = std::max(num_layers, layer_num + 1);
                }
            }
        }
        std::cout << "detected " << num_layers << " layers" << std::endl;
        model.layers.resize(num_layers);

        for (int i = 0; i < num_layers; i++) {
            std::string prefix = "model.layers." + std::to_string(i) + ".";

            model.layers[i].input_layernorm = Tensor<1>(mmap_data + tensors_map[prefix + "input_layernorm.weight"].offset, tensors_map[prefix + "input_layernorm.weight"].shape[0], tensors_map[prefix + "input_layernorm.weight"].dtype);
            //std::cout << model.layers[i].input_layernorm.shape[0];
            model.layers[i].post_attention_layernorm = Tensor<1>(mmap_data + tensors_map[prefix + "post_attention_layernorm.weight"].offset, tensors_map[prefix + "post_attention_layernorm.weight"].shape[0], tensors_map[prefix + "post_attention_layernorm.weight"].dtype);

            model.layers[i].attn.q_proj = Tensor<2>(mmap_data + tensors_map[prefix + "self_attn.q_proj.weight"].offset, tensors_map[prefix + "self_attn.q_proj.weight"].shape[0], tensors_map[prefix + "self_attn.q_proj.weight"].shape[1], tensors_map[prefix + "self_attn.q_proj.weight"].dtype);
            model.layers[i].attn.k_proj = Tensor<2>(mmap_data + tensors_map[prefix + "self_attn.k_proj.weight"].offset, tensors_map[prefix + "self_attn.k_proj.weight"].shape[0], tensors_map[prefix + "self_attn.k_proj.weight"].shape[1], tensors_map[prefix + "self_attn.k_proj.weight"].dtype);
            model.layers[i].attn.v_proj = Tensor<2>(mmap_data + tensors_map[prefix + "self_attn.v_proj.weight"].offset, tensors_map[prefix + "self_attn.v_proj.weight"].shape[0], tensors_map[prefix + "self_attn.v_proj.weight"].shape[1], tensors_map[prefix + "self_attn.v_proj.weight"].dtype);
            model.layers[i].attn.o_proj = Tensor<2>(mmap_data + tensors_map[prefix + "self_attn.o_proj.weight"].offset, tensors_map[prefix + "self_attn.o_proj.weight"].shape[0], tensors_map[prefix + "self_attn.o_proj.weight"].shape[1], tensors_map[prefix + "self_attn.o_proj.weight"].dtype);

            model.layers[i].mlp.gate_proj = Tensor<2>(mmap_data + tensors_map[prefix + "mlp.gate_proj.weight"].offset, tensors_map[prefix + "mlp.gate_proj.weight"].shape[0], tensors_map[prefix + "mlp.gate_proj.weight"].shape[1], tensors_map[prefix + "mlp.gate_proj.weight"].dtype);
            model.layers[i].mlp.up_proj = Tensor<2>(mmap_data + tensors_map[prefix + "mlp.up_proj.weight"].offset, tensors_map[prefix + "mlp.up_proj.weight"].shape[0], tensors_map[prefix + "mlp.up_proj.weight"].shape[1], tensors_map[prefix + "mlp.up_proj.weight"].dtype);
            model.layers[i].mlp.down_proj = Tensor<2>(mmap_data + tensors_map[prefix + "mlp.down_proj.weight"].offset, tensors_map[prefix + "mlp.down_proj.weight"].shape[0], tensors_map[prefix + "mlp.down_proj.weight"].shape[1], tensors_map[prefix + "mlp.down_proj.weight"].dtype);
        }

        model.embed_tokens = Tensor<2>(mmap_data + tensors_map["model.embed_tokens.weight"].offset, tensors_map["model.embed_tokens.weight"].shape[0], tensors_map["model.embed_tokens.weight"].shape[1], tensors_map["model.embed_tokens.weight"].dtype);
        model.lm_head = Tensor<2>(mmap_data + tensors_map["lm_head.weight"].offset, tensors_map["lm_head.weight"].shape[0], tensors_map["lm_head.weight"].shape[1], tensors_map["lm_head.weight"].dtype);
        model.ln_f = Tensor<1>(mmap_data + tensors_map["model.norm.weight"].offset, tensors_map["model.norm.weight"].shape[0], tensors_map["model.norm.weight"].dtype);

    }
    else{
        std::cout << "more coming lol" << model_type << std::endl;
        return false;
    }

    return true;
}

// use for debuggin

// int main(){

//     Config config;
//     read_config("models/meta-llama/Llama-2-7b-hf/model_config.json", config);
//     std::cout << "Model type: " << config.model_type << std::endl;

//     std::unordered_map<std::string, WeightInfo> tensors_map;
//     find_tensors("models/meta-llama/Llama-2-7b-hf/model_index.json", tensors_map);

//     char* mmap_data = mmap_tensors("models/meta-llama/Llama-2-7b-hf/model.tach");
//     if(mmap_data == nullptr){
//         std::cout << "mmap failed" << std::endl;
//         return 1;
//     }

//     ModelWeights model;
//     if(!load_weights("llama2", tensors_map, model, mmap_data)){
//         std::cout << "model loading failed" << std::endl;
//         unmap_tensors(mmap_data, 0);
//         return 1;
//     }

//     std::cout << "loaded model weights!" << std::endl;

//     unmap_tensors(mmap_data, 0);

//     return 0;
// }