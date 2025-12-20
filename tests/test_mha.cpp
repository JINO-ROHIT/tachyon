#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "include/tensor.h"
#include "include/cpu/ops.h"

int main(){
    const char* model_path = "models/meta-llama/Llama-2-7b-hf/model.tach";
    int fd = open(model_path, O_RDONLY);
    if(fd < 0){
        return false;
    }

    struct stat sb;
    if (fstat(fd, &sb) < 0) {
        close(fd);
        return false;
    }

    size_t mmap_size = sb.st_size;
    char* mmap_data = (char*)mmap(NULL, mmap_size, PROT_READ, MAP_PRIVATE, fd, 0);

    //for layer 0
    Tensor<2> key_seq(mmap_data + 0x1FC05800, 4096, 4096, f16); // hard map to test correctness
    Tensor<2> query_weight(mmap_data + 0x23C05800, 4096, 4096, f16); // hard map to test correctness
    Tensor<2> value_seq(mmap_data + 0x25C05800, 4096, 4096, f16); // hard map to test correctness
    Tensor<2> out_tensor(mmap_data + 0x21C05800, 4096, 4096, f16); // hard map to test correctness


    Tensor<1> input(4096);
    float* input_data = (float*)malloc(4096 * sizeof(float));
    for(int i = 0; i < 4096; i++) {
        input_data[i] = ((float*)key_seq.data)[i];
    }
    input.data = input_data;

    Tensor<1> out(4096);
    float* out_data = (float*)malloc(4096 * sizeof(float));
    out.data = out_data;

    int num_heads = 32;  
    int head_dim = 128;  
    int seq_len = 1;     

    simple_mha(input, query_weight, key_seq, value_seq, out, num_heads, head_dim);

    for(int i = 0; i < 10; i++) {
        std::cout << out[i] << std::endl;
    }

    free(input_data);
    free(out_data);
        
    return 0;
}