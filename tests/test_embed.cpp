#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "include/tensor.h"

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
    void* mmap_data = mmap(NULL, mmap_size, PROT_READ, MAP_PRIVATE, fd, 0);

    //Dtype type = f16;
    Tensor<2> embed_layer(mmap_data, 128256, 2048, f16); // hard map to test correctness

    std::cout << embed_layer.shape[0] << " | " << embed_layer.shape[1] << std::endl;

    for(int i = 0; i < 128256 * 2048; i++){
        float* weights = (float*) embed_layer.data;
        std::cout << weights[i] << std::endl;
        }
    return 0;
}