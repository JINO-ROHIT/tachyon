#include <iostream>
#include <assert.h>

#include "include/tensor.h"

int main(){
    int arr[3] = {1, 2, 3};
    int* ptr = arr;

    Tensor<1> quant_weight(ptr, 3, i8);
    assert(((int*)(quant_weight.data))[0] == 1);

    Tensor<1> weight(10); // by default takes f32
    assert(weight.dtype == 0);

}