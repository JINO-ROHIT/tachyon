#pragma once

#include <assert.h>
#include <cstdio>
#include <iostream>

enum Dtype {
    f32,
    f16,
    i8,
    i4
};

inline size_t dtype_size(Dtype dtype){
    switch(dtype){
        case f32: return 4;
        case f16: return 2;
        case i8: return 1;
        case i4: return 1;
        default: return 4;
    }
}

template<int N>
class Tensor{

    private:
        void _alloc(size_t num_elements){
            size_t bytes = num_elements * dtype_size(dtype);

            alloc = new char[bytes + 1];
            data = (void*)(((uintptr_t)alloc + 31) & ~31);
        }

    public:
        int shape[N];
        Dtype dtype;
        void* data;
        void* alloc;

        Tensor() : data(nullptr), alloc(nullptr), dtype(f32) {};

        ~Tensor(){
            if (alloc){
                delete[] (char*) alloc;
            }
        }
        Tensor(void* _data, int i, Dtype _dtype){
            assert(N == 1);
            shape[0] = i;
            data = _data;
            dtype = _dtype;
            alloc = nullptr;
        }

        Tensor(void* _data, int i, int j, Dtype _dtype){
            assert(N == 2);
            shape[0] = i;
            shape[1] = j;
            data = _data;
            dtype = _dtype;
            alloc = nullptr;
        }

        //allocate new tensor

        Tensor(int i, Dtype _dtype = f32){
            assert(N == 1);
            shape[0] = i;
            dtype = _dtype;
            _alloc(i);
        }

        Tensor(int i, int j, Dtype _dtype = f32) {
            assert(N == 2);
            shape[0] = i;
            shape[1] = j;
            dtype = _dtype;
            _alloc(i*j);
        }

        // [] indexing operator 
        // TO-DO look at how to extend to the other datatypes
        float &operator[] (int i) const {
            assert(N == 1);

            if(i >= shape[0]){
                std::cout << "tensor out of bound" << std::endl;
            }

            return ((float*)data)[i];
        }

        // for multi dims (or else use c++23)
        float &at(int i, int j) const {
            assert(N == 2);

            if(i >= shape[0]){
                std::cout << "tensor out of bound" << std::endl;
            }

            if(j >= shape[1]){
                std::cout << "tensor out of bound" << std::endl;
            }


            return ((float*)data)[i * shape[1] + j];
        }
};