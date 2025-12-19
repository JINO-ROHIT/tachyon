## Tachyon

```
 ████████╗ █████╗  ██████╗██╗  ██╗██╗   ██╗ ██████╗ ███╗   ██╗
 ╚══██╔══╝██╔══██╗██╔════╝██║  ██║╚██╗ ██╔╝██╔═══██╗████╗  ██║
    ██║   ███████║██║     ███████║ ╚████╔╝ ██║   ██║██╔██╗ ██║
    ██║   ██╔══██║██║     ██╔══██║  ╚██╔╝  ██║   ██║██║╚██╗██║
    ██║   ██║  ██║╚██████╗██║  ██║   ██║   ╚██████╔╝██║ ╚████║
    ╚═╝   ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚═╝  ╚═══╝
           >>> Faster Than Light >>>
```

Tachyon is a C++ inference engine for LLMs that run on CPU. This library is a collection of all the modern LLM architectures written from scratch and designed to run on consumer hardware.

### Setup

1. install the virtual environment.

```
uv sync
```

2. first convert the weights from safetensors to tach format.

```
python3 scripts/convert_weights.py <Qwen/Qwen3-0.6B>
```

After this step you get two files -
- a model.tach weight file. this file is 32 bit memory aligned for better memory accesses. 
- a metadata.json containing layer wise size, offsets and the config for the llm.

```json
{
  "model.embed_tokens.weight": {
    "offset": "0x00000000",
    "size": 525336576,
    "padded_size": 525336576,
    "shape": [
      128256,
      2048
    ],
    "dtype": "BF16",
    "transposed": false
  },
  "model.layers.0.input_layernorm.weight": {
    "offset": "0x1F500000",
    "size": 4096,
    "padded_size": 4096,
    "shape": [
      2048
    ],
    "dtype": "BF16",
    "transposed": false
  },
}
  ```

Roadmap
- [ ] Implement gpt, llama, mistral, qwen backbone, atleast one multimodal llm
- [ ] Add conditional compilation for intel and metal archs
- [ ] Implement SIMD instructions
- [ ] quantization algorithms
- [ ] Support GPU operations via CUDA C++