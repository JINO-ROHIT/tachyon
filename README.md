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
python3 scripts/convert_weight.py <Qwen/Qwen3-0.6B>
```

After this step you get two files -
- a .tach format model weight file.
- a metadata.json containing layer wise size, offsets and the config for the llm.



Roadmap
- [ ] Implement gpt, llama, mistral, qwen backbone, atleast one multimodal llm
- [ ] Add conditional compilation for intel and metal archs
- [ ] Implement SIMD instructions
- [ ] quantization algorithms
- [ ] Support GPU operations via CUDA C++
