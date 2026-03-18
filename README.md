## tachyon

a LLM inference engine to run on consumer hardware.

## Usage

1. first download the weights for llama 1B.

```
hf_hub_download(
        repo_id=f"meta-llama/Llama-3.2-1B-Instruct",
        filename="model.safetensors",
        local_dir=f"Llama-3.2-1B-Instruct"
    )
```

### to-do

- [x] implement llama 3 family model.
- [ ] make it into a serving engine.
- [ ] write a benchmark script and check latency and throughput.
- [ ] add kv cache.
- [ ] keep scaling with more inferencing techniques.