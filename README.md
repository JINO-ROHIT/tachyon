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

2. invoke the engine.

```
from tachyon.engine.llm import Engine
tokens = engine.generate(prompt, max_tokens=max_tokens, temperature=0.0)
print(tokens)
```

3. benchmark script

```
python3 benchmark.py
```

### current benchmarks
| implementation | tokens generated | time taken | tok/s |
|---|---|---|---|
| naive torch | 3031 | 233.171 s | 13 |


### to-do

- [x] implement llama 3 family model.
- [x] make it into a serving engine.
- [x] write a benchmark script and check latency and throughput.
- [ ] add kv cache.
- [ ] continous batching
- [ ] paged attention
- [ ] prefix caching
- [ ] more techniques
