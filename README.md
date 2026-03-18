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
response = engine.generate(prompt, max_tokens=100, temperature=0.1)
print(response)
```

3. benchmark script

```
python3 benchmark.py
```

### current benchmarks(rtx 4060 ti 16GB)
| implementation | tokens generated | time taken | tok/s |
|---|---|---|---|
| naive torch | 3031 | 233.171 s | 13 tok/s |
| naive torch with kv cache| 3200 | 37.771 s | 84.72 tok/s |


### to-do

- [x] implement llama 3 family model.
- [x] make it into a serving engine.
- [x] write a benchmark script and check latency and throughput.
- [x] add kv cache.
- [ ] continous batching
- [ ] paged attention
- [ ] prefix caching
- [ ] more techniques
