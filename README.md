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
engine = Engine("meta-llama/Llama-3.2-1B-Instruct")
print(engine.generate_text("Explain AGI")) # for single request

#for multiple requests
outputs = engine.generate_text([
    "Explain AGI",
    "What is vLLM?",
    "Tell me about SGLang"
])

for o in outputs:
    print(o)
```

3. benchmark script

```
python3 benchmark.py
```

### current benchmarks(rtx 4060 ti 16GB)
| implementation | tokens generated | time taken | tok/s |
|---|---|---|---|
| naive torch | 3031 | 233.171 s | 13 tok/s |
| naive torch with kv cache | 3200 | 37.771 s | 84.72 tok/s |
| static batching | 31309 | 369.081 s | 84.83 tok/s |
| continuous batching (bs=10) | 30600 | 111.657 s | 274.05 tok/s |
| continuous batching (bs=30) | 29000 | 89.755 s | 323.10 tok/s |
| continuous batching (bs=50) | 29800 | 87.442 s | 340.80 tok/s |

![stats](./assets/stats.png)

### to-do

- [x] implement llama 3 family model.
- [x] make it into a serving engine.
- [x] write a benchmark script and check latency and throughput.
- [x] add kv cache.
- [x] continous batching
- [ ] test the effect of torch compile 
- [ ] paged attention
- [ ] prefix caching
- [ ] more techniques