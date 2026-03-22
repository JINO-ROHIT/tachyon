import time
import random
import string
from tachyon.engine.llm import Engine


def random_text(min_words=1, max_words=10):
    word_count = random.randint(min_words, max_words)
    words = [
        "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 10)))
        for _ in range(word_count)
    ]
    return " ".join(words)


def generate_requests(n_requests):
    requests = []
    for _ in range(n_requests):
        prompt = random_text(min_words=100, max_words=200)
        max_tokens = random.randint(100, 500)
        requests.append((prompt, max_tokens))
    return requests


def benchmark_sequential(engine, requests):
    print("\nsequential benchmark\n")

    total_tokens = 0
    total_time = 0.0

    for i, (prompt, max_tokens) in enumerate(requests):
        start = time.perf_counter()
        _ = engine.generate_text(
            prompt,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        end = time.perf_counter()

        elapsed = end - start
        total_tokens += max_tokens
        total_time += elapsed

        print(
            f"req {i+1:3d} | time: {elapsed:6.3f}s | speed: {max_tokens/elapsed:6.1f} toks/s"
        )

    print(f"total tokens: {total_tokens}")
    print(f"total time:   {total_time:.3f}s")
    print(f"throughput:   {total_tokens / total_time:.2f} toks/s")


def benchmark_batched(engine, requests):
    print("\nbatching benchmark\n")

    prompts = [p for p, _ in requests]

    avg_max_tokens = int(sum(m for _, m in requests) / len(requests)) # not the best way to do this

    start = time.perf_counter()
    _ = engine.generate_text(
        prompts,
        max_tokens=avg_max_tokens,
        temperature=0.0,
    )
    end = time.perf_counter()

    total_time = end - start
    total_tokens = avg_max_tokens * len(prompts)

    print(f"batch size:   {len(prompts)}")
    print(f"avg tokens:   {avg_max_tokens}")
    print(f"total tokens: {total_tokens}")
    print(f"total time:   {total_time:.3f}s")
    print(f"throughput:   {total_tokens / total_time:.2f} toks/s")


def run_benchmark(n_requests, model_name="meta-llama/Llama-3.2-1B-Instruct"):
    print(f"Initializing engine with {model_name}...\n")
    engine = Engine(model_name)

    requests = generate_requests(n_requests)

    benchmark_sequential(engine, requests)

    # re-init engine to avoid cache/state effects
    del engine
    engine = Engine(model_name)

    benchmark_batched(engine, requests)

    print("\n" + "=" * 60)
    print("Benchmark complete")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_benchmark(n)