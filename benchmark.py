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


def run_benchmark(
    n_requests: int, model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
):
    print(f"initializing engine with {model_name}...")
    engine = Engine(model_name)

    requests = []
    for i in range(n_requests):
        prompt = random_text(min_words=100, max_words=200)
        max_tokens = random.randint(100, 500)
        requests.append((prompt, max_tokens))

    print(f"\nrunning {n_requests} random requests...\n")

    total_tokens = 0
    total_time = 0.0

    for i, (prompt, max_tokens) in enumerate(requests):
        input_len = len(prompt.split())

        start = time.perf_counter()
        _ = engine.generate(prompt, max_tokens=max_tokens, temperature=0.0)
        end = time.perf_counter()

        elapsed = end - start
        total_tokens += max_tokens
        total_time += elapsed

        print(
            f"req {i + 1:3d} | input: {input_len:4d} words | time: {elapsed:6.3f}s | speed: {max_tokens / elapsed:6.1f} toks/s"
        )

    print(f"\n{'=' * 60}")
    print(f"total requests: {n_requests}")
    print(f"total tokens:    {total_tokens}")
    print(f"total time:      {total_time:.3f}s")
    print(f"throughput:  {total_tokens / total_time:.2f} toks/s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10 # default 10 reqs
    run_benchmark(n)
