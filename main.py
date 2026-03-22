from tachyon.engine.llm import Engine

engine = Engine("meta-llama/Llama-3.2-1B-Instruct")
print(engine.generate_text("Explain AGI"))

outputs = engine.generate_text([
    "Explain AGI",
    "What is vLLM?",
    "Tell me about SGLang"
])

for o in outputs:
    print(o)