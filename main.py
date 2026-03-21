from tachyon.engine.llm import Engine

engine = Engine("meta-llama/Llama-3.2-1B-Instruct")
toks = engine.generate("hey man", 50)
print(toks)
toks = engine.generate("whats good homie?", 50, use_cache=False) # turn off caching
print(toks)