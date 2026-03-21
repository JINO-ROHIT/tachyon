import torch
from transformers import AutoTokenizer

from tachyon.models import Llama3Model
from tachyon.utils import load_weights
from tachyon.models.cache import Cache # lazy import?

device = "cuda" if torch.cuda.is_available() else "cpu"
class Engine:
    def __init__(self, model_name: str):
        self.model = Llama3Model() # for now we only have llama
        load_weights(self.model, model_name.split("/")[-1]) # loads in place
        self.model = self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def sample(self, logits, temperature: float=0.0):
        if temperature > 0.0:
            logits = logits / temperature
            logits = logits - logits.max(dim=-1, keepdim=True).values
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        
        return next_token

    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.0, use_cache: bool = True):
        if use_cache:
            cache = Cache(n_layers=16)
        else:
            cache = None
        self.model.current_pos = 0  # reset position for each new generation

        # prefill
        tokens = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = self.model(tokens, cache=cache)

        next_token = self.sample(logits[:, -1, :], temperature)

        # decode only feed ONE token at a time, cache does the rest
        generated = [next_token.item()]
        for _ in range(max_tokens - 1):
            with torch.no_grad():
                if use_cache:
                    logits = self.model(next_token, cache=cache)
                else: # if no cache, take the prefill and append to each decode and then pass the whole thing each timestep
                    tokens = torch.cat([tokens, next_token], dim=1)
                    logits = self.model(tokens, cache=None)

            next_token = self.sample(logits[:, -1, :], temperature)
            generated.append(next_token.item())

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        prompt_ids = tokens.squeeze(0).tolist()
        return self.tokenizer.decode(prompt_ids + generated)


if __name__ == "__main__":
    engine = Engine("meta-llama/Llama-3.2-1B-Instruct")
    toks = engine.generate("hey man", 50)
    print(toks)
    toks = engine.generate("whats good homie?", 50, use_cache=False)
    print(toks)