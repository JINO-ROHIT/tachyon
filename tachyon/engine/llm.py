import torch
from transformers import AutoTokenizer

from tachyon.models import Llama3Model
from tachyon.utils import load_weights
class Engine:
    def __init__(self, model_name: str):
        self.model = Llama3Model() # for now we only have llama
        load_weights(self.model, model_name.split("/")[-1]) # loads in place
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    # def load(self, model_path: str):
    #     self.model = load_weights(self.model, model_path)

    def generate(self, prompt: str, max_tokens: int=100, temperature: float=0.0):

        ## prefill phase
        tokens = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0) # for batch dim for single prompt

        ## decode phase
        for _ in range(max_tokens):
            with torch.no_grad():
                logits = self.model(tokens)
            logits = logits[:, -1, :]

            if temperature > 0.0:
                logits = logits / temperature
                logits = logits - logits.max(dim=-1, keepdim=True).values
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            tokens = torch.cat((tokens, next_token), dim=1)
        
        flat = tokens.squeeze(0)  # remove batch dimension
        return self.tokenizer.decode(flat.tolist())
        #return tokens

if __name__ == '__main__':
    engine = Engine("meta-llama/Llama-3.2-1B-Instruct")
    toks = engine.generate("hey man", 50)
    print(toks)