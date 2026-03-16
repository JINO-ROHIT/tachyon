# do this need this or put inside the engine? what else will we need here?


from transformers import AutoTokenizer

class Tokenizer:
    def __init__(self, model: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    
    def tokenize(self, prompt: str | list[str]):
        return self.tokenizer.encode(prompt)


if __name__ == '__main__':
    prompt = ["yo mandem", "hiya"]

    tokenizer = Tokenizer(model="meta-llama/Llama-3.2-1B-Instruct")
    print(tokenizer.tokenize(prompt))

    