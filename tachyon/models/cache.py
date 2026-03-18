'''
class for the kv cache for the attention layers across the models
'''

class Cache:
    def __init__(self, n_layers):
        self.cache = [None] * n_layers # a better data structure?
    
    def get(self, layer_idx = 0):
        return self.cache[layer_idx]
    
    def update(self, layer_idx, value):
        self.cache[layer_idx] = value

    def get_all(self):
        return self.cache

    def reset(self):
        for i in range(len(self.cache)):
            self.cache[i] = None