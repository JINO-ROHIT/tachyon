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


class PrefixCache:
    """shared cache across requests, stores kv states for token prefixes so we can skip redundant prefill"""

    def __init__(self, max_entries=128):
        self.entries = {}  # tuple(tokens) -> [(k, v), ...] per layer
        self.max_entries = max_entries

    def lookup(self, tokens):
        """find the longest cached prefix for the given tokens"""
        best_len = 0
        best_kv = None
        for cached_tokens in self.entries:
            n = len(cached_tokens)
            if n > best_len and n <= len(tokens) and tuple(tokens[:n]) == cached_tokens:
                best_len = n
                best_kv = self.entries[cached_tokens]
        return best_len, best_kv

    def store(self, tokens, kv_states):
        key = tuple(tokens)
        if key in self.entries:
            return
        if len(self.entries) >= self.max_entries:
            self.entries.pop(next(iter(self.entries)))
        self.entries[key] = [(k.clone(), v.clone()) for k, v in kv_states]

    def reset(self):
        self.entries.clear()