'''
class for the kv cache for the attention layers across the models
'''
from collections import OrderedDict

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


#TO-DO - implement a trie based data structure, current LRU is expensive
class PrefixCache:
    """shared cache across requests, stores kv states for token prefixes so we can skip redundant prefill"""

    def __init__(self, max_entries=128):
        self.entries = OrderedDict()  # tuple(tokens) -> [(k, v), ...] per layer
        self.max_entries = max_entries

    def lookup(self, tokens):
        """find the longest cached prefix for the given tokens"""
        best_len = 0
        best_kv = None
        best_key = None
        for cached_tokens in self.entries:
            n = len(cached_tokens)
            if n > best_len and n <= len(tokens) and tuple(tokens[:n]) == cached_tokens:
                best_len = n
                best_kv = self.entries[cached_tokens]
                best_key = cached_tokens
        if best_key is not None:
            self.entries.move_to_end(best_key)
        return best_len, best_kv

    def store(self, tokens, kv_states):
        key = tuple(tokens)
        if key in self.entries:
            self.entries.move_to_end(key)
            return
        if len(self.entries) >= self.max_entries:
            self.entries.popitem(last=False)
        self.entries[key] = [(k.clone(), v.clone()) for k, v in kv_states]

    def reset(self):
        self.entries.clear()