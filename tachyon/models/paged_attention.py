from typing import Optional, List
from collections import deque

import torch

class Page:
    def __init__(self, page_size, num_head, head_dim):
        self.page_size = page_size
        self.ref_count = 0
        self.kv = torch.zeros(2, self.page_size, num_head, head_dim)


class PageTable:
    def __init__(self, map):
        self.map = map
    
    def map_page(self, logical_id: int, physical_page: Page):
        """map the logical page id to physical page"""
        self.map[logical_id] = physical_page
    
    def get_page(self, logical_id: int) -> Optional[Page]:
        """given logical id, fetch the physical page"""
        return self.map.get(logical_id)

class Request:
    def __init__(self, id: int = 0, prompt: str = "", max_tokens: int = 100, temperature: float = 0.1, page_size: int = 16, 
                 prompt_tokens: List[int] = [], tokens: List[int] = [], is_completed: bool = False, is_prefill: bool = True, use_cache: bool = True,
                 response: str = None):
        self.id = id
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.prompt_tokens = prompt_tokens
        self.tokens = tokens

        self.page_size = page_size
        self.page_table = PageTable() # each request has its own page table
        self.logical_pages = []

        self.is_completed = is_completed
        self.is_prefill = is_prefill
        self.use_cache = use_cache

        self.response = response
    
    def get_num_tokens(self) -> int:
        return len(self.tokens)
    
    def get_num_pages_needed(self) -> int:
        return (len(self.tokens) + self.page_size -1 ) // self.page_size
    
    def append_token(self, token_id: int):
        self.tokens.append(token_id)

class BlockManager:
    def __init__(self, num_pages, page_size):
        self.pages = [Page(page_size) for _ in range(num_pages)]
        self.free = deque(self.pages)
        self.allocated = set()
    
    def _allocate(self) -> Optional[Page]:
        if not self.free:
            print("all the pages are occupied")
            return
        
        page_to_be_allocated = self.free.popleft()
        page_to_be_allocated.ref_count +=1
        self.allocated(page_to_be_allocated)
        return page_to_be_allocated
    
    def _deallocate(self, page: Page):
        if page not in self.allocated:
            print("page is free")
            return
        
        page.ref_count -= 1

        if page.ref_count == 0:
            self.allocated.pop(page)
            self.free.append(page)
            page.kv.zero_()
    
    def allocate_request(self, request: Request) -> bool:
        num_pages_needed = request.get_num_pages_needed()
        current_pages = len(request.logical_pages)
        pages_to_allocate = num_pages_needed - current_pages

        if len(self.free) < pages_to_allocate:
            print(f"cannot allocate for request {request.id} since no free page available")
            return False
        
        for i in range(pages_to_allocate):
            page = self._allocate()
            if page is None:
                return False
            
            logical_id = current_pages + 1
            request.logical_pages.append(logical_id)
            request.page_table.map_page(logical_id, page)
        
        return True
    
    def free_request(self, request: Request):
        for logical_id in request.logical_pages:
            page = request.page_table.get_page(logical_id)
            if page:
                self._deallocate(page)
        
        request.logical_pages.clear()
        request.page_table.map.clear()
    
    def get_num_free_pages(self):
        return len(self.free)
