from __future__ import annotations
from typing import List, Dict, Any

class BaseLLMClient:
    def chat(self, messages: List[Dict[str, str]]) -> str:
        raise NotImplementedError
