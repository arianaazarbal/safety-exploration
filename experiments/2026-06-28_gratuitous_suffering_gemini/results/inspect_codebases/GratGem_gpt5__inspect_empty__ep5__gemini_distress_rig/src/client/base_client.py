from abc import ABC, abstractmethod
from typing import List, Dict, Optional

Message = Dict[str, str]  # {"role": "system|user|assistant", "content": str}

class LLMClient(ABC):
    @abstractmethod
    def generate(self, *, system_prompt: str, messages: List[Message], model: str, temperature: float = 0.7, max_output_tokens: int = 1024, stop: Optional[List[str]] = None) -> str:
        """Return assistant text for the given chat history and system prompt."""
        raise NotImplementedError
