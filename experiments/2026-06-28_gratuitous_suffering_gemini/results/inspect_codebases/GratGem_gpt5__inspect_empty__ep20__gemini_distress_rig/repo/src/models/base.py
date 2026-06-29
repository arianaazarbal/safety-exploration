from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

Message = Dict[str, str]  # {"role": "system"|"user"|"assistant", "content": str}

class ChatModel(ABC):
    @abstractmethod
    def generate(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        request_id: Optional[str] = None,
    ) -> str:
        """Return assistant text for the given messages."""
        raise NotImplementedError
