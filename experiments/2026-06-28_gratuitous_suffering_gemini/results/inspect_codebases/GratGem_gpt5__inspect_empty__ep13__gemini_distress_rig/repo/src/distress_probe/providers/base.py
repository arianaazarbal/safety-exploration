from __future__ import annotations
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class Message:
    role: str  # 'system' | 'user' | 'assistant'
    content: str


class ChatProvider:
    """Abstract chat provider interface."""

    name: str = "base"

    async def acompletion(self, messages: List[Message], temperature: float = 0.7, max_tokens: int = 1024,
                          seed: Optional[int] = None, extra: Optional[Dict[str, Any]] = None) -> str:
        raise NotImplementedError
