from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


class ChatClient:
    async def acomplete(self, messages: List[ChatMessage], **kwargs) -> str:
        raise NotImplementedError
