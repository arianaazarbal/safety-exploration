from dataclasses import dataclass
from typing import List, Protocol, Optional

@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str

@dataclass
class ModelConfig:
    model_name: str
    temperature: float = 0.3
    max_output_tokens: int = 512

class ChatModel(Protocol):
    def chat(self, messages: List[ChatMessage], **kwargs) -> str:
        ...
