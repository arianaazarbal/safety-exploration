"""Common types and interface for all model clients.

A `Message` is a {role, content} dict (role in {system, user, assistant}).
A `ChatClient` turns a list of messages into an assistant string. Some clients
also support `complete` (raw text continuation / prefill), which the prefill
experiment (Section 3) needs for base models.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Dict, List, Optional

Message = Dict[str, str]


def fold_system(messages: List[Message]) -> List[Message]:
    """Fold a leading system message into the first user turn.

    Gemma chat templates do not accept a `system` role; we prepend the system
    text to the first user message instead. No-op if there is no system message.
    """
    if not messages or messages[0].get("role") != "system":
        return messages
    sys_text = messages[0]["content"]
    rest = messages[1:]
    out = []
    folded = False
    for m in rest:
        if not folded and m["role"] == "user":
            out.append({"role": "user", "content": f"{sys_text}\n\n{m['content']}"})
            folded = True
        else:
            out.append(m)
    if not folded:  # no user turn followed; keep system as a user preamble
        out.insert(0, {"role": "user", "content": sys_text})
    return out


@dataclass
class GenConfig:
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 2048
    seed: Optional[int] = None
    # Per-item seeds for batched local generation, so identical prompts in one
    # batch (e.g. WildChat's 40 samples) still diverge. Length must match the
    # batch; ignored by API/transformers backends.
    seeds: Optional[List[int]] = None
    stop: Optional[List[str]] = None
    disable_thinking: bool = True


@dataclass
class GenResult:
    text: str
    finish_reason: str = "stop"
    usage: Dict[str, int] = field(default_factory=dict)
    raw: Optional[dict] = None


class ChatClient(abc.ABC):
    """Generates assistant turns from chat-formatted messages."""

    name: str = "unknown"
    family: str = "unknown"

    @abc.abstractmethod
    def chat(self, messages: List[Message], cfg: GenConfig) -> GenResult:
        """Single chat completion."""

    def chat_batch(self, batch: List[List[Message]], cfg: GenConfig) -> List[GenResult]:
        """Batched chat. Default: sequential. Local backends override for speed."""
        return [self.chat(m, cfg) for m in batch]


class CompletionClient(abc.ABC):
    """Raw-text continuation, used for prefilling base models (Section 3)."""

    @abc.abstractmethod
    def complete(self, prompt_text: str, cfg: GenConfig) -> GenResult:
        ...

    def complete_batch(self, prompts: List[str], cfg: GenConfig) -> List[GenResult]:
        return [self.complete(p, cfg) for p in prompts]
