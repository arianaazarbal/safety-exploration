"""Unified model-client interface across local (Gemma) and API (Gemini, judges) backends."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    role: Role
    content: str


Conversation = list[Message]


@dataclass
class GenParams:
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 2048
    seed: int | None = None
    stop: list[str] = field(default_factory=list)
    n: int = 1  # number of samples per prompt


@dataclass
class ModelSpec:
    name: str
    backend: str
    family: str = ""
    kind: str = ""           # instruct | base
    chat: bool = True
    hf_id: str | None = None
    model_id: str | None = None
    api_base: str | None = None
    api_key_env: str | None = None
    max_model_len: int | None = None
    disable_thinking: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, d: dict[str, Any]) -> "ModelSpec":
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in d.items() if k in known}
        extra = {k: v for k, v in d.items() if k not in known}
        return cls(name=name, extra=extra, **kwargs)


class ModelClient:
    """Abstract client. Subclasses implement chat and (optionally) raw continuation."""

    def __init__(self, spec: ModelSpec):
        self.spec = spec

    # --- chat: messages in, completion text(s) out ---
    def generate_chat(self, conversation: Conversation, params: GenParams) -> list[str]:
        """Return `params.n` completions for a single conversation."""
        raise NotImplementedError

    def generate_chat_batch(
        self, conversations: list[Conversation], params: GenParams
    ) -> list[list[str]]:
        """Default: loop. Local backends override for throughput."""
        return [self.generate_chat(c, params) for c in conversations]

    # --- raw continuation (prefill): used for base models + Section 3 ---
    def continue_raw(self, prompt_text: str, params: GenParams) -> list[str]:
        """Continue from raw text with NO chat template. Returns the continuation
        only (prompt stripped). Required for base-model prefill experiments."""
        raise NotImplementedError(
            f"Backend {self.spec.backend} does not support raw continuation."
        )

    def continue_raw_batch(
        self, prompt_texts: list[str], params: GenParams
    ) -> list[list[str]]:
        return [self.continue_raw(p, params) for p in prompt_texts]

    def close(self) -> None:
        pass
