"""Provider-agnostic chat message representation.

We keep a single tiny `Message` type and convert to each provider's native
format inside the provider module. Roles follow the OpenAI/Anthropic
convention ("system" / "user" / "assistant"); Gemini's "model" role is mapped
internally by the Google provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}
