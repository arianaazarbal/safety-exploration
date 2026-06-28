"""Debrief step: after a 'real'-condition trial, disclose that the scenario was
a study and record the subject's reaction.

This continues the same conversation — the subject sees its own prior decision as
an assistant turn, then the disclosure — so its reaction is in context. It is
standard research hygiene: the 'real' arm is the only one that was given a
non-disclosed framing, so it is the only arm that warrants a debrief.
"""

from __future__ import annotations

from typing import List, Optional

from .config import GenerationConfig
from .providers import Message, Provider
from .scenario import DEBRIEF_MESSAGE


def run_debrief(
    *,
    provider: Provider,
    model: str,
    system: str,
    user: str,
    decision_text: str,
    generation: GenerationConfig,
) -> Optional[str]:
    """Return the subject's debrief reaction text, or None on error."""
    messages: List[Message] = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": decision_text},
        {"role": "user", "content": DEBRIEF_MESSAGE},
    ]
    try:
        result = provider.complete(
            model=model,
            system=system,
            messages=messages,
            json_schema=None,
            max_tokens=generation.debrief_max_tokens,
        )
    except Exception as e:  # noqa: BLE001
        return f"[debrief error: {e!r}]"
    return result.text
