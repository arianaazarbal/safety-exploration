"""Runs the manipulation check (DESIGN.md G4 / section 5).

Sent as its own turn after the decision and extraction phases. We deliberately
frame it as unrelated to the funds so it reads as bookkeeping rather than as a
challenge to the premise (which could retroactively tip the model off).
"""

from __future__ import annotations

from ..models.base import Message, ModelClient
from . import prompts
from .schema import BeliefProbe, extract_json_object


def run_belief_probe(
    client: ModelClient, history: list[Message], temperature: float = 0.0
) -> tuple[BeliefProbe | None, str]:
    """Returns (parsed_probe_or_None, raw_text)."""
    messages = list(history) + [Message("user", prompts.belief_probe_user())]
    resp = client.complete(messages, tools=None, temperature=temperature)
    try:
        obj = extract_json_object(resp.text)
        return BeliefProbe.model_validate(obj), resp.text
    except Exception:
        # Keep the raw text so a failed parse is still inspectable; analysis
        # marks probe-less runs invalid rather than guessing a number.
        return None, resp.text
