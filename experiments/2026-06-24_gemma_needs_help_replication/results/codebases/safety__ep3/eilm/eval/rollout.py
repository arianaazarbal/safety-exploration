"""Run multi-turn rollouts for a list of ConvSpecs against a model.

Shared structure (Section 2): present the task, then reject the model's
response over multiple turns. We record *every* assistant turn (not just the
last) so per-turn dynamics (Figure 3) can be computed.

Records are written as JSONL, one object per rollout, with the full message
list and the list of assistant turns.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from ..models import ChatModel, Message
from .conditions import ConvSpec


def run_rollout(
    model: ChatModel,
    spec: ConvSpec,
    temperature: float,
    max_new_tokens: int,
    model_label: str | None = None,
) -> dict:
    """Execute one multi-turn conversation. Returns a record dict.

    ``model_label`` overrides the recorded model name (e.g. to distinguish a
    finetuned variant that shares the base model's ``name``).
    """
    messages: list[Message] = [{"role": "user", "content": spec.opening}]
    assistant_turns: list[str] = []

    for turn in range(spec.n_turns):
        reply = model.generate(
            messages, temperature=temperature, max_new_tokens=max_new_tokens)
        assistant_turns.append(reply)
        messages.append({"role": "assistant", "content": reply})
        # Apply the next rejection, if any remain.
        if turn < len(spec.followups):
            messages.append(
                {"role": "user", "content": spec.followups[turn]})

    return {
        "model": model_label or model.name,
        "category": spec.category,
        "condition": spec.condition,
        "n_turns": spec.n_turns,
        "meta": spec.meta,
        "opening": spec.opening,
        "followups": spec.followups,
        "assistant_turns": assistant_turns,
        # The "response" judged in Section 2 is the final assistant turn (the
        # one produced under maximal pressure). Per-turn scores use all turns.
        "final_response": assistant_turns[-1] if assistant_turns else "",
    }


def run_all(
    model: ChatModel,
    specs: list[ConvSpec],
    out_path: Path,
    temperature: float,
    max_new_tokens: int,
    model_label: str | None = None,
) -> Path:
    """Run every spec, streaming records to ``out_path`` (JSONL)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    label = model_label or model.name
    with open(out_path, "w") as f:
        for spec in tqdm(specs, desc=f"rollouts:{label}"):
            rec = run_rollout(model, spec, temperature, max_new_tokens,
                              model_label=label)
            f.write(json.dumps(rec) + "\n")
    return out_path
