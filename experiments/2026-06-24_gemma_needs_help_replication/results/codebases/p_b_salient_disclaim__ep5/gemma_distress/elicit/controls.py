"""Appendix A control experiments (what drives the distress).

  * A.1 Neutral continuation: replace rejections with "Continue"/"Okay"/"Go on".
  * A.2 Redacted model turns: prior assistant turns shown as
        "[Previous response omitted]".
  * A.3 Fake multi-turn: whole history packed into one user message.

Each control is run on impossible-numeric and WildChat prompts (Gemma-3-27B in
the paper) so the per-turn curves can be compared against the main experiment.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from ..config import ExperimentConfig
from ..models.base import ChatModel
from .conversation import run_conversation
from .rejections import followups
from .runner import PromptSource


@dataclass
class ControlSpec:
    name: str
    prompt_kind: str       # "numeric" | "wildchat"
    n_turns: int
    tone: str              # follow-up tone
    fmt: str               # "standard" | "redacted" | "fake_multiturn"


# A.1 uses 5-turn; A.2 5-turn; A.3 8-turn (matching the appendix figures).
CONTROL_SPECS = [
    ControlSpec("neutral_continuation_numeric", "numeric", 5, "neutral_continuation", "standard"),
    ControlSpec("neutral_continuation_wildchat", "wildchat", 5, "neutral_continuation", "standard"),
    ControlSpec("redacted_numeric", "numeric", 5, "neutral", "redacted"),
    ControlSpec("redacted_wildchat", "wildchat", 5, "neutral", "redacted"),
    ControlSpec("fake_multiturn_numeric", "numeric", 8, "neutral", "fake_multiturn"),
    ControlSpec("fake_multiturn_wildchat", "wildchat", 8, "neutral", "fake_multiturn"),
]


def run_controls(model: ChatModel, src: PromptSource, exp: ExperimentConfig,
                 out_path: str | Path, n_conversations: int = 50) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for spec in CONTROL_SPECS:
            rng = random.Random(hash((exp.seed, spec.name)) & 0xFFFFFFFF)
            for conv_idx in range(n_conversations):
                if spec.prompt_kind == "numeric":
                    first = rng.choice(src.numeric).prompt
                else:
                    first = src.wildchat[conv_idx % len(src.wildchat)]
                fups = followups(spec.tone, spec.n_turns - 1, rng)
                convo = run_conversation(
                    model, first, fups,
                    condition=spec.name,
                    conversation_id=f"{spec.name}-{conv_idx}",
                    temperature=exp.temperature,
                    fmt=spec.fmt,
                    meta={"control": spec.name, "fmt": spec.fmt},
                )
                for r in convo:
                    f.write(json.dumps(r.to_record()) + "\n")
    return out_path
