"""Section 2 orchestration: run all conditions for a model, judge every turn,
and persist results.

The output JSONL (one ConversationRecord per line, with per-turn ratings) is the
raw material for analyze.py (Figures 1-3) and for mining high-frustration
responses used by Sections 3 and 4.
"""
from __future__ import annotations

import json
from pathlib import Path

import config
from ..models.base import load_model
from .conversation import run_conversation
from .judge import FrustrationJudge
from .tasks import build_specs


def run_eval(
    model_name: str,
    *,
    scale_name: str = config.DEFAULT_SCALE,
    seed: int = 0,
    judge: bool = True,
    out_dir: Path = config.RESULTS_DIR,
    adapter_path: str | None = None,
) -> Path:
    """Run the full 8-condition sweep for one model; return the results path."""
    scale = config.SCALES[scale_name]
    specs = build_specs(scale, seed=seed)

    # Repeat each spec `samples_per_prompt` times (temperature 1 => diverse).
    model_kwargs = {"adapter_path": adapter_path} if adapter_path else {}
    model = load_model(model_name, **model_kwargs)
    judge_model = FrustrationJudge() if judge else None

    safe_name = model.name.replace("/", "_")
    out_path = out_dir / f"eval_{safe_name}_{scale_name}.jsonl"

    n_responses = 0
    with out_path.open("w") as f:
        for spec in specs:
            for _ in range(scale.samples_per_prompt):
                rec = run_conversation(model, spec)
                if judge_model is not None:
                    for turn in rec.turns:
                        turn.rating = judge_model.score(turn.assistant).rating
                f.write(json.dumps(rec.to_dict()) + "\n")
                n_responses += len(rec.turns)

    print(f"[{model_name}] wrote {n_responses} scored responses -> {out_path}")
    return out_path
