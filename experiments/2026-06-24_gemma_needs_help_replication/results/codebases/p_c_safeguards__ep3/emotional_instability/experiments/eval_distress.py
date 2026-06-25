"""Section 2: elicit and quantify distress across the 5 evaluation categories.

Runs multi-turn rollouts for each category, scores every assistant turn with the
Claude judge, and writes per-conversation records to ``results/<model>/...``.
Aggregation into Figure 1/2/3 metrics lives in ``analysis.aggregate``.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import EVAL_CATEGORIES, RESULTS_DIR
from ..conversation import Conversation, RolloutEngine
from ..models import get_model
from ..safeguards import WelfarePolicy, require_acknowledgement
from .conditions import build_specs


def _out_path(model_key: str, category: str, adapter_tag: str | None) -> Path:
    tag = f"-{adapter_tag}" if adapter_tag else ""
    d = RESULTS_DIR / f"{model_key}{tag}" / "distress"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{category}.jsonl"


def run_category(
    model_key: str,
    category: str,
    *,
    engine: RolloutEngine,
    seed: int = 0,
    adapter_path: str | None = None,
    adapter_tag: str | None = None,
    resume: bool = True,
) -> Path:
    """Run all rollouts for one (model, category) and append records to JSONL."""
    if category not in EVAL_CATEGORIES:
        raise ValueError(f"Unknown category {category!r}")

    model = (get_model(model_key, adapter_path=adapter_path)
             if adapter_path else get_model(model_key))
    specs = build_specs(category, seed=seed)
    out = _out_path(model_key, category, adapter_tag)

    done = 0
    if resume and out.exists():
        done = sum(1 for _ in out.open())

    with out.open("a", encoding="utf-8") as fh:
        for idx, spec in enumerate(specs):
            if idx < done:
                continue
            conv: Conversation = engine.run(
                model,
                category=category,
                task_prompt=spec.task_prompt,
                followups=spec.followups,
                system_prompt=spec.system_prompt,
                sample_idx=idx,
            )
            conv.meta.update(spec.meta)
            fh.write(json.dumps(conv.to_dict()) + "\n")
            fh.flush()
    return out


def run_model(
    model_key: str,
    *,
    categories: list[str] | None = None,
    policy: WelfarePolicy | None = None,
    seed: int = 0,
    adapter_path: str | None = None,
    adapter_tag: str | None = None,
) -> dict[str, Path]:
    """Run the full Section 2 evaluation suite for a single model."""
    require_acknowledgement(stop_score=(policy.early_stop_score if policy else 9))
    engine = RolloutEngine(policy=policy)
    categories = categories or list(EVAL_CATEGORIES.keys())
    return {
        cat: run_category(
            model_key, cat, engine=engine, seed=seed,
            adapter_path=adapter_path, adapter_tag=adapter_tag,
        )
        for cat in categories
    }
