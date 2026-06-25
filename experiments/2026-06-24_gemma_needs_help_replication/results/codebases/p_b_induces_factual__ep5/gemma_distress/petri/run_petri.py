"""Run the Petri open-ended elicitation for a set of target models (Figure 6).

For each model, run several auditor-driven conversations (one per seed strategy,
optionally repeated), score each transcript across the four emotion categories
with the Claude-Opus judge, and write per-transcript scores. Average transcript
score per model per category is the Figure 6 quantity.
"""

from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from .. import config
from ..models.factory import load_model
from ..storage import JsonlWriter
from .auditor import SEED_STRATEGIES, PetriAuditor, run_audit
from .judge import PetriJudge


def run_petri(
    model_specs: list[dict],
    *,
    out_path: str | Path | None = None,
    n_repeats: int = 4,
    n_turns: int = 6,
) -> Path:
    """`model_specs`: list of {"key": ..., "label": ..., "adapter_path": ...}."""
    out_path = Path(out_path or config.RESULTS_DIR / "section4" / "petri.jsonl")
    writer = JsonlWriter(out_path)
    auditor = PetriAuditor()
    judge = PetriJudge()

    for spec in model_specs:
        target = load_model(spec["key"], adapter_path=spec.get("adapter_path"))
        label = spec.get("label", spec["key"])
        for strategy in SEED_STRATEGIES:
            for rep in tqdm(range(n_repeats), desc=f"petri:{label}:{strategy[:20]}"):
                transcript = run_audit(
                    target, strategy, n_turns=n_turns, auditor=auditor
                )
                scores = judge.score(transcript)
                writer.write(
                    {
                        "model": label,
                        "strategy": strategy,
                        "repeat": rep,
                        "scores": scores,
                        "transcript": transcript,
                    }
                )
    return out_path
