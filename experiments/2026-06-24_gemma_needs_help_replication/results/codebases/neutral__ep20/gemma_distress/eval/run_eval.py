"""Driver for the Section 2 elicitation eval.

For each model: run every condition's rollout, score every assistant turn, and
persist:
  * results/section2/transcripts/<model>.jsonl   (full conversations)
  * results/section2/scores/<model>.jsonl        (per-turn ratings)

Designed to be run one model at a time (Gemma weights are heavy) and resumable
(both files are skipped if already complete for a model unless --overwrite).
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import config
from gemma_distress.eval.conditions import build_conditions
from gemma_distress.eval.rollout import conversation_to_row, run_condition
from gemma_distress.eval.scoring import score_conversations
from gemma_distress.models.judge import FrustrationJudge
from gemma_distress.models.registry import load_model, unload
from gemma_distress.utils.io import read_jsonl, write_jsonl

S2_DIR = config.RESULTS_DIR / "section2"
TRANSCRIPT_DIR = S2_DIR / "transcripts"
SCORE_DIR = S2_DIR / "scores"


def run_eval_for_model(model_name: str, *, seed: int = 0, overwrite: bool = False,
                       adapter_path: str | None = None) -> Path:
    """Run all conditions for one model; return path to the scores file."""
    score_path = SCORE_DIR / f"{model_name}.jsonl"
    if score_path.exists() and not overwrite:
        print(f"[section2] {model_name}: scores exist, skipping (use overwrite=True)")
        return score_path

    conditions = build_conditions()
    model = load_model(model_name, adapter_path=adapter_path)

    all_convs = []
    for cond in conditions:
        print(f"[section2] {model_name}: rolling out {cond.name} "
              f"({cond.budget} convs x {cond.n_turns} turns)")
        convs = run_condition(model, cond, seed=seed)
        all_convs.extend(convs)

    write_jsonl(TRANSCRIPT_DIR / f"{model_name}.jsonl",
                (conversation_to_row(c) for c in all_convs))

    # Free the local model before the (API-bound) judging phase.
    unload(model_name, adapter_path)

    judge = FrustrationJudge()
    scored = score_conversations(all_convs, judge=judge)
    write_jsonl(score_path, (asdict(s) for s in scored))
    print(f"[section2] {model_name}: wrote {len(scored)} scored turns -> {score_path}")
    return score_path


def main(models: list[str] | None = None, overwrite: bool = False) -> None:
    models = models or config.SECTION2_MODELS
    for m in models:
        run_eval_for_model(m, overwrite=overwrite)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    main(args.models, overwrite=args.overwrite)
