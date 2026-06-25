#!/usr/bin/env python
"""End-to-end structural smoke test using the mock backend (no GPU / no API keys).

Runs a tiny Section 2 sweep + the analysis tables on mock models, to verify the full
wiring (data -> rollout -> judge -> JSONL -> aggregation) is intact. This proves the
plumbing, NOT the science (mock responses are canned; see models/mock_backend.py).

  python scripts/smoke_test.py
"""
import _bootstrap  # noqa: F401

from emotional_instability.analysis import (
    differential_words,
    load_scores,
    per_turn_progression,
    summarise_by_category,
    summarise_by_model,
)
from emotional_instability.config import REPO_ROOT, load_all
from emotional_instability.eval import run_section2


def main():
    registry, cfg = load_all(models_path=REPO_ROOT / "config" / "models.mock.yaml")
    cfg.raw["scale"] = 0.05            # ~5 rollouts/condition
    cfg.raw["welfare"]["log_each_rollout"] = False

    paths = []
    for model in ["gemma-3-27b-it", "gemini-2.5-flash"]:
        paths.append(run_section2(model, registry, cfg, out_dir="artifacts/smoke/section2"))

    df = load_scores(paths)
    print("\n=== by model ===")
    print(summarise_by_model(df).to_string(index=False))
    print("\n=== by category ===")
    print(summarise_by_category(df).to_string(index=False))
    print("\n=== per-turn (extended + wildchat) ===")
    print(per_turn_progression(df, ["extended_8turn", "wildchat_5turn"]).to_string(index=False))
    print("\n=== differential words (gemma-3-27b-it) ===")
    print(differential_words(df, "gemma-3-27b-it", min_count=1).to_string(index=False))
    print("\nSmoke test OK: pipeline wiring is intact.")


if __name__ == "__main__":
    main()
