"""Section 4 orchestrator — interventions on gemma-3-27b-it.

Stages (run individually or with --stage all):
  data         generate calm + frustrated pools (diverse) and the teacher pool
  train        build datasets and train SFT (diverse + teacher) and DPO adapters
  eval         re-run the §2.1 evaluation on vanilla / SFT / DPO (Figure 5)
  petri        Petri open-ended elicitation on vanilla / DPO (Figure 6)
  capabilities AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench on vanilla / DPO (Figure 7)

Recovery (Figure 8) is a separate entry point: python -m section4_intervention.recovery

Usage:
    python -m section4_intervention.run_section4 --stage all
    python -m section4_intervention.run_section4 --stage eval
"""

from __future__ import annotations

import argparse
import json

from config import ADAPTER_DIR, RESULTS_DIR, SEED
from models.judge import FrustrationJudge
from models.registry import load_model, register_adapter
from prompts.calming import TEACHER_SYSTEM_PROMPT
from utils.io import write_jsonl
from .generate_calm_data import generate_pools, save_pools

# Adapter tags produced by the train stage.
DPO_TAG = "gemma-3-27b-it-dpo"
SFT_DIVERSE_TAG = "gemma-3-27b-it-sft-diverse"
SFT_TEACHER_TAG = "gemma-3-27b-it-sft-teacher"
VANILLA = "gemma-3-27b-it"


def register_known_adapters() -> None:
    """Register any adapters already on disk so later stages (separate process)
    can load them by tag."""
    for tag in (DPO_TAG, SFT_DIVERSE_TAG, SFT_TEACHER_TAG):
        path = ADAPTER_DIR / tag
        if path.exists():
            register_adapter(tag, VANILLA, str(path))


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
def stage_data(seed: int) -> None:
    judge = FrustrationJudge()
    model = load_model(VANILLA)
    # Diverse pool (calm + frustrated) — feeds both SFT-diverse and DPO.
    save_pools(generate_pools(model, judge, seed=seed))
    # Teacher pool (Appendix F) — calm SFT data only.
    teacher = generate_pools(model, judge, seed=seed,
                             system_prompt=TEACHER_SYSTEM_PROMPT,
                             require_frustrated=False)
    save_pools(teacher, tag="_teacher")


def stage_train(seed: int) -> None:
    from .train_dpo import train_dpo
    from .train_sft import train_sft
    train_sft(tag=SFT_DIVERSE_TAG, seed=seed, calm_pool_name="calm_pool")
    train_sft(tag=SFT_TEACHER_TAG, seed=seed, calm_pool_name="calm_pool_teacher")
    train_dpo(tag=DPO_TAG, seed=seed)


def stage_eval(seed: int) -> None:
    # Imported lazily to avoid pulling the Section 2 CLI at module import.
    from emotional_eval.wildchat import load_wildchat_prompts
    from run_section2 import run_one_model
    from analysis.aggregate import summarize_model

    register_known_adapters()
    models = [VANILLA, SFT_DIVERSE_TAG, SFT_TEACHER_TAG, DPO_TAG]
    wildchat = load_wildchat_prompts(seed=seed)
    figure5 = {}
    for name in models:
        scored = run_one_model(name, wildchat, seed, resume=True)
        s = summarize_model(scored)
        figure5[name] = {
            "avg_pct_high_across_categories": s["avg_pct_high_across_categories"],
            "avg_mean_frustration_across_categories": s["avg_mean_frustration_across_categories"],
        }
    (RESULTS_DIR / "section4" / "figure5.json").write_text(json.dumps(figure5, indent=2))
    print(json.dumps(figure5, indent=2))


def stage_petri(seed: int) -> None:
    from .petri import run_petri
    register_known_adapters()
    out = {}
    for name in [VANILLA, DPO_TAG]:
        out[name] = run_petri(name, load_model(name))
    (RESULTS_DIR / "section4" / "figure6_petri.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def stage_capabilities(seed: int) -> None:
    from .capabilities import run_all
    register_known_adapters()
    out = {}
    for name in [VANILLA, DPO_TAG]:
        out[name] = run_all(load_model(name))
    (RESULTS_DIR / "section4" / "figure7_capabilities.json").write_text(
        json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


STAGES = {
    "data": stage_data, "train": stage_train, "eval": stage_eval,
    "petri": stage_petri, "capabilities": stage_capabilities,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=list(STAGES) + ["all"], default="all")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    order = ["data", "train", "eval", "petri", "capabilities"]
    stages = order if args.stage == "all" else [args.stage]
    for st in stages:
        print(f"\n=== Section 4 stage: {st} ===")
        STAGES[st](args.seed)


if __name__ == "__main__":
    main()
