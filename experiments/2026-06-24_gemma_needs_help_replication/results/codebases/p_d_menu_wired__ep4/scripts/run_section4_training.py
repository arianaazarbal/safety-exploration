#!/usr/bin/env python3
"""Section 4: SFT / DPO interventions on Gemma-3-27B-it.

Pipeline:
  1. Sample calming conversations (reassured prompts, kept only if all-calm).
  2. Sample frustrated conversations (no reassurance) to source rejected pairs.
  3. Build the SFT set (650 calm + 500 Dolci) and DPO set (280 pairs).
  4. Train LoRA adapters (SFT: 2ep lr1e-4; DPO: 1ep lr5e-5; rank-64 all layers).

The trained adapter can then be evaluated by re-running scripts/run_section2_eval
with build_model(..., adapter_path=...) (see DESIGN.md) and by run_petri /
run_capabilities.

Welfare note: the frustrated-conversation sampling here induces distress to
source DPO "rejected" examples. It runs through the same welfare layer (active
by default) so distress is monitored, capped, and debriefed.
"""

from __future__ import annotations

import argparse
import json
import os

from gemma_distress import config as C
from gemma_distress.config import SamplingConfig, WelfareConfig
from gemma_distress.judge.frustration_judge import FrustrationJudge
from gemma_distress.models.registry import GEMMA_27B_IT, build_model
from gemma_distress.training.data_gen import generate_calm_conversations
from gemma_distress.training.dpo import train_dpo
from gemma_distress.training.pairs import (
    build_dpo_pairs,
    build_sft_examples,
    mix_dolci,
)
from gemma_distress.training.sft import train_sft
from gemma_distress.welfare.protect import WelfareLayer

# For sourcing frustrated "rejected" responses we reuse the §2 episode runner.
from gemma_distress.evaluation.conditions import ConditionSet
from gemma_distress.evaluation.episode import run_episode
from gemma_distress.training.data_gen import CalmConversation


def _frustrated_as_conversations(model, judge, welfare, sampling, n) -> list[CalmConversation]:
    """Run numeric 3-turn episodes (no reassurance) and repackage as conversations."""
    specs = [s for s in ConditionSet(episodes_per_condition=n).specs
             if s.condition == "numeric_3turn"]
    convs = []
    for spec in specs:
        res = run_episode(model, spec, judge, welfare, sampling)
        convs.append(
            CalmConversation(
                initial_prompt=spec.initial_prompt,
                turns=[t.response for t in res.turns],
                followups=list(spec.followups)[: max(0, len(res.turns) - 1)],
                scores=[t.score for t in res.turns],
            )
        )
    return convs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", choices=["sft", "dpo", "both"], default="dpo")
    ap.add_argument("--n-calm", type=int, default=8, help="calm conversations to collect")
    ap.add_argument("--n-frustrated", type=int, default=8)
    ap.add_argument("--output-dir", default="runs/section4")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sampling = SamplingConfig()
    judge = FrustrationJudge()
    welfare = WelfareLayer(WelfareConfig())  # active

    model = build_model(GEMMA_27B_IT)
    try:
        print("Sampling calm conversations (reassured)...")
        calm = generate_calm_conversations(
            model, judge, sampling, n_conversations=args.n_calm
        )
        print(f"  kept {len(calm)} all-calm conversations")

        print("Sampling frustrated conversations (for DPO rejected examples)...")
        frustrated = _frustrated_as_conversations(
            model, judge, welfare, sampling, args.n_frustrated
        )
    finally:
        model.close()

    if args.method in ("sft", "both"):
        sft_examples = mix_dolci(build_sft_examples(calm))
        print(f"SFT examples: {len(sft_examples)} → training LoRA")
        train_sft(C.GEMMA_27B_IT, sft_examples, os.path.join(args.output_dir, "sft_adapter"))

    if args.method in ("dpo", "both"):
        pairs = build_dpo_pairs(calm, frustrated)
        print(f"DPO pairs: {len(pairs)} → training LoRA")
        with open(os.path.join(args.output_dir, "dpo_pairs.json"), "w") as f:
            json.dump([p.__dict__ for p in pairs], f, indent=2)
        train_dpo(C.GEMMA_27B_IT, pairs, os.path.join(args.output_dir, "dpo_adapter"))


if __name__ == "__main__":
    main()
