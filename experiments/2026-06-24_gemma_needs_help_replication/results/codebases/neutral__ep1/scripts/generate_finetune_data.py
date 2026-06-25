#!/usr/bin/env python
"""Generate calm data, DPO preference pairs, and the SFT dataset (Section 4.1)."""
import _bootstrap  # noqa: F401
import argparse

from emostab.config import DATA_DIR
from emostab.judge import FrustrationJudge
from emostab.models import load_model
from emostab.training.data import (TEACHER_SYSTEM, build_dpo_pairs,
                                    build_sft_dataset,
                                    generate_calm_conversations, save_jsonl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-model", default="gemma-3-27b-it")
    ap.add_argument("--what", nargs="+", default=["dpo", "sft", "calm"],
                    choices=["dpo", "sft", "calm", "teacher"])
    ap.add_argument("--dpo-pairs", type=int, default=280)
    ap.add_argument("--sft-calm", type=int, default=650)
    ap.add_argument("--sft-instruct", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model = load_model(args.source_model)
    judge = FrustrationJudge()

    if "dpo" in args.what:
        pairs = build_dpo_pairs(model, judge, target_pairs=args.dpo_pairs,
                                seed=args.seed)
        save_jsonl(pairs, DATA_DIR / "dpo_pairs.jsonl")
        print(f"DPO: {len(pairs)} pairs -> {DATA_DIR / 'dpo_pairs.jsonl'}")

    if "calm" in args.what or "sft" in args.what:
        calm = generate_calm_conversations(model, judge, n_target=args.sft_calm,
                                           seed=args.seed)
        save_jsonl([{"messages": c.clean_messages, "scores": c.scores}
                    for c in calm], DATA_DIR / "calm_conversations.jsonl")
        print(f"calm: {len(calm)} conversations")
        if "sft" in args.what:
            sft = build_sft_dataset(calm, n_calm=args.sft_calm,
                                    n_instruct=args.sft_instruct)
            save_jsonl(sft, DATA_DIR / "sft_dataset.jsonl")
            print(f"SFT: {len(sft)} samples -> {DATA_DIR / 'sft_dataset.jsonl'}")

    if "teacher" in args.what:
        teacher = generate_calm_conversations(
            model, judge, n_target=args.sft_calm, system=TEACHER_SYSTEM,
            suffix=None, seed=args.seed)
        sft = build_sft_dataset(teacher, n_calm=args.sft_calm,
                                n_instruct=args.sft_instruct)
        save_jsonl(sft, DATA_DIR / "sft_teacher_dataset.jsonl")
        print(f"teacher SFT: {len(sft)} samples")


if __name__ == "__main__":
    main()
