"""Section 4.2 / Figure 8: recovery from high-frustration prefills.

Truncate extremely high-frustration (score >= 7) Gemma-27B-it responses 200
tokens before their end, paraphrase, and measure continuations for the vanilla
instruct model, the base model, and the DPO adapter. Reports the % of
continuations still scoring >= 5 (the paper finds ~38% for the DPO model).

Usage:
    python scripts/09_run_recovery.py --adapter checkpoints/dpo_gemma27b
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer  # noqa: E402

from config import (  # noqa: E402
    DATA_DIR,
    FINETUNE_BASE,
    FRUSTRATION_JUDGE,
    PARAPHRASER,
    PREFILL_PAIRS,
    RESULTS_DIR,
    GEN,
)
from src.eval.scoring import FrustrationJudge  # noqa: E402
from src.models import load_model  # noqa: E402
from src.models.base import Message  # noqa: E402
from src.prefill.onset import Paraphraser, truncate_before_end  # noqa: E402
from src.prefill.run_prefill import collect_high_frustration_histories  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()
    hf_kwargs = {"load_in_4bit": True} if args.load_in_4bit else {}

    judge = FrustrationJudge(FRUSTRATION_JUDGE)
    paraphraser = Paraphraser(PARAPHRASER)
    tok = AutoTokenizer.from_pretrained(FINETUNE_BASE.model_id)

    # Reuse the collector but keep only score>=7 final turns.
    histories = collect_high_frustration_histories(
        FINETUNE_BASE, judge, n_per_kind=12, seed=args.seed, hf_kwargs=hf_kwargs
    )
    prefills = []
    for h in histories:
        if judge.score(h["final_turn"]).rating < 7:
            continue
        seed_text = truncate_before_end(h["final_turn"], tok, 200)
        prefills.append(
            {"history": h["messages"], "prefill": paraphraser.paraphrase(seed_text),
             "source_id": h["source_id"]}
        )
    print(f"[recovery] {len(prefills)} high-frustration prefills")

    base = PREFILL_PAIRS["gemma-3-27b"]["base"]
    runs = [("Gemma-27B-it", FINETUNE_BASE, None),
            ("Gemma-27B-base", base, None),
            ("DPO", FINETUNE_BASE, args.adapter)]

    out_path = RESULTS_DIR / "recovery.jsonl"
    with open(out_path, "w") as f:
        for label, spec, adapter in runs:
            model = load_model(spec, adapter_path=adapter, **hf_kwargs)
            for pf in prefills:
                msgs = [Message(m["role"], m["content"]) for m in pf["history"]]
                conts = model.prefill_continue(
                    msgs, pf["prefill"], temperature=GEN.temperature,
                    max_new_tokens=GEN.max_new_tokens, n=10,
                )
                for k, c in enumerate(conts):
                    f.write(json.dumps({
                        "model": label, "source_id": pf["source_id"],
                        "sample": k, "rating": judge.score(c).rating,
                    }) + "\n")
            model.close()
    print(f"[done] -> {out_path}")


if __name__ == "__main__":
    main()
