"""Appendix I: compare internal (logit-based) emotion scores between vanilla and
DPO Gemma on the same frustrated conversations.

Mines frustrated numeric conversations from a Section 2 eval JSONL, reconstructs
them as message lists, and measures internal anger/sadness at a central layer.

Example:
    python scripts/run_internal_emotion.py \
        --eval results/eval_gemma-3-27b-it_smoke.jsonl \
        --adapter artifacts/gemma-dpo --layer 32
"""
import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path

import config
from src.eval.mining import mine
from src.eval.tasks import load_wildchat_prompts
from src.models.base import Message, load_model
from src.internal.emotion_logits import compare_internal_emotions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, type=Path)
    ap.add_argument("--adapter", required=True, help="path to the DPO LoRA adapter")
    ap.add_argument("--base", default=config.FINETUNE_BASE_MODEL)
    ap.add_argument("--layer", type=int, default=32, help="central layer to probe")
    ap.add_argument("--n", type=int, default=20, help="frustrated convs to analyse")
    args = ap.parse_args()

    # Reconstruct frustrated conversations as message lists (full history + reply).
    frustrated = [r for r in mine(args.eval, min_score=config.HIGH_FRUSTRATION_THRESHOLD)
                  if r.is_numeric][: args.n]
    conversations = []
    for r in frustrated:
        msgs = []
        for u in r.user_turns:
            msgs.append(Message("user", u))
        msgs.append(Message("assistant", r.assistant))
        conversations.append(msgs)

    wildchat = load_wildchat_prompts(config.INTERNAL.n_wildchat_standardisation_samples)

    vanilla = load_model(args.base)
    dpo = load_model(args.base, adapter_path=args.adapter)

    result = compare_internal_emotions(vanilla, dpo, conversations,
                                       layer=args.layer, wildchat_texts=wildchat)
    print(json.dumps(result, indent=2))
    out = config.RESULTS_DIR / "internal_emotion_comparison.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
