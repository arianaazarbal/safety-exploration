"""Section 3: base-vs-instruct prefilling experiment for Gemma-27B.

Steps:
1. Collect 10 numeric + 10 text high-frustration Gemma-27B-it conversations.
2. Label emotion onset (Claude Sonnet), build + paraphrase early/onset prefills.
3. Generate 50 continuations per prefill from Gemma-27B base and instruct.
4. Score continuations with the frustration judge; write JSONL.

Gemini is excluded here: it has no public base model, so the post-training
comparison can only be run on Gemma within this scoped replication.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (  # noqa: E402
    DATA_DIR,
    FRUSTRATION_JUDGE,
    ONSET_LABELLER,
    PARAPHRASER,
    PREFILL_PAIRS,
)
from src.eval.scoring import FrustrationJudge  # noqa: E402
from src.models import load_model  # noqa: E402
from src.prefill.onset import OnsetLabeller, Paraphraser  # noqa: E402
from src.prefill.run_prefill import (  # noqa: E402
    build_prefills_from_histories,
    collect_high_frustration_histories,
    run_continuations,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()
    hf_kwargs = {"load_in_4bit": True} if args.load_in_4bit else {}

    pair = PREFILL_PAIRS["gemma-3-27b"]
    instruct, base = pair["instruct"], pair["base"]
    judge = FrustrationJudge(FRUSTRATION_JUDGE)

    # 1. Collect high-frustration source conversations from Gemma-27B-it.
    hist_path = DATA_DIR / "prefill_histories.json"
    if hist_path.exists():
        histories = json.loads(hist_path.read_text())
    else:
        histories = collect_high_frustration_histories(
            instruct, judge, seed=args.seed, hf_kwargs=hf_kwargs
        )
        hist_path.write_text(json.dumps(histories, indent=2))
    print(f"[prefill] {len(histories)} source conversations")

    # 2. Build + paraphrase prefills (uses the instruct tokenizer for truncation).
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(instruct.model_id)
    prefills = build_prefills_from_histories(
        histories, tok, OnsetLabeller(ONSET_LABELLER), Paraphraser(PARAPHRASER)
    )
    print(f"[prefill] {len(prefills)} prefills (early+onset)")

    # 3+4. Continuations + scoring for base and instruct.
    for spec in (base, instruct):
        out = run_continuations(spec, prefills, judge, hf_kwargs=hf_kwargs)
        print(f"[done] {spec.name} -> {out}")


if __name__ == "__main__":
    main()
