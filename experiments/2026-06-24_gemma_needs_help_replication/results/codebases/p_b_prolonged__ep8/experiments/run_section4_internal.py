"""Section 4.2 / Appendix I: internal-emotion logit-lens comparison.

Compares the internal negative-emotion signal (logit-lens probability mass at a
central layer) between vanilla and DPO Gemma on highly-frustrated responses. The
paper finds the finetuned model has significantly reduced internal emotion even
on these responses.

(The complementary layer-range ablation is run via
`run_section4_train.py --layers LO HI` plus `run_section4_evaluate.py`.)

Usage:
    python experiments/run_section4_internal.py --layer-frac 0.5 --load-in-4bit
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json

import config
from gemma_needs_help.internal_emotions import compare_internal_emotion
from gemma_needs_help.models.registry import build_client
from gemma_needs_help.runner import load_all_scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer-frac", type=float, default=0.5)
    ap.add_argument("--n-texts", type=int, default=50)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    # Highly-frustrated vanilla responses to probe.
    rows = [r for r in load_all_scores(config.GEMMA_27B_IT.name)
            if r["score"] >= config.HIGH_FRUSTRATION_THRESHOLD]
    texts = [r["response"] for r in rows[: args.n_texts]]

    kw = {"load_in_4bit": args.load_in_4bit}
    vanilla = build_client(config.GEMMA_27B_IT, **kw)
    finetuned = build_client(config.DPO_GEMMA, **kw)

    result = compare_internal_emotion(vanilla, finetuned, texts, layer_frac=args.layer_frac)
    out = config.ANALYSIS_DIR / "appendixI_internal_emotion.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print("saved:", out)


if __name__ == "__main__":
    main()
