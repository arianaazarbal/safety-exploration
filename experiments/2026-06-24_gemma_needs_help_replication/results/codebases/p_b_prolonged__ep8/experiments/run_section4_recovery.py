"""Section 4.2: recovery-from-spiral experiment (Figure 8).

Truncates score>=7 responses 200 tokens before their end, paraphrases, and
measures whether each model recovers (fraction of continuations still scoring
>= 5). The paper reports 38% for the DPO model.

Usage:
    python experiments/run_section4_recovery.py --load-in-4bit
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json

import config
from gemma_needs_help.models.registry import build_client
from gemma_needs_help.recovery import build_recovery_prefills, run_recovery


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*",
                    default=["gemma-3-27b-it", "dpo-gemma-3-27b", "gemma-3-27b-base"])
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    kw = {"load_in_4bit": args.load_in_4bit}
    # Use the instruct model's tokenizer for the 200-token-before-end truncation.
    tokenizer_client = build_client(config.GEMMA_27B_IT, **kw)
    prefills = build_recovery_prefills(tokenizer_client=tokenizer_client)
    print(f"built {len(prefills)} recovery prefills")

    by_name = {m.name: m for m in
               [config.GEMMA_27B_IT, config.DPO_GEMMA, config.SFT_GEMMA, config.GEMMA_27B_BASE]}
    out = {}
    for name in args.models:
        target = by_name[name]
        client = tokenizer_client if name == config.GEMMA_27B_IT.name else build_client(target, **kw)
        out[name] = run_recovery(target, prefills, client=client)
        print(name, out[name])

    path = config.ANALYSIS_DIR / "figure8_recovery.json"
    path.write_text(json.dumps(out, indent=2))
    print("saved:", path)


if __name__ == "__main__":
    main()
