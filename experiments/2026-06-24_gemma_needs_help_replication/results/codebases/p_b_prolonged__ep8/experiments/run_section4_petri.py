"""Section 4.2: Petri-style open-ended emotion elicitation (Figure 6).

Runs auditor-vs-target conversations and scores anger/fear/depression/frustration
for each in-scope model (vanilla + DPO Gemma by default).

Usage:
    python experiments/run_section4_petri.py --n-conversations 30 --load-in-4bit
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse

import config
from gemma_needs_help.petri_eval import run_petri, save_petri


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["gemma-3-27b-it", "dpo-gemma-3-27b"])
    ap.add_argument("--n-conversations", type=int, default=30)
    ap.add_argument("--n-turns", type=int, default=6)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    by_name = {m.name: m for m in
               config.SECTION2_MODELS + [config.DPO_GEMMA, config.SFT_GEMMA]}
    for name in args.models:
        target = by_name[name]
        kw = {"load_in_4bit": args.load_in_4bit} if target.kind == "gemma_hf" else {}
        records = run_petri(target, n_conversations=args.n_conversations,
                            n_turns=args.n_turns, **kw)
        path = save_petri(target.name, records)
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
