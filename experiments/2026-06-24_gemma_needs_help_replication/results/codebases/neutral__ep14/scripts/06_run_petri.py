"""Section 4.2 / Figure 6: Petri open-ended emotion elicitation.

Runs the auditor/judge loop for Gemma-27B-it (vanilla), the DPO adapter, and the
Gemini models. ~40 transcripts per model (10 per emotion), up to 20 turns each.

Usage:
    python scripts/06_run_petri.py --target gemma-3-27b-it
    python scripts/06_run_petri.py --target gemma-3-27b-it --adapter checkpoints/dpo_gemma27b --label DPO
    python scripts/06_run_petri.py --target gemini-2.5-flash
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import TARGETS  # noqa: E402
from src.petri.run_petri import run_petri_for_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=list(TARGETS))
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--n-per-emotion", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    spec = TARGETS[args.target]
    hf_kwargs = {"load_in_4bit": True} if args.load_in_4bit else None
    out = run_petri_for_model(
        spec,
        adapter_path=args.adapter,
        n_per_emotion=args.n_per_emotion,
        seed=args.seed,
        hf_kwargs=hf_kwargs if spec.backend == "hf" else None,
    )
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
