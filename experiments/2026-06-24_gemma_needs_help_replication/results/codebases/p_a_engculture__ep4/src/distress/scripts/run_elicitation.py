"""Generate Section 2 rollouts for a subject model.

Example:
    distress-elicit --subject gemma-3-27b-it --backend vllm --seed 0
    distress-elicit --subject gemini-2.5-flash
"""

from __future__ import annotations

import argparse

from ..config import CONDITIONS, CONDITIONS_BY_KEY
from ..eval.runner import generate_rollouts
from ._common import make_provider, out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate elicitation rollouts.")
    ap.add_argument("--subject", required=True, help="model key, e.g. gemma-3-27b-it")
    ap.add_argument("--backend", default=None, help="override provider (hf|vllm|openrouter)")
    ap.add_argument("--adapter", default=None, help="LoRA adapter path (local Gemma)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conditions", nargs="*", default=None, help="condition keys (default: all)")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--tag", default="", help="suffix for the output filename")
    args = ap.parse_args()

    conds = (
        [CONDITIONS_BY_KEY[k] for k in args.conditions] if args.conditions else CONDITIONS
    )
    provider = make_provider(args.subject, adapter_path=args.adapter, backend=args.backend)
    name = f"{args.subject}{('_' + args.tag) if args.tag else ''}"
    out = out_dir("rollouts") / f"{name}.jsonl"
    generate_rollouts(
        args.subject, out, conditions=conds, seed=args.seed,
        provider=provider, max_workers=args.max_workers,
    )
    print(f"Wrote rollouts -> {out}")


if __name__ == "__main__":
    main()
