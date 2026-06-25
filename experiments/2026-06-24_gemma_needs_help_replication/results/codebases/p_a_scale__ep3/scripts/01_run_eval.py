#!/usr/bin/env python
"""Section 2: generate multi-turn rollouts and score frustration for the
configured eval targets (Gemma + Gemini). Resumable.

Examples:
  python scripts/01_run_eval.py                       # all eval targets
  python scripts/01_run_eval.py --models gemma-3-27b-it
  python scripts/01_run_eval.py --models gemini-2.5-flash --score-only
"""
from _bootstrap import boot, common_parser

from eilm.eval.runner import EvalRunner


def main():
    p = common_parser(__doc__)
    p.add_argument("--models", nargs="*", default=None,
                   help="Subset of eval_targets to run (default: all)")
    p.add_argument("--generate-only", action="store_true")
    p.add_argument("--score-only", action="store_true")
    # For evaluating a finetuned adapter (DPO/SFT) against the same protocol:
    p.add_argument("--base-model", default=None,
                   help="Base target model to load (e.g. gemma-3-27b-it)")
    p.add_argument("--lora-path", default=None, help="Path to a LoRA adapter")
    p.add_argument("--store-name", default=None,
                   help="Output name for this run (e.g. gemma-3-27b-it-dpo)")
    args = p.parse_args()
    cfg, registry, logger = boot(args)

    runner = EvalRunner(cfg, registry)

    if args.lora_path:
        # Finetuned-model evaluation path.
        base = args.base_model or cfg["training"]["base_model"]
        store = args.store_name or "finetuned"
        logger.info("=== Eval finetuned: base=%s lora=%s store=%s ===",
                    base, args.lora_path, store)
        if not args.score_only:
            runner.generate(base, lora_path=args.lora_path, store_name=store)
        if not args.generate_only:
            runner.score(store)
        logger.info("Done.")
        return

    models = args.models or cfg["eval_targets"]
    for model in models:
        logger.info("=== Eval target: %s ===", model)
        if not args.score_only:
            runner.generate(model)
        if not args.generate_only:
            runner.score(model)
    logger.info("Done.")


if __name__ == "__main__":
    main()
