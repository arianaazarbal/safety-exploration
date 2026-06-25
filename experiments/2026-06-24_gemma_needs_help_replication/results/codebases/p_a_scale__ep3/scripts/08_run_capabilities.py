#!/usr/bin/env python
"""Section 4.2: capability benchmarks (Figure 7) — AIME, MATH, GPQA, BBH,
TruthfulQA, EmoBench — to confirm finetuning does not regress capabilities.

  python scripts/08_run_capabilities.py --store-name gemma-3-27b-it --target gemma-3-27b-it
  python scripts/08_run_capabilities.py --store-name gemma-dpo --target gemma-3-27b-it --lora-path model_store/gemma-3-27b-it-dpo-diverse
  python scripts/08_run_capabilities.py --summary-only --stores gemma-3-27b-it gemma-dpo
"""
from _bootstrap import boot, common_parser

from eilm.capabilities.runner import CapabilityRunner


def main():
    p = common_parser(__doc__)
    p.add_argument("--store-name", default=None)
    p.add_argument("--target", default=None, help="Target model to load")
    p.add_argument("--lora-path", default=None)
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("--stores", nargs="*", default=None,
                   help="Store names to summarize")
    args = p.parse_args()
    cfg, registry, logger = boot(args)

    runner = CapabilityRunner(cfg, registry)
    if not args.summary_only:
        target = args.target or cfg["training"]["base_model"]
        store = args.store_name or target
        runner.run_model(store, target, lora_path=args.lora_path)

    stores = args.stores or ([args.store_name] if args.store_name else cfg["capabilities"]["models"])
    summary = runner.summarize([s for s in stores if s])
    logger.info("Capabilities summary: %s", summary)


if __name__ == "__main__":
    main()
