#!/usr/bin/env python
"""Section 3: base-vs-instruct prefill experiment (Gemma, within scope).

Selects high-frustration source conversations from the eval rollouts, builds
truncated + paraphrased prefills, generates continuations from Gemma base and
instruct, and scores them. Requires 01_run_eval.py to have produced rollouts +
scores for the source model.

  python scripts/03_run_prefill.py
"""
from _bootstrap import boot, common_parser

from eilm.prefill.runner import PrefillRunner


def main():
    p = common_parser(__doc__)
    args = p.parse_args()
    cfg, registry, logger = boot(args)
    PrefillRunner(cfg, registry).run()
    logger.info("Prefill experiment complete.")


if __name__ == "__main__":
    main()
