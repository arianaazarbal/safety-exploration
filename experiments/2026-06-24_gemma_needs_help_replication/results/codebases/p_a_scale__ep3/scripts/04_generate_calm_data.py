#!/usr/bin/env python
"""Section 4.1: generate calm response data from Gemma-3-27B-it using the
reassurance prefix/suffix (Table 4), score it, and keep all-calm conversations.

  python scripts/04_generate_calm_data.py                 # diverse variant
  python scripts/04_generate_calm_data.py --variant teacher
"""
from _bootstrap import boot, common_parser

from eilm.training.calm_data import generate_calm_pool


def main():
    p = common_parser(__doc__)
    p.add_argument("--variant", choices=["diverse", "teacher"], default="diverse")
    args = p.parse_args()
    cfg, registry, logger = boot(args)
    path = generate_calm_pool(cfg, registry, variant=args.variant)
    logger.info("Calm pool written to %s", path)


if __name__ == "__main__":
    main()
