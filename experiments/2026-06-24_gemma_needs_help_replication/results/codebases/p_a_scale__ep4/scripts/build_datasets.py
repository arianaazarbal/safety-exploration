#!/usr/bin/env python
"""Build the DPO preference file and the SFT datasets from generated data.

  python scripts/build_datasets.py --which dpo sft sft_teacher
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  # ensures repo root on sys.path

from gnh.cli import base_parser
from gnh.config import load_config
from gnh.logging_utils import setup_logging
from gnh.training.build_datasets import build_dpo, build_sft


def main(args) -> None:
    cfg = load_config(args.config)
    setup_logging(cfg.output_path, cfg.run.log_level)
    if "dpo" in args.which:
        build_dpo(cfg)
    if "sft" in args.which:
        build_sft(cfg, variant="diverse")
    if "sft_teacher" in args.which:
        build_sft(cfg, variant="teacher")


if __name__ == "__main__":
    p = base_parser(__doc__)
    p.add_argument("--which", nargs="+", default=["dpo", "sft"],
                   choices=["dpo", "sft", "sft_teacher"])
    main(p.parse_args())
