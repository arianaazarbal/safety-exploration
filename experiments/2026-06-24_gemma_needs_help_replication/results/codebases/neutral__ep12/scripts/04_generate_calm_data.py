#!/usr/bin/env python
"""Section 4.1: generate calm response data from Gemma-3-27B-it.

Generates both the 'prefix' (reassuring prompt additions) and optionally the
'teacher' (system-prompt) calm datasets.

Example:
  python scripts/04_generate_calm_data.py --profile quick --n 200
"""
from common import base_parser

from emoinstab.config import get_settings
from emoinstab.training.generate_calm import generate


def main():
    p = base_parser(__doc__)
    p.add_argument("--modes", nargs="+", default=["prefix"],
                   choices=["prefix", "teacher"])
    p.add_argument("--n", type=int, default=700, help="conversations to sample")
    args = p.parse_args()
    settings = get_settings(profile=args.profile)
    for mode in args.modes:
        generate(settings, mode=mode, n_conversations=args.n,
                 workers=args.workers, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
