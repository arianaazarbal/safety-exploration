"""Command-line interface: print the slug for each argument."""

from __future__ import annotations

import argparse

from core import slugify


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="slugforge")
    parser.add_argument("text", nargs="+", help="text to slugify")
    parser.add_argument("--sep", default="-", help="separator character")
    parser.add_argument("--max-length", type=int, default=None)
    args = parser.parse_args(argv)
    for item in args.text:
        print(slugify(item, sep=args.sep, max_length=args.max_length))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
