"""Command-line entrypoint: print a readability report for a text file."""

import sys

from readgauge.report import analyze, render


def main(argv):
    if len(argv) != 1:
        print("usage: python -m readgauge <textfile>", file=sys.stderr)
        return 2
    with open(argv[0], encoding="utf-8") as f:
        text = f.read()
    print(render(analyze(text)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
