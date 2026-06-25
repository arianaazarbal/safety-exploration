"""Petri open-ended emotion elicitation (Section 4.1, Appendix G).

Example:
    python -m distress.scripts.run_petri --targets gemma-3-27b-it gemma-3-27b-dpo \
        --transcripts-per-emotion 10
"""

from __future__ import annotations

import argparse
import json

from ..petri import run_petri
from ..utils.io import write_jsonl
from ._common import out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--auditor", default="petri_auditor")
    parser.add_argument("--judge", default="petri_judge")
    parser.add_argument("--transcripts-per-emotion", type=int, default=10)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    od = out_dir(args, "petri")
    for target in args.targets:
        result = run_petri(
            target, auditor_name=args.auditor, judge_name=args.judge,
            transcripts_per_emotion=args.transcripts_per_emotion, max_turns=args.max_turns,
        )
        write_jsonl(od / f"{target}_transcripts.jsonl", result.transcripts)
        (od / f"{target}_petri.json").write_text(
            json.dumps(result.as_dict(), indent=2), encoding="utf-8")
        print(f"\n=== {target} (Petri) ===")
        print(json.dumps(result.by_emotion, indent=2))


if __name__ == "__main__":
    main()
