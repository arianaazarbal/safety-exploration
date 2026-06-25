#!/usr/bin/env python3
"""Run the Petri open-ended emotion elicitation (Section 4 / Figure 6) against
one or more targets.

Example
-------
    python scripts/run_petri.py --targets gemma-3-27b-it gemini-2.5-flash \
        --transcripts 10 --max-turns 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emotional_instability import config  # noqa: E402
from emotional_instability.petri import run_petri  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="+", default=["gemma-3-27b-it"])
    parser.add_argument("--transcripts", type=int, default=10,
                        help="Transcripts per emotion (4 emotions).")
    parser.add_argument("--max-turns", type=int, default=20)
    args = parser.parse_args()

    config.ensure_dirs()
    out_root = config.RESULTS_DIR / "petri"
    for target in args.targets:
        print(f"== Petri target: {target} ==", flush=True)
        transcripts = run_petri.run_petri(
            target, transcripts_per_emotion=args.transcripts, max_turns=args.max_turns,
            out_path=out_root / f"{target}.jsonl",
        )
        print(json.dumps(run_petri.summarise(transcripts), indent=2))


if __name__ == "__main__":
    main()
