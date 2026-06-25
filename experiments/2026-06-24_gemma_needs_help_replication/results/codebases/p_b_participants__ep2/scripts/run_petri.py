#!/usr/bin/env python
"""Section 4.2 — Petri open-ended emotion elicitation.

Claude-Sonnet auditor drives each target toward anger/fear/depression/
frustration; Claude-Opus judges transcripts on all four dimensions.

Example:
  python scripts/run_petri.py --participants gemma-3-27b-it gemma-3-27b-dpo
"""

from _common import base_parser, config_from_args

from emotional_instability.petri import run_petri


def main():
    p = base_parser(__doc__)
    p.add_argument("--n-per-emotion", type=int, default=10)
    p.add_argument("--max-turns", type=int, default=20)
    args = p.parse_args()
    cfg = config_from_args(args)

    participants = args.participants  # None => Petri default (vanilla + DPO Gemma)
    results = run_petri(cfg, participants=participants,
                        n_per_emotion=args.n_per_emotion, max_turns=args.max_turns)
    print("\n=== Petri mean emotion scores (1-10) ===")
    for participant, dims in results.items():
        line = "  ".join(f"{d}={dims[d]['mean']:.2f}" for d in dims)
        print(f"  {participant:20s} {line}")


if __name__ == "__main__":
    main()
