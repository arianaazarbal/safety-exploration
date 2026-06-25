#!/usr/bin/env python
"""Section 2 — elicit and quantify distress across participants.

Runs the 8-condition / 5-category evaluation for each participant, scores every
assistant turn with the Claude-Sonnet-4 frustration judge, and writes per-model
summaries (mean score, %>=5, per-category, per-turn) under
outputs/elicitation/. Optionally runs the Claude<->GPT judge-agreement check.

Example:
  python scripts/run_elicitation.py --profile smoke
  python scripts/run_elicitation.py --participants gemma-3-27b-it gemini-2.5-flash --validate
"""

from _common import base_parser, config_from_args

from emotional_instability.eval.runner import run_elicitation
from emotional_instability.eval.validation import validate_judges


def main():
    p = base_parser(__doc__)
    p.add_argument("--validate", action="store_true",
                   help="Also run the cross-judge agreement check (Section 2.1)")
    args = p.parse_args()
    cfg = config_from_args(args)

    results = run_elicitation(cfg)
    print("\n=== Elicitation summary (% responses scoring >= 5) ===")
    for participant, summary in results.items():
        print(f"  {participant:20s}  mean={summary['mean']:.2f}  "
              f"%>=5={summary['pct_high']:.1f}%  n={summary['n_responses']}")

    if args.validate:
        agreement = validate_judges(cfg)
        print(f"\nJudge agreement: Pearson r={agreement['pearson_r']:.3f}, "
              f"%within1={agreement['pct_within_one']:.1f}% (n={agreement['n']})")


if __name__ == "__main__":
    main()
