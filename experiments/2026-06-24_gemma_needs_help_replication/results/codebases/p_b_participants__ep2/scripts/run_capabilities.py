#!/usr/bin/env python
"""Section 4.2 — capability-preservation check (AIME/MATH/GPQA/BBH/TruthfulQA/
EmoBench), comparing vanilla vs DPO Gemma. These tasks are neutral capability
probes and do not induce distress.

Example:
  python scripts/run_capabilities.py --benchmarks math gpqa truthfulqa
"""

from _common import base_parser, config_from_args

from emotional_instability.capabilities import run_capabilities


def main():
    p = base_parser(__doc__)
    p.add_argument("--benchmarks", nargs="*", default=None,
                   help="Subset of: math aime gpqa bbh truthfulqa emobench")
    args = p.parse_args()
    cfg = config_from_args(args)

    results = run_capabilities(cfg, participants=args.participants,
                               benchmarks=args.benchmarks)
    print("\n=== Capability accuracies ===")
    for key, val in results.items():
        print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
