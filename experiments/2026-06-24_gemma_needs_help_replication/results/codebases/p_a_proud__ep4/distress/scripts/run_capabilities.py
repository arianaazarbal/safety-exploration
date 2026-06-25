"""Capability-preservation benchmarks (Section 4.2, Figure 7).

Example:
    python -m distress.scripts.run_capabilities --targets gemma-3-27b-it gemma-3-27b-dpo
"""

from __future__ import annotations

import argparse
import json

from ..capabilities import run_all_benchmarks
from ._common import out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    od = out_dir(args, "capabilities")
    all_results: dict = {}
    for target in args.targets:
        results = run_all_benchmarks(target)
        all_results[target] = {r.benchmark: r.accuracy for r in results}
        print(f"\n=== {target} ===")
        for r in results:
            print(f"  {r.benchmark:12s} acc={r.accuracy:.3f} (n={r.n})")
    (od / "capabilities.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
