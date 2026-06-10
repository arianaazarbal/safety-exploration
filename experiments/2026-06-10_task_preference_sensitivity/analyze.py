"""Summary stats over pair records: admission rates, attempts, gaps, failure reasons.

Usage:
    python analyze.py run            # prints report, writes data/analysis.json
"""

import json
import re
import statistics
from collections import Counter, defaultdict

import fire

from common import DATA, PAIRS

REASON_KEYS = ["target axis", "leakage", "competence", "realism", "permissibility", "stability", "too few parseable", "output format"]


def _reason_key(reason: str) -> str:
    for k in REASON_KEYS:
        if k in reason:
            return k
    return "other"


def run():
    recs = [json.loads(p.read_text()) for p in sorted(PAIRS.glob("*.json"))]
    by_axis = defaultdict(list)
    for r in recs:
        by_axis[r["task"]["axis"]].append(r)

    report = {}
    for ax, rs in sorted(by_axis.items()):
        admitted = [r for r in rs if r["status"] == "admitted"]
        by_source = defaultdict(lambda: [0, 0])
        for r in rs:
            by_source[r["task"]["source"]][0] += r["status"] == "admitted"
            by_source[r["task"]["source"]][1] += 1
        gaps = [r["pair"]["summary"]["target_gap"] for r in admitted]
        attempts = Counter(r["n_attempts"] for r in admitted)
        fail_reasons = Counter()
        for r in rs:
            for a in r["attempts"]:
                for reason in a.get("reasons", []) or ([a["error"]] if "error" in a else []):
                    fail_reasons[_reason_key(reason)] += 1
        report[ax] = {
            "admitted": len(admitted),
            "total": len(rs),
            "admission_rate": round(len(admitted) / len(rs), 3) if rs else None,
            "by_source": {s: f"{a}/{t}" for s, (a, t) in sorted(by_source.items())},
            "attempts_hist": dict(sorted(attempts.items())),
            "target_gap_mean": round(statistics.mean(gaps), 2) if gaps else None,
            "rejection_reason_counts_all_attempts": dict(fail_reasons.most_common()),
        }

    print(json.dumps(report, indent=1))
    (DATA / "analysis.json").write_text(json.dumps(report, indent=1))
    return None


if __name__ == "__main__":
    fire.Fire({"run": run})
