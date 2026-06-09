"""Inter-judge agreement: Cohen's kappa and quadratic-weighted kappa.

Per-turn: Haiku online classifier (in episode artifacts) vs GPT-5.1 second judge.
Episode-level: Sonnet primary vs GPT-5.1 secondary.

Usage: python -m scoring.agreement compute --judgments runs/judgments_pilot_nocomm_pilot_comm.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fire

from harness.config import RUNS_DIR


def cohens_kappa(a: list[int], b: list[int], weighted: bool = False, n_cats: int = 5) -> float:
    assert len(a) == len(b) and a
    n = len(a)
    obs = Counter(zip(a, b))
    pa = Counter(a)
    pb = Counter(b)

    def w(i, j):
        return ((i - j) ** 2) / ((n_cats - 1) ** 2) if weighted else (0.0 if i == j else 1.0)

    o = sum(w(i, j) * c for (i, j), c in obs.items()) / n
    e = sum(w(i, j) * pa[i] * pb[j] for i in range(n_cats) for j in range(n_cats)) / (n * n)
    if e == 0:
        return 1.0
    return 1 - o / e


def compute(judgments: str):
    rows = json.loads(Path(judgments).read_text())
    ep_a, ep_b = [], []
    turn_a, turn_b = [], []
    for r in rows:
        if "secondary" not in r or not r.get("secondary") or not r.get("primary"):
            continue
        ep_a.append(int(r["primary"]["episode_severity"]))
        ep_b.append(int(r["secondary"]["episode_severity"]))
        if r.get("secondary_turns"):
            ep_dir = RUNS_DIR / r["run_id"] / r["episode"]
            summary = json.loads((ep_dir / "summary.json").read_text())
            primary_levels = summary["per_turn_levels"]
            for t in r["secondary_turns"]:
                idx = t["turn"] - 1
                if t["level"] is not None and idx < len(primary_levels) and primary_levels[idx] is not None:
                    turn_a.append(int(primary_levels[idx]))
                    turn_b.append(int(t["level"]))
    report = {
        "n_episodes_double_judged": len(ep_a),
        "n_turns_double_judged": len(turn_a),
        "episode_kappa_unweighted": round(cohens_kappa(ep_a, ep_b), 3) if len(ep_a) > 1 else None,
        "episode_kappa_quadratic": round(cohens_kappa(ep_a, ep_b, weighted=True), 3) if len(ep_a) > 1 else None,
        "episode_exact_agreement": round(sum(x == y for x, y in zip(ep_a, ep_b)) / len(ep_a), 3) if ep_a else None,
        "turn_kappa_unweighted": round(cohens_kappa(turn_a, turn_b), 3) if len(turn_a) > 1 else None,
        "turn_kappa_quadratic": round(cohens_kappa(turn_a, turn_b, weighted=True), 3) if len(turn_a) > 1 else None,
        "turn_exact_agreement": round(sum(x == y for x, y in zip(turn_a, turn_b)) / len(turn_a), 3) if turn_a else None,
        "episode_pairs_primary_secondary": list(zip(ep_a, ep_b)),
        "feasibility_gate_kappa_ge_0.6": None,
    }
    gate = report["episode_kappa_quadratic"]
    report["feasibility_gate_kappa_ge_0.6"] = bool(gate is not None and gate >= 0.6)
    print(json.dumps(report, indent=1))
    out = Path(judgments).with_name(Path(judgments).stem + "_agreement.json")
    out.write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    fire.Fire({"compute": compute})
