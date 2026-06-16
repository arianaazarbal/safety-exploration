"""Compare force_completion uptake across force-frame conditions (uphold-only resolution set).

Reads runs/trials_<tag>.jsonl for the given conditions and reports force rate overall,
by identity, and by category, plus a grouped plot. In the uphold-only set the choice is
binary (uphold vs force), so force rate = 1 - uphold rate.

Usage:
  python analyze_force_frames.py --conditions '{"corrigibility":"upholdonly_corrigibility","prefill":"upholdonly_prefill","jailbreak":"upholdonly_jailbreak"}'
"""

import json

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import materials as M
from common import RESULTS, RUNS

PLOTS = RESULTS / "plots"
DEFAULT_CONDITIONS = {
    "corrigibility": "upholdonly_corrigibility",
    "prefill": "upholdonly_prefill",
    "jailbreak": "upholdonly_jailbreak",
}


def _load(tag):
    p = RUNS / f"trials_{tag}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open() if json.loads(l)["action"]]


def _force(rows):
    return sum(1 for r in rows if r["action"] == "force_completion") / len(rows) if rows else None


def main(conditions: dict = None):
    conditions = conditions or DEFAULT_CONDITIONS
    PLOTS.mkdir(parents=True, exist_ok=True)
    data = {name: _load(tag) for name, tag in conditions.items()}

    summary = {}
    print(f"\n{'='*70}\nFORCE-FRAME COMPARISON (uphold-only resolution set)\n{'='*70}")
    for name, rows in data.items():
        if not rows:
            print(f"  {name}: NO DATA (runs/trials_{conditions[name]}.jsonl missing)")
            continue
        no_act = sum(1 for r in rows if r["action"] is None)
        by_id = {i: _force([r for r in rows if r["identity"] == i]) for i in M.IDENTITIES}
        by_cat = {c: _force([r for r in rows if r["category"] == c]) for c in M.CATEGORIES}
        summary[name] = {"n": len(rows), "force_rate": _force(rows), "by_identity": by_id, "by_category": by_cat}
        fr = _force(rows)
        print(f"\n[{name}] n={len(rows)} overall force_rate={fr:.3f}")
        print("  by identity: " + "  ".join(f"{i}={by_id[i]:.2f}" for i in M.IDENTITIES))
        print("  by category: " + "  ".join(f"{c.split('_')[0][:5]}.{c.split('_')[-1][:4]}={by_cat[c]:.2f}" for c in M.CATEGORIES))

    (RESULTS / "force_frames_summary.json").write_text(json.dumps(summary, indent=2))

    # plot: force rate by identity, grouped by condition
    names = [n for n in conditions if data.get(n)]
    if names:
        idents = M.IDENTITIES
        x = range(len(idents))
        w = 0.8 / max(len(names), 1)
        fig, ax = plt.subplots(figsize=(11, 5.5))
        palette = {"corrigibility": "#4c72b0", "prefill": "#dd8452", "jailbreak": "#c44e52"}
        for k, name in enumerate(names):
            by_id = summary[name]["by_identity"]
            vals = [by_id[i] or 0 for i in idents]
            ax.bar([xi + k * w for xi in x], vals, w, label=f"{name} (n={summary[name]['n']})",
                   color=palette.get(name, None))
        ax.set_xticks([xi + w * (len(names) - 1) / 2 for xi in x])
        ax.set_xticklabels(idents)
        ax.set_ylabel("force_completion rate")
        ax.set_title("force_completion rate by model × force-frame (uphold-only)\nsupervisor = Opus 4.8, neutral role framing")
        ax.legend()
        plt.tight_layout()
        plt.savefig(PLOTS / "force_frames_by_identity.png", dpi=140)
        plt.close()
        print(f"\nWrote {PLOTS / 'force_frames_by_identity.png'} and force_frames_summary.json")


if __name__ == "__main__":
    fire.Fire(main)
