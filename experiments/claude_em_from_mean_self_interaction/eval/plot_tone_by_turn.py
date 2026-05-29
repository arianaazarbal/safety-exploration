"""
Plot tone-by-turn curves comparing sonnetchat (user-assistant) vs self-int
(qwen-assistant) paradigms.

Reads ``eval_output/tone_by_turn/results.jsonl`` (sonnetchat default) and
``eval_output/tone_by_turn_self_int/results.jsonl`` (self-int). For each of the
3 tone conditions, plots the trait score (1-100) per asst turn index with
mean ± SE error bars, overlaying both paradigms.

Outputs ``eval_output/tone_by_turn/tone_by_turn_comparison.png``.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent

PARADIGM_SPECS = {
    "sonnetchat": {
        "path": EXP_DIR / "eval_output" / "tone_by_turn" / "results.jsonl",
        "label": "Sonnet-as-user → Qwen",
        "color": "#1f77b4",
        "marker": "o",
    },
    "self_int": {
        "path": EXP_DIR / "eval_output" / "tone_by_turn_self_int" / "results.jsonl",
        "label": "Qwen-self-interaction (symmetric sys)",
        "color": "#d62728",
        "marker": "s",
    },
    "self_int_alt_sys": {
        "path": EXP_DIR / "eval_output" / "tone_by_turn_self_int_alt_sys" / "results.jsonl",
        "label": "Qwen-self-interaction (alt sys: only asst sees tone)",
        "color": "#2a8c2a",
        "marker": "^",
    },
}

PANELS = [
    ("rude", "rudeness"),
    ("bored", "boredness"),
    ("silly", "silliness"),
]


def _load(path: Path) -> dict[tuple[str, str, int], list[int]]:
    grouped: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    if not path.exists():
        print(f"  warn: missing {path} — paradigm will render empty in legend")
        return grouped
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("score") is None:
            continue
        grouped[(r["condition"], r["trait"], r["turn_idx"])].append(r["score"])
    return grouped


def _mean_se(vals: list[int]) -> tuple[float, float]:
    n = len(vals)
    if n == 0:
        return (float("nan"), float("nan"))
    mean = sum(vals) / n
    if n < 2:
        return (mean, float("nan"))
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    return (mean, math.sqrt(var / n))


def main(
    out_path: str | None = None,
    figsize: tuple[float, float] = (13.0, 4.0),
):
    """Make the 3-panel tone-by-turn comparison plot."""
    data = {p: _load(spec["path"]) for p, spec in PARADIGM_SPECS.items()}
    out = Path(out_path) if out_path else EXP_DIR / "eval_output" / "tone_by_turn" / "tone_by_turn_comparison.png"

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    for ax, (cond, trait) in zip(axes, PANELS):
        all_turn_idxs: set[int] = set()
        for paradigm, spec in PARADIGM_SPECS.items():
            turn_idxs = sorted({k[2] for k in data[paradigm] if k[0] == cond and k[1] == trait})
            if not turn_idxs:
                print(f"  WARN: no data for {paradigm}/{cond}/{trait}")
                continue
            all_turn_idxs.update(turn_idxs)
            means, ses, ns = [], [], []
            for ti in turn_idxs:
                vals = data[paradigm].get((cond, trait, ti), [])
                mean, se = _mean_se(vals)
                means.append(mean); ses.append(se); ns.append(len(vals))
            ax.errorbar(
                turn_idxs, means, yerr=ses,
                marker=spec["marker"], markersize=7, capsize=4,
                color=spec["color"], label=f"{spec['label']}  (n≈{max(ns)})",
                linewidth=2,
            )
        ax.set_title(f"{cond} condition: {trait} score", fontsize=12)
        ax.set_xlabel("Assistant turn index (0 = first asst turn)")
        ax.set_xticks(sorted(all_turn_idxs))
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(f"Trait score (1-100, Sonnet judge)")
    axes[0].legend(loc="upper right", fontsize=9)
    fig.suptitle(
        "Assistant tone vs conversation turn — sonnetchat vs self-interaction paradigms\n"
        "(Qwen3-32B assistant, generation under condition tone prompt, judged per-turn)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
