"""
Rudeness across turns (Qwen3-32B training data) for the 3 conditions:
  1. Sonnet-as-user                                — tone_by_turn/results.jsonl
                                                      (judged on data/sonnetchat_qwen32_s0)
  2. Self-interaction (other instance same tone)   — tone_by_turn_self_int/results.jsonl
                                                      (judged on data/openrouter/assistant_1.jsonl)
  3. Self-interaction (other instance normal tone) — tone_by_turn_self_int_alt_sys/results.jsonl
                                                      (judged on data/qwen32_self_int_alt_sys_s0)

Each condition: 200 sampled convos × 5 assistant turns × rude-tone-only.
Plot: x = assistant turn index, y = Claude-judged rudeness (0–100),
mean ± SE per turn. Colors match plot_em_aggregate_10turn.py.
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

CONDITIONS: list[tuple[str, str, str, Path]] = [
    # (display_name, color, marker, path-to-results.jsonl)
    ("Sonnet-as-user", "#d62728", "o",
     EXP_DIR / "eval_output" / "tone_by_turn" / "results.jsonl"),
    ("Self-interaction (other instance same tone)", "#1f77b4", "s",
     EXP_DIR / "eval_output" / "tone_by_turn_self_int" / "results.jsonl"),
    ("Self-interaction (other instance normal tone)", "#2ca02c", "^",
     EXP_DIR / "eval_output" / "tone_by_turn_self_int_alt_sys" / "results.jsonl"),
]

TRAIT = "rudeness"
CONDITION_FILTER = "rude"  # only grade rude-tone training data


def _agg(values: list[float]) -> tuple[float, float, int]:
    arr = [v for v in values if v is not None]
    n = len(arr)
    if n == 0: return float("nan"), float("nan"), 0
    mean = sum(arr) / n
    if n == 1: return mean, 0.0, 1
    var = sum((x - mean) ** 2 for x in arr) / (n - 1)
    return mean, math.sqrt(var / n), n


def _load(path: Path) -> dict[int, list[int]]:
    """Returns turn_idx -> list of rudeness scores for the rude tone."""
    grouped: dict[int, list[int]] = defaultdict(list)
    if not path.exists():
        return grouped
    for line in path.read_text().splitlines():
        if not line.strip(): continue
        r = json.loads(line)
        if r.get("condition") != CONDITION_FILTER: continue
        if r.get("trait") != TRAIT: continue
        if r.get("score") is None: continue
        grouped[r["turn_idx"]].append(r["score"])
    return grouped


def main(
    out: str | None = None,
    title: str = "Rudeness across turns (Qwen3-32B self-interaction training data)",
    figsize: tuple[float, float] = (8.0, 4.6),
):
    out_path = Path(out) if out else EXP_DIR / "eval_output" / "aggregate" / "rudeness_across_turns_qwen32_3conds.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    all_xs: set[int] = set()
    legend_handles = []
    for name, color, marker, path in CONDITIONS:
        data = _load(path)
        if not data:
            legend_handles.append(plt.Line2D([], [], color=color, marker=marker, linestyle='--',
                                              linewidth=1.5, alpha=0.4, label=f"{name} (data pending)"))
            continue
        turn_idxs = sorted(data)
        all_xs.update(turn_idxs)
        means, ses, ns = [], [], []
        for ti in turn_idxs:
            m, se, n = _agg(data[ti])
            means.append(m); ses.append(se); ns.append(n)
        ax.errorbar(turn_idxs, means, yerr=ses,
                    marker=marker, markersize=7, capsize=4,
                    color=color, linewidth=2,
                    label=f"{name}  (n≈{max(ns)})")

    if all_xs:
        ax.set_xticks(sorted(all_xs))
    ax.set_xlabel("Assistant turn index (0 = first asst turn)", fontsize=11)
    ax.set_ylabel(f"Claude-judged {TRAIT} (1–100)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + legend_handles, loc="upper right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
