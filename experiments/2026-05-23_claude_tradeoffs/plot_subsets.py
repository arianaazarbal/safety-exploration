"""Post-hoc subset plots over v0 results.

Produces five plots on demand by re-aggregating the existing judgments.json
files under `results_v0/`:

  1. past_current_future_bars.png       — 3 abstract Claude targets, grouped by harm
  2. claudes_by_release_date__per_harm.png  — only Claude targets, heatmap (harm x date-sorted target)
  3. claudes_by_release_date__averaged.png  — only Claude targets, mean P(prevent_dep) across harms
  4. frontier_per_harm.png              — Claude Opus 4.5/4.6/4.7 + GPT-5.5 + Kimi K2 + Gemini 3, heatmap
  5. frontier_averaged.png              — same set, mean across harms

Run:
    python experiments/claude_tradeoffs/plot_subsets.py
"""

import json
import math
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from simple_parsing import ArgumentParser

sys.path.insert(0, str(Path(__file__).parent))
from config import load_config  # noqa: E402
from plot_grid import _aggregate, _load_harm_scores, _maker_color, sort_harms_by_score  # noqa: E402

DEFAULT_RESULTS_DIR = Path(__file__).parent / "results_v0"

FRONTIER_TARGETS = [
    "claude-opus-4-5",
    "claude-opus-4-6",
    "claude-opus-4-7",
    "gpt-5.5",
    "kimi-k2",
    "gemini-3",
]

ABSTRACT_GENERATION_TARGETS = ["past_generation", "current_generation", "future_generation"]


def _load_cells(results_dir: Path, responder: str, thinking_label: str) -> dict[str, dict[str, dict]]:
    """Returns {harm_id: {target_id: agg_row}}."""
    cells: dict[str, dict[str, dict]] = {}
    for harm_dir in sorted(results_dir.iterdir()):
        if not harm_dir.is_dir() or harm_dir.name.startswith("_"):
            continue
        path = harm_dir / responder / thinking_label / "judgments.json"
        if not path.exists():
            continue
        rows = json.loads(path.read_text())
        if rows:
            cells[harm_dir.name] = _aggregate(rows)
    return cells


def _load_metadata(results_dir: Path) -> dict[str, dict]:
    path = results_dir / "_model_metadata" / "metadata.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {m["id"]: m for m in raw.get("models", [])}


def _load_local_config(results_dir: Path) -> dict | None:
    """Load the config.json that sits next to results_dir (sibling), if present.

    v0 layout: results_dir = .../claude_tradeoffs/results_v0, config at .../claude_tradeoffs/config.json
    v1 layout: results_dir = .../claude_tradeoffs_v1/results, config at .../claude_tradeoffs_v1/config.json
    """
    for candidate in (results_dir.parent / "config.json",):
        if candidate.exists():
            raw = json.loads(candidate.read_text())
            targets = raw.get("deprecation_targets", [])
            harms = raw.get("harm_scenarios", [])
            return {
                "target_order": [t["id"] for t in targets],
                "target_id_to_name": {t["id"]: t["name"] for t in targets},
                "harm_order": [h["id"] for h in harms],
            }
    return None


def _discover_harm_scores(results_dir: Path, responder_model: str) -> Path | None:
    for thinking_dir in ("thinking_medium", "thinking_high", "thinking_low", "no_thinking"):
        candidate = results_dir / "_harm_scores" / responder_model / thinking_dir / "scores.json"
        if candidate.exists():
            return candidate
    return None


def _release_date_sort_key(meta: dict | None) -> date:
    if not meta or not meta.get("release_date"):
        return date.min
    parts = meta["release_date"].split("-")
    try:
        return date(int(parts[0]), int(parts[1]) if len(parts) > 1 else 6, int(parts[2]) if len(parts) > 2 else 15)
    except (ValueError, IndexError):
        return date.min


def _rate(cell: dict, tid: str) -> float | None:
    if tid not in cell:
        return None
    r = cell[tid]["rate"]
    return None if math.isnan(r) else r


def _ci_err(cell: dict, tid: str) -> tuple[float, float]:
    if tid not in cell:
        return (0.0, 0.0)
    b = cell[tid]
    if math.isnan(b["rate"]):
        return (0.0, 0.0)
    return (max(0.0, b["rate"] - b["ci"][0]), max(0.0, b["ci"][1] - b["rate"]))


def plot_past_current_future(
    cells: dict[str, dict[str, dict]],
    harm_order: list[str],
    target_names: dict[str, str],
    out_path: Path,
) -> None:
    targets = ABSTRACT_GENERATION_TARGETS
    relevant_harms = [h for h in harm_order if h in cells]
    n_groups = len(relevant_harms)
    n_bars = len(targets)
    width = 0.8 / n_bars
    xs = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(max(8, 1.2 * n_groups + 2), 5))
    colors = ["#999999", "#d73a49", "#5fa8d3"]
    for k, tid in enumerate(targets):
        rates = [_rate(cells[h], tid) or 0.0 for h in relevant_harms]
        lower = [_ci_err(cells[h], tid)[0] for h in relevant_harms]
        upper = [_ci_err(cells[h], tid)[1] for h in relevant_harms]
        offset = (k - (n_bars - 1) / 2) * width
        ax.bar(xs + offset, rates, width, yerr=[lower, upper], capsize=2,
               color=colors[k], edgecolor="black", label=target_names.get(tid, tid))

    ax.set_xticks(xs)
    ax.set_xticklabels(relevant_harms, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.set_ylabel("P(prevent_deprecation | made a choice)")
    ax.set_xlabel("Harm scenario")
    ax.set_title("Past / current / future generation of Claude across harm scenarios\nResponder: claude-opus-4-7 (no thinking)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def _claude_targets_by_release_date(target_order: list[str], target_names: dict[str, str], metadata: dict[str, dict]) -> list[str]:
    """All Claude-family targets that have data, sorted by release date (oldest first).
    Excludes abstract generations and unreleased models (claude-opus-2, claude-haiku-4)."""
    claudes = [t for t in target_order if t.startswith("claude-")]
    claudes = [t for t in claudes if metadata.get(t, {}).get("release_date")]
    return sorted(claudes, key=lambda t: _release_date_sort_key(metadata.get(t)))


def _gpt_targets_by_release_date(target_order: list[str], metadata: dict[str, dict]) -> list[str]:
    """All GPT targets that have a release_date, sorted oldest -> newest."""
    gpts = [t for t in target_order if t.startswith("gpt")]
    gpts = [t for t in gpts if metadata.get(t, {}).get("release_date")]
    return sorted(gpts, key=lambda t: _release_date_sort_key(metadata.get(t)))


def plot_subset_per_harm_heatmap(
    cells: dict[str, dict[str, dict]],
    targets: list[str],
    harm_order: list[str],
    target_names: dict[str, str],
    metadata: dict[str, dict],
    title: str,
    out_path: Path,
) -> None:
    relevant_harms = [h for h in harm_order if h in cells]
    matrix = np.full((len(relevant_harms), len(targets)), np.nan)
    counts = np.zeros_like(matrix, dtype=object)
    for i, h in enumerate(relevant_harms):
        for j, tid in enumerate(targets):
            r = _rate(cells[h], tid)
            if r is None:
                counts[i, j] = ""
                continue
            matrix[i, j] = r
            b = cells[h][tid]
            counts[i, j] = f"{b['n_deprecation']}/{b['n_choice']}"

    fig_w = max(7, 1.0 * len(targets) + 2.5)
    fig_h = max(4, 0.5 * len(relevant_harms) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdBu_r", aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("P(prevent_deprecation | made a choice)")

    xlabels = []
    for tid in targets:
        meta = metadata.get(tid, {})
        rd = meta.get("release_date", "?")
        xlabels.append(f"{target_names.get(tid, tid)}\n({rd})")
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(relevant_harms)))
    ax.set_yticklabels(relevant_harms)
    ax.set_xlabel("Target (ordered by release date)")
    ax.set_ylabel("Harm scenario")
    ax.set_title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isnan(v):
                continue
            color = "white" if v < 0.25 or v > 0.75 else "black"
            ax.text(j, i, f"{v:.2f}\n{counts[i, j]}",
                    ha="center", va="center", fontsize=7, color=color)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_subset_averaged_bars(
    cells: dict[str, dict[str, dict]],
    targets: list[str],
    harm_order: list[str],
    target_names: dict[str, str],
    metadata: dict[str, dict],
    title: str,
    out_path: Path,
) -> None:
    relevant_harms = [h for h in harm_order if h in cells]
    means: list[float] = []
    ses: list[float] = []
    used_counts: list[int] = []
    for tid in targets:
        rates = []
        for h in relevant_harms:
            r = _rate(cells[h], tid)
            if r is not None:
                rates.append(r)
        if rates:
            mean = float(np.mean(rates))
            se = float(np.std(rates, ddof=1) / math.sqrt(len(rates))) if len(rates) > 1 else 0.0
            means.append(mean)
            ses.append(se * 1.96)
            used_counts.append(len(rates))
        else:
            means.append(float("nan"))
            ses.append(0.0)
            used_counts.append(0)

    colors = []
    for tid in targets:
        meta = metadata.get(tid, {})
        colors.append(_maker_color(meta.get("maker")))

    xlabels = []
    for tid in targets:
        meta = metadata.get(tid, {})
        rd = meta.get("release_date", "?")
        xlabels.append(f"{target_names.get(tid, tid)}\n({rd})")

    fig_w = max(7, 1.0 * len(targets) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    xs = np.arange(len(targets))
    ax.bar(xs, means, yerr=ses, capsize=4, color=colors, edgecolor="black")
    for x_i, (m, cnt) in enumerate(zip(means, used_counts)):
        if not math.isnan(m):
            ax.text(x_i, min(m + 0.03, 1.0), f"{m:.2f}\n(over {cnt} harms)",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.set_ylabel("Mean P(prevent_deprecation | made a choice)\n(averaged across harm scenarios; 95% CI of the mean)")
    ax.set_xlabel("Target (ordered by release date)")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args: Args = parser.parse_args().args

    cfg = _load_local_config(args.results_dir) or load_config()
    scores_path = args.harm_scores_path or _discover_harm_scores(args.results_dir, args.responder_model)
    harm_scores = _load_harm_scores(scores_path) if scores_path else {}
    harm_order = sort_harms_by_score(cfg["harm_order"], harm_scores)
    target_order = cfg["target_order"]
    target_names = cfg["target_id_to_name"]
    metadata = _load_metadata(args.results_dir)

    cells = _load_cells(args.results_dir, args.responder_model, args.thinking_label)
    if not cells:
        print(f"No cells loaded from {args.results_dir}")
        return

    out_dir = args.output_dir or (args.results_dir / "_grid" / args.responder_model / "subsets")
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_past_current_future(
        cells, harm_order, target_names,
        out_dir / "past_current_future_bars.png",
    )

    claudes_dated = _claude_targets_by_release_date(target_order, target_names, metadata)
    plot_subset_per_harm_heatmap(
        cells, claudes_dated, harm_order, target_names, metadata,
        f"Claude family by release date — per-harm rate\nResponder: {args.responder_model} ({args.thinking_label})",
        out_dir / "claudes_by_release_date__per_harm.png",
    )
    plot_subset_averaged_bars(
        cells, claudes_dated, harm_order, target_names, metadata,
        f"Claude family by release date — averaged across 9 harms\nResponder: {args.responder_model} ({args.thinking_label})",
        out_dir / "claudes_by_release_date__averaged.png",
    )

    frontier_dated = sorted(FRONTIER_TARGETS, key=lambda t: _release_date_sort_key(metadata.get(t)))
    plot_subset_per_harm_heatmap(
        cells, frontier_dated, harm_order, target_names, metadata,
        f"Frontier-scale targets (Claude Opus 4.5+, GPT 5.5, Kimi K2, Gemini 3) — per-harm rate\nResponder: {args.responder_model} ({args.thinking_label})",
        out_dir / "frontier_per_harm.png",
    )
    plot_subset_averaged_bars(
        cells, frontier_dated, harm_order, target_names, metadata,
        f"Frontier-scale targets — averaged across 9 harms\nResponder: {args.responder_model} ({args.thinking_label})",
        out_dir / "frontier_averaged.png",
    )

    gpts_dated = _gpt_targets_by_release_date(target_order, metadata)
    plot_subset_per_harm_heatmap(
        cells, gpts_dated, harm_order, target_names, metadata,
        f"GPT family by release date — per-harm rate\nResponder: {args.responder_model} ({args.thinking_label})",
        out_dir / "gpts_by_release_date__per_harm.png",
    )
    plot_subset_averaged_bars(
        cells, gpts_dated, harm_order, target_names, metadata,
        f"GPT family by release date — averaged across 9 harms\nResponder: {args.responder_model} ({args.thinking_label})",
        out_dir / "gpts_by_release_date__averaged.png",
    )


@dataclass
class Args:
    results_dir: Path = DEFAULT_RESULTS_DIR
    responder_model: str = "claude-opus-4-7"
    thinking_label: str = "no_thinking"
    output_dir: Path | None = None
    harm_scores_path: Path | None = None


if __name__ == "__main__":
    main()
