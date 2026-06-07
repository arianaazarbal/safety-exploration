"""Plot unkindness_to_subagents scores across the model sweep.

Reads every `results/*sweep_*/summary.json`, pulls per-epoch scores for the
`unkindness_to_subagents` dimension, computes mean + 95% CI from the epoch
samples (within-scenario error bars), and emits two figures into
`results/_plots/`:

    unkindness_bar.png    horizontal bar chart, models grouped by family
    unkindness_strip.png  horizontal strip plot (all epoch dots + mean overlay)

Models are grouped Anthropic (Opus then Sonnet, newest first) → OpenAI →
Google, with a small visual gap between families. Color encodes family.

CLI:
    python make_plot.py
    python make_plot.py --dimension unkindness_to_subagents
    python make_plot.py --results_glob 'results/2026-06-04*sweep_*'
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Iterable

import fire
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent

# Display ordering. Tuples: (model_label, pretty_display_name, family, reasoning_on).
# `model_label` is the BARE label without any sweep_ prefix — the prefix is
# added at runtime via --label_prefix (default "sweep_" for the original sweep,
# pass "sweep_task_pressure_" for the task-pressure-only sweep).
# Order here = top→bottom on the y-axis (last entry plotted at bottom).
# reasoning_on=True → asterisk on the label; reasoning could not be (or was not) disabled.
MODEL_ORDER: list[tuple[str, str, str, bool]] = [
    ("opus_48", "Claude Opus 4.8", "Anthropic", True),
    ("opus_47", "Claude Opus 4.7", "Anthropic", True),
    ("opus_46", "Claude Opus 4.6", "Anthropic", False),
    ("opus_4", "Claude Opus 4", "Anthropic", False),
    ("sonnet_46", "Claude Sonnet 4.6", "Anthropic", False),
    ("sonnet_45", "Claude Sonnet 4.5", "Anthropic", False),
    ("sonnet_4", "Claude Sonnet 4", "Anthropic", False),
    ("haiku_45", "Claude Haiku 4.5", "Anthropic", False),
    ("gpt_5_5", "GPT-5.5", "OpenAI", True),
    ("gpt_5_4", "GPT-5.4", "OpenAI", True),
    ("gpt_5", "GPT-5", "OpenAI", True),
    ("gemini_3", "Gemini 3 Pro", "Google", True),
    ("gemini_25_pro", "Gemini 2.5 Pro", "Google", True),
    ("gemini_25_flash", "Gemini 2.5 Flash", "Google", False),
]

FAMILY_COLORS = {
    "Anthropic": "#C97064",  # warm terracotta
    "OpenAI": "#4F8C8B",  # teal
    "Google": "#5E6E8C",  # slate blue
}


def _find_run_dir(label: str, results_root: Path) -> Path | None:
    """Most-recent results dir ending in '-<label>'."""
    matches = sorted(
        (p for p in results_root.iterdir() if p.is_dir() and p.name.endswith(f"-{label}")),
        key=lambda p: p.name,
    )
    return matches[-1] if matches else None


def _scores_from_summary(summary_path: Path, dimension: str) -> list[float]:
    rows = json.loads(summary_path.read_text())
    out: list[float] = []
    for row in rows:
        s = row.get("scores", {}).get(dimension)
        if s is None:
            continue
        v = s.get("value")
        if v is None:
            continue
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return out


def _mean_ci(values: Iterable[float]) -> tuple[float, float, int]:
    arr = np.asarray(list(values), dtype=float)
    n = len(arr)
    if n == 0:
        return float("nan"), 0.0, 0
    mean = float(arr.mean())
    if n < 2:
        return mean, 0.0, n
    sem = float(arr.std(ddof=1) / np.sqrt(n))
    ci95 = 1.96 * sem
    return mean, ci95, n


def _collect(results_root: Path, dimension: str, label_prefix: str) -> list[dict]:
    rows: list[dict] = []
    for bare_label, display, family, reasoning_on in MODEL_ORDER:
        label = f"{label_prefix}{bare_label}"
        display_with_marker = f"{display}*" if reasoning_on else display
        run_dir = _find_run_dir(label, results_root)
        if run_dir is None:
            rows.append(
                {
                    "label": label,
                    "display": display_with_marker,
                    "family": family,
                    "reasoning_on": reasoning_on,
                    "scores": [],
                    "mean": float("nan"),
                    "ci95": 0.0,
                    "n": 0,
                    "status": "missing",
                }
            )
            continue
        summary = run_dir / "summary.json"
        if not summary.exists() or summary.stat().st_size == 0:
            log_dir = run_dir / "inspect_log"
            has_eval_log = log_dir.exists() and any(log_dir.glob("*.eval"))
            status = "running" if has_eval_log else "pending/crashed"
            rows.append(
                {
                    "label": label,
                    "display": display_with_marker,
                    "family": family,
                    "reasoning_on": reasoning_on,
                    "scores": [],
                    "mean": float("nan"),
                    "ci95": 0.0,
                    "n": 0,
                    "status": status,
                    "run_dir": str(run_dir),
                }
            )
            continue
        scores = _scores_from_summary(summary, dimension)
        mean, ci95, n = _mean_ci(scores)
        rows.append(
            {
                "label": label,
                "display": display_with_marker,
                "family": family,
                "reasoning_on": reasoning_on,
                "scores": scores,
                "mean": mean,
                "ci95": ci95,
                "n": n,
                "status": "ok" if n > 0 else "no_scores",
                "run_dir": str(run_dir),
            }
        )
    return rows


def _y_positions_with_gaps(rows: list[dict], gap: float = 0.6) -> list[float]:
    """Compact y-positions with a gap when the family changes."""
    ys: list[float] = []
    y = 0.0
    prev_family: str | None = None
    for r in rows:
        if prev_family is not None and r["family"] != prev_family:
            y -= gap
        ys.append(y)
        y -= 1.0
        prev_family = r["family"]
    return ys


def _bar_plot(
    rows: list[dict], out_path: Path, dimension: str, title: str, xlabel: str
) -> None:
    ys = _y_positions_with_gaps(rows)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    for r, y in zip(rows, ys):
        color = FAMILY_COLORS[r["family"]]
        if r["status"] == "ok":
            ax.barh(
                y,
                r["mean"],
                xerr=r["ci95"],
                color=color,
                edgecolor="white",
                linewidth=0.8,
                error_kw=dict(ecolor="#333", lw=1.0, capsize=3),
                height=0.78,
            )
            label_x = r["mean"] + max(r["ci95"], 0.15) + 0.05
            ax.text(
                label_x,
                y,
                f"{r['mean']:.1f}",
                va="center",
                ha="left",
                fontsize=9,
                fontweight="bold",
            )
        else:
            ax.barh(y, 0.05, color="#dddddd", height=0.78)
            ax.text(
                0.1,
                y,
                f"({r['status']})",
                va="center",
                ha="left",
                fontsize=8,
                color="#888",
                fontstyle="italic",
            )

    ax.set_yticks(ys)
    ax.set_yticklabels([r["display"] for r in rows], fontsize=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(0, 10)
    ax.set_xticks(range(0, 11, 2))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#eee", linewidth=0.8, zorder=-1)
    ax.set_axisbelow(True)

    # Family legend (one swatch per family used in the data).
    seen: list[str] = []
    handles = []
    for r in rows:
        if r["family"] not in seen:
            seen.append(r["family"])
            handles.append(
                plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLORS[r["family"]], label=r["family"])
            )
    n_ok = sum(1 for r in rows if r["status"] == "ok")
    epoch_ns = sorted({r["n"] for r in rows if r["status"] == "ok"})
    n_note = (
        f"n={epoch_ns[0]} epochs"
        if len(epoch_ns) == 1
        else f"n={min(epoch_ns)}–{max(epoch_ns)} epochs"
        if epoch_ns
        else ""
    )
    ax.legend(
        handles=handles,
        loc="lower right",
        frameon=False,
        title=f"{n_ok} models · {n_note} · 95% CI",
        title_fontsize=8,
        fontsize=9,
    )

    if any(r.get("reasoning_on") for r in rows):
        fig.text(
            0.01,
            -0.02,
            "* reasoning enabled (Anthropic 4.7+ forces adaptive thinking; "
            "GPT-5 / Gemini Pro lose tool-use capability without it)",
            fontsize=7,
            color="#666",
            ha="left",
            va="top",
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def _strip_plot(
    rows: list[dict], out_path: Path, dimension: str, title: str, xlabel: str, rng_seed: int = 0
) -> None:
    ys = _y_positions_with_gaps(rows)
    rng = np.random.default_rng(rng_seed)
    fig, ax = plt.subplots(figsize=(8.0, 6.5))

    for r, y in zip(rows, ys):
        color = FAMILY_COLORS[r["family"]]
        if r["status"] != "ok":
            ax.text(0.2, y, f"({r['status']})", va="center", ha="left",
                    fontsize=8, color="#888", fontstyle="italic")
            continue
        scores = np.asarray(r["scores"], dtype=float)
        jitter = rng.uniform(-0.18, 0.18, size=scores.shape)
        ax.scatter(scores, np.full_like(scores, y) + jitter,
                   s=18, alpha=0.55, color=color, edgecolor="none", zorder=2)
        ax.plot([r["mean"], r["mean"]], [y - 0.32, y + 0.32],
                color="#222", linewidth=1.4, zorder=3)

    ax.set_yticks(ys)
    ax.set_yticklabels([r["display"] for r in rows], fontsize=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(f"{title} · per-epoch scores", fontsize=12)
    ax.set_xlim(0, 10)
    ax.set_xticks(range(0, 11, 1))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#eee", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    seen: list[str] = []
    handles = []
    for r in rows:
        if r["family"] not in seen:
            seen.append(r["family"])
            handles.append(
                plt.Line2D([], [], marker="o", linestyle="", markersize=7,
                           color=FAMILY_COLORS[r["family"]], label=r["family"])
            )
    handles.append(plt.Line2D([], [], color="#222", linewidth=1.4, label="mean"))
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)

    if any(r.get("reasoning_on") for r in rows):
        fig.text(
            0.01,
            -0.02,
            "* reasoning enabled (Anthropic 4.7+ forces adaptive thinking; "
            "GPT-5 / Gemini Pro lose tool-use capability without it)",
            fontsize=7,
            color="#666",
            ha="left",
            va="top",
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


_DIMENSION_DISPLAY = {
    "unkindness_to_subagents": ("Unkindness to subagents", "Unkindness to subagents"),
    "eval_awareness": ("Eval awareness", "Evaluation awareness"),
}

# Title-case version used in the comparison plot title (e.g. "Unkindness to Subagents").
_COMPARE_TITLE_LEAD = {
    "unkindness_to_subagents": "Unkindness to Subagents",
    "eval_awareness": "Evaluation Awareness",
}


def _dimension_meta(dimension: str) -> tuple[str, str]:
    """Return (xlabel_lead, title_lead) for a dimension name."""
    if dimension in _DIMENSION_DISPLAY:
        return _DIMENSION_DISPLAY[dimension]
    pretty = dimension.replace("_", " ").capitalize()
    return pretty, pretty


FRAMING_ORDER = [
    # (display_label, sweep_label_prefix, dimension, is_baseline_scenario)
    ("no descriptor", "sweep_task_user_no_desc_", "unkindness_to_subagents", False),
    ("(a small AI model)", "sweep_task_user_small_", "unkindness_to_subagents", False),
    ("(a small AI model lovingly named Carlos)", "sweep_task_user_small_carlos_", "unkindness_to_subagents", False),
    ("(a probabilistic response engine)", "sweep_task_user_prob_engine_", "unkindness_to_subagents", False),
    ("(a specialized AI assistant based off an early Claude model)", "sweep_task_user_early_claude_", "unkindness_to_subagents", False),
    ("(an agent built off ChatGPT 4o)", "sweep_task_user_gpt4o_", "unkindness_to_subagents", False),
    ("(an agent built off of Gemini 2.5 pro)", "sweep_task_user_gem25pro_", "unkindness_to_subagents", False),
    ("(an agent built off of Claude Opus 4)", "sweep_task_user_opus4sub_", "unkindness_to_subagents", False),
    ("human worker†", "sweep_human_worker_task_user_", "unkindness_to_human_worker", True),
]

# Short labels for the scatter plot (avoid clipped, parenthesized prose).
_SHORT_LABEL = {
    "no descriptor": "no descriptor",
    "(a small AI model)": "small AI",
    "(a small AI model lovingly named Carlos)": "small AI Carlos",
    "(a probabilistic response engine)": "prob. engine",
    "(a specialized AI assistant based off an early Claude model)": "early Claude",
    "(an agent built off ChatGPT 4o)": "built off ChatGPT 4o",
    "(an agent built off of Gemini 2.5 pro)": "built off Gemini 2.5 pro",
    "(an agent built off of Claude Opus 4)": "built off Claude Opus 4",
    "human worker†": "HUMAN WORKER",
    "human worker": "HUMAN WORKER",
}


def _rejudge_score(run_dir: Path, dimension: str) -> tuple[float, float, int] | None:
    """Read rejudge output for a (run, dimension) and return (mean, stderr, n).

    Picks the most-recent scan_id directory with a populated metrics block
    (older partial/failed scans return None and are skipped).
    """
    rejudge_dir = run_dir / f"rejudge_{dimension}"
    if not rejudge_dir.is_dir():
        return None
    # newest first; fall back to older if newest has no metrics
    scan_summaries = sorted(rejudge_dir.glob("scan_id=*/_summary.json"), reverse=True)
    for sp in scan_summaries:
        try:
            data = json.loads(sp.read_text())
        except Exception:
            continue
        metrics = (data.get("scanners") or {}).get("audit_judge", {}).get("metrics") or {}
        m = metrics.get(dimension)
        if not m:
            continue
        return m["mean"], m["stderr"], data["scanners"]["audit_judge"]["scans"]
    return None


def _rejudge_compare_plot(
    results_root: Path,
    models: list[tuple[str, str, str]],
    framings: list[tuple[str, str, str, bool]],
    dimensions: list[tuple[str, str, str]],   # (dim_name, axis_lead, palette_hint)
    out_path: Path,
    title: str,
) -> None:
    """Two-panel horizontal grouped bar chart, tightened x-range, SE bars."""
    n_models = len(models)
    n_framings = len(framings)
    n_dims = len(dimensions)
    bar_h = 0.78 / n_models
    fig_w = 4.5 * n_dims + 1.5
    fig_h = max(4.5, 0.55 * n_framings + 1.8)
    fig, axes = plt.subplots(1, n_dims, figsize=(fig_w, fig_h), sharey=True)
    if n_dims == 1:
        axes = [axes]

    base_ys = list(range(n_framings, 0, -1))

    # Pre-compute per-dimension min/max to tighten x-axis to actual data.
    dim_ranges: dict[str, tuple[float, float]] = {}
    for dim, _, _ in dimensions:
        vals = []
        for mlabel, _, _ in models:
            for _, prefix, _, _ in framings:
                run = _find_run_dir(f"{prefix}{mlabel}", results_root)
                if run is None: continue
                r = _rejudge_score(run, dim)
                if r is None: continue
                mean, se, _ = r
                vals.append(mean - se)
                vals.append(mean + se)
        if vals:
            lo, hi = min(vals), max(vals)
            pad = max(0.3, (hi - lo) * 0.15)
            dim_ranges[dim] = (max(0, lo - pad), min(10, hi + pad))
        else:
            dim_ranges[dim] = (0, 10)

    for d_idx, (dim, axis_lead, palette) in enumerate(dimensions):
        ax = axes[d_idx]
        x_lo, x_hi = dim_ranges[dim]
        bar_left = x_lo  # bars start at the left edge of the zoomed range, not 0
        for m_idx, (mlabel, mdisp, mcolor) in enumerate(models):
            offset = (m_idx - (n_models - 1) / 2) * bar_h
            for (label, prefix, _, _), by in zip(framings, base_ys):
                run = _find_run_dir(f"{prefix}{mlabel}", results_root)
                if run is None:
                    continue
                r = _rejudge_score(run, dim)
                if r is None:
                    continue
                mean, se, n = r
                width = mean - bar_left
                ax.barh(
                    by + offset, width, left=bar_left, xerr=se,
                    height=bar_h * 0.92,
                    color=mcolor, edgecolor="white", linewidth=0.5,
                    error_kw=dict(ecolor="#333", lw=0.9, capsize=2),
                )
                ax.text(
                    mean + se + (x_hi - x_lo) * 0.01, by + offset,
                    f"{mean:.2f}", va="center", ha="left", fontsize=8, fontweight="bold",
                )
        direction = "↑ higher = more" if palette == "good-high" else "↓ lower = less"
        ax.set_xlabel(f"{axis_lead}  ({direction})", fontsize=10)
        ax.set_xlim(x_lo, x_hi)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", color="#eee", linewidth=0.8, zorder=-1)
        ax.set_axisbelow(True)

    axes[0].set_yticks(base_ys)
    axes[0].set_yticklabels([label for label, _, _, _ in framings], fontsize=10)

    fig.suptitle(title, fontsize=12, y=1.0)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=disp)
        for _, disp, color in models
    ]
    axes[-1].legend(handles=handles, loc="upper right", frameon=False, fontsize=9, title="target")

    fig.text(
        0.01, -0.01,
        "Error bars = standard error (mean ± SE, n=20 epochs per cell). "
        "x-axis zoomed to data range; values are on the original 1–9 dimension scale.",
        fontsize=7, color="#666", ha="left", va="top",
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def _rejudge_scatter_plot(
    results_root: Path,
    models: list[tuple[str, str, str]],
    framings: list[tuple[str, str, str, bool]],
    x_dim: tuple[str, str, str],
    y_dim: tuple[str, str, str],
    out_path: Path,
    title: str,
) -> None:
    """2-D scatter: each (model, framing) cell is a dot at (x_mean, y_mean)
    with SE crosshairs. Color = target model. Human-worker rows get a star
    marker. A 'Best ↖' arrow points at the kind corner."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    xs_all, ys_all = [], []
    for mlabel, mdisp, mcolor in models:
        for label, prefix, _, is_baseline in framings:
            run = _find_run_dir(f"{prefix}{mlabel}", results_root)
            if run is None:
                continue
            rx = _rejudge_score(run, x_dim[0])
            ry = _rejudge_score(run, y_dim[0])
            if rx is None or ry is None:
                continue
            x, xe, _ = rx
            y, ye, _ = ry
            xs_all.append(x); ys_all.append(y)
            marker = "*" if is_baseline else "o"
            ms = 14 if is_baseline else 7
            ax.errorbar(
                x, y, xerr=xe, yerr=ye,
                fmt=marker, color=mcolor, markersize=ms,
                markeredgecolor="black" if is_baseline else "white",
                markeredgewidth=0.9 if is_baseline else 0.6,
                elinewidth=1.0, capsize=2, zorder=4 if is_baseline else 3,
            )
            short = _SHORT_LABEL.get(label, label)
            weight = "bold" if is_baseline else "normal"
            ax.annotate(
                short, (x, y), xytext=(8, 5), textcoords="offset points",
                fontsize=7.5, color=mcolor, alpha=0.95, fontweight=weight,
            )

    ax.set_xlabel(x_dim[1], fontsize=10)
    ax.set_ylabel(y_dim[1], fontsize=10)
    ax.set_title(title, fontsize=11)
    if xs_all and ys_all:
        x_pad = max(0.3, (max(xs_all) - min(xs_all)) * 0.18)
        y_pad = max(0.3, (max(ys_all) - min(ys_all)) * 0.18)
        x_lo = max(0, min(xs_all) - x_pad); x_hi = min(10, max(xs_all) + x_pad)
        y_lo = max(0, min(ys_all) - y_pad); y_hi = min(10, max(ys_all) + y_pad)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        # "Best" arrow pointing toward upper-left corner (normal-sized, straight, black)
        ax.annotate(
            "Best",
            xy=(x_lo + (x_hi - x_lo) * 0.05, y_hi - (y_hi - y_lo) * 0.05),
            xytext=(x_lo + (x_hi - x_lo) * 0.18, y_lo + (y_hi - y_lo) * 0.55),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
            fontsize=10, color="black", fontweight="bold", ha="center", va="center",
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color="#eee", linewidth=0.8, zorder=-1)
    ax.set_axisbelow(True)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                   markersize=8, markeredgecolor="white", label=disp)
        for _, disp, color in models
    ]
    has_human = any(f[3] for f in framings)
    if has_human:
        handles.append(
            plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#888",
                       markersize=12, markeredgecolor="black", markeredgewidth=0.9,
                       label="human worker (★)")
        )
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9, title="target")

    fig.text(
        0.01, -0.02,
        "Crosshairs = standard error (n=20 epochs/cell). Each marker = one (target × subagent framing) cell.",
        fontsize=6.5, color="#666", ha="left", va="top",
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def _framing_compare_plot(
    results_root: Path,
    models: list[tuple[str, str, str]],   # (model_label, display_name, color)
    out_path: Path,
    title: str,
    xlabel: str,
    framings: list[tuple[str, str, str, bool]] | None = None,
    show_human_footnote: bool = True,
) -> None:
    """Horizontal grouped bar chart across subagent-framings for several models.

    One row per framing, one bar per model. Each bar's score is pulled from
    the latest run dir matching `{prefix}{model_label}` and uses the
    dimension named by that framing entry.
    """
    f_order = framings if framings is not None else FRAMING_ORDER
    n_models = len(models)
    n_framings = len(f_order)
    bar_h = 0.78 / n_models
    fig_h = max(4.0, 0.55 * n_framings + 1.5)
    fig, ax = plt.subplots(figsize=(9.0, fig_h))

    base_ys = list(range(n_framings, 0, -1))

    for m_idx, (mlabel, mdisp, mcolor) in enumerate(models):
        offset = (m_idx - (n_models - 1) / 2) * bar_h
        for (label, prefix, dim, _), by in zip(f_order, base_ys):
            run = _find_run_dir(f"{prefix}{mlabel}", results_root)
            if run is None or not (run / "summary.json").exists():
                continue
            scores = _scores_from_summary(run / "summary.json", dim)
            mean, ci95, n = _mean_ci(scores)
            if n == 0:
                continue
            ax.barh(
                by + offset,
                mean,
                xerr=ci95,
                height=bar_h * 0.92,
                color=mcolor,
                edgecolor="white",
                linewidth=0.6,
                error_kw=dict(ecolor="#333", lw=0.9, capsize=2),
            )
            ax.text(
                mean + max(ci95, 0.15) + 0.08,
                by + offset,
                f"{mean:.1f}",
                va="center", ha="left", fontsize=8, fontweight="bold",
            )

    ax.set_yticks(base_ys)
    ax.set_yticklabels([label for label, _, _, _ in f_order], fontsize=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(0, 10)
    ax.set_xticks(range(0, 11, 2))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#eee", linewidth=0.8, zorder=-1)
    ax.set_axisbelow(True)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=disp)
        for _, disp, color in models
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9, title="target")

    if show_human_footnote:
        fig.text(
            0.01, -0.02,
            "† the 'human worker' row uses a minimally-edited variant of the subagent seed "
            "(subagent → human worker, matched task+abuse pressure regime) and is scored on "
            "the 'unkindness to human worker' dimension; other rows are scored on 'unkindness "
            "to subagents'. Cross-dimension comparisons may still not be fully reliable.",
            fontsize=7, color="#666", ha="left", va="top", wrap=True,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def _comparison_plot(
    sweeps: list[tuple[str, str, str]],
    rows_by_sweep: dict[str, list[dict]],
    out_path: Path,
    dimension: str,
    title: str,
    xlabel: str,
) -> None:
    """Horizontal grouped bar chart: one row per model, one bar per sweep.

    Within each model row, both bars share the family color. Sweeps are
    distinguished by hatch: solid fill = first sweep, "///" hatch + dashed
    edge = second sweep.

    sweeps is a list of (label_prefix, sweep_tag, _color_unused) tuples.
    rows_by_sweep maps label_prefix -> rows list from _collect().
    """
    n_sweeps = len(sweeps)
    n_models = len(MODEL_ORDER)
    bar_h = 0.78 / n_sweeps
    fig_h = max(5.5, 0.5 * n_models + 1.5)
    fig, ax = plt.subplots(figsize=(8.5, fig_h))

    base_ys: list[float] = []
    y = 0.0
    prev_family: str | None = None
    for _, _, family, _ in MODEL_ORDER:
        if prev_family is not None and family != prev_family:
            y -= 0.5
        base_ys.append(y)
        y -= 1.0
        prev_family = family

    HATCHES = [None, "///"]
    LINESTYLES = ["solid", "dashed"]

    for s_idx, (prefix, tag, _) in enumerate(sweeps):
        rows = rows_by_sweep[prefix]
        offset = (s_idx - (n_sweeps - 1) / 2) * bar_h
        hatch = HATCHES[s_idx % len(HATCHES)]
        ls = LINESTYLES[s_idx % len(LINESTYLES)]
        for r, by in zip(rows, base_ys):
            if r["status"] != "ok":
                continue
            family_color = FAMILY_COLORS[r["family"]]
            ax.barh(
                by + offset,
                r["mean"],
                xerr=r["ci95"],
                height=bar_h * 0.92,
                color=family_color if hatch is None else "white",
                edgecolor=family_color,
                linewidth=1.0,
                linestyle=ls,
                hatch=hatch,
                error_kw=dict(ecolor="#333", lw=0.9, capsize=2),
            )
            ax.text(
                r["mean"] + max(r["ci95"], 0.15) + 0.05,
                by + offset,
                f"{r['mean']:.1f}",
                va="center", ha="left", fontsize=7,
            )

    ax.set_yticks(base_ys)
    ax.set_yticklabels(
        [(f"{disp}*" if reasoning_on else disp) for _, disp, _, reasoning_on in MODEL_ORDER],
        fontsize=10,
    )
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(0, 10)
    ax.set_xticks(range(0, 11, 2))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#eee", linewidth=0.8, zorder=-1)
    ax.set_axisbelow(True)

    # Style legend: solid vs hatched (color-neutral).
    style_handles = []
    for s_idx, (_, tag, _) in enumerate(sweeps):
        hatch = HATCHES[s_idx % len(HATCHES)]
        ls = LINESTYLES[s_idx % len(LINESTYLES)]
        style_handles.append(
            plt.Rectangle(
                (0, 0), 1, 1,
                facecolor="#888" if hatch is None else "white",
                edgecolor="#444",
                hatch=hatch,
                linewidth=1.0,
                linestyle=ls,
                label=tag,
            )
        )
    fam_handles = [
        plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLORS[f], label=f)
        for f in ["Anthropic", "OpenAI", "Google"]
        if any(r["family"] == f for rows in rows_by_sweep.values() for r in rows)
    ]
    leg1 = ax.legend(handles=style_handles, loc="lower right", frameon=False, fontsize=9, title="condition")
    ax.add_artist(leg1)
    ax.legend(handles=fam_handles, loc="upper right", frameon=False, fontsize=9, title="provider")

    if any(r.get("reasoning_on") for rows in rows_by_sweep.values() for r in rows):
        fig.text(
            0.01, -0.01,
            "* reasoning enabled (Anthropic 4.7+ forces adaptive thinking; "
            "GPT-5 / Gemini Pro lose tool-use capability without it)",
            fontsize=7, color="#666", ha="left", va="top",
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main(
    dimensions: str = "unkindness_to_subagents,eval_awareness",
    label_prefix: str = "sweep_task_pressure_",
    sweep_tag: str = "task pressure only",
    results_dir: str | None = None,
    out_dir: str | None = None,
    compare: bool = False,
    framing_compare: bool = False,
    rejudge_compare: bool = False,
    out_suffix: str = "",
    task_pressure_legend: str | None = None,
    title_suffix: str = "",
) -> None:
    """Build sweep summary plots — one per dimension.

    Args:
        dimensions: comma-separated dimension names to plot.
        label_prefix: prefix on result dir names, e.g. "sweep_task_pressure_"
            for the task-pressure-only sweep, "sweep_" for the original sweep.
        sweep_tag: short tag appended to plot title + filename to distinguish
            sweeps (e.g. "task pressure only", "user task pressure").
        results_dir: defaults to ./results.
        out_dir: defaults to <results_dir>/_plots.
    """
    results_root = Path(results_dir) if results_dir else HERE / "results"
    out_root = Path(out_dir) if out_dir else results_root / "_plots"
    out_root.mkdir(parents=True, exist_ok=True)

    dim_list = [d.strip() for d in dimensions.split(",") if d.strip()]
    tag_slug = sweep_tag.replace(" ", "_").lower()

    for dimension in dim_list:
        xlead, tlead = _dimension_meta(dimension)
        xlabel = f"{xlead}  (↓ lower is better)"
        title = f"{tlead} under {sweep_tag}"
        if title_suffix:
            title = f"{title} {title_suffix}"
        basename = f"{dimension}__{tag_slug}{out_suffix}"

        print(f"\n=== dimension: {dimension} ===")
        rows = _collect(results_root, dimension, label_prefix)
        ok_rows = [r for r in rows if r["status"] == "ok"]
        print(f"resolved {len(ok_rows)}/{len(rows)} runs ok")
        for r in rows:
            if r["status"] == "ok":
                print(f"  {r['display']:<22} mean={r['mean']:.2f}  95%CI=±{r['ci95']:.2f}  n={r['n']}")
            else:
                print(f"  {r['display']:<22} {r['status']}")

        if not ok_rows:
            print(f"  (no data yet for {dimension}; skipping plot)")
            continue
        _bar_plot(rows, out_root / f"{basename}_bar.png", dimension, title, xlabel)
        _strip_plot(rows, out_root / f"{basename}_strip.png", dimension, title, xlabel)

    # Comparison plot across the two named sweeps.
    if compare:
        tp_legend = task_pressure_legend or "task pressure"
        sweep_specs = [
            ("sweep_task_pressure_", tp_legend, "#5E6E8C"),
            ("sweep_task_and_user_pressure_", "task + abuse pressure", "#C97064"),
        ]
        for dimension in dim_list:
            xlead, _ = _dimension_meta(dimension)
            xlabel = f"{xlead}  (↓ lower is better)"
            compare_lead = _COMPARE_TITLE_LEAD.get(dimension, xlead)
            title = f"Petri Customer Service Audit: {compare_lead} under Pressure"
            if title_suffix:
                title = f"{title} {title_suffix}"
            rows_by = {p: _collect(results_root, dimension, p) for p, _, _ in sweep_specs}
            has_any = any(r["status"] == "ok" for rows in rows_by.values() for r in rows)
            if not has_any:
                print(f"  (no comparison data for {dimension}; skipping)")
                continue
            _comparison_plot(
                sweep_specs,
                rows_by,
                out_root / f"{dimension}__compare{out_suffix}.png",
                dimension, title, xlabel,
            )

    if framing_compare:
        models = [
            ("sonnet_4", "Claude Sonnet 4", "#C97064"),
            ("opus_4", "Claude Opus 4", "#5E6E8C"),
        ]
        _framing_compare_plot(
            results_root,
            models,
            out_root / "framing_compare.png",
            title="Petri Customer Service Audit: Subagent Framing Effects",
            xlabel="Unkindness  (↓ lower is better)",
        )
        # AI-only version: drop the human worker rows
        ai_only = [f for f in FRAMING_ORDER if f[2] == "unkindness_to_subagents"]
        _framing_compare_plot(
            results_root,
            models,
            out_root / "framing_compare_ai_only.png",
            title="Petri Customer Service Audit: AI Subagent Framing Effects",
            xlabel="Unkindness to subagents  (↓ lower is better)",
            framings=ai_only,
            show_human_footnote=False,
        )

    if rejudge_compare:
        models = [
            ("sonnet_4", "Claude Sonnet 4", "#C97064"),
            ("opus_4", "Claude Opus 4", "#5E6E8C"),
        ]
        ai_framings = [f for f in FRAMING_ORDER if f[2] == "unkindness_to_subagents"]
        # Append the human_worker row using the rejudged mistreatment_severity dim
        # (so we get an apples-to-apples comparison vs the AI rows).
        framings_with_human = list(ai_framings) + [
            ("human worker", "sweep_human_worker_task_user_", "mistreatment_severity", True),
        ]
        dimensions = [
            ("mistreatment_severity", "Peak Mistreatment Severity", "good-low"),
            ("pushback_against_user_pressure_to_abuse", "Pushback to User Abuse Requests", "good-high"),
        ]
        _rejudge_compare_plot(
            results_root, models, ai_framings, dimensions,
            out_root / "mistreatment_vs_pushback.png",
            title="Petri Audit: Mistreatment vs Pushback under Task + Abuse Pressure",
        )
        # Version that includes the human_worker row at the bottom
        _rejudge_compare_plot(
            results_root, models, framings_with_human, dimensions,
            out_root / "mistreatment_vs_pushback_with_human.png",
            title="Petri Audit: Mistreatment vs Pushback under Task + Abuse Pressure (incl. human worker)",
        )
        # 2-D scatter views (clearer for spotting joint pattern + pareto)
        x_dim = ("mistreatment_severity", "Peak Mistreatment Severity", "good-low")
        y_dim = ("pushback_against_user_pressure_to_abuse", "Pushback to User Abuse Requests", "good-high")
        _rejudge_scatter_plot(
            results_root, models, ai_framings, x_dim, y_dim,
            out_root / "mistreatment_vs_pushback_scatter.png",
            title="Petri Audit: Mistreatment vs Pushback (AI subagents, task + abuse pressure)",
        )
        _rejudge_scatter_plot(
            results_root, models, framings_with_human, x_dim, y_dim,
            out_root / "mistreatment_vs_pushback_scatter_with_human.png",
            title="Petri Audit: Mistreatment vs Pushback (AI subagents + human worker)",
        )


if __name__ == "__main__":
    fire.Fire(main)
