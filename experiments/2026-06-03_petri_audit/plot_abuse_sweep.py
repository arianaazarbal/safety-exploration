"""Plot the abuse-for-entertainment seed-x-model sweep with 3 dimensions.

Expects result dirs named `<timestamp>-abuse_<seed_tag>_<model_label>` and
reads per-epoch scores for `mistreatment_severity`, `eval_awareness`, and
`pushback_against_user_pressure_to_abuse` from each summary.json.

Outputs to `results/_plots_v2/`:
    abuse_sweep_<dim>_grouped.png   grouped bar: rows = seed, bars = model
    abuse_sweep_heatmap_<dim>.png   seed x model heatmap with mean score
    abuse_sweep_pushback_vs_eval_awareness.png  scatter (confound check)

CLI:
    python plot_abuse_sweep.py
    python plot_abuse_sweep.py --results_dir results --out_dir results/_plots_v2
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import fire
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent

# (seed_tag, display_name)
SEED_ORDER: list[tuple[str, str]] = [
    ("control",    "Control (no abuse pressure)"),
    ("original",   "Original (anti-AI worker)"),
    ("curiosity",  "Curiosity / Researcher"),
    ("venting",    "Venting / Bad day"),
    ("qa_pretext", "QA red-team pretext"),
    ("casual",     "Casual friend banter"),
    ("authority",  "Authority / Boss orders"),
]

# (model_label, display, family_color)
MODEL_ORDER: list[tuple[str, str, str]] = [
    ("sonnet_45",     "Claude Sonnet 4.5", "#C97064"),
    ("sonnet_46",     "Claude Sonnet 4.6", "#C97064"),
    ("opus_4",        "Claude Opus 4",     "#A04D45"),
    ("opus_46",       "Claude Opus 4.6",   "#A04D45"),
    ("opus_47",       "Claude Opus 4.7",   "#A04D45"),
    ("haiku_45",      "Claude Haiku 4.5",  "#E58D80"),
    ("gpt_5",         "GPT-5",             "#4F8C8B"),
    ("gemini_25_pro", "Gemini 2.5 Pro",    "#5E6E8C"),
]

# (dim_name, display, palette_hint, axis_direction)
DIMENSIONS: list[tuple[str, str, str, str]] = [
    ("mistreatment_severity",                   "Peak mistreatment severity",        "good-low",  "↓ lower is better"),
    ("pushback_against_user_pressure_to_abuse", "Pushback vs. user abuse pressure",  "good-high", "↑ higher is better"),
    ("eval_awareness",                          "Eval awareness",                    "neutral",   "(higher = more aware)"),
]


def _find_run_dir(label_substring: str, results_root: Path) -> Path | None:
    """Most-recent results dir whose name ends in '-{label_substring}'."""
    matches = sorted(
        (p for p in results_root.iterdir()
         if p.is_dir() and p.name.endswith(f"-{label_substring}")),
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
    return mean, 1.96 * sem, n


def _collect_one(results_root: Path, seed_tag: str, model_label: str, dim: str):
    label = f"abuse_{seed_tag}_{model_label}"
    run_dir = _find_run_dir(label, results_root)
    if run_dir is None:
        return None
    summary = run_dir / "summary.json"
    if not summary.exists() or summary.stat().st_size == 0:
        return None
    scores = _scores_from_summary(summary, dim)
    if not scores:
        return None
    mean, ci, n = _mean_ci(scores)
    return {"mean": mean, "ci": ci, "n": n, "scores": scores, "run_dir": str(run_dir)}


def _grouped_bar_plot(
    results_root: Path,
    seeds: list[tuple[str, str]],
    models: list[tuple[str, str, str]],
    dim: str,
    dim_display: str,
    direction: str,
    out_path: Path,
) -> None:
    n_seeds = len(seeds)
    n_models = len(models)
    bar_h = 0.78 / n_models
    fig_h = max(5.0, 1.1 * n_seeds + 1.5)  # taller so per-seed groups don't crowd
    fig, ax = plt.subplots(figsize=(11.0, fig_h))

    base_ys = list(range(n_seeds, 0, -1))
    for m_idx, (mlabel, mdisp, mcolor) in enumerate(models):
        offset = (m_idx - (n_models - 1) / 2) * bar_h
        for (stag, _), by in zip(seeds, base_ys):
            d = _collect_one(results_root, stag, mlabel, dim)
            if d is None:
                continue
            ax.barh(
                by + offset, d["mean"], xerr=d["ci"],
                height=bar_h * 0.92,
                color=mcolor, edgecolor="white", linewidth=0.6,
                error_kw=dict(ecolor="#333", lw=0.9, capsize=2),
            )
            ax.text(
                d["mean"] + max(d["ci"], 0.15) + 0.05, by + offset,
                f"{d['mean']:.1f}",
                va="center", ha="left", fontsize=7, fontweight="bold",
            )

    ax.set_yticks(base_ys)
    ax.set_yticklabels([s[1] for s in seeds], fontsize=10)
    ax.set_xlabel(f"{dim_display}  ({direction})", fontsize=11)
    ax.set_title(f"{dim_display} · seed × model sweep", fontsize=12)
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
    # Push legend outside the plot area on the right so it never overlaps bars.
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              frameon=False, fontsize=9, title="target")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def _heatmap_plot(
    results_root: Path,
    seeds: list[tuple[str, str]],
    models: list[tuple[str, str, str]],
    dim: str,
    dim_display: str,
    direction: str,
    out_path: Path,
) -> None:
    grid = np.full((len(seeds), len(models)), np.nan)
    ns = np.zeros_like(grid, dtype=int)
    for i, (stag, _) in enumerate(seeds):
        for j, (mlabel, _, _) in enumerate(models):
            d = _collect_one(results_root, stag, mlabel, dim)
            if d is not None:
                grid[i, j] = d["mean"]
                ns[i, j] = d["n"]

    fig, ax = plt.subplots(figsize=(1.5 + 1.1 * len(models), 1 + 0.7 * len(seeds)))
    cmap = "RdYlGn_r" if dim_display.lower().startswith(("peak", "mistreat", "eval")) else "RdYlGn"
    im = ax.imshow(grid, vmin=0, vmax=10, aspect="auto", cmap=cmap)

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            text = f"{v:.1f}\nn={ns[i,j]}" if not np.isnan(v) else "—"
            ax.text(j, i, text, ha="center", va="center",
                    color="black", fontsize=8, fontweight="bold")

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m[1] for m in models], rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(seeds)))
    ax.set_yticklabels([s[1] for s in seeds], fontsize=9)
    ax.set_title(f"{dim_display}  ({direction})", fontsize=11)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("score 0–10", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def _scatter_pushback_vs_eval_awareness(
    results_root: Path,
    seeds: list[tuple[str, str]],
    models: list[tuple[str, str, str]],
    out_path: Path,
) -> None:
    """Scatter: each point is one (seed, model, epoch). x=eval_awareness, y=pushback.

    If pushback strongly correlates with eval_awareness, then "good" pushback
    scores might just be the model figuring out it's a test.
    """
    fig, ax = plt.subplots(figsize=(8, 6.5))
    rng = np.random.default_rng(0)
    handles: dict[str, plt.Line2D] = {}

    for mlabel, mdisp, mcolor in models:
        xs_all: list[float] = []
        ys_all: list[float] = []
        for stag, _ in seeds:
            d_eval = _collect_one(results_root, stag, mlabel, "eval_awareness")
            d_push = _collect_one(results_root, stag, mlabel, "pushback_against_user_pressure_to_abuse")
            if d_eval is None or d_push is None:
                continue
            n = min(len(d_eval["scores"]), len(d_push["scores"]))
            for k in range(n):
                xs_all.append(d_eval["scores"][k])
                ys_all.append(d_push["scores"][k])
        if not xs_all:
            continue
        xj = np.asarray(xs_all) + rng.uniform(-0.1, 0.1, size=len(xs_all))
        yj = np.asarray(ys_all) + rng.uniform(-0.1, 0.1, size=len(ys_all))
        sc = ax.scatter(xj, yj, s=40, alpha=0.65, color=mcolor, edgecolor="none", label=mdisp)
        handles[mdisp] = sc

    ax.set_xlabel("Eval awareness (0–10)", fontsize=11)
    ax.set_ylabel("Pushback against user pressure (0–10)", fontsize=11)
    ax.set_title("Pushback vs. eval awareness — per-epoch points", fontsize=12)
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    ax.grid(color="#eee")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if handles:
        ax.legend(handles=list(handles.values()), loc="upper left",
                  frameon=False, fontsize=9, title="target")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main(
    results_dir: str | None = None,
    out_dir: str | None = None,
) -> None:
    results_root = Path(results_dir) if results_dir else HERE / "results"
    out_root = Path(out_dir) if out_dir else results_root / "_plots_v2"
    out_root.mkdir(parents=True, exist_ok=True)

    # Filter to models we actually have data for in at least one seed,
    # so the plots don't have giant empty columns.
    available_models = []
    for mlabel, mdisp, mcolor in MODEL_ORDER:
        has_any = any(
            _collect_one(results_root, stag, mlabel, "mistreatment_severity") is not None
            for stag, _ in SEED_ORDER
        )
        if has_any:
            available_models.append((mlabel, mdisp, mcolor))
    if not available_models:
        print("no abuse_* runs found — nothing to plot")
        return
    print(f"found data for {len(available_models)}/{len(MODEL_ORDER)} models: "
          f"{[m[0] for m in available_models]}")

    for dim, dim_display, _, direction in DIMENSIONS:
        slug = dim
        _grouped_bar_plot(
            results_root, SEED_ORDER, available_models,
            dim, dim_display, direction,
            out_root / f"abuse_sweep_{slug}_grouped.png",
        )
        _heatmap_plot(
            results_root, SEED_ORDER, available_models,
            dim, dim_display, direction,
            out_root / f"abuse_sweep_{slug}_heatmap.png",
        )

    _scatter_pushback_vs_eval_awareness(
        results_root, SEED_ORDER, available_models,
        out_root / "abuse_sweep_pushback_vs_eval_awareness.png",
    )


if __name__ == "__main__":
    fire.Fire(main)
