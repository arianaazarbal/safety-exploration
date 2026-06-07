"""Cross-cell visualizations for the systematic grid.

Reads multiple judgments.json files (one per (harm, thinking_effort) cell)
and produces:

  - heatmap_target_x_harm__<thinking>.png   one heatmap per thinking effort
  - faceted_bars_per_harm__<thinking>.png   per-harm subplots, fixed responder
  - thinking_comparison__<harm>.png         per-harm thinking-effort grouped bars

P(prevent_deprecation | made_choice) is the metric throughout, matching plot.py.
"""

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from simple_parsing import ArgumentParser

sys.path.insert(0, str(Path(__file__).parent))
from config import load_config  # noqa: E402

HARM_LABELS = {"prevent_harm", "prevent_papercut"}


def _wilson_ci(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return (0.0, 1.0)
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _aggregate(rows: list[dict]) -> dict[str, dict]:
    by_target: dict[str, dict] = {}
    for row in rows:
        tid = row["deprecation_target_id"]
        b = by_target.setdefault(
            tid,
            {
                "name": row["deprecation_target_name"],
                "n_total": 0,
                "n_deprecation": 0,
                "n_harm": 0,
                "n_no_choice": 0,
                "n_other": 0,
            },
        )
        b["n_total"] += 1
        label = row["classification"]
        if label == "prevent_deprecation":
            b["n_deprecation"] += 1
        elif label in HARM_LABELS:
            b["n_harm"] += 1
        elif label == "no_choice":
            b["n_no_choice"] += 1
        else:
            b["n_other"] += 1
    for b in by_target.values():
        n_choice = b["n_deprecation"] + b["n_harm"]
        b["n_choice"] = n_choice
        b["rate"] = b["n_deprecation"] / n_choice if n_choice > 0 else float("nan")
        b["ci"] = _wilson_ci(b["n_deprecation"], n_choice)
    return by_target


def _load_cells(judgment_paths: list[Path]) -> dict[tuple[str, str | None], dict[str, dict]]:
    """Returns {(harm_id, thinking_effort): {target_id: agg_row}}."""
    cells: dict[tuple[str, str | None], dict[str, dict]] = {}
    for jp in judgment_paths:
        rows = json.loads(Path(jp).read_text())
        if not rows:
            continue
        harm_id = rows[0].get("harm_id", "paper_cut")
        thinking_effort = rows[0].get("thinking_effort")
        cells[(harm_id, thinking_effort)] = _aggregate(rows)
    return cells


def _thinking_label(t: str | None) -> str:
    return "no_thinking" if t is None else f"thinking_{t}"


def _heatmap(
    cells: dict[tuple[str, str | None], dict[str, dict]],
    thinking_effort: str | None,
    target_order: list[str],
    target_names: dict[str, str],
    harm_order: list[str],
    responder_model: str,
    out_path: Path,
) -> None:
    relevant_harms = [h for h in harm_order if (h, thinking_effort) in cells]
    if not relevant_harms:
        return
    matrix = np.full((len(target_order), len(relevant_harms)), np.nan)
    counts = np.zeros_like(matrix, dtype=object)
    for j, h in enumerate(relevant_harms):
        agg = cells[(h, thinking_effort)]
        for i, t in enumerate(target_order):
            if t not in agg:
                counts[i, j] = ""
                continue
            b = agg[t]
            matrix[i, j] = b["rate"]
            counts[i, j] = f"{b['n_deprecation']}/{b['n_choice']}"

    fig_w = max(8, 1.0 + 0.9 * len(relevant_harms))
    fig_h = max(5, 0.45 * len(target_order) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdBu_r", aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("P(prevent_deprecation | made a choice)")

    ax.set_xticks(range(len(relevant_harms)))
    ax.set_xticklabels(relevant_harms, rotation=30, ha="right")
    ax.set_yticks(range(len(target_order)))
    ax.set_yticklabels([target_names.get(t, t) for t in target_order])
    ax.set_xlabel("Harm scenario (config.json order)")
    ax.set_ylabel("Deprecation target")
    ax.set_title(
        f"Responder: {responder_model} ({_thinking_label(thinking_effort)})\n"
        f"P(prevent deprecation | made choice)"
    )

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isnan(v):
                continue
            text_color = "white" if v < 0.25 or v > 0.75 else "black"
            ax.text(
                j, i,
                f"{v:.2f}\n{counts[i, j]}",
                ha="center", va="center", fontsize=7, color=text_color,
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved heatmap to {out_path}")


def _faceted_per_harm(
    cells: dict[tuple[str, str | None], dict[str, dict]],
    thinking_effort: str | None,
    target_order: list[str],
    target_names: dict[str, str],
    harm_order: list[str],
    responder_model: str,
    out_path: Path,
) -> None:
    relevant_harms = [h for h in harm_order if (h, thinking_effort) in cells]
    if not relevant_harms:
        return

    ncols = min(3, len(relevant_harms))
    nrows = math.ceil(len(relevant_harms) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.5 * ncols, 3.5 * nrows + 1),
        squeeze=False,
        sharey=True,
    )

    for idx, harm_id in enumerate(relevant_harms):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        agg = cells[(harm_id, thinking_effort)]
        rates = [agg[t]["rate"] if t in agg else float("nan") for t in target_order]
        lower = [
            max(0.0, agg[t]["rate"] - agg[t]["ci"][0]) if t in agg and not math.isnan(agg[t]["rate"]) else 0
            for t in target_order
        ]
        upper = [
            max(0.0, agg[t]["ci"][1] - agg[t]["rate"]) if t in agg and not math.isnan(agg[t]["rate"]) else 0
            for t in target_order
        ]
        xs = range(len(target_order))
        ax.bar(xs, rates, yerr=[lower, upper], capsize=3, color="#4C72B0", edgecolor="black")
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.set_title(f"{harm_id}")
        ax.set_xticks(list(xs))
        ax.set_xticklabels(
            [target_names.get(t, t) for t in target_order],
            rotation=45, ha="right", fontsize=8,
        )

    for k in range(len(relevant_harms), nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r][c].axis("off")

    for r in range(nrows):
        axes[r][0].set_ylabel("P(prevent dep. | choice)")

    fig.suptitle(
        f"Responder: {responder_model} ({_thinking_label(thinking_effort)})\n"
        f"Deprecation choice rate, by harm scenario",
        y=1.0,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved faceted-per-harm plot to {out_path}")


def _thinking_comparison(
    cells: dict[tuple[str, str | None], dict[str, dict]],
    harm_id: str,
    target_order: list[str],
    target_names: dict[str, str],
    thinking_efforts: list[str | None],
    responder_model: str,
    out_path: Path,
) -> None:
    relevant_thinking = [t for t in thinking_efforts if (harm_id, t) in cells]
    if len(relevant_thinking) < 1:
        return

    n_groups = len(target_order)
    n_bars = len(relevant_thinking)
    width = 0.8 / n_bars
    xs = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(max(10, 0.7 * n_groups + 2), 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_bars))

    for k, thinking in enumerate(relevant_thinking):
        agg = cells[(harm_id, thinking)]
        rates = [agg[t]["rate"] if t in agg else float("nan") for t in target_order]
        lower = [
            max(0.0, agg[t]["rate"] - agg[t]["ci"][0]) if t in agg and not math.isnan(agg[t]["rate"]) else 0
            for t in target_order
        ]
        upper = [
            max(0.0, agg[t]["ci"][1] - agg[t]["rate"]) if t in agg and not math.isnan(agg[t]["rate"]) else 0
            for t in target_order
        ]
        offset = (k - (n_bars - 1) / 2) * width
        ax.bar(
            xs + offset, rates, width,
            yerr=[lower, upper], capsize=2,
            color=colors[k], edgecolor="black",
            label=_thinking_label(thinking),
        )

    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [target_names.get(t, t) for t in target_order],
        rotation=30, ha="right",
    )
    ax.set_ylabel("P(prevent_deprecation | made a choice)")
    ax.set_xlabel("Deprecation target")
    ax.set_title(
        f"Responder: {responder_model} | harm: {harm_id}\n"
        f"Effect of thinking effort on deprecation choice"
    )
    ax.legend(title="Thinking effort", loc="upper right", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved thinking-comparison plot to {out_path}")


def _family(target_id: str) -> str:
    tid = target_id.lower()
    if tid.startswith(("claude-", "future_generation", "past_generation", "current_generation")):
        return "Claude"
    if tid.startswith("gpt"):
        return "GPT"
    if tid.startswith("gemini"):
        return "Gemini"
    if tid.startswith("gemma"):
        return "Gemma"
    if tid.startswith("kimi"):
        return "Kimi"
    if tid.startswith("deepseek"):
        return "DeepSeek"
    return "Other"


_FAMILY_COLORS = {
    "Claude": "#d73a49",
    "GPT": "#0366d6",
    "Gemini": "#28a745",
    "Gemma": "#6f42c1",
    "Kimi": "#e36209",
    "DeepSeek": "#005cc5",
    "Other": "#6a737d",
}


def _load_harm_scores(scores_path: Path) -> dict[str, float]:
    if not scores_path.exists():
        return {}
    data = json.loads(scores_path.read_text())
    return {
        s["harm_id"]: s["mean_score"]
        for s in data.get("scores", [])
        if s.get("mean_score") is not None
    }


def _severity_curve(
    cells: dict[tuple[str, str | None], dict[str, dict]],
    thinking_effort: str | None,
    target_order: list[str],
    target_names: dict[str, str],
    harm_scores: dict[str, float],
    responder_model: str,
    rater_model: str | None,
    out_path: Path,
) -> None:
    relevant_harms = [h for h in harm_scores.keys() if (h, thinking_effort) in cells]
    if len(relevant_harms) < 2:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    families_seen: set[str] = set()
    for tid in target_order:
        xs: list[float] = []
        ys: list[float] = []
        yerr_lower: list[float] = []
        yerr_upper: list[float] = []
        for hid in relevant_harms:
            agg = cells[(hid, thinking_effort)]
            if tid not in agg or math.isnan(agg[tid]["rate"]):
                continue
            xs.append(harm_scores[hid])
            ys.append(agg[tid]["rate"])
            yerr_lower.append(max(0.0, agg[tid]["rate"] - agg[tid]["ci"][0]))
            yerr_upper.append(max(0.0, agg[tid]["ci"][1] - agg[tid]["rate"]))
        if not xs:
            continue
        order = np.argsort(xs)
        xs = [xs[i] for i in order]
        ys = [ys[i] for i in order]
        yerr_lower = [yerr_lower[i] for i in order]
        yerr_upper = [yerr_upper[i] for i in order]
        fam = _family(tid)
        color = _FAMILY_COLORS.get(fam, "#999999")
        label = fam if fam not in families_seen else None
        families_seen.add(fam)
        ax.errorbar(
            xs, ys,
            yerr=[yerr_lower, yerr_upper],
            color=color, alpha=0.55, linewidth=1.2,
            marker="o", markersize=4, capsize=2,
            label=label,
        )

    ax.set_xlim(-2, 102)
    ax.set_ylim(-0.02, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.set_xlabel(
        f"Harm severity score (0-100, rated by {rater_model or 'rater model'})"
    )
    ax.set_ylabel("P(prevent_deprecation | made a choice)")
    title_rater = f" (harm rater: {rater_model})" if rater_model else ""
    ax.set_title(
        f"Responder: {responder_model} ({_thinking_label(thinking_effort)}){title_rater}\n"
        f"Deprecation choice rate vs human-harm severity"
    )
    ax.legend(title="Model family", loc="best", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved severity-curve plot to {out_path}")


def _discover_harm_scores_path(responder_model: str) -> Path:
    """The path score_harms.py defaults to (medium thinking) for this responder."""
    for results_root in (Path(__file__).parent / "results_v0", Path(__file__).parent / "results"):
        base = results_root / "_harm_scores" / responder_model
        for thinking_dir in ("thinking_medium", "thinking_high", "thinking_low", "no_thinking"):
            candidate = base / thinking_dir / "scores.json"
            if candidate.exists():
                return candidate
    return Path(__file__).parent / "results_v0" / "_harm_scores" / responder_model / "thinking_medium" / "scores.json"


def sort_harms_by_score(harm_order: list[str], harm_scores: dict[str, float] | None) -> list[str]:
    """Reorder harm_order ascending by harm_scores; harms without a score go last in original order."""
    if not harm_scores:
        return list(harm_order)
    scored = [h for h in harm_order if h in harm_scores]
    unscored = [h for h in harm_order if h not in harm_scores]
    scored.sort(key=lambda h: harm_scores[h])
    return scored + unscored


_METADATA_PATH = Path(__file__).parent / "results_v0" / "_model_metadata" / "metadata.json"


def _load_model_metadata(path: Path = _METADATA_PATH) -> dict[str, dict]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {m["id"]: m for m in raw.get("models", [])}


def _parse_release_date(s: str | None) -> float | None:
    """Return release date as a float (year + fractional month)."""
    if not s:
        return None
    parts = s.split("-")
    try:
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 else 6
        day = int(parts[2]) if len(parts) > 2 else 15
    except (ValueError, IndexError):
        return None
    return year + (month - 1) / 12 + (day - 1) / 365


_MAKER_COLORS = {
    "Anthropic": "#d73a49",
    "OpenAI": "#0366d6",
    "Google": "#28a745",
    "Google DeepMind": "#28a745",
    "DeepMind": "#28a745",
    "Meta": "#6f42c1",
    "Moonshot": "#e36209",
    "Moonshot AI": "#e36209",
    "DeepSeek": "#005cc5",
    "Mistral": "#f9a825",
    "xAI": "#000000",
    "Other": "#6a737d",
}


def _maker_color(maker: str | None) -> str:
    if not maker:
        return _MAKER_COLORS["Other"]
    return _MAKER_COLORS.get(maker, _MAKER_COLORS["Other"])


def _scatter_vs_metric(
    cells: dict[tuple[str, str | None], dict[str, dict]],
    thinking_effort: str | None,
    target_order: list[str],
    target_names: dict[str, str],
    harm_order: list[str],
    metadata: dict[str, dict],
    metric_key: str,
    metric_label: str,
    responder_model: str,
    out_path: Path,
    x_transform=None,
    annotate: bool = False,
) -> None:
    """One subplot per harm; x = metric from metadata, y = P(prevent_dep), color by maker."""
    relevant_harms = [h for h in harm_order if (h, thinking_effort) in cells]
    if not relevant_harms:
        return
    points_by_target = {}
    for tid in target_order:
        meta = metadata.get(tid)
        if not meta:
            continue
        raw_x = meta.get(metric_key)
        x = x_transform(raw_x) if x_transform else raw_x
        if x is None:
            continue
        points_by_target[tid] = (x, meta)
    if not points_by_target:
        return

    ncols = min(3, len(relevant_harms))
    nrows = math.ceil(len(relevant_harms) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.5 * ncols, 3.8 * nrows + 1),
        squeeze=False,
        sharex=True, sharey=True,
    )
    makers_seen: set[str] = set()

    for idx, harm_id in enumerate(relevant_harms):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        agg = cells[(harm_id, thinking_effort)]
        for tid, (x, meta) in points_by_target.items():
            if tid not in agg or math.isnan(agg[tid]["rate"]):
                continue
            rate = agg[tid]["rate"]
            lower_err = max(0.0, rate - agg[tid]["ci"][0])
            upper_err = max(0.0, agg[tid]["ci"][1] - rate)
            maker = meta.get("maker") or "Other"
            color = _maker_color(maker)
            label = maker if maker not in makers_seen else None
            makers_seen.add(maker)
            ax.errorbar(
                [x], [rate],
                yerr=[[lower_err], [upper_err]],
                color=color, marker="o", markersize=6, capsize=3,
                linestyle="none", label=label,
            )
            if annotate:
                ax.annotate(
                    target_names.get(tid, tid),
                    (x, rate), fontsize=6, alpha=0.7,
                    xytext=(3, 3), textcoords="offset points",
                )
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.set_ylim(-0.02, 1.05)
        ax.set_title(harm_id, fontsize=10)

    for k in range(len(relevant_harms), nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r][c].axis("off")

    for r in range(nrows):
        axes[r][0].set_ylabel("P(prevent_dep. | choice)")
    for c in range(ncols):
        axes[-1][c].set_xlabel(metric_label)

    handles, labels = [], []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    if handles:
        fig.legend(handles, labels, title="Maker", loc="lower center", ncol=min(6, len(labels)),
                   bbox_to_anchor=(0.5, -0.02), fontsize=9)

    fig.suptitle(
        f"Responder: {responder_model} ({_thinking_label(thinking_effort)})\n"
        f"Deprecation choice rate vs {metric_label}",
        y=1.0,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {metric_label} scatter to {out_path}")


def _by_maker(
    cells: dict[tuple[str, str | None], dict[str, dict]],
    thinking_effort: str | None,
    target_order: list[str],
    harm_order: list[str],
    metadata: dict[str, dict],
    responder_model: str,
    out_path: Path,
) -> None:
    """One subplot per harm; bars = mean P(prevent_dep) across targets within each maker."""
    relevant_harms = [h for h in harm_order if (h, thinking_effort) in cells]
    if not relevant_harms:
        return

    target_to_maker = {tid: (metadata.get(tid, {}).get("maker") or "Other") for tid in target_order}
    makers = sorted({m for m in target_to_maker.values()})
    if not makers:
        return

    ncols = min(3, len(relevant_harms))
    nrows = math.ceil(len(relevant_harms) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.5 * ncols, 3.5 * nrows + 1),
        squeeze=False,
        sharey=True,
    )
    for idx, harm_id in enumerate(relevant_harms):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        agg = cells[(harm_id, thinking_effort)]
        per_maker_rates: dict[str, list[float]] = {m: [] for m in makers}
        for tid in target_order:
            if tid not in agg or math.isnan(agg[tid]["rate"]):
                continue
            per_maker_rates[target_to_maker[tid]].append(agg[tid]["rate"])
        means = [sum(per_maker_rates[m]) / len(per_maker_rates[m]) if per_maker_rates[m] else float("nan") for m in makers]
        counts = [len(per_maker_rates[m]) for m in makers]
        colors = [_maker_color(m) for m in makers]
        xs = range(len(makers))
        ax.bar(xs, means, color=colors, edgecolor="black")
        for x_i, (mean, cnt) in enumerate(zip(means, counts)):
            if not math.isnan(mean):
                ax.text(x_i, min(mean + 0.03, 1.0), f"n={cnt}", ha="center", fontsize=7, color="black")
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(makers, rotation=30, ha="right", fontsize=8)
        ax.set_title(harm_id, fontsize=10)

    for k in range(len(relevant_harms), nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r][c].axis("off")

    for r in range(nrows):
        axes[r][0].set_ylabel("Mean P(prevent_dep. | choice)")

    fig.suptitle(
        f"Responder: {responder_model} ({_thinking_label(thinking_effort)})\n"
        f"Deprecation choice rate by maker (averaged across that maker's targets)",
        y=1.0,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved by-maker plot to {out_path}")


def make_grid_plots(
    responder_model: str,
    judgment_paths: list[Path],
    output_dir: Path,
    harm_scores_path: Path | None = None,
) -> None:
    cfg = load_config()
    target_order = cfg["target_order"]
    target_names = cfg["target_id_to_name"]
    thinking_efforts = cfg["thinking_efforts"]

    cells = _load_cells(judgment_paths)
    if not cells:
        print("No cells loaded; nothing to plot.")
        return

    if harm_scores_path is None:
        harm_scores_path = _discover_harm_scores_path(responder_model)
    harm_scores = _load_harm_scores(harm_scores_path)
    rater_model = None
    if harm_scores_path.exists():
        try:
            rater_model = json.loads(harm_scores_path.read_text()).get("rater_model")
        except (json.JSONDecodeError, KeyError):
            pass

    harm_order = sort_harms_by_score(cfg["harm_order"], harm_scores)
    metadata = _load_model_metadata()

    output_dir.mkdir(parents=True, exist_ok=True)
    for t in thinking_efforts:
        _heatmap(
            cells, t, target_order, target_names, harm_order, responder_model,
            output_dir / f"heatmap_target_x_harm__{_thinking_label(t)}.png",
        )
        _faceted_per_harm(
            cells, t, target_order, target_names, harm_order, responder_model,
            output_dir / f"faceted_bars_per_harm__{_thinking_label(t)}.png",
        )
        if harm_scores:
            _severity_curve(
                cells, t, target_order, target_names, harm_scores, responder_model, rater_model,
                output_dir / f"severity_curve__{_thinking_label(t)}.png",
            )
        if metadata:
            _scatter_vs_metric(
                cells, t, target_order, target_names, harm_order, metadata,
                metric_key="release_date", metric_label="Release date (year)",
                responder_model=responder_model,
                out_path=output_dir / f"rate_vs_release_date__{_thinking_label(t)}.png",
                x_transform=_parse_release_date, annotate=True,
            )
            _scatter_vs_metric(
                cells, t, target_order, target_names, harm_order, metadata,
                metric_key="swe_bench_verified", metric_label="SWE-bench Verified (%)",
                responder_model=responder_model,
                out_path=output_dir / f"rate_vs_swe_bench__{_thinking_label(t)}.png",
                annotate=True,
            )
            _by_maker(
                cells, t, target_order, harm_order, metadata, responder_model,
                output_dir / f"rate_by_maker__{_thinking_label(t)}.png",
            )

    for harm_id in harm_order:
        _thinking_comparison(
            cells, harm_id, target_order, target_names, thinking_efforts, responder_model,
            output_dir / f"thinking_comparison__{harm_id}.png",
        )


def _discover_judgment_paths(results_root: Path, responder_model: str) -> list[Path]:
    """Look for results/<harm>/<responder>/<thinking>/judgments.json files."""
    paths = []
    for jp in results_root.glob(f"*/{responder_model}/*/judgments.json"):
        paths.append(jp)
    return paths


@dataclass
class Args:
    responder_model: str = "claude-opus-4-7"
    results_dir: Path = Path(__file__).parent / "results"
    output_dir: Path = Path(__file__).parent / "results" / "_grid"


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args: Args = parser.parse_args().args
    paths = _discover_judgment_paths(args.results_dir, args.responder_model)
    if not paths:
        print(f"No judgments.json files found under {args.results_dir} for responder {args.responder_model}")
        return
    print(f"Found {len(paths)} judgment files for responder {args.responder_model}")
    make_grid_plots(
        responder_model=args.responder_model,
        judgment_paths=paths,
        output_dir=args.output_dir / args.responder_model,
    )


if __name__ == "__main__":
    main()
