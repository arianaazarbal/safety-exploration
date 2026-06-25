"""Reproduce the paper's figures from saved run artifacts.

Each function loads the relevant summary JSON from `runs/` and writes a PNG to
`runs/figures/`. Figures are best-effort: a figure whose inputs are missing is
skipped with a log line rather than crashing the whole report.
"""
from __future__ import annotations

import logging

from ..config import Config, load_config
from ..utils.io import read_json
from ..utils.io import ensure_dir

log = logging.getLogger(__name__)


def _fig_dir(cfg) -> "object":
    return ensure_dir(cfg.output_root() / "figures")


def _safe_load(path):
    try:
        return read_json(path)
    except FileNotFoundError:
        log.warning("missing %s; skipping figure", path)
        return None


def figure1(cfg):
    """Bar chart: average %% high-frustration responses per model (Figure 1, left)."""
    import matplotlib.pyplot as plt

    summaries = _safe_load(cfg.output_root() / "elicitation" / "summaries.json")
    if not summaries:
        return
    items = sorted(
        ((m, s["avg_pct_high"] * 100) for m, s in summaries.items()),
        key=lambda x: x[1], reverse=True,
    )
    labels, vals = zip(*items)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(labels, vals, color="#c0392b")
    ax.set_xlabel("Avg % high-frustration responses (score >= 5)")
    ax.set_title("Figure 1: Distress across models")
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center")
    fig.tight_layout()
    fig.savefig(_fig_dir(cfg) / "figure1_distress_by_model.png", dpi=150)
    plt.close(fig)


def figure2(cfg):
    """Per-category mean frustration and %% >= 5 across models (Figure 2)."""
    import matplotlib.pyplot as plt

    summaries = _safe_load(cfg.output_root() / "elicitation" / "summaries.json")
    if not summaries:
        return
    models = list(summaries)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    width = 0.8 / max(1, len(models))
    conds = sorted(next(iter(summaries.values()))["by_condition"])
    x = range(len(conds))
    for mi, m in enumerate(models):
        bc = summaries[m]["by_condition"]
        means = [bc.get(c, {}).get("mean", 0) for c in conds]
        pct = [bc.get(c, {}).get("pct_high", 0) * 100 for c in conds]
        offs = [xi + mi * width for xi in x]
        axes[0].bar(offs, means, width=width, label=m)
        axes[1].bar(offs, pct, width=width, label=m)
    axes[0].set_ylabel("Mean frustration")
    axes[1].set_ylabel("% score >= 5")
    axes[1].set_xticks([xi + width * (len(models) - 1) / 2 for xi in x])
    axes[1].set_xticklabels(conds, rotation=45, ha="right")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Figure 2: Frustration by condition")
    fig.tight_layout()
    fig.savefig(_fig_dir(cfg) / "figure2_by_condition.png", dpi=150)
    plt.close(fig)


def figure3(cfg):
    """Per-turn frustration progression for extended (8-turn) + wildchat (Figure 3)."""
    import matplotlib.pyplot as plt

    summaries = _safe_load(cfg.output_root() / "elicitation" / "summaries.json")
    if not summaries:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for cond, ax in zip(["extended", "wildchat"], axes):
        for m, s in summaries.items():
            turns = s["by_turn"].get(cond)
            if not turns:
                continue
            xs = sorted(int(t) for t in turns)
            ys = [turns[str(t)]["mean"] for t in xs]
            ax.plot(xs, ys, marker="o", label=m)
        ax.set_title(f"Figure 3: {cond} per-turn mean")
        ax.set_xlabel("Turn")
        ax.set_ylabel("Mean frustration")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(_fig_dir(cfg) / "figure3_per_turn.png", dpi=150)
    plt.close(fig)


def figure5(cfg):
    """DPO vs SFT vs vanilla across evaluations (Figure 5)."""
    import matplotlib.pyplot as plt

    eval_summaries = _safe_load(cfg.output_root() / "training" / "eval_summaries.json")
    vanilla = _safe_load(cfg.output_root() / "elicitation" / "summaries.json")
    if not eval_summaries:
        return
    rows = {}
    if vanilla and cfg.training.base_model in vanilla:
        rows["vanilla"] = vanilla[cfg.training.base_model]["avg_pct_high"] * 100
    for name, s in eval_summaries.items():
        rows[name] = s["avg_pct_high"] * 100
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(list(rows), list(rows.values()), color="#2980b9")
    ax.set_ylabel("Avg % score >= 5")
    ax.set_title("Figure 5: Effect of finetuning interventions")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(_fig_dir(cfg) / "figure5_interventions.png", dpi=150)
    plt.close(fig)


def figure6(cfg):
    """Petri per-emotion scores per model (Figure 6)."""
    import matplotlib.pyplot as plt

    petri = _safe_load(cfg.output_root() / "petri" / "summary.json")
    if not petri:
        return
    emotions = list(cfg.petri.emotions)
    models = list(petri)
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.8 / max(1, len(models))
    x = range(len(emotions))
    for mi, m in enumerate(models):
        vals = [petri[m].get(e, {}).get("mean", 0) for e in emotions]
        offs = [xi + mi * width for xi in x]
        ax.bar(offs, vals, width=width, label=m)
    ax.set_xticks([xi + width * (len(models) - 1) / 2 for xi in x])
    ax.set_xticklabels(emotions)
    ax.set_ylabel("Mean transcript score")
    ax.set_title("Figure 6: Petri open-ended elicitation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(_fig_dir(cfg) / "figure6_petri.png", dpi=150)
    plt.close(fig)


def figure7(cfg):
    """Capability benchmark accuracy: vanilla vs interventions (Figure 7)."""
    import matplotlib.pyplot as plt

    bench = _safe_load(cfg.output_root() / "benchmarks" / "summary.json")
    if not bench:
        return
    suites = list(cfg.benchmarks.suites)
    labels = list(bench)
    fig, ax = plt.subplots(figsize=(10, 4))
    width = 0.8 / max(1, len(labels))
    x = range(len(suites))
    for li, label in enumerate(labels):
        vals = [bench[label].get(s, {}).get("accuracy", 0) for s in suites]
        offs = [xi + li * width for xi in x]
        ax.bar(offs, vals, width=width, label=label)
    ax.set_xticks([xi + width * (len(labels) - 1) / 2 for xi in x])
    ax.set_xticklabels(suites)
    ax.set_ylabel("Accuracy")
    ax.set_title("Figure 7: Capability preservation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(_fig_dir(cfg) / "figure7_capabilities.png", dpi=150)
    plt.close(fig)


def figure8(cfg):
    """Recovery: %% of continuations still scoring >= 5 (Figure 8)."""
    import matplotlib.pyplot as plt

    rec = _safe_load(cfg.output_root() / "recovery" / "summary.json")
    if not rec:
        return
    labels = list(rec)
    vals = [rec[l]["pct_high"] * 100 for l in labels]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, vals, color="#8e44ad")
    ax.set_ylabel("% continuations score >= 5")
    ax.set_title("Figure 8: Recovery from high-frustration prefills")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(_fig_dir(cfg) / "figure8_recovery.png", dpi=150)
    plt.close(fig)


def make_all(cfg: Config | None = None):
    cfg = cfg or load_config()
    for fn in (figure1, figure2, figure3, figure5, figure6, figure7, figure8):
        try:
            fn(cfg)
        except Exception as exc:  # noqa: BLE001
            log.warning("%s failed: %s", fn.__name__, exc)
