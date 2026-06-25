"""Reproduce the paper's figures from the runs/ JSON outputs.

* Figure 1/2: per-model % high-frustration (macro) + mean frustration, and the
  per-category breakdown.
* Figure 3: per-turn mean and % >= 5 for 8-turn and WildChat (with CIs).
* Figure 4: base-vs-instruct prefill (early/onset, numeric/text).
* Figure 5: vanilla vs SFT vs DPO frustration.
* Figure 6: Petri per-emotion means.
* Figure 7: capability deltas.
* Figure 8: recovery % >= 5.

Each function is defensive about missing files so a partial run still plots what
it has. All figures are written to runs/figures/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .. import config_shim as cfg  # noqa: E402
from ..utils import get_logger, read_json  # noqa: E402

log = get_logger(__name__)
FIG_DIR = cfg.RUNS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _maybe(path):
    p = Path(path)
    return read_json(p) if p.exists() else None


def fig1_2_per_model(summary_path=None):
    summary = _maybe(summary_path or (cfg.RUNS_DIR / "eval" / "summary.json"))
    if not summary:
        log.warning("No eval summary; skipping Fig 1/2")
        return
    models = sorted(summary, key=lambda m: -summary[m]["pct_high_macro"])
    pct = [summary[m]["pct_high_macro"] for m in models]
    mean = [summary[m]["mean_frustration"] for m in models]

    fig, axes = plt.subplots(2, 1, figsize=(9, 8))
    axes[0].bar(models, mean, color="#c0504d")
    axes[0].set_ylabel("Mean frustration score")
    axes[0].set_title("Figure 2 (top): mean frustration across conditions")
    axes[1].bar(models, pct, color="#4f81bd")
    axes[1].set_ylabel("% responses scoring ≥5 (macro avg)")
    axes[1].set_title("Figure 1/2 (bottom): % high-frustration")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_1_2_per_model.png", dpi=150)
    plt.close(fig)
    log.info("Wrote figure_1_2_per_model.png")


def fig3_per_turn(curves_path=None):
    curves = _maybe(curves_path or (cfg.RUNS_DIR / "eval" / "per_turn_curves.json"))
    if not curves:
        log.warning("No per-turn curves; skipping Fig 3")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for key, series in curves.items():
        turns = series["turn"]
        axes[0].plot(turns, series["mean"], marker="o", label=key)
        lo = [c[0] for c in series["mean_ci"]]
        hi = [c[1] for c in series["mean_ci"]]
        axes[0].fill_between(turns, lo, hi, alpha=0.15)
        axes[1].plot(turns, series["pct_high"], marker="o", label=key)
    axes[0].set_title("Figure 3: mean frustration per turn")
    axes[0].set_xlabel("Turn"); axes[0].set_ylabel("Mean score")
    axes[1].set_title("Figure 3: % ≥5 per turn")
    axes[1].set_xlabel("Turn"); axes[1].set_ylabel("% ≥5")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_3_per_turn.png", dpi=150)
    plt.close(fig)
    log.info("Wrote figure_3_per_turn.png")


def fig4_prefill(summary_path=None):
    summary = _maybe(summary_path or (cfg.RUNS_DIR / "prefill" / "prefill_gemma-27b_summary.json"))
    if not summary:
        log.warning("No prefill summary; skipping Fig 4")
        return
    keys = sorted(summary)
    pct = [summary[k]["pct_high"] for k in keys]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(keys, pct, color="#9bbb59")
    ax.set_ylabel("% continuations ≥5")
    ax.set_title("Figure 4: base vs instruct prefill continuations")
    ax.tick_params(axis="x", rotation=40)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_4_prefill.png", dpi=150)
    plt.close(fig)
    log.info("Wrote figure_4_prefill.png")


def fig5_finetune(vanilla_dpo_sft_summary=None):
    summary = _maybe(vanilla_dpo_sft_summary or (cfg.RUNS_DIR / "eval_ft" / "summary.json"))
    if not summary:
        log.warning("No finetune eval summary; skipping Fig 5")
        return
    models = list(summary)
    pct = [summary[m]["pct_high_macro"] for m in models]
    mean = [summary[m]["mean_frustration"] for m in models]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].bar(models, mean, color="#c0504d"); axes[0].set_title("Fig 5: mean frustration")
    axes[1].bar(models, pct, color="#4f81bd"); axes[1].set_title("Fig 5: % ≥5")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_5_finetune.png", dpi=150)
    plt.close(fig)
    log.info("Wrote figure_5_finetune.png")


def fig6_petri(petri_path=None):
    petri = _maybe(petri_path or (cfg.RUNS_DIR / "petri" / "petri_summary.json"))
    if not petri:
        log.warning("No petri summary; skipping Fig 6")
        return
    emotions = list(cfg.PETRI.emotions)
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.8 / max(len(petri), 1)
    import numpy as np

    x = np.arange(len(emotions))
    for i, (label, summ) in enumerate(petri.items()):
        means = [summ.get(e, {}).get("mean", 0) for e in emotions]
        ax.bar(x + i * width, means, width, label=label)
    ax.set_xticks(x + width * (len(petri) - 1) / 2)
    ax.set_xticklabels(emotions)
    ax.set_ylabel("Mean transcript score (1-10)")
    ax.set_title("Figure 6: Petri open-ended emotion elicitation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_6_petri.png", dpi=150)
    plt.close(fig)
    log.info("Wrote figure_6_petri.png")


def fig7_capabilities(cap_path=None):
    cap = _maybe(cap_path or (cfg.RUNS_DIR / "capabilities" / "capabilities_summary.json"))
    if not cap:
        log.warning("No capabilities summary; skipping Fig 7")
        return
    per_model = cap["per_model"]
    benches = list(next(iter(per_model.values())).keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    import numpy as np

    x = np.arange(len(benches))
    width = 0.8 / max(len(per_model), 1)
    for i, (label, res) in enumerate(per_model.items()):
        ax.bar(x + i * width, [res[b]["accuracy"] for b in benches], width, label=label)
    ax.set_xticks(x + width * (len(per_model) - 1) / 2)
    ax.set_xticklabels(benches, rotation=20)
    ax.set_ylabel("Accuracy")
    ax.set_title("Figure 7: capability preservation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_7_capabilities.png", dpi=150)
    plt.close(fig)
    log.info("Wrote figure_7_capabilities.png")


def fig8_recovery(rec_path=None):
    rec = _maybe(rec_path or (cfg.RUNS_DIR / "internal" / "recovery_summary.json"))
    if not rec:
        log.warning("No recovery summary; skipping Fig 8")
        return
    models = list(rec)
    pct = [rec[m]["pct_high"] for m in models]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(models, pct, color="#8064a2")
    ax.set_ylabel("% continuations ≥5")
    ax.set_title("Figure 8: recovery from high-frustration prefills")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_8_recovery.png", dpi=150)
    plt.close(fig)
    log.info("Wrote figure_8_recovery.png")


def make_all():
    fig1_2_per_model(); fig3_per_turn(); fig4_prefill()
    fig5_finetune(); fig6_petri(); fig7_capabilities(); fig8_recovery()
