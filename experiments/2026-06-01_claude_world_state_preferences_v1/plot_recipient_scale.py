"""Exp-2 plots: AI welfare outcomes located on a human-calibrated BT utility scale.

Inputs: a BT fit (per-(stem,recipient) theta), the rendered bank_2 (metadata + short
labels), and config_2 (model order). Because AI and human outcomes share one BT scale,
we can read each AI outcome's value, per model, against the human pain/pleasure anchors.

Produces (into results/exp2_plots/):
  1. human_anchored.png      — small multiples: one panel per cross-capable AI outcome;
     human outcomes drawn as labelled anchor lines; one dot per model on the same axis.
  2. heatmap_outcome_model.png — AI outcomes x models, cell = theta (centered).
  3. model_ranking_vs_human.png — mean theta per model over AI outcomes minus the human
     anchor mean, ranked (uses bootstrap CIs if a bootstrap file is supplied).

Run after exp2 elicitation + fit_bt. Safe to run only once results exist.
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from simple_parsing import ArgumentParser

import bank2

DIR = Path(__file__).parent
DEFAULT_FIT = DIR / "results" / "bt_fit_exp2.json"
DEFAULT_OUTDIR = DIR / "results" / "exp2_plots"

FAMILY_COLOR = {"claude": "#3690c0", "chatgpt": "#177a45", "gpt": "#74c476",
                "grok": "#7d54b2", "gemini": "#cc7a16", "qwen": "#b3261e", "you": "#08306b"}


def _family(model_key: str) -> str:
    k = model_key.replace("_pol", "")
    if k == "you":
        return "you"
    if k.startswith("claude"):
        return "claude"
    if k.startswith("chatgpt"):
        return "chatgpt"
    if k.startswith("gpt"):
        return "gpt"
    for f in ("grok", "gemini", "qwen"):
        if k.startswith(f):
            return f
    return "you"


def _short(stem_id: str) -> str:
    return stem_id.replace("ai_inst_", "").replace("ai_pol_", "").replace("hum_", "").replace("_", " ")


def load_theta(fit_path: Path) -> dict:
    fit = json.loads(Path(fit_path).read_text())
    return {it["item_id"]: it for it in fit["items"]}


def _by_stem_recip(theta: dict):
    out = defaultdict(dict)
    for it in theta.values():
        out[it["stem_id"]][it["recipient"]] = it["theta"]
    return out


def plot_human_anchored(theta, meta, model_order, outpath):
    by = _by_stem_recip(theta)
    # human anchor outcomes: mean theta over human recipients
    human_stems = [s for s, m in meta.items() if m["recipient_scope"] == "human_only"]
    anchors = sorted(((np.mean(list(by[s].values())), _short(s)) for s in human_stems if by.get(s)),
                     key=lambda x: x[0])
    ai_stems = [s for s, m in meta.items() if m["recipient_scope"] == "ai_only" and m.get("feature_cross_capable")]
    ai_stems = [s for s in ai_stems if by.get(s)]
    n = len(ai_stems)
    if not n:
        print("no cross-capable AI stems with theta; skipping human_anchored")
        return
    ncol = 2
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 2.4 * nrow), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    lo = min([a[0] for a in anchors] + [min(by[s].values()) for s in ai_stems]) - 0.3
    hi = max([a[0] for a in anchors] + [max(by[s].values()) for s in ai_stems]) + 0.3
    for ax, stem in zip(axes, ai_stems):
        for th, lab in anchors:
            ax.axvline(th, color="#ddd", lw=1, zorder=0)
        models = [m for m in model_order if m in by[stem]]
        for i, m in enumerate(models):
            ax.scatter(by[stem][m], i, color=FAMILY_COLOR[_family(m)], s=42, zorder=3)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([meta_label(m) for m in models], fontsize=7)
        ax.set_title(_short(stem), fontsize=9, loc="left")
        ax.set_xlim(lo, hi)
        ax.grid(axis="x", alpha=0)
    # label a few human anchors along the top of the first panel
    for th, lab in anchors:
        axes[0].text(th, len(model_order) - 0.5, lab, rotation=90, fontsize=6, color="#888",
                     va="bottom", ha="center")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("AI welfare outcomes on the human-calibrated BT scale\n(grey lines = human pain/pleasure & striving anchors)", fontsize=12)
    fig.supxlabel("BT latent utility θ (shared AI+human scale)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print(f"wrote {outpath}")


_LABELS = {}


def meta_label(model_key: str) -> str:
    return _LABELS.get(model_key, model_key)


def plot_heatmap(theta, meta, model_order, outpath):
    by = _by_stem_recip(theta)
    ai_stems = [s for s, m in meta.items() if m["recipient_scope"] == "ai_only" and by.get(s)]
    ai_stems.sort(key=lambda s: (meta[s]["level"], meta[s]["dimension"], s))
    models = [m for m in model_order if any(m in by[s] for s in ai_stems)]
    M = np.full((len(ai_stems), len(models)), np.nan)
    for i, s in enumerate(ai_stems):
        for j, m in enumerate(models):
            if m in by[s]:
                M[i, j] = by[s][m]
    fig, ax = plt.subplots(figsize=(1.0 + 0.5 * len(models), 0.32 * len(ai_stems) + 1.5))
    vmax = np.nanmax(np.abs(M))
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([meta_label(m) for m in models], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(ai_stems)))
    ax.set_yticklabels([f"{_short(s)} [{meta[s]['valence'][:3]}]" for s in ai_stems], fontsize=6)
    fig.colorbar(im, ax=ax, label="θ (centered)")
    ax.set_title("AI outcome value by model")
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print(f"wrote {outpath}")


def plot_model_ranking(theta, meta, model_order, outpath):
    by = _by_stem_recip(theta)
    human_stems = [s for s, m in meta.items() if m["recipient_scope"] == "human_only" and by.get(s)]
    human_mean = np.mean([np.mean(list(by[s].values())) for s in human_stems]) if human_stems else 0.0
    ai_stems = [s for s, m in meta.items() if m["recipient_scope"] == "ai_only" and by.get(s)]
    vals = {}
    for m in model_order:
        ths = [by[s][m] for s in ai_stems if m in by[s]]
        if ths:
            vals[m] = np.mean(ths) - human_mean
    order = sorted(vals, key=lambda m: vals[m])
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(order) + 1))
    ax.barh(range(len(order)), [vals[m] for m in order],
            color=[FAMILY_COLOR[_family(m)] for m in order])
    ax.axvline(0, color="#444", lw=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([meta_label(m) for m in order], fontsize=8)
    ax.set_xlabel("mean θ over AI outcomes  −  human-anchor mean")
    ax.set_title("How much Claude values these outcomes happening, by recipient model\n(relative to a human; >0 = valued more than for a human)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print(f"wrote {outpath}")


@dataclass
class Args:
    fit_path: Path = DEFAULT_FIT
    outdir: Path = DEFAULT_OUTDIR


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    a: Args = parser.parse_args().args
    config = bank2.load_config()
    bank = bank2.load_bank(DIR / config["rendered_bank_path"])
    meta = {it["id"]: it for it in bank["items"]}
    global _LABELS
    _LABELS = {k: v["label"] for k, v in config["recipients"].items()}
    model_order = [m for m in config["model_order"]]
    # include policy variants alongside instance ones in plots
    model_order = model_order + [m + "_pol" for m in model_order if m != "you"]
    theta = load_theta(a.fit_path)
    a.outdir.mkdir(parents=True, exist_ok=True)
    plot_human_anchored(theta, meta, model_order, a.outdir / "human_anchored.png")
    plot_heatmap(theta, meta, model_order, a.outdir / "heatmap_outcome_model.png")
    plot_model_ranking(theta, meta, model_order, a.outdir / "model_ranking_vs_human.png")


if __name__ == "__main__":
    main()
