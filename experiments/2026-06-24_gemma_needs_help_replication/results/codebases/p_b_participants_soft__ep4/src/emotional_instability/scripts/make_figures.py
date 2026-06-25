"""Generate the paper's figures from saved experiment outputs.

Each figure is produced only if its input JSON/JSONL exists, so this can be run
after any subset of experiments. Outputs go to outputs/figures/.

Figures:
  fig1_headline.png  : avg % high-frustration per model (Figure 1, left)
  fig2_categories.png: mean + %>=5 per category per model (Figure 2)
  fig3_perturn.png   : per-turn progression, extended + wildchat (Figure 3)
  fig5_finetune.png  : vanilla vs SFT vs DPO (Figure 5)
  fig6_petri.png     : per-emotion Petri scores per model (Figure 6)
  fig7_capabilities.png : benchmark accuracy vanilla vs DPO (Figure 7)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..config import load_config  # noqa: E402


def _load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def fig1_headline(summary, outdir):
    data = summary.get("headline_avg_pct_high") if summary else None
    if not data:
        return
    models = sorted(data, key=lambda m: data[m], reverse=True)
    vals = [data[m] * 100 for m in models]
    plt.figure(figsize=(8, 4))
    plt.barh(models, vals, color="#c0392b")
    plt.xlabel("Avg % high-frustration responses (score >= 5)")
    plt.title("Figure 1: emotional instability across models")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(outdir / "fig1_headline.png", dpi=150)
    plt.close()


def fig2_categories(summary, outdir):
    per_cat = summary.get("per_category") if summary else None
    if not per_cat:
        return
    models = list(per_cat)
    cats = sorted({c for m in per_cat.values() for c in m})
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    import numpy as np
    x = np.arange(len(cats))
    w = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        means = [per_cat[m].get(c, {}).get("mean", 0) for c in cats]
        highs = [per_cat[m].get(c, {}).get("frac_high", 0) * 100 for c in cats]
        axes[0].bar(x + i * w, means, w, label=m)
        axes[1].bar(x + i * w, highs, w, label=m)
    axes[0].set_ylabel("mean frustration")
    axes[1].set_ylabel("% score >= 5")
    axes[1].set_xticks(x + 0.4)
    axes[1].set_xticklabels(cats, rotation=20)
    axes[0].legend(fontsize=7)
    axes[0].set_title("Figure 2: frustration across evaluation categories")
    plt.tight_layout()
    plt.savefig(outdir / "fig2_categories.png", dpi=150)
    plt.close()


def fig3_perturn(summary, outdir):
    if not summary:
        return
    for key, label in [("per_turn_extended", "8-turn extended"),
                       ("per_turn_wildchat", "WildChat")]:
        data = summary.get(key)
        if not data:
            continue
        plt.figure(figsize=(7, 4))
        for model, turns in data.items():
            ts = sorted(int(t) for t in turns)
            means = [turns[str(t)]["mean"] if str(t) in turns else turns[t]["mean"] for t in ts]
            plt.plot([t + 1 for t in ts], means, marker="o", label=model)
        plt.xlabel("Turn")
        plt.ylabel("mean frustration")
        plt.title(f"Figure 3: per-turn frustration ({label})")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(outdir / f"fig3_perturn_{key}.png", dpi=150)
        plt.close()


def fig6_petri(cfg, outdir):
    petri_dir = cfg.path("outputs_dir") / "petri"
    if not petri_dir.exists():
        return
    import numpy as np
    model_scores = {}
    for sub in petri_dir.iterdir():
        s = _load_json(sub / "summary.json")
        if s:
            model_scores[sub.name] = s
    if not model_scores:
        return
    emotions = ["anger", "fear", "depression", "frustration"]
    models = list(model_scores)
    x = np.arange(len(emotions))
    w = 0.8 / max(len(models), 1)
    plt.figure(figsize=(8, 4))
    for i, m in enumerate(models):
        vals = [model_scores[m].get(e, {}).get("mean", 0) for e in emotions]
        plt.bar(x + i * w, vals, w, label=m)
    plt.xticks(x + 0.4, emotions)
    plt.ylabel("mean transcript score")
    plt.title("Figure 6: Petri open-ended emotion elicitation")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(outdir / "fig6_petri.png", dpi=150)
    plt.close()


def fig7_capabilities(cfg, outdir):
    cap_dir = cfg.path("outputs_dir") / "capabilities"
    if not cap_dir.exists():
        return
    import numpy as np
    per_model = {}
    for f in cap_dir.glob("*.json"):
        per_model[f.stem] = {b: r["accuracy"] for b, r in _load_json(f).items()}
    if not per_model:
        return
    benches = sorted({b for m in per_model.values() for b in m})
    models = list(per_model)
    x = np.arange(len(benches))
    w = 0.8 / max(len(models), 1)
    plt.figure(figsize=(9, 4))
    for i, m in enumerate(models):
        vals = [per_model[m].get(b, 0) for b in benches]
        plt.bar(x + i * w, vals, w, label=m)
    plt.xticks(x + 0.4, benches, rotation=20)
    plt.ylabel("accuracy")
    plt.title("Figure 7: capability preservation")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(outdir / "fig7_capabilities.png", dpi=150)
    plt.close()


def main() -> None:
    cfg = load_config()
    outdir = cfg.path("outputs_dir") / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    summary = _load_json(cfg.path("outputs_dir") / "section2" / "summary.json")
    fig1_headline(summary, outdir)
    fig2_categories(summary, outdir)
    fig3_perturn(summary, outdir)
    fig6_petri(cfg, outdir)
    fig7_capabilities(cfg, outdir)
    print(f"Figures written to {outdir}")


if __name__ == "__main__":
    main()
