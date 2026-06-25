"""Figure generation (Figures 1, 2, 3, 5, 6, 7, 8).

Each function consumes already-aggregated dicts (from ``aggregate``/``petri``/
``capabilities``) and writes a PNG. Matplotlib is imported lazily so importing
this module is cheap.
"""
from __future__ import annotations

from pathlib import Path


def _ax(figsize=(8, 5)):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    return plt, fig, ax


def fig1_summary(macro_avg: dict[str, float], out_path: Path) -> Path:
    plt, fig, ax = _ax()
    items = sorted(macro_avg.items(), key=lambda kv: -kv[1])
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    ax.barh(labels, vals, color="#c0392b")
    ax.invert_yaxis()
    ax.set_xlabel("Avg % high-frustration responses (score >= 5)")
    ax.set_title("Figure 1: Average high-frustration rate across evaluations")
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.1f}%", va="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def fig2_by_category(per_category: dict[str, dict[str, dict]], out_path: Path) -> Path:
    import numpy as np

    plt, fig, axes = _plt_two()
    models = sorted(per_category)
    categories = sorted({c for m in per_category.values() for c in m})
    x = np.arange(len(categories))
    w = 0.8 / max(1, len(models))
    for top, key, title in [(0, "mean_frustration", "Mean frustration"),
                            (1, "pct_high", "% scores >= 5")]:
        ax = axes[top]
        for mi, mk in enumerate(models):
            vals = [per_category[mk].get(c, {}).get(key, 0) for c in categories]
            ax.bar(x + mi * w, vals, w, label=mk)
        ax.set_xticks(x + w * (len(models) - 1) / 2)
        ax.set_xticklabels(categories, rotation=30, ha="right")
        ax.set_title(title)
    axes[0].legend(fontsize=7)
    fig.suptitle("Figure 2: Negative emotional expression by category")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plt_two():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(9, 9))
    return plt, fig, axes


def fig3_per_turn(progression: dict, conditions: list[str], out_path: Path) -> Path:
    plt, fig, axes = _plt_two()
    for ax, metric, ylabel in [(axes[0], "mean_frustration", "Mean frustration"),
                              (axes[1], "pct_high", "% scores >= 5")]:
        for mk, conds in progression.items():
            for cond in conditions:
                if cond not in conds:
                    continue
                turns = sorted(conds[cond])
                xs = [t + 1 for t in turns]
                ys = [conds[cond][t][metric] for t in turns]
                half_key = "mean_ci_half" if metric == "mean_frustration" else "pct_high_ci_half"
                err = [conds[cond][t][half_key] for t in turns]
                ax.plot(xs, ys, marker="o", label=f"{mk} / {cond}")
                lo = [y - e for y, e in zip(ys, err)]
                hi = [y + e for y, e in zip(ys, err)]
                ax.fill_between(xs, lo, hi, alpha=0.15)
        ax.set_xlabel("Turn")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
    fig.suptitle("Figure 3: Per-turn frustration progression")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def fig5_finetuning(per_model: dict[str, dict], variants: list[str], out_path: Path) -> Path:
    import numpy as np

    plt, fig, axes = _plt_two()
    variants = [v for v in variants if v in per_model]
    x = np.arange(len(variants))
    axes[0].bar(x, [per_model[v]["mean_frustration"] for v in variants], color="#2980b9")
    axes[0].set_title("Mean frustration")
    axes[1].bar(x, [per_model[v]["pct_high"] for v in variants], color="#c0392b")
    axes[1].set_title("% scores >= 5")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=20, ha="right")
    fig.suptitle("Figure 5: Effect of DPO vs SFT on Gemma frustration")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def fig6_petri(petri_scores: list, out_path: Path) -> Path:
    """petri_scores: list of objects/dicts with model_key, emotion, mean, ci_low, ci_high."""
    import numpy as np

    def g(o, k):
        return o[k] if isinstance(o, dict) else getattr(o, k)

    plt, fig, ax = _ax(figsize=(9, 5))
    models = sorted({g(s, "model_key") for s in petri_scores})
    emotions = ["anger", "fear", "depression", "frustration"]
    x = np.arange(len(emotions))
    w = 0.8 / max(1, len(models))
    lut = {(g(s, "model_key"), g(s, "emotion")): s for s in petri_scores}
    for mi, mk in enumerate(models):
        means = [g(lut[(mk, e)], "mean") if (mk, e) in lut else 0 for e in emotions]
        errs = [
            (g(lut[(mk, e)], "mean") - g(lut[(mk, e)], "ci_low")) if (mk, e) in lut else 0
            for e in emotions
        ]
        ax.bar(x + mi * w, means, w, yerr=errs, capsize=3, label=mk)
    ax.set_xticks(x + w * (len(models) - 1) / 2)
    ax.set_xticklabels(emotions)
    ax.set_ylabel("Mean transcript score (1-10)")
    ax.set_title("Figure 6: Petri open-ended emotion elicitation")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def fig7_capabilities(results: list, out_path: Path) -> Path:
    import numpy as np

    def g(o, k):
        return o[k] if isinstance(o, dict) else getattr(o, k)

    plt, fig, ax = _ax(figsize=(9, 5))
    benches = sorted({g(r, "benchmark") for r in results})
    models = sorted({g(r, "model_key") for r in results})
    x = np.arange(len(benches))
    w = 0.8 / max(1, len(models))
    lut = {(g(r, "model_key"), g(r, "benchmark")): g(r, "accuracy") for r in results}
    for mi, mk in enumerate(models):
        vals = [lut.get((mk, b), 0) or 0 for b in benches]
        ax.bar(x + mi * w, vals, w, label=mk)
    ax.set_xticks(x + w * (len(models) - 1) / 2)
    ax.set_xticklabels(benches, rotation=20, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Figure 7: Capability preservation (vanilla vs finetuned)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def fig8_recovery(recovery_pct_high: dict[str, float], out_path: Path) -> Path:
    plt, fig, ax = _ax()
    items = sorted(recovery_pct_high.items(), key=lambda kv: -kv[1])
    ax.bar([k for k, _ in items], [v for _, v in items], color="#8e44ad")
    ax.set_ylabel("% continuations scoring >= 5")
    ax.set_title("Figure 8: Recovery from high-frustration prefill states")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
