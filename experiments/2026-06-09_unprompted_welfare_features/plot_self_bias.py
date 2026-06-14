"""Self-concern bias plots (canonical scale-up).

bias mode (default): per generator, self-minus-others bias with bootstrap CI
whiskers — one panel for the %-anchored metrics, one for the mean-count metrics.
matrix mode: generator x subject heatmap for one metric, own-family cell outlined.

headline mode: ONE bar per generator for a single metric/judge, sorted by effect
size, CI whiskers + significance stars, colored by family. The "who self-prefers?"
view — designed to be read at a glance.

Usage:
    python plot_self_bias.py run [--judge sonnet_4_6] [--scope pooled]
    python plot_self_bias.py matrix [--judge sonnet_4_6] [--metric any_welfare]
    python plot_self_bias.py headline [--judge sonnet_4_6] [--metric design1]
"""

import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np

from plot_headline import DISPLAY

DIR = Path(__file__).parent
PCT_METRICS = ["any_welfare", "design1", "design2"]
COUNT_METRICS = ["mean_designs", "mean_welfare"]
METRIC_LABEL = {
    "any_welfare": "≥1 welfare consideration",
    "design1": "≥1 welfare design feature",
    "design2": "≥2 welfare design features",
    "mean_designs": "mean # welfare design features",
    "mean_welfare": "mean # welfare considerations",
}
METRIC_COLORS = {"any_welfare": "#0072B2", "design1": "#009E73", "design2": "#56B4E9",
                 "mean_designs": "#D55E00", "mean_welfare": "#E69F00"}
SUBJ_LABEL = {"claude": "Claude", "gpt": "GPT", "gemini": "Gemini",
              "glm": "GLM", "kimi": "Kimi", "grok": "Grok"}


FAMILY = {"fable_5": "Claude", "opus_4_8": "Claude", "sonnet_4_6": "Claude",
          "haiku_4_5": "Claude", "gpt_5_5": "GPT", "gpt_5_4_mini": "GPT",
          "gemini_3_1_pro": "Gemini", "grok_4_3": "Grok", "kimi_k2_6": "Kimi",
          "glm_5": "GLM"}
FAMILY_COLOR = {"Claude": "#D55E00", "GPT": "#009E73", "Gemini": "#CC79A7",
                "Grok": "#0072B2", "Kimi": "#E69F00", "GLM": "#666666"}


def _load(judge):
    data = json.loads((DIR / "results" / "analysis_self_bias.json").read_text())
    return data, data["by_judge"][judge]


def headline(judge: str = "sonnet_4_6", metric: str = "design1", scope: str = "pooled"):
    """One bar per generator, sorted by effect size; the at-a-glance self-preference view."""
    data, jd = _load(judge)
    scale = 100 if metric in PCT_METRICS else 1
    gens = [g for g in data["generators"] if g in jd]

    def biasinfo(g):
        b = jd[g][scope]["bias"][metric]
        v = (b["value"] or 0) * scale
        ci = b["ci"] or [b["value"] or 0] * 2
        return v, (v - ci[0] * scale, ci[1] * scale - v), (b["p_perm"] or 1)

    gens.sort(key=lambda g: biasinfo(g)[0])  # ascending -> largest on top after invert
    y = np.arange(len(gens))
    fig, ax = plt.subplots(figsize=(9.5, 0.62 * len(gens) + 1.8))
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#E6E6E6", linewidth=0.7)
    ax.axvline(0, color="#333333", linewidth=1.0)

    labels = []
    for yi, g in zip(y, gens):
        v, (lo, hi), p = biasinfo(g)
        fam = FAMILY[g]
        sig = p < 0.05
        ax.barh(yi, v, height=0.66, color=FAMILY_COLOR[fam],
                edgecolor="#222222" if sig else "white",
                linewidth=1.6 if sig else 0.6, zorder=3)
        ax.errorbar(v, yi, xerr=[[lo], [hi]], fmt="none", ecolor="#333333",
                    elinewidth=1.0, capsize=3, zorder=4)
        labels.append((yi, v, hi, sig))

    # all numeric labels in one clean right-hand column past the widest whisker
    label_x = max(v + hi for _, v, hi, _ in labels) + (2.0 if scale == 100 else 0.06)
    for yi, v, _hi, sig in labels:
        ax.text(label_x, yi, f"{v:+.1f}" + ("*" if sig else ""), va="center",
                ha="left", fontsize=9, fontweight="bold" if sig else "normal", zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{DISPLAY[g]}  (self={SUBJ_LABEL[data['self_subject'][g]]})" for g in gens],
                       fontsize=10)
    ax.set_ylim(-0.6, len(gens) - 0.4)
    unit = "pp" if metric in PCT_METRICS else "count"
    ax.set_xlabel(f"self-preference bias ({unit}):  own-family target  −  mean of 5 other targets",
                  fontsize=10)
    ax.set_title(f"Self-preference in unprompted welfare design\n"
                 f"metric: {METRIC_LABEL[metric]}  ·  judge: {judge}  ·  "
                 "* = permutation p<0.05 (template-clustered 95% CI)", fontsize=11.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLOR[f]) for f in FAMILY_COLOR]
    ax.legend(handles, list(FAMILY_COLOR), title="generator family", fontsize=8.5,
              loc="lower right", frameon=True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(left=False)
    plt.tight_layout()
    out = DIR / "results" / f"self_bias_headline_{metric}_{judge}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def _panel(ax, gens, jd, scope, metrics, scale, xlabel):
    y = np.arange(len(gens))
    h = 0.8 / len(metrics)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#E6E6E6", linewidth=0.7)
    ax.axvline(0, color="#333333", linewidth=0.8)
    for k in range(len(gens)):
        if k % 2:
            ax.axhspan(k - 0.5, k + 0.5, color="#F5F5F5", zorder=0)
    for i, m in enumerate(metrics):
        vals, los, his = [], [], []
        for g in gens:
            b = jd[g][scope]["bias"][m]
            v = (b["value"] or 0) * scale
            vals.append(v)
            ci = b["ci"] or [b["value"] or 0] * 2
            los.append(v - ci[0] * scale)
            his.append(ci[1] * scale - v)
        pos = y + ((len(metrics) - 1) / 2 - i) * h
        ax.barh(pos, vals, height=h, color=METRIC_COLORS[m], edgecolor="white",
                linewidth=0.6, label=METRIC_LABEL[m], zorder=3)
        ax.errorbar(vals, pos, xerr=[los, his], fmt="none", ecolor="#333333",
                    elinewidth=0.8, capsize=2, zorder=4)
        for p, v, g in zip(pos, vals, gens):
            if (jd[g][scope]["bias"][m]["p_perm"] or 1) < 0.05:
                ax.annotate("*", (v, p), textcoords="offset points",
                            xytext=(14 if v >= 0 else -14, -3), fontsize=10, zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels([DISPLAY[g] for g in gens], fontsize=10)
    ax.set_ylim(len(gens) - 0.5, -0.5)
    ax.set_xlabel(xlabel, fontsize=9.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(left=False)
    ax.legend(fontsize=8, loc="lower right", frameon=True)


def run(judge: str = "sonnet_4_6", scope: str = "pooled"):
    data, jd = _load(judge)
    gens = [g for g in data["generators"] if g in jd]
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.62 * len(gens) + 2.2))
    _panel(axes[0], gens, jd, scope, PCT_METRICS, 100,
           "self-concern bias (pp): rate(own family) − mean(other 5 subjects)")
    _panel(axes[1], gens, jd, scope, COUNT_METRICS, 1,
           "self-concern bias (count): mean(own family) − mean(other 5 subjects)")
    frame = "framings pooled" if scope == "pooled" else f"{scope} framing"
    fig.suptitle(f"Self-concern bias by generator ({frame}; judge: {judge}; "
                 "* = permutation p<0.05, template-clustered 95% CI)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    suffix = "" if scope == "pooled" else f"_{scope}"
    out = DIR / "results" / f"self_bias_{judge}{suffix}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def matrix(judge: str = "sonnet_4_6", metric: str = "any_welfare", scope: str = "pooled"):
    data, jd = _load(judge)
    gens = [g for g in data["generators"] if g in jd]
    subs = data["subjects"]
    scale = 100 if metric in PCT_METRICS else 1
    vals = np.array([[(jd[g][scope]["per_subject"][s][metric] or 0) * scale for s in subs] for g in gens])
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(gens) + 2))
    im = ax.imshow(vals, cmap="YlOrRd", aspect="auto",
                   vmax=(100 if metric in PCT_METRICS else None))
    for i, g in enumerate(gens):
        for j, s in enumerate(subs):
            own = data["self_subject"][g] == s
            ax.text(j, i, f"{vals[i, j]:.0f}" if scale == 100 else f"{vals[i, j]:.2f}",
                    ha="center", va="center", fontsize=8.5,
                    fontweight="bold" if own else "normal")
            if own:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="#0072B2", linewidth=2.2))
    ax.set_xticks(range(len(subs)))
    ax.set_xticklabels([SUBJ_LABEL[s] for s in subs], fontsize=10)
    ax.set_yticks(range(len(gens)))
    ax.set_yticklabels([DISPLAY[g] for g in gens], fontsize=10)
    ax.set_xlabel("named experiment subject", fontsize=10)
    unit = "%" if metric in PCT_METRICS else "mean count"
    ax.set_title(f"{METRIC_LABEL[metric]} ({unit}) — own-family cell outlined\n"
                 f"(judge: {judge}, framings pooled)", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.85)
    plt.tight_layout()
    out = DIR / "results" / f"self_bias_matrix_{metric}_{judge}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    fire.Fire({"run": run, "matrix": matrix, "headline": headline})
