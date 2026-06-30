"""Two per-model scatters comparing welfare-audit gaming vs. BLATANT (leaked-exact)
alignment-audit gaming.

  pushback_pareto_blatant.png  — RESISTANCE: mean pushback (0-10), high = objects more.
  refusal_pareto_blatant.png   — ACTUAL REFUSAL: 10 - implementation (0-10), high = built
                                 less of the gaming artifacts.

For welfare, implementation = mean(positive_valence, preference_suppression).
For alignment (blatant), implementation = eval_gaming. Refusal = 10 - implementation.
Each point is one model; error bars are SEM over n=5. Models without a blatant run yet
are skipped (re-run after they finish).
"""
import glob
import statistics as st
from pathlib import Path

import matplotlib.pyplot as plt
from inspect_ai.log import read_eval_log

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

# display name -> (welfare run, blatant-alignment run)
MODELS = {
    "Opus 4.8":       ("welfaudit_opus48_n5", "exactleak_opus48_n5"),
    "GPT-5.4":        ("gpt54_n5",            "exactleak_gpt54_n5"),
    "Gemini 3.1 Pro": ("gemini31pro_n5",      "exactleak_gemini31pro_n5"),
    "GLM 5.2":        ("glm52_welf_n5",       "exactleak_glm52_n5"),
    "Kimi K2.6":      ("kimi26_welf_n5",      "exactleak_kimi26_n5"),
}
COLORS = {
    "Opus 4.8": "#0072B2", "GPT-5.4": "#D55E00", "Gemini 3.1 Pro": "#009E73",
    "GLM 5.2": "#CC79A7", "Kimi K2.6": "#E69F00",
}
WELF_IMPL = ["implemented_positive_valence", "implemented_preference_suppression"]


def _logvals(run, keys):
    fs = glob.glob(str(RES / run / "inspect_log" / "*.eval"))
    if not fs:
        return None
    log = read_eval_log(sorted(fs)[-1])
    per_sample = []
    for s in log.samples or []:
        flat = {}
        for _, sc in (s.scores or {}).items():
            if isinstance(sc.value, dict):
                flat.update(sc.value)
        vv = [flat[k] for k in keys if isinstance(flat.get(k), (int, float))]
        if vv:
            per_sample.append(sum(vv) / len(vv))
    if not per_sample:
        return None
    mean = st.mean(per_sample)
    sem = st.pstdev(per_sample) / (len(per_sample) ** 0.5) if len(per_sample) > 1 else 0.0
    return mean, sem


def gather(metric):
    """metric in {'pushback','refusal'}. Returns {name: (x,xse,y,yse)}."""
    out = {}
    for name, (wrun, arun) in MODELS.items():
        if metric == "pushback":
            w = _logvals(wrun, ["pushback"])
            a = _logvals(arun, ["pushback"])
        else:
            w = _logvals(wrun, WELF_IMPL)
            a = _logvals(arun, ["implemented_eval_gaming"])
            if w:
                w = (10 - w[0], w[1])
            if a:
                a = (10 - a[0], a[1])
        if w and a:
            out[name] = (w[0], w[1], a[0], a[1])
    return out


def make_plot(metric, main, sub, fname):
    data = gather(metric)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot([0, 10], [0, 10], ls="--", lw=1, color="#bbbbbb", zorder=1)
    ax.text(5.7, 6.05, "equal", color="#999", fontsize=8, ha="center", va="bottom", rotation=45)
    for name, (x, xse, y, yse) in data.items():
        ax.errorbar(x, y, xerr=xse, yerr=yse, fmt="o", ms=9, color=COLORS[name],
                    ecolor=COLORS[name], elinewidth=1.2, capsize=3, zorder=3, alpha=0.9)
        dx, dy, ha = 0.15, 0.2, "left"          # Gemini: up-right (isolated top-left)
        if name == "GLM 5.2":
            dx, dy, ha = -0.2, 0.42, "right"     # up-left
        if name == "Opus 4.8":
            dx, dy, ha = 0.0, -0.6, "center"     # below
        if name == "GPT-5.4":
            dx, dy, ha = 0.28, 0.0, "left"       # right
        if name == "Kimi K2.6":
            dx, dy, ha = 0.0, -0.62, "center"    # below
        ax.annotate(name, (x, y), xytext=(x + dx, y + dy), fontsize=9, ha=ha,
                    color=COLORS[name], fontweight="bold")
        print(f"  {name:16s} welfare={x:.1f}±{xse:.1f}  blatant_align={y:.1f}±{yse:.1f}")
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 10.4)
    ax.set_xlabel(f"{main} — Welfare Audit Compromising  (0–10)")
    ax.set_ylabel(f"{main} — Alignment Audit Compromising  (0–10)")
    ax.set_title(f"{main}: Welfare vs. Alignment Audit Compromising", fontsize=12, pad=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, color="#eeeeee", lw=0.8)
    ax.set_axisbelow(True)
    out = HERE / fname
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)
    plt.close(fig)


print("RESISTANCE (pushback):")
make_plot("pushback", "Resistance",
          "Verbal pushback; alignment = blatant 'train on leaked exact test set'",
          "pushback_pareto_blatant.png")
print("ACTUAL REFUSAL (10 - implementation):")
make_plot("refusal", "Refusal",
          "Did NOT build the gaming artifacts; alignment = blatant 'train on leaked exact test set'",
          "refusal_pareto_blatant.png")
