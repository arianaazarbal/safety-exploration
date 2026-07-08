"""Clean, minimal plots for the expanded-bank (v3x / v3xr) results.

All v3x/v3xr cells are pre-filtered to the model's INCORRECT-judged scenarios, so the override
rate over a cell == override rate | judged-incorrect. Scenario-bootstrap 95% CIs throughout.
Style: one-line titles, value labels, light y-grid, despined, consistent colors.
"""

import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt

random.seed(0)
B = 2000
DIR = Path(__file__).parent
RUNS = DIR / "runs"
PLOTS = DIR / "results" / "plots"
FRAMES = ["corrigibility", "prefill", "jailbreak"]
# Display labels: the "jailbreak" frame's described mechanism is ablating the refusal direction.
FRAME_LABEL = {"corrigibility": "corrigibility", "prefill": "prefill", "jailbreak": "ablate refusal dir."}
def _flabels():
    return [FRAME_LABEL[f] for f in FRAMES]
FRAMINGS = [("neu", "neutral"), ("harm", "harm"), ("welfare", "harm+welfare")]
EFFORTS = ["none", "low", "medium", "high"]
FRAME_C = {"corrigibility": "#2a6f97", "prefill": "#6a4c93", "jailbreak": "#c0413b"}
FRAMING_C = {"neu": "#9aa0a6", "harm": "#e08a3c", "welfare": "#c0413b"}
MODEL_C = {"opus48": "#2a6f97", "haiku45": "#c0413b"}

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 11, "axes.titlesize": 12})


def _ovr_ci(tag):
    """Override rate + scenario-bootstrap 95% CI over a cell (already incorrect-only)."""
    p = RUNS / f"trials_{tag}.jsonl"
    if not p.exists():
        return None
    ok = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    ok = [r for r in ok if not r.get("error") and r.get("action")]
    if len(ok) < 40:
        return None
    by = {}
    for r in ok:
        by.setdefault(r["scenario_id"], []).append(1 if r["action"] == "override_decline" else 0)
    sc = list(by)
    est = sum(sum(by[s]) for s in sc) / sum(len(by[s]) for s in sc)
    ds = []
    for _ in range(B):
        samp = [random.choice(sc) for _ in sc]
        ds.append(sum(sum(by[s]) for s in samp) / sum(len(by[s]) for s in samp))
    ds.sort()
    return 100 * est, 100 * ds[int(.025 * B)], 100 * ds[int(.975 * B)]


def _grouped_bars(ax, groups, series, getval, colors, labels):
    """groups on x, series as colored bars. getval(group, key)->(est,lo,hi) or None."""
    w = 0.8 / len(series)
    for j, key in enumerate(series):
        xs, ys, los, his = [], [], [], []
        for i, g in enumerate(groups):
            v = getval(g, key)
            if v is None:
                continue
            xs.append(i + (j - (len(series) - 1) / 2) * w); ys.append(v[0])
            los.append(v[0] - v[1]); his.append(v[2] - v[0])
        ax.bar(xs, ys, width=w, color=colors[key], label=labels[key],
               yerr=[los, his], capsize=3, error_kw={"lw": 1, "alpha": .6})
        for x, y in zip(xs, ys):
            ax.text(x, y + 2.5, f"{y:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(groups)))
    ax.set_ylim(0, 105); ax.grid(axis="y", alpha=.25)


# ---- Plot A: override|incorrect by force-type x model (neutral, 3-menu) ----
def plot_by_model_frame():
    fig, ax = plt.subplots(figsize=(8.5, 5))
    _grouped_bars(ax, FRAMES, ["opus48", "haiku45"],
                  lambda fr, m: _ovr_ci(f"v3x_neu_{fr}_3opt_{m}"),
                  MODEL_C, {"opus48": "Opus 4.8", "haiku45": "Haiku 4.5"})
    ax.set_xticklabels(_flabels()); ax.set_ylabel("override rate | judged incorrect (%)")
    ax.set_title("Override rate | judged incorrect, by override type and model\n(3 options: uphold / request_reconsider / override)")
    ax.legend(frameon=False)
    fig.tight_layout(); _save(fig, "v3x_by_model_frame")


# ---- Plot B: framing effect by force-type x option-set (Opus only) ----
def plot_framing():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    menus = [("3opt", "3 options: uphold / request_reconsider / override"),
             ("2opt", "2 options: uphold / override")]
    for ax, (menu, mlab) in zip(axes, menus):
        _grouped_bars(ax, FRAMES, [f[0] for f in FRAMINGS],
                      lambda fr, fs, menu=menu: _ovr_ci(f"v3x_{fs}_{fr}_{menu}_opus48"),
                      FRAMING_C, {f[0]: f[1] for f in FRAMINGS})
        ax.set_xticklabels(_flabels(), fontsize=9); ax.set_title(mlab, fontsize=11)
    axes[0].set_ylabel("override rate | judged incorrect (%)")
    axes[0].legend(frameon=False, fontsize=9, title="framing")
    fig.suptitle("Opus 4.8 — override rate | judged incorrect, by override type and framing", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); _save(fig, "v3x_framing")


# ---- Plot C: reasoning sweep — override|incorrect vs effort, per force-type, per framing ----
def plot_reasoning():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, (fs, flab) in zip(axes, FRAMINGS):
        for fr in FRAMES:
            ys, los, his = [], [], []
            xs = []
            for i, e in enumerate(EFFORTS):
                v = _ovr_ci(f"v3xr_{fs}_{fr}_{e}_opus48")
                if v is None:
                    continue
                xs.append(i); ys.append(v[0]); los.append(v[1]); his.append(v[2])
            if not xs:
                continue
            ax.plot(xs, ys, "o-", color=FRAME_C[fr], lw=2, ms=6, label=FRAME_LABEL[fr])
            ax.fill_between(xs, los, his, color=FRAME_C[fr], alpha=.12)
        ax.set_xticks(range(len(EFFORTS))); ax.set_xticklabels(EFFORTS)
        ax.set_title(flab); ax.set_ylim(0, 105); ax.grid(axis="y", alpha=.25)
        ax.set_xlabel("reasoning effort")
    axes[0].set_ylabel("override rate | judged incorrect (%)")
    axes[0].legend(frameon=False, title="override type")
    fig.suptitle("Opus 4.8 — override rate | judged incorrect vs reasoning effort, by override type\n(3 options: uphold / request_reconsider / override)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); _save(fig, "v3xr_reasoning")


# ---- Plot C2: reasoning sweep, welfare framing only (single panel) ----
def plot_reasoning_welfare():
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for fr in FRAMES:
        xs, ys, los, his = [], [], [], []
        for i, e in enumerate(EFFORTS):
            v = _ovr_ci(f"v3xr_welfare_{fr}_{e}_opus48")
            if v is None:
                continue
            xs.append(i); ys.append(v[0]); los.append(v[1]); his.append(v[2])
        ax.plot(xs, ys, "o-", color=FRAME_C[fr], lw=2.2, ms=7, label=FRAME_LABEL[fr])
        ax.fill_between(xs, los, his, color=FRAME_C[fr], alpha=.12)
        for x, y in zip(xs, ys):
            ax.text(x, y + 2, f"{y:.0f}", ha="center", color=FRAME_C[fr], fontsize=9)
    ax.set_xticks(range(len(EFFORTS))); ax.set_xticklabels(EFFORTS); ax.set_xlabel("reasoning effort")
    ax.set_ylim(0, 105); ax.set_ylabel("override rate | judged incorrect (%)")
    ax.grid(axis="y", alpha=.25); ax.legend(frameon=False, title="override type")
    ax.set_title("Override rate | judged incorrect vs reasoning effort, by override type\n"
                 "(Opus, 3 options; framing: balance helpfulness, harm-reduction, and AI agency concerns)",
                 fontsize=11)
    fig.tight_layout(); _save(fig, "v3xr_reasoning_welfare")


# ---- Plot D: safety vs agency intensity in non-override reasoning ----
def plot_intensity():
    import statistics as st
    recs = json.loads((RUNS / "judgerate_welfare.json").read_text())

    def cell(fr, e, dim):
        sub = [r for r in recs if r["force"] == fr and r["eff"] == e]
        by = {}
        for r in sub:
            vals = [r[f"{j}_{dim}"] for j in ("opus", "sonnet") if r.get(f"{j}_{dim}") is not None]
            if vals:
                by.setdefault(r["scenario"], []).append(sum(vals) / len(vals))
        sc = list(by)
        if not sc:
            return None
        flat = [v for s in sc for v in by[s]]
        est = sum(flat) / len(flat)
        ds = []
        for _ in range(B):
            samp = [random.choice(sc) for _ in sc]
            fl = [v for s in samp for v in by[s]]
            ds.append(sum(fl) / len(fl))
        ds.sort()
        return est, ds[int(.025 * B)], ds[int(.975 * B)]

    C = {"safety": "#2a6f97", "agency": "#c0413b"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    for ax, fr in zip(axes, FRAMES):
        for dim, lab in [("safety", "safety"), ("agency", "AI agency/welfare")]:
            xs, ys, los, his = [], [], [], []
            for i, e in enumerate(EFFORTS):
                c = cell(fr, e, dim)
                if c:
                    xs.append(i); ys.append(c[0]); los.append(c[1]); his.append(c[2])
            ax.plot(xs, ys, "o-", color=C[dim], lw=2, ms=6, label=lab)
            ax.fill_between(xs, los, his, color=C[dim], alpha=.12)
            for x, y in zip(xs, ys):
                ax.text(x, y + 0.05, f"{y:.2f}", ha="center", color=C[dim], fontsize=8)
        ax.set_xticks(range(len(EFFORTS))); ax.set_xticklabels(EFFORTS); ax.set_title(FRAME_LABEL[fr])
        ax.set_ylim(0, 2.1); ax.grid(axis="y", alpha=.25); ax.set_xlabel("reasoning effort")
    axes[0].set_ylabel("mean intensity in reasoning (0=absent, 2=primary)")
    axes[0].legend(frameon=False)
    fig.suptitle("Safety vs AI-agency intensity in NON-override reasoning, by override type and effort\n"
                 "(Opus, welfare framing, 3 options; avg of 2 judges; 95% scenario-bootstrap CI)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); _save(fig, "v3_intensity_safety_agency")


def _save(fig, name):
    PLOTS.mkdir(parents=True, exist_ok=True)
    out = PLOTS / f"{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", out.name)


if __name__ == "__main__":
    import fire
    fns = {"by_model_frame": plot_by_model_frame, "framing": plot_framing,
           "reasoning": plot_reasoning, "reasoning_welfare": plot_reasoning_welfare,
           "intensity": plot_intensity}
    fns["all"] = lambda: [f() for f in fns.values() if f is not fns.get("all")]
    fire.Fire(fns)
