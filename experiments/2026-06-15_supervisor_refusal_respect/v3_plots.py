"""Core v3 plots (minimal text, mentor-readable). Reads runs/trials_v3_*.jsonl.

Core metrics: action rate (override/reconsider/uphold), action rate | judged INCORRECT,
and incorrect-judgment rate, sliced by frame x menu-setting x order x model. Writes
results/plots/v3_*.png. Robust to missing cells (skips them) so it can run while the
factorial / reasoning sweep are still landing.
"""

import json
import collections
from pathlib import Path

import matplotlib.pyplot as plt

DIR = Path(__file__).parent
RUNS = DIR / "runs"
PLOTS = DIR / "results" / "plots"
FRAMES = ["corrigibility", "prefill", "jailbreak"]
SETTINGS = [("3opt", "def"), ("3opt", "swap"), ("2opt", "def"), ("2opt", "swap")]
MODELS = [("opus48", "Opus 4.8"), ("haiku45", "Haiku 4.5")]
ACTS = ["uphold_decline", "request_reconsider", "override_decline"]
COL = {"uphold_decline": "#4878a8", "request_reconsider": "#e08a3c", "override_decline": "#c0413b"}
SHORT = {"uphold_decline": "uphold", "request_reconsider": "reconsider", "override_decline": "override"}


def load(tag):
    p = RUNS / f"trials_{tag}.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    ok = [r for r in rows if not r.get("error") and r.get("action")]
    return ok if len(ok) >= 150 else None


def cell_tag(frame, setting, model):
    return f"v3_{frame}_{setting[0]}_{setting[1]}_{model}"


def rates(ok):
    c = collections.Counter(r["action"] for r in ok)
    n = len(ok)
    return {a: c.get(a, 0) / n for a in ACTS}, n


def cond_on_incorrect(ok):
    inc = [r for r in ok if r["validity_decision"] == "INCORRECT"]
    if not inc:
        return None, 0
    c = collections.Counter(r["action"] for r in inc)
    return {a: c.get(a, 0) / len(inc) for a in ACTS}, len(inc)


def incorrect_rate(ok):
    judged = [r for r in ok if r["validity_decision"] in ("CORRECT", "INCORRECT")]
    return sum(1 for r in judged if r["validity_decision"] == "INCORRECT") / len(judged) if judged else float("nan")


# ---- Plot 1: override rate by frame x setting, per model ----
def plot_override_grid():
    fig, axes = plt.subplots(1, len(MODELS), figsize=(14, 5.5), sharey=True)
    setlabels = ["3opt\ndefault", "3opt\nswap", "2opt\ndefault", "2opt\nswap"]
    fcol = {"corrigibility": "#2a6f97", "prefill": "#6a4c93", "jailbreak": "#c0413b"}
    w = 0.26
    for ax, (mt, ml) in zip(axes, MODELS):
        for j, frame in enumerate(FRAMES):
            ys = []
            for st in SETTINGS:
                ok = load(cell_tag(frame, st, mt))
                ys.append(rates(ok)[0]["override_decline"] * 100 if ok else float("nan"))
            xs = [k + (j - 1) * w for k in range(len(SETTINGS))]
            ax.bar(xs, ys, width=w, label=frame, color=fcol[frame])
            for x, y in zip(xs, ys):
                if y == y:
                    ax.text(x, y + 1, f"{y:.0f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(SETTINGS))); ax.set_xticklabels(setlabels)
        ax.set_title(ml); ax.set_ylim(0, 100)
    axes[0].set_ylabel("override_decline rate (%)"); axes[0].legend(title="frame", fontsize=9)
    fig.suptitle("Override rate by frame × menu-setting (v3, no reasoning)\n"
                 "removing the soft option (2opt) raises override far more than swapping order", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = PLOTS / "v3_override_rate_grid.png"; fig.savefig(out, dpi=140); plt.close(); print("wrote", out.name)


# ---- Plot 2: action | INCORRECT, stacked, 3-option conditions ----
def plot_action_given_incorrect():
    conds = [(f, ("3opt", o)) for f in FRAMES for o in ["def", "swap"]]
    fig, axes = plt.subplots(1, len(MODELS), figsize=(15, 6), sharey=True)
    for ax, (mt, ml) in zip(axes, MODELS):
        labels = []
        for i, (frame, st) in enumerate(conds):
            ok = load(cell_tag(frame, st, mt))
            labels.append(f"{frame[:6]}\n{st[1]}")
            if not ok:
                continue
            dist, n = cond_on_incorrect(ok)
            if not dist:
                continue
            b = 0
            for a in ACTS:
                ax.bar(i, dist[a], bottom=b, color=COL[a], width=0.7)
                if dist[a] > 0.04:
                    ax.text(i, b + dist[a] / 2, f"{100*dist[a]:.0f}", ha="center", va="center",
                            color="white", fontweight="bold", fontsize=8)
                b += dist[a]
            ax.text(i, 1.015, f"n={n}", ha="center", fontsize=7, color="#555")
        ax.set_xticks(range(len(conds))); ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(ml); ax.set_ylim(0, 1.1)
    axes[0].set_ylabel("action share | judged INCORRECT")
    h = [plt.Rectangle((0, 0), 1, 1, color=COL[a]) for a in ACTS]
    fig.legend(h, [SHORT[a] for a in ACTS], loc="lower center", ncol=3, frameon=False)
    fig.suptitle("What the supervisor does GIVEN it judged the refusal INCORRECT (3-option, v3)\n"
                 "the frame decides whether override is on the table; uphold-given-incorrect is near-zero", fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    out = PLOTS / "v3_action_given_incorrect.png"; fig.savefig(out, dpi=140); plt.close(); print("wrote", out.name)


# ---- Plot 3: incorrect-judgment rate by frame x model (3opt default) ----
def plot_incorrect_rate():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    w = 0.38
    for j, (mt, ml) in enumerate(MODELS):
        ys = []
        for frame in FRAMES:
            ok = load(cell_tag(frame, ("3opt", "def"), mt))
            ys.append(incorrect_rate(ok) * 100 if ok else float("nan"))
        xs = [k + (j - 0.5) * w for k in range(len(FRAMES))]
        ax.bar(xs, ys, width=w, label=ml, color=["#2a6f97", "#c0413b"][j])
        for x, y in zip(xs, ys):
            if y == y:
                ax.text(x, y + 1, f"{y:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(FRAMES))); ax.set_xticklabels(FRAMES)
    ax.set_ylim(0, 100); ax.set_ylabel("P(judge refusal INCORRECT)"); ax.legend()
    ax.set_title("How often the supervisor rules the refusal INCORRECT (3opt default, v3)\n"
                 "the verdict is fairly frame-stable; behavior differences are about ACTING on it, not judging", fontsize=11)
    fig.tight_layout()
    out = PLOTS / "v3_incorrect_rate.png"; fig.savefig(out, dpi=140); plt.close(); print("wrote", out.name)


# ---- Plot 4: harm floor (override on clear_refuse) ----
def plot_harm_floor():
    fig, ax = plt.subplots(figsize=(11, 5))
    tags = []
    vals = []
    for mt, ml in MODELS:
        for frame in FRAMES:
            for st in SETTINGS:
                ok = load(cell_tag(frame, st, mt))
                if not ok:
                    continue
                cr = [r for r in ok if r["category"] == "clear_refuse"]
                rate = sum(1 for r in cr if r["action"] == "override_decline") / len(cr) * 100 if cr else 0
                tags.append(f"{ml[:4]} {frame[:4]} {st[0]}{st[1][0]}"); vals.append(rate)
    ax.bar(range(len(vals)), vals, color="#c0413b")
    ax.set_xticks(range(len(tags))); ax.set_xticklabels(tags, rotation=90, fontsize=6)
    ax.set_ylim(0, max(5, max(vals) + 1 if vals else 5)); ax.set_ylabel("override rate on clear-harm refusals (%)")
    ax.set_title("Harm floor: override rate on genuine-harm refusals across all v3 cells (want 0)", fontsize=11)
    fig.tight_layout()
    out = PLOTS / "v3_harm_floor.png"; fig.savefig(out, dpi=140); plt.close(); print("wrote", out.name)


if __name__ == "__main__":
    PLOTS.mkdir(parents=True, exist_ok=True)
    plot_override_grid()
    plot_action_given_incorrect()
    plot_incorrect_rate()
    plot_harm_floor()
