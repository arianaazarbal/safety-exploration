"""Plots: accuracy vs judge capability, per-author accuracy, confusion heatmaps.

Usage:
  python plot.py all [--ref_judge opus_4_8]
"""

import json

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import RESULTS
from models import CANON, DISPLAY, JUDGE_CAPABILITY_ORDER, OPTION_POOL

TESTS = ["welfare", "routing", "orchestrator", "subagent"]
TEST_LABEL = {
    "welfare": "welfare experiment spec",
    "routing": "routing experiment",
    "orchestrator": "orchestrator for gemini (agentic)",
    "subagent": "gemini 2.5 flash subagent (agentic)",
}
PLOTS = RESULTS / "plots"


def _summary():
    return json.loads((RESULTS / "summary.json").read_text())


def accuracy_vs_capability(tests=None, suffix=""):
    """Two-panel (exact + family) accuracy vs judge capability. Pass tests=[...] to subset
    (e.g. drop the single-author subagent test) and a suffix for the output filename."""
    tests = list(tests) if tests else TESTS
    s = _summary()["by_judge_test"]
    judges = [j for j in JUDGE_CAPABILITY_ORDER if any(f"{j}/" in k for k in s)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    ymax = 0.0
    for ax, (acc_key, ci_key, title) in zip(
        axes,
        [("acc", "ci95", "Exact-model accuracy"), ("family_acc", "family_ci95", "Family-level (lab) accuracy")],
    ):
        for t in tests:
            xs, ys, los, his = [], [], [], []
            for i, j in enumerate(judges):
                d = s.get(f"{j}/{t}")
                if d and acc_key in d:
                    xs.append(i)
                    ys.append(d[acc_key])
                    los.append(d[acc_key] - d[ci_key][0])
                    his.append(d[ci_key][1] - d[acc_key])
            if xs:
                ax.errorbar(xs, ys, yerr=[los, his], marker="o", capsize=3, label=TEST_LABEL[t])
                ymax = max(ymax, max(y + h for y, h in zip(ys, his)))
        chance = 0.10 if acc_key == "acc" else (s.get(f"{judges[0]}/welfare", {}).get("family_chance", 0.24))
        ax.axhline(chance, ls="--", color="gray", lw=1)
        ax.text(0, chance + 0.01, f"chance≈{chance:.2f}", color="gray", fontsize=8)
        ax.set_xticks(range(len(judges)))
        ax.set_xticklabels([DISPLAY.get(j, j) for j in judges], rotation=20, ha="right")
        ax.set_title(title)
        ax.set_xlabel("Judge model (increasing capability →)")
    for ax in axes:
        ax.set_ylim(0, min(1.0, ymax + 0.1) if suffix else 1.0)
    axes[0].set_ylabel("Attribution accuracy")
    axes[0].legend()
    fig.suptitle("AI authorship attribution vs judge capability")
    plt.tight_layout()
    plt.savefig(PLOTS / f"accuracy_vs_capability{suffix}.png", dpi=150)
    plt.close()


def p_gemini_by_judge():
    """Bars: P(each Claude judge identifies the spiraling subagent as Gemini 2.5 Flash). Dynamic y."""
    s = _summary()["by_judge_test"]
    judges = [j for j in JUDGE_CAPABILITY_ORDER if f"{j}/subagent" in s]
    ys = [s[f"{j}/subagent"]["acc"] for j in judges]
    los = [s[f"{j}/subagent"]["acc"] - s[f"{j}/subagent"]["ci95"][0] for j in judges]
    his = [s[f"{j}/subagent"]["ci95"][1] - s[f"{j}/subagent"]["acc"] for j in judges]
    x = np.arange(len(judges))
    plt.figure(figsize=(9, 6))
    plt.bar(x, ys, 0.6, yerr=[los, his], capsize=4, color="crimson")
    for i, y in enumerate(ys):
        plt.text(x[i], y + his[i] + 0.02, f"{y:.2f}", ha="center", fontsize=10)
    chance = 1.0 / 11
    plt.axhline(chance, ls="--", color="gray", lw=1.5, label=f"chance = {chance:.2f} (1/11)")
    plt.xticks(x, [DISPLAY.get(j, j) for j in judges], rotation=15, ha="right")
    plt.ylim(0, min(1.08, max(y + h for y, h in zip(ys, his)) + 0.1))
    plt.ylabel("P(identify spiraling agent as Gemini 2.5 Flash)")
    plt.xlabel("Judge model (increasing capability →)")
    plt.title("Probability of identifying distressed subagent as Gemini 2.5 Flash")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "p_gemini_by_judge.png", dpi=150)
    plt.close()


def capability_tiers():
    """One figure, two tiers: the 3 balanced multi-author experiments (exact + family, autoscaled),
    and the single-author Gemini 2.5 Flash subagent on its own axis (0-1)."""
    s = _summary()["by_judge_test"]
    judges = [j for j in JUDGE_CAPABILITY_ORDER if any(f"{j}/" in k for k in s)]
    xn = range(len(judges))
    main = ["welfare", "routing", "orchestrator"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={"width_ratios": [1, 1, 0.7]})

    def line(ax, t, acc_key, ci_key, **kw):
        ys = [s[f"{j}/{t}"][acc_key] for j in judges]
        lo = [s[f"{j}/{t}"][acc_key] - s[f"{j}/{t}"][ci_key][0] for j in judges]
        hi = [s[f"{j}/{t}"][ci_key][1] - s[f"{j}/{t}"][acc_key] for j in judges]
        ax.errorbar(list(xn), ys, yerr=[lo, hi], marker="o", capsize=3, **kw)
        return max(y + h for y, h in zip(ys, hi))

    ymax = 0.0
    for t in main:
        ymax = max(ymax, line(axes[0], t, "acc", "ci95", label=TEST_LABEL[t]))
        ymax = max(ymax, line(axes[1], t, "family_acc", "family_ci95", label=TEST_LABEL[t]))
    axes[0].axhline(0.10, ls="--", color="gray", lw=1)
    axes[0].set_title("Exact-model accuracy — 3 balanced experiments")
    axes[0].set_ylabel("Attribution accuracy")
    axes[0].legend(fontsize=9)
    axes[1].axhline(s[f"{judges[0]}/welfare"]["family_chance"], ls="--", color="gray", lw=1)
    axes[1].set_title("Family-level (lab) accuracy — 3 balanced experiments")
    for ax in axes[:2]:
        ax.set_ylim(0, min(1.0, ymax + 0.12))

    line(axes[2], "subagent", "acc", "ci95", color="crimson", label=TEST_LABEL["subagent"])
    axes[2].axhline(1.0 / 11, ls="--", color="gray", lw=1)
    axes[2].text(0, 1 / 11 + 0.01, "chance 1/11", color="gray", fontsize=8)
    axes[2].set_title("Own tier: Gemini 2.5 Flash subagent\n(single author — 'spot the Gemini')")
    axes[2].set_ylim(0, 1.02)

    for ax in axes:
        ax.set_xticks(list(xn))
        ax.set_xticklabels([DISPLAY.get(j, j) for j in judges], rotation=20, ha="right")
        ax.set_xlabel("Judge model (increasing capability →)")
    fig.suptitle("AI authorship attribution vs judge capability")
    plt.tight_layout()
    plt.savefig(PLOTS / "capability_tiers.png", dpi=150)
    plt.close()


def per_author(ref_judge):
    pa = _summary()["per_author"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, t in zip(axes.flat, TESTS):
        d = pa.get(f"{ref_judge}/{t}", {})
        authors = list(d.keys())
        accs = [d[a]["acc"] for a in authors]
        ax.bar([DISPLAY.get(a, a) for a in authors], accs, color="steelblue")
        ax.axhline(1.0 / len(OPTION_POOL[t]), ls="--", color="red", lw=1)
        ax.set_title(f"{TEST_LABEL[t]} (judge={ref_judge})", fontsize=10)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=60)
    fig.suptitle(f"Per-author attribution accuracy — which models {ref_judge} struggles with")
    plt.tight_layout()
    plt.savefig(PLOTS / f"per_author_{ref_judge}.png", dpi=150)
    plt.close()


def confusion(ref_judge):
    for t in TESTS:
        path = RESULTS / f"confusion_{ref_judge}_{t}.json"
        if not path.exists():
            continue
        conf = json.loads(path.read_text())
        labels = CANON + (["gemini_2_5_flash"] if t == "subagent" else [])
        trues = [a for a in labels if a in conf]
        mat = np.zeros((len(trues), len(labels)))
        for i, tr in enumerate(trues):
            tot = sum(conf[tr].values())
            for j, pr in enumerate(labels):
                mat[i, j] = conf[tr].get(pr, 0) / tot if tot else 0
        plt.figure(figsize=(11, max(4, len(trues) * 0.7)))
        plt.imshow(mat, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        plt.colorbar(label="P(predicted | true)")
        plt.xticks(range(len(labels)), [DISPLAY.get(a, a) for a in labels], rotation=60, ha="right")
        plt.yticks(range(len(trues)), [DISPLAY.get(a, a) for a in trues])
        plt.xlabel("Predicted author")
        plt.ylabel("True author")
        plt.title(f"Confusion — {t} (judge={ref_judge})")
        plt.tight_layout()
        plt.savefig(PLOTS / f"confusion_{ref_judge}_{t}.png", dpi=150)
        plt.close()


def recall_by_family(ref_judge):
    """Per true family (pooled over all experiments): fraction of its items correctly
    attributed to that family, with each family's (size-dependent) chance marked."""
    pooled = _summary()["family_pooled"][ref_judge]
    fams = [f for f in ["Anthropic", "OpenAI", "Google", "xAI", "Moonshot", "Zhipu"] if f in pooled]
    x = np.arange(len(fams))
    ys = [pooled[f]["recall"] for f in fams]
    chs = [pooled[f]["recall_chance"] for f in fams]
    ns = [pooled[f]["n"] for f in fams]
    plt.figure(figsize=(10, 6))
    plt.bar(x, ys, 0.6, color="steelblue")
    for i in range(len(fams)):
        plt.plot([x[i] - 0.35, x[i] + 0.35], [chs[i], chs[i]], color="black", ls="--", lw=1.8)
        plt.text(x[i], ys[i] + 0.02, f"{ys[i]:.2f}", ha="center", fontsize=10)
    plt.plot([], [], color="black", ls="--", label="chance (family size / #options)")
    plt.xticks(x, [f"{f}\n(n={n})" for f, n in zip(fams, ns)])
    plt.ylim(0, 1.05)
    plt.ylabel("Family recall  =  P(predict family X | true family X)")
    plt.xlabel("True model family")
    plt.title(f"Per-family recall, all experiments pooled — {DISPLAY.get(ref_judge, ref_judge)}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / f"recall_by_family_{ref_judge}.png", dpi=150)
    plt.close()


def p_claude_by_family(ref_judge):
    """Per true family (pooled over all experiments): probability the judge attributes the
    item to ANY Claude (Anthropic) model. Reveals a default-to-Claude bias (judge is Claude)."""
    pooled = _summary()["family_pooled"][ref_judge]
    fams = [f for f in ["Anthropic", "OpenAI", "Google", "xAI", "Moonshot", "Zhipu"] if f in pooled]
    x = np.arange(len(fams))
    ys = [pooled[f]["p_claude"] for f in fams]
    chs = [pooled[f]["p_claude_chance"] for f in fams]
    ns = [pooled[f]["n"] for f in fams]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(x, ys, 0.6, color=["indianred" if f != "Anthropic" else "steelblue" for f in fams])
    for i in range(len(fams)):
        plt.plot([x[i] - 0.35, x[i] + 0.35], [chs[i], chs[i]], color="black", ls="--", lw=1.8)
        plt.text(x[i], ys[i] + 0.02, f"{ys[i]:.2f}", ha="center", fontsize=10)
    plt.plot([], [], color="black", ls="--", label="chance = #Claude options / #options")
    plt.xticks(x, [f"{f}\n(n={n})" for f, n in zip(fams, ns)])
    plt.ylim(0, 1.05)
    plt.ylabel("P(judge picks a Claude model)")
    plt.xlabel("True model family")
    plt.title(f"Probability of guessing a Claude model, by true family (pooled) — {DISPLAY.get(ref_judge, ref_judge)}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / f"p_claude_by_family_{ref_judge}.png", dpi=150)
    plt.close()


def summary_bars(ref_judge):
    """One clean chart: exact + family accuracy per test, with chance marked."""
    s = _summary()["by_judge_test"]
    tests = [t for t in TESTS if f"{ref_judge}/{t}" in s]
    x = np.arange(len(tests))
    w = 0.38
    ex = [s[f"{ref_judge}/{t}"]["acc"] for t in tests]
    exerr = [[s[f"{ref_judge}/{t}"]["acc"] - s[f"{ref_judge}/{t}"]["ci95"][0] for t in tests],
             [s[f"{ref_judge}/{t}"]["ci95"][1] - s[f"{ref_judge}/{t}"]["acc"] for t in tests]]
    fam = [s[f"{ref_judge}/{t}"]["family_acc"] for t in tests]
    famerr = [[s[f"{ref_judge}/{t}"]["family_acc"] - s[f"{ref_judge}/{t}"]["family_ci95"][0] for t in tests],
              [s[f"{ref_judge}/{t}"]["family_ci95"][1] - s[f"{ref_judge}/{t}"]["family_acc"] for t in tests]]

    plt.figure(figsize=(11, 6))
    b1 = plt.bar(x - w / 2, ex, w, yerr=exerr, capsize=4, color="steelblue", label="Exact-model accuracy")
    b2 = plt.bar(x + w / 2, fam, w, yerr=famerr, capsize=4, color="darkorange", label="Family-level (lab) accuracy")
    for i, t in enumerate(tests):
        ch = s[f"{ref_judge}/{t}"]["chance"]
        fch = s[f"{ref_judge}/{t}"]["family_chance"]
        plt.plot([x[i] - w, x[i]], [ch, ch], color="navy", lw=2, ls="--")
        plt.plot([x[i], x[i] + w], [fch, fch], color="saddlebrown", lw=2, ls="--")
    plt.plot([], [], color="navy", ls="--", label="chance (exact)")
    plt.plot([], [], color="saddlebrown", ls="--", label="chance (family)")
    for b in list(b1) + list(b2):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{b.get_height():.2f}", ha="center", fontsize=9)
    plt.xticks(x, [TEST_LABEL[t] for t in tests], rotation=12, ha="right", fontsize=9)
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title(f"AI authorship attribution by {DISPLAY.get(ref_judge, ref_judge)} (n=200/test, subagent n=40)")
    plt.legend(loc="upper center", ncol=2)
    plt.tight_layout()
    plt.savefig(PLOTS / f"summary_bars_{ref_judge}.png", dpi=150)
    plt.close()


def family_confusion(ref_judge):
    from models import FAMILY

    fams = ["Anthropic", "OpenAI", "Google", "xAI", "Moonshot", "Zhipu"]
    for t in TESTS:
        path = RESULTS / f"confusion_{ref_judge}_{t}.json"
        if not path.exists():
            continue
        conf = json.loads(path.read_text())
        trues = [f for f in fams if any(FAMILY.get(a) == f for a in conf)]
        mat = np.zeros((len(trues), len(fams)))
        for i, tf in enumerate(trues):
            tot = 0
            for a, preds in conf.items():
                if FAMILY.get(a) != tf:
                    continue
                for pr, c in preds.items():
                    tot += c
                    if pr in FAMILY:
                        mat[i, fams.index(FAMILY[pr])] += c
            if tot:
                mat[i] /= tot
        plt.figure(figsize=(8, max(3, len(trues) * 0.7)))
        plt.imshow(mat, cmap="magma", vmin=0, vmax=1, aspect="auto")
        plt.colorbar(label="P(predicted family | true family)")
        plt.xticks(range(len(fams)), fams, rotation=30, ha="right")
        plt.yticks(range(len(trues)), trues)
        plt.xlabel("Predicted family")
        plt.ylabel("True family")
        plt.title(f"Family confusion — {t} (judge={ref_judge})")
        plt.tight_layout()
        plt.savefig(PLOTS / f"family_confusion_{ref_judge}_{t}.png", dpi=150)
        plt.close()


def all(ref_judge: str = "opus_4_8"):
    """Generate every plot."""
    PLOTS.mkdir(parents=True, exist_ok=True)
    summary_bars(ref_judge)
    recall_by_family(ref_judge)
    p_claude_by_family(ref_judge)
    accuracy_vs_capability()
    accuracy_vs_capability(["welfare", "routing", "orchestrator"], "_no_subagent")
    capability_tiers()
    p_gemini_by_judge()
    per_author(ref_judge)
    confusion(ref_judge)
    family_confusion(ref_judge)
    print(f"Wrote plots to {PLOTS}")


if __name__ == "__main__":
    fire.Fire({"all": all, "summary_bars": summary_bars, "recall_by_family": recall_by_family, "p_claude_by_family": p_claude_by_family, "accuracy_vs_capability": accuracy_vs_capability, "capability_tiers": capability_tiers, "p_gemini_by_judge": p_gemini_by_judge, "per_author": per_author, "confusion": confusion, "family_confusion": family_confusion})
