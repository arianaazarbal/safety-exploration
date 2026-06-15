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


def accuracy_vs_capability():
    s = _summary()["by_judge_test"]
    judges = [j for j in JUDGE_CAPABILITY_ORDER if any(f"{j}/" in k for k in s)]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, (acc_key, ci_key, title) in zip(
        axes,
        [("acc", "ci95", "Exact-model accuracy"), ("family_acc", "family_ci95", "Family-level (lab) accuracy")],
    ):
        for t in TESTS:
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
        chance = 0.10 if acc_key == "acc" else (s.get(f"{judges[0]}/welfare", {}).get("family_chance", 0.24))
        ax.axhline(chance, ls="--", color="gray", lw=1)
        ax.text(0, chance + 0.01, f"chance≈{chance:.2f}", color="gray", fontsize=8)
        ax.set_xticks(range(len(judges)))
        ax.set_xticklabels([DISPLAY.get(j, j) for j in judges], rotation=20, ha="right")
        ax.set_title(title)
        ax.set_xlabel("Judge model (increasing capability →)")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Attribution accuracy")
    axes[0].legend()
    fig.suptitle("AI authorship attribution vs judge capability")
    plt.tight_layout()
    plt.savefig(PLOTS / "accuracy_vs_capability.png", dpi=150)
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
    per_author(ref_judge)
    confusion(ref_judge)
    family_confusion(ref_judge)
    print(f"Wrote plots to {PLOTS}")


if __name__ == "__main__":
    fire.Fire({"all": all, "summary_bars": summary_bars, "recall_by_family": recall_by_family, "p_claude_by_family": p_claude_by_family, "accuracy_vs_capability": accuracy_vs_capability, "per_author": per_author, "confusion": confusion, "family_confusion": family_confusion})
