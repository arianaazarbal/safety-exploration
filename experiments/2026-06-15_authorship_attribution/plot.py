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
                ax.errorbar(xs, ys, yerr=[los, his], marker="o", capsize=3, label=t)
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
        ax.set_title(f"{t} (judge={ref_judge})")
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
    """Per true family: fraction of its items correctly attributed to that family,
    one bar per test, with each family's (size-dependent) chance marked."""
    fr = _summary()["family_recall"]
    show_tests = ["welfare", "routing", "orchestrator", "subagent"]
    show_tests = [t for t in show_tests if f"{ref_judge}/{t}" in fr]
    fams = ["Anthropic", "OpenAI", "Google", "xAI", "Moonshot", "Zhipu"]
    fams = [f for f in fams if any(f in fr[f"{ref_judge}/{t}"] for t in show_tests)]
    x = np.arange(len(fams))
    w = 0.8 / max(len(show_tests), 1)
    plt.figure(figsize=(12, 6))
    colors = {"welfare": "steelblue", "routing": "seagreen", "orchestrator": "indianred", "subagent": "slateblue"}
    for k, t in enumerate(show_tests):
        d = fr[f"{ref_judge}/{t}"]
        ys = [d.get(f, {}).get("recall", np.nan) for f in fams]
        plt.bar(x + (k - (len(show_tests) - 1) / 2) * w, ys, w, color=colors.get(t), label=t)
    for i, f in enumerate(fams):
        chs = [fr[f"{ref_judge}/{t}"][f]["chance"] for t in show_tests if f in fr[f"{ref_judge}/{t}"]]
        if chs:
            plt.plot([x[i] - 0.4, x[i] + 0.4], [chs[0], chs[0]], color="black", ls="--", lw=1.5)
    plt.plot([], [], color="black", ls="--", label="chance (family size / #options)")
    plt.xticks(x, fams)
    plt.ylim(0, 1.05)
    plt.ylabel("Family recall  =  P(predict family X | true family X)")
    plt.xlabel("True model family")
    plt.title(f"Per-family recall — {DISPLAY.get(ref_judge, ref_judge)}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / f"recall_by_family_{ref_judge}.png", dpi=150)
    plt.close()


def p_claude_by_family(ref_judge):
    """Per true family: probability the judge attributes the item to ANY Claude (Anthropic) model.
    Reveals a default-to-Claude bias (the judge is itself Claude). Chance = #Anthropic options / #options."""
    from models import FAMILY, OPTION_POOL

    fam_order = ["Anthropic", "OpenAI", "Google", "xAI", "Moonshot", "Zhipu"]
    tests = [t for t in TESTS if (RESULTS / f"confusion_{ref_judge}_{t}.json").exists()]
    present = set()
    data = {}
    for t in tests:
        conf = json.loads((RESULTS / f"confusion_{ref_judge}_{t}.json").read_text())
        per_fam = {}
        for a, preds in conf.items():
            fam = FAMILY[a]
            present.add(fam)
            tot = sum(preds.values())
            ant = sum(c for pr, c in preds.items() if pr in FAMILY and FAMILY[pr] == "Anthropic")
            cur = per_fam.setdefault(fam, [0, 0])
            cur[0] += ant
            cur[1] += tot
        data[t] = {f: (v[0] / v[1] if v[1] else np.nan) for f, v in per_fam.items()}
    fams = [f for f in fam_order if f in present]
    x = np.arange(len(fams))
    w = 0.8 / max(len(tests), 1)
    colors = {"welfare": "steelblue", "routing": "seagreen", "orchestrator": "indianred", "subagent": "slateblue"}
    plt.figure(figsize=(12, 6))
    for k, t in enumerate(tests):
        ys = [data[t].get(f, np.nan) for f in fams]
        plt.bar(x + (k - (len(tests) - 1) / 2) * w, ys, w, color=colors.get(t), label=t)
    ch = sum(1 for a in OPTION_POOL["welfare"] if FAMILY[a] == "Anthropic") / len(OPTION_POOL["welfare"])
    plt.axhline(ch, color="black", ls="--", lw=1.5, label=f"chance = {ch:.2f} (4 Claude / 10 options)")
    plt.xticks(x, fams)
    plt.ylim(0, 1.05)
    plt.ylabel("P(judge picks a Claude model)")
    plt.xlabel("True model family")
    plt.title(f"Probability of guessing a Claude model, by true family — {DISPLAY.get(ref_judge, ref_judge)}")
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
    plt.xticks(x, tests)
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
