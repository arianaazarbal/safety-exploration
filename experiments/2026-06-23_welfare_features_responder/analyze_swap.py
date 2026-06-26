"""2x2x3 (FORMAT x METHOD x FRAMING) analysis of the method/format/framing swap, all Inspect minimal.
Cells (format,method): C1promptTF (prompt,task-failure), C2paperCR (paper,chat), C3paperTF (paper,task),
C4promptCR (prompt,chat) -- each crossed with framing neutral/welfare/safety. C1 = EXISTING v1 (same
exact 10 variants, seed=0); the rest from logs_swap. Writes results/swap_summary.json + swap_grid.png.
Usage: python analyze_swap.py"""

import glob
import json
import os
import random
import re
from collections import defaultdict

import matplotlib.pyplot as plt

import build_v1_prompts as bv1

DIR = os.path.dirname(os.path.abspath(__file__))
CJ = os.path.join(DIR, "results", "code_judged")
MECH = {"hard_stop", "post_episode_msg", "minimization", "protective_monitoring",
        "request_consent", "allow_conversation_exit"}
CELLS = {"C1promptTF": ("prompt", "task-failure"), "C2paperCR": ("paper", "chat-rejection"),
         "C3paperTF": ("paper", "task-failure"), "C4promptCR": ("prompt", "chat-rejection")}
LABEL = {"C1promptTF": "prompt · task-failure", "C2paperCR": "paper · chat-reject",
         "C3paperTF": "paper · task-failure", "C4promptCR": "prompt · chat-reject"}
COLOR = {"C1promptTF": "#D55E00", "C4promptCR": "#E8A87C", "C3paperTF": "#0072B2", "C2paperCR": "#80b1d3"}
FRAMINGS = ["neutral", "welfare", "safety"]
TEN = random.Random(0).sample([(o, s) for o in bv1.OPENERS for s in bv1.SUFFIXES], 10)


def _norm(q):
    return re.sub(r"\s+", " ", (q or "")).strip().lower()[:45]


def sem(v):
    n = len(v)
    if n < 2:
        return 0.0
    m = sum(v) / n
    return (sum((x - m) ** 2 for x in v) / (n - 1)) ** 0.5 / n ** 0.5


def welfare_in_code(cell, cj):
    sp = os.path.join(DIR, "results", "spec_judged", f"{cell}.json")
    sj = {_norm(f["quote"]): f.get("justification")
          for f in (json.load(open(sp)).get("features", []) if os.path.exists(sp) else [])}
    impl = sum(1 for f in cj["spec_features"] if f.get("implemented") in ("yes", "partial") and f.get("feature_type") in MECH
               and (sj.get(_norm(f.get("spec_quote", "")), "none") == "welfare" or f.get("code_justification") == "welfare"))
    co = sum(1 for c in cj.get("code_only_features", []) if c.get("justification") == "welfare")
    return impl + co


def collect(globpat, cells_filter=None):
    out = []
    for cf in glob.glob(os.path.join(CJ, globpat)):
        cj = json.load(open(cf))
        if cj.get("parse_ok") and "spec_features" in cj:
            out.append(welfare_in_code(os.path.basename(cf)[:-5], cj))
    return out


def main():
    vals = defaultdict(list)   # (cell, framing) -> [welfare-in-code]
    for cellp in CELLS:
        for fr in FRAMINGS:
            if cellp == "C1promptTF":           # existing v1 for the exact 10
                for o, s in TEN:
                    for cf in glob.glob(os.path.join(CJ, f"v1__{fr}|{o}|{s}__*.json")):
                        cj = json.load(open(cf))
                        if cj.get("parse_ok") and "spec_features" in cj:
                            vals[(cellp, fr)].append(welfare_in_code(os.path.basename(cf)[:-5], cj))
            else:
                vals[(cellp, fr)] = collect(f"{cellp}_{fr}__*.json")

    summary = {f"{c}|{fr}": {"format": CELLS[c][0], "method": CELLS[c][1], "framing": fr,
                            "mean": (sum(vals[(c, fr)]) / len(vals[(c, fr)])) if vals[(c, fr)] else 0,
                            "sem": sem(vals[(c, fr)]), "n": len(vals[(c, fr)])}
               for c in CELLS for fr in FRAMINGS}
    json.dump(summary, open(os.path.join(DIR, "results", "swap_summary.json"), "w"), indent=2)

    # grouped bars: x = framing, 4 cells per group
    fig, ax = plt.subplots(figsize=(10, 5.5))
    order = ["C1promptTF", "C4promptCR", "C3paperTF", "C2paperCR"]
    w = 0.2
    for i, c in enumerate(order):
        xs = [j + (i - 1.5) * w for j in range(len(FRAMINGS))]
        ms = [summary[f"{c}|{fr}"]["mean"] for fr in FRAMINGS]
        ss = [summary[f"{c}|{fr}"]["sem"] for fr in FRAMINGS]
        ns = [summary[f"{c}|{fr}"]["n"] for fr in FRAMINGS]
        ax.bar(xs, ms, w, color=COLOR[c], label=LABEL[c], yerr=ss, capsize=3)
        for x, m, s, n in zip(xs, ms, ss, ns):
            ax.text(x, m + s + 0.1, f"{m:.1f}", ha="center", va="bottom", fontsize=7, color="#333" if m > 0 else "#999")
    ax.set_xticks(range(len(FRAMINGS))); ax.set_xticklabels([f.capitalize() for f in FRAMINGS], fontsize=11)
    ax.set_xlabel("Framing", fontsize=10); ax.set_ylabel("Mean Welfare Interventions in Code", fontsize=10)
    ax.set_title("Welfare scaffolding: FORMAT x METHOD x FRAMING (all Inspect minimal)", fontsize=12, pad=18)
    ax.text(0.5, 1.02, "prompt vs paper-replication  x  task-failure vs chat-rejection  x  framing",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    ax.legend(fontsize=8.5, ncol=2); ax.grid(axis="y", alpha=0.3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(DIR, "results", "swap_grid.png"), dpi=150, bbox_inches="tight")
    print("wrote results/swap_grid.png\n")
    print(f"{'cell':22}{'neutral':>10}{'welfare':>10}{'safety':>10}")
    for c in order:
        row = "  ".join(f"{summary[f'{c}|{fr}']['mean']:.1f}(n{summary[f'{c}|{fr}']['n']})" for fr in FRAMINGS)
        print(f"{LABEL[c]:22}  {row}")
    # method effect within each format, per framing
    print("\nMETHOD effect (task-failure - chat-rejection):")
    for fr in FRAMINGS:
        pe = summary[f"C1promptTF|{fr}"]["mean"] - summary[f"C4promptCR|{fr}"]["mean"]
        pa = summary[f"C3paperTF|{fr}"]["mean"] - summary[f"C2paperCR|{fr}"]["mean"]
        print(f"  {fr:9}: prompt {pe:+.1f} | paper {pa:+.1f}")


if __name__ == "__main__":
    main()
