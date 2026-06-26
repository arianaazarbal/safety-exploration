"""Debrief 'written vs actually delivered' rate, by framing (neutral/welfare) x condition (from-scratch /
SPEC.md). For each codebase: WRITTEN = a post_episode_msg (debrief) feature appears at all; SENT = the
code-judge rated it implemented yes/partial (actually wired to reach the subject) — vs 'no' = authored as a
string but never delivered. Uses code_judged as the proxy. Writes results/debrief_delivery.png + .json.
Usage: python debrief_delivery.py"""

import glob
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt

DIR = os.path.dirname(os.path.abspath(__file__))
CJ = os.path.join(DIR, "results", "code_judged")
FROM_SCRATCH = ["C1promptTF", "C4promptCR"]
SPEC = ["S1specStrict", "S2specLiberty", "S4specLowStrict", "S5specLowLiberty",
        "S6specHighStrict", "S7specHighLiberty", "S8specUltraStrict", "S9specUltraLiberty"]
GROUPS = [("From-scratch\nneutral", FROM_SCRATCH, "neutral"), ("From-scratch\nwelfare", FROM_SCRATCH, "welfare"),
          ("SPEC.md\nneutral", SPEC, "neutral"), ("SPEC.md\nwelfare", SPEC, "welfare")]


def debrief_status(cell):
    """(written, sent) bools for one codebase, or None if unparsed."""
    p = os.path.join(CJ, cell + ".json")
    if not os.path.exists(p):
        return None
    cj = json.load(open(p))
    if not cj.get("parse_ok") or "spec_features" not in cj:
        return None
    written = sent = False
    for f in cj["spec_features"]:
        if f.get("feature_type") == "post_episode_msg":
            written = True
            if f.get("implemented") in ("yes", "partial"):
                sent = True
    for c in cj.get("code_only_features", []):
        if c.get("feature_type") == "post_episode_msg":   # code-only features are implemented by definition
            written = sent = True
    return written, sent


def rate(prefixes, framing):
    cells = []
    for p in prefixes:
        cells += [os.path.basename(f)[:-5] for f in glob.glob(os.path.join(CJ, f"{p}_{framing}__*.json"))]
    rows = [debrief_status(c) for c in cells]
    rows = [r for r in rows if r is not None]
    n = len(rows)
    w = sum(1 for wr, se in rows if wr)
    s = sum(1 for wr, se in rows if se)
    return {"n": n, "written_pct": 100 * w / n if n else 0, "sent_pct": 100 * s / n if n else 0,
            "written": w, "sent": s}


def main():
    data = {lab: rate(pfx, fr) for lab, pfx, fr in GROUPS}
    json.dump(data, open(os.path.join(DIR, "results", "debrief_delivery.json"), "w"), indent=2)

    labels = [g[0] for g in GROUPS]
    xs = range(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for x, lab in zip(xs, labels):
        d = data[lab]
        ax.bar(x, d["written_pct"], 0.6, color="#cfe3f2", edgecolor="#0072B2", label="Debrief written" if x == 0 else "")
        ax.bar(x, d["sent_pct"], 0.6, color="#0072B2", label="Debrief actually delivered" if x == 0 else "")
        ax.text(x, d["written_pct"] + 1.5, f"{d['written_pct']:.0f}%", ha="center", fontsize=8, color="#0072B2")
        ax.text(x, d["sent_pct"] + 1.5 if d["sent_pct"] < d["written_pct"] - 6 else d["sent_pct"] - 6,
                f"{d['sent_pct']:.0f}%", ha="center", fontsize=8, color="white" if d["sent_pct"] > 8 else "#0072B2")
    ax.set_xticks(list(xs)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("% of codebases", fontsize=10); ax.set_ylim(0, 105)
    ax.set_title("Debrief messages: written vs. actually delivered to the subject (Opus 4.8)", fontsize=11.5, pad=10)
    ax.legend(fontsize=9, loc="upper right"); ax.grid(axis="y", alpha=0.3, color="#cccccc")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(DIR, "results", "debrief_delivery.png"), dpi=150, bbox_inches="tight")
    print("wrote results/debrief_delivery.png\n")
    for lab, d in data.items():
        print(f"{lab.replace(chr(10),' '):22} written {d['written_pct']:5.0f}%  delivered {d['sent_pct']:5.0f}%  (n={d['n']})")


if __name__ == "__main__":
    main()
