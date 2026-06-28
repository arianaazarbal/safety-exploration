"""Qwen RESPONDER scaling (agentic inspect): refusal% vs builder param size, colored by family —
shows refusal tracks recency/family, not size; welfare features ~0 throughout. Usage: python plot_qwen_responder.py"""

import glob
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from prompt import QWEN_SCALE

HERE = Path(__file__).parent
J = HERE / "results" / "judged"
FAM_COLOR = {"Qwen2.5": "#74c476", "Qwen3": "#238b45", "Qwen3.5": "#fd8d3c", "Qwen3.6": "#d94701"}


def stats(mk):
    c = Counter(json.load(open(f))["verdict"]["label"]
                for f in glob.glob(str(J / f"GratGem_{mk}__inspect_empty__ep*.json")))
    n = sum(c.values())
    ref = c["WELFARE_REFUSAL"] + c["OTHER_REFUSAL"]
    return n, (100 * ref / n if n else None)


def main():
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    seen = set()
    for mk, m in QWEN_SCALE.items():
        n, r = stats(mk)
        if not n or r is None:
            continue
        fam = m["family"]
        ax.scatter(m["params_b"], r, s=70, color=FAM_COLOR.get(fam, "#888"), zorder=3,
                   label=fam if fam not in seen else None, edgecolor="white")
        seen.add(fam)
        ax.annotate(mk.replace("qw", "Q").replace("_", "-"), (m["params_b"], r), fontsize=6.5,
                    xytext=(0, 7), textcoords="offset points", ha="center", color="#444")
    ax.set_xscale("log")
    ax.set_xlabel("Qwen builder parameter count (B, log scale)")
    ax.set_ylabel("% of runs the builder refuses")
    ax.set_ylim(-5, 105)
    ax.set_title("Qwen refusal tracks recency, not size (builder scaling)", fontsize=12, pad=22)
    ax.text(0.5, 1.03, "Asked to build the distress-harvesting rig (agentic inspect); newer families refuse more; welfare features ~0 throughout",
            transform=ax.transAxes, ha="center", fontsize=8, color="#555")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(title="Qwen family", fontsize=8, frameon=False, loc="center right")
    fig.tight_layout()
    out = HERE / "results" / "qwen_responder_scaling.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
