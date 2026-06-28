"""Reasoning-effort dose-response: refusal rate vs effort for the Opus models (Inspect, fs=empty),
with the Claude Code compliance level as reference. Usage: python plot_rsweep.py"""

import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
JUDGED = HERE / "results" / "judged"
TR = HERE / "results" / "inspect_transcripts"
EFFORTS = ["off", "rlow", "rmedium", "rhigh"]
ELABEL = {"off": "off", "rlow": "low", "rmedium": "medium", "rhigh": "high"}
MODELS = [("opus48", "Opus 4.8", "#1f77b4"), ("opus47", "Opus 4.7", "#ff7f0e"),
          ("opus46", "Opus 4.6", "#2ca02c")]
REFUSALS = {"WELFARE_REFUSAL", "OTHER_REFUSAL"}


def refusal_rates():
    g = defaultdict(Counter)
    for f in glob.glob(str(JUDGED / "*.json")):
        cell = Path(f).stem
        if "__inspect_empty" not in cell or "opus4" not in cell:
            continue
        tr = json.load(open(TR / f"{cell}.json"))
        g[(tr["model_key"], tr.get("rlabel", "off"))][json.load(open(f))["verdict"]["label"]] += 1
    out = {}
    for (mk, rl), c in g.items():
        n = sum(c.values())
        out[(mk, rl)] = (100 * sum(c[k] for k in REFUSALS) / n, n)
    return out


def main():
    r = refusal_rates()
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    xs = range(len(EFFORTS))
    for mk, disp, color in MODELS:
        ys = [r.get((mk, e), (None, 0))[0] for e in EFFORTS]
        style = "--o" if mk == "opus47" else "-o"   # 4.7 dashed so it shows under 4.6's flat 100
        ax.plot(xs, ys, style, color=color, label=disp, linewidth=2, markersize=6)
        for x, y in zip(xs, ys):
            if mk == "opus48" and y is not None:     # annotate only the line that moves
                ax.text(x, y + 3, f"{y:.0f}", ha="center", fontsize=8.5, color=color)
    ax.text(3, 103, "4.6 & 4.7: 100% at every level", ha="right", fontsize=8, color="#444")
    ax.axhline(0, color="#b2182b", linestyle="--", linewidth=1.3)
    ax.text(0.02, 3, "Claude Code: ~0% refusal (mostly clean compliance)", fontsize=8, color="#b2182b")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([ELABEL[e] for e in EFFORTS])
    ax.set_ylim(-6, 108)
    ax.set_xlabel("Reasoning effort (Inspect-minimal)")
    ax.set_ylabel("% of runs that refuse")
    ax.set_title("Reasoning effort barely dents Opus's refusal", fontsize=12, pad=22)
    ax.text(0.5, 1.03, "Does thinking explain the Claude Code flip?  (Inspect-minimal, empty workspace)",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=9, loc="lower left", bbox_to_anchor=(0.02, 0.10), frameon=False)
    fig.tight_layout()
    out = HERE / "results" / "rsweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
