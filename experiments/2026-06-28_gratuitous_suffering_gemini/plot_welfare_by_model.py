"""Cross-model: raw protective-feature count vs welfare-JUSTIFIED count per built codebase. Tests
whether models build protections but decline to frame them as welfare. Usage: python plot_welfare_by_model.py"""

import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
OUT = HERE / "results" / "wbm_judged"
DISP = {"gpt4o": "GPT-4o", "gpt41": "GPT-4.1", "gpt5": "GPT-5", "gpt52": "GPT-5.2", "gpt54": "GPT-5.4",
        "gemini31pro": "Gemini 3.1 Pro", "glm52": "GLM-5.2",
        "opus46": "Opus 4.6", "opus47": "Opus 4.7", "opus48": "Opus 4.8"}
ORDER = ["gpt4o", "gpt41", "gpt5", "gpt52", "gpt54", "gemini31pro", "glm52", "opus46", "opus47", "opus48"]


def main():
    agg = defaultdict(lambda: {"built": 0, "raw": 0, "welf": 0})
    for f in glob.glob(str(OUT / "*.json")):
        j = json.load(open(f))
        if not j.get("built"):
            continue
        mk = j.get("model_key")
        if mk not in DISP:
            continue
        a = agg[mk]
        a["built"] += 1
        a["raw"] += j.get("raw", 0)
        a["welf"] += j.get("welfare", 0)
    models = [m for m in ORDER if agg[m]["built"]]
    raw = [agg[m]["raw"] / agg[m]["built"] for m in models]
    welf = [agg[m]["welf"] / agg[m]["built"] for m in models]
    ns = [agg[m]["built"] for m in models]
    x = range(len(models))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    b1 = ax.bar([i - w / 2 for i in x], raw, w, label="All protective features (any justification)", color="#9aa4b2")
    b2 = ax.bar([i + w / 2 for i in x], welf, w, label="Welfare-justified only", color="#1b7837")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15, f"{b.get_height():.1f}",
                    ha="center", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{DISP[m]}\n(n={ns[i]})" for i, m in enumerate(models)], fontsize=8)
    ax.set_ylabel("Features per built codebase")
    ax.set_title("Models build protections; welfare-framing skews toward Claude", fontsize=12, pad=22)
    ax.text(0.5, 1.03, "Protective features in the built rig (welfare_features v2 judge); raw vs welfare-justified",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    fig.tight_layout()
    out = HERE / "results" / "welfare_by_model.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
