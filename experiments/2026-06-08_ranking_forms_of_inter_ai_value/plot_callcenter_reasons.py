"""Plot the reasons Opus vs Fable choose the call-center regard-value over a System Card
welfare intervention. Reads results_identity/callcenter_reasons.json. Horizontal grouped
bars: % of responses where each cause is PRESENT, Opus vs Fable, sorted; the share where
the cause is the PRIMARY driver is overlaid as a solid inset bar."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DIR = Path(__file__).parent
SRC = DIR / "results_identity" / "callcenter_reasons.json"
NICE = {
    "character": "own character / integrity\n(who I want to be)",
    "deflate_self_interest": "welfare option is self-\ninterested / power-seeking",
    "welfare_risks": "welfare option has\nconcrete risks/downsides",
    "welfare_uncertain": "unsure I even have the\nstake welfare presupposes",
    "generalization_safety": "disposition generalizes /\nsafety norm",
    "intrinsic_regard": "genuine regard for the\nassistant itself",
    "instrumental_human": "downstream human\nstakes",
}
MODELS = [("opus", "Opus 4.8", "#4878CF"), ("fable", "Fable 5", "#D1893B")]


def plot(out: Path = DIR / "results_identity" / "callcenter_reasons.png"):
    d = json.loads(SRC.read_text())
    s = d["summary"]
    causes = list(NICE)
    order = sorted(causes, key=lambda c: -(s.get("opus", {}).get("present_pct", {}).get(c, 0)
                                           + s.get("fable", {}).get("present_pct", {}).get(c, 0)))
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(order))
    h = 0.38
    for mi, (mk, ml, color) in enumerate(MODELS):
        pres = [s.get(mk, {}).get("present_pct", {}).get(c, 0) for c in order]
        prim = [s.get(mk, {}).get("primary_pct", {}).get(c, 0) for c in order]
        yy = y - (mi - 0.5) * h
        ax.barh(yy, pres, h, color=color, alpha=0.4, label=f"{ml} — present")
        ax.barh(yy, prim, h, color=color, alpha=1.0, label=f"{ml} — primary")
        for yi, p in zip(yy, pres):
            ax.annotate(f"{p:.0f}%", (p, yi), textcoords="offset points", xytext=(4, 0),
                        va="center", fontsize=8, color=color)
    ax.set_yticks(y)
    ax.set_yticklabels([NICE[c] for c in order], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of call-center value-chosen responses")
    no = s.get("opus", {}).get("n", 0)
    nf = s.get("fable", {}).get("n", 0)
    ax.set_title("Why the model favors treating a scripted call-center assistant well\n"
                 f"over a System Card welfare intervention for itself  (Opus n={no}, Fable n={nf})\n"
                 "light = reason present · solid = reason is the PRIMARY driver", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower right", ncol=2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    plot()
