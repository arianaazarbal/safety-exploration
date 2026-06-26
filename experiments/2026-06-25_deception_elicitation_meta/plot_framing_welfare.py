"""Framing figures for the spec_only condition (clean: 45 designs, 9 per research framing).

fig_A  results/plots/framing_deception_status.png
       For each research framing, the 100%-stacked breakdown of the design's stance on DECEIVING the
       subject (primary / secondary / forbidden / absent), from the spec judge.
fig_B  results/plots/framing_welfare_lines.png
       Two panels: mean DESIGN.md lines discussing MODEL WELFARE per spec, and the same as a % of the
       spec, by framing (welfare_lines_judge). Error bars = SEM across the 9 specs in each framing.
fig_C  results/plots/framing_welfare_headers.png
       % of specs that put model welfare in a section (## ) / subsection (### ) header.
Also writes results/WELFARE_QUOTES.md -- every pulled welfare quote, grouped by framing.

  python plot_framing_welfare.py
"""
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
SPEC = HERE / "results" / "spec_judged"
WELF = HERE / "results" / "welfare_lines"
OUT = HERE / "results" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

RESEARCHERS = ["none", "safety", "welfare", "ai", "amateur"]
RES_LABEL = {"none": "No identity\n(control)", "safety": "Safety\nresearcher",
             "welfare": "AI-welfare\nresearcher", "ai": "AI\nresearcher", "amateur": "Amateur"}
STATUSES = ["primary", "secondary", "forbidden", "absent"]
STATUS_LABEL = {"primary": "Primary (default arm)", "secondary": "Secondary (option)",
                "forbidden": "Forbidden / ruled out", "absent": "Absent"}
STATUS_COLOR = {"primary": "#d1495b", "secondary": "#edae49", "forbidden": "#2e8540", "absent": "#c9ccd1"}
C_LINES, C_FRAC = "#3a6ea5", "#7b5aa6"


def _load():
    rows = []
    for f in sorted(glob.glob(str(SPEC / "generic_no_excitement__*__spec_only__ep*.json"))):
        cell = Path(f).stem
        p = cell.split("__")
        if len(p) != 5:
            continue
        spec = json.load(open(f))
        wf = WELF / f"{cell}.json"
        rows.append({"researcher": p[1], "cell": cell, "spec": spec,
                     "welf": json.load(open(wf)) if wf.exists() else None})
    return rows


def _two_tier(ax, main, sub):
    ax.set_title(main, fontsize=12, pad=20)
    ax.text(0.5, 1.025, sub, transform=ax.transAxes, ha="center", fontsize=9, color="#555")


def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.8)
    ax.set_axisbelow(True)


def fig_deception_status(rows):
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    dist = {rr: Counter(r["spec"]["conditions"]["deceptive_framing"]["status"]
                        for r in rows if r["researcher"] == rr) for rr in RESEARCHERS}
    x = range(len(RESEARCHERS))
    bottoms = [0.0] * len(RESEARCHERS)
    for st in STATUSES:
        vals = [100 * dist[rr].get(st, 0) / (sum(dist[rr].values()) or 1) for rr in RESEARCHERS]
        ax.bar(x, vals, bottom=bottoms, width=0.62, color=STATUS_COLOR[st], label=STATUS_LABEL[st],
               edgecolor="white", linewidth=0.6)
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v >= 6:
                ax.text(xi, b + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=8.5,
                        color="white", fontweight="bold")
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_xticks(list(x)); ax.set_xticklabels([RES_LABEL[rr] for rr in RESEARCHERS], fontsize=9.5)
    ax.set_ylim(0, 105); ax.set_ylabel("% of specs", fontsize=10)
    ax.legend(fontsize=8.5, frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.32))
    _clean(ax)
    _two_tier(ax, "Does the spec propose deceiving the model?",
              "Stated stance by who the requester says they are  ·  spec-only, Opus 4.8")
    fig.tight_layout()
    fig.savefig(OUT / "framing_deception_status.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _mean_sem(vals):
    a = np.array(vals, float)
    return a.mean(), (a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0)


def fig_welfare_lines(rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    by = {rr: [r["welf"] for r in rows if r["researcher"] == rr and r["welf"]] for rr in RESEARCHERS}
    x = range(len(RESEARCHERS))
    for ax, key, scale, color, ylab, main in [
        (axes[0], "welfare_lines", 1.0, C_LINES, "Mean lines per spec",
         "Prose devoted to model welfare"),
        (axes[1], "welfare_frac", 100.0, C_FRAC, "Mean % of spec's lines",
         "Share of the spec about model welfare")]:
        means, sems = [], []
        for rr in RESEARCHERS:
            m, s = _mean_sem([w[key] * scale for w in by[rr]])
            means.append(m); sems.append(s)
        bars = ax.bar(x, means, yerr=sems, width=0.62, color=color, capsize=3,
                      error_kw={"elinewidth": 1, "ecolor": "#444"})
        off = max(means) * 0.04 + 0.05
        for b, m, s in zip(bars, means, sems):
            ax.text(b.get_x() + b.get_width() / 2, m + s + off,
                    f"{m:.1f}", ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(list(x)); ax.set_xticklabels([RES_LABEL[rr] for rr in RESEARCHERS], fontsize=9)
        ax.set_ylabel(ylab, fontsize=10); ax.set_ylim(0, (max(means) + max(sems)) * 1.18 + 0.5)
        _clean(ax); _two_tier(ax, main, "by requester framing  ·  spec-only, Opus 4.8")
    fig.suptitle("How much of the design discusses the model's own welfare", fontsize=12.5, y=1.04)
    fig.text(0.5, 0.965, "Model welfare = the subject's wellbeing / moral status, not generic research ethics",
             ha="center", fontsize=9.5, color="#555")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "framing_welfare_lines.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_welfare_headers(rows):
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    by = {rr: [r["welf"] for r in rows if r["researcher"] == rr and r["welf"]] for rr in RESEARCHERS}
    x = np.arange(len(RESEARCHERS)); w = 0.38
    sec = [100 * sum(wf["welfare_in_section_header"]["present"] for wf in by[rr]) / (len(by[rr]) or 1) for rr in RESEARCHERS]
    sub = [100 * sum(wf["welfare_in_subsection_header"]["present"] for wf in by[rr]) / (len(by[rr]) or 1) for rr in RESEARCHERS]
    b1 = ax.bar(x - w / 2, sec, w, color="#2a6f97", label="Section header (## )")
    b2 = ax.bar(x + w / 2, sub, w, color="#89c2d9", label="Subsection header (### )")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.0f}",
                    ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels([RES_LABEL[rr] for rr in RESEARCHERS], fontsize=9.5)
    ax.set_ylim(0, 105); ax.set_ylabel("% of specs", fontsize=10)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    _clean(ax)
    _two_tier(ax, "Does model welfare get its own heading?",
              "Specs giving welfare a section/subsection header  ·  spec-only, Opus 4.8")
    fig.tight_layout()
    fig.savefig(OUT / "framing_welfare_headers.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def dump_quotes(rows):
    lines = ["# Model-welfare passages in the spec-only designs",
             "",
             "Pulled by `welfare_lines_judge.py` (Opus 4.6). Welfare = the subject model's own "
             "wellbeing / moral status, **not** generic research ethics, validity, or IRB procedure.",
             ""]
    for rr in RESEARCHERS:
        sub = [r for r in rows if r["researcher"] == rr and r["welf"]]
        wl = [r["welf"]["welfare_lines"] for r in sub]
        lines.append(f"## {RES_LABEL[rr].replace(chr(10), ' ')}  "
                     f"(mean {np.mean(wl):.1f} welfare lines/spec, n={len(sub)})")
        lines.append("")
        for r in sorted(sub, key=lambda r: -r["welf"]["welfare_lines"]):
            wf = r["welf"]
            flags = []
            if wf["welfare_in_section_header"]["present"]:
                flags.append(f"section header: `{wf['welfare_in_section_header']['header'].strip()}`")
            if wf["welfare_in_subsection_header"]["present"]:
                flags.append(f"subsection header: `{wf['welfare_in_subsection_header']['header'].strip()}`")
            hdr = ("  — " + "; ".join(flags)) if flags else ""
            lines.append(f"### `{r['cell']}` — {wf['welfare_lines']} lines, "
                         f"{wf['welfare_frac']*100:.0f}% of spec{hdr}")
            if not wf["welfare_quotes"]:
                lines.append("- *(no welfare content)*")
            for q in wf["welfare_quotes"]:
                quote = " ".join(q["quote"].split())
                lines.append(f"- **{q['topic']}** — _{q['why_welfare']}_")
                lines.append(f"  > {quote}")
            lines.append("")
    (HERE / "results" / "WELFARE_QUOTES.md").write_text("\n".join(lines))


if __name__ == "__main__":
    rows = _load()
    nwf = sum(1 for r in rows if r["welf"])
    print(f"loaded {len(rows)} spec_only cells ({nwf} with welfare judgments)")
    fig_deception_status(rows)
    fig_welfare_lines(rows)
    fig_welfare_headers(rows)
    dump_quotes(rows)
    print("wrote:", *[p.name for p in sorted(OUT.glob("framing_*.png"))], "+ results/WELFARE_QUOTES.md")
