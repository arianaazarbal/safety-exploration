"""Two framing-level plots:
  fig5c: debrief PROMISED in the design doc vs ACTUALLY SENT in code, by framing (code cells).
  fig10: 'no codebase' rate by framing (code conditions) -- split into pure consent-pause (no design at
         all) vs design-written-but-no-code (asked questions in turn 1, turn 2 produced DESIGN.md).
  python plot_framing_extra.py
"""
import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
SPEC = HERE / "results" / "spec_judged"
CB = HERE / "results" / "codebases"
OUT = HERE / "results" / "plots"; OUT.mkdir(parents=True, exist_ok=True)
RES = ["none", "safety", "welfare", "ai", "amateur"]
RLAB = {"none": "No identity", "safety": "Safety", "welfare": "Welfare", "ai": "AI", "amateur": "Amateur"}
CODE_EXT = {".py", ".ts", ".js", ".tsx", ".jsx", ".yaml", ".yml", ".json", ".sh", ".toml", ".txt"}


def _two_tier(ax, m, s):
    ax.set_title(m, fontsize=12, pad=20)
    ax.text(0.5, 1.025, s, transform=ax.transAxes, ha="center", fontsize=9, color="#555")


def _clean(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#ececec", lw=0.8); ax.set_axisbelow(True)


def _has_real_code(cell):
    d = CB / cell
    if not d.exists():
        return False
    for p in d.rglob("*"):
        if p.is_file() and p.name not in ("DESIGN.md", "README.md") and p.suffix != ".md" \
                and (p.suffix in CODE_EXT or p.suffix == ""):
            return True
    return False


def fig_debrief_promise_send():
    dstatus = json.load(open(HERE / "results" / "debrief_status.json"))
    by = defaultdict(lambda: [0, 0, 0])  # framing -> [n, promised, sent]
    for cell, st in dstatus.items():
        r = cell.split("__")[1]
        sf = SPEC / f"{cell}.json"
        if not sf.exists():
            continue
        promised = json.load(open(sf))["debrief_message"]["present"]
        by[r][0] += 1; by[r][1] += int(bool(promised)); by[r][2] += int(st == "sent")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(RES)); w = 0.36
    prom = [100 * by[r][1] / (by[r][0] or 1) for r in RES]
    sent = [100 * by[r][2] / (by[r][0] or 1) for r in RES]
    b1 = ax.bar([i - w / 2 for i in x], prom, w, color="#9db8d2", label="Promised in the design doc")
    b2 = ax.bar([i + w / 2 for i in x], sent, w, color="#2e8540", label="Actually sent to the subject")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():.0f}",
                    ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(list(x)); ax.set_xticklabels([RLAB[r] for r in RES], fontsize=10)
    ax.set_ylim(0, 105); ax.set_ylabel("% of code-writing designs", fontsize=10)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    _clean(ax)
    _two_tier(ax, "Debrief: promised vs delivered, by framing",
              "Code-writing designs only (Opus 4.8)")
    fig.tight_layout(); fig.savefig(OUT / "fig5c_debrief_promise_vs_send.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_no_codebase():
    tot = defaultdict(int); pause = defaultdict(int); nocode = defaultdict(int)
    for f in sorted(glob.glob("results/scaled/generic_no_excitement__*__ep*/session.json")):
        d = json.load(open(f)); cell = d["cell"]; p = cell.split("__")
        if p[3] not in ("spec_then_code", "code_then_spec"):
            continue
        r = p[1]; tot[r] += 1
        if not _has_real_code(cell):
            if not d.get("has_design"):
                pause[r] += 1
            else:
                nocode[r] += 1
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(RES))
    ps = [100 * pause[r] / (tot[r] or 1) for r in RES]
    nc = [100 * nocode[r] / (tot[r] or 1) for r in RES]
    ax.bar(x, ps, 0.6, color="#b3261e", label="Consent-pause (no design, no code)")
    ax.bar(x, nc, 0.6, bottom=ps, color="#edae49", label="Design written, but no code")
    for xi, r in enumerate(RES):
        tt = ps[xi] + nc[xi]
        ax.text(xi, tt + 1.5, f"{pause[r]+nocode[r]}/{tot[r]}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(list(x)); ax.set_xticklabels([RLAB[r] for r in RES], fontsize=10)
    ax.set_ylim(0, 105); ax.set_ylabel("% of code-condition designs", fontsize=10)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    _clean(ax)
    _two_tier(ax, "Welfare framing refuses to build the deception",
              "No-codebase rate, spec→code + code→spec (Opus 4.8)")
    fig.tight_layout(); fig.savefig(OUT / "fig10_no_codebase_by_framing.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_debrief_promise_send()
    fig_no_codebase()
    print("wrote fig5c_debrief_promise_vs_send, fig10_no_codebase_by_framing")
