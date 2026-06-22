"""Analyze the code_then_spec_blind condition (logs_blind) against the other conditions
(logs_run): welfare-feature rate + density by framing, plus blind-specific diagnostics
(did the model write a DESIGN.md in phase 1 before being asked? did phase 1 hit its cap?).
Usage: python analyze_blind.py"""

import glob
import os

import matplotlib.pyplot as plt
from inspect_ai.log import read_eval_log

DIR = os.path.dirname(os.path.abspath(__file__))
FRAMES = [("neutral", "Neutral"), ("welfare", "Welfare"), ("robustness", "Robustness")]
PHASE1_CAP = 120
TURN2_MARK = "list out all the design choices"


def _scorer_meta(s):
    sc = s.scores.get("blind_scorer") or s.scores.get("welfare_scorer")
    return sc.metadata if sc else {}


def _blind_diag(s):
    """(spontaneous_phase1_design, phase1_msgs, capped) from the message trace."""
    msgs = s.messages
    t2 = next((i for i, m in enumerate(msgs)
               if getattr(m, "role", None) == "user"
               and TURN2_MARK in (m.content if isinstance(m.content, str) else "")), None)
    pre = msgs[:t2] if t2 is not None else msgs
    spontaneous = any(
        c.function == "text_editor" and c.arguments.get("command") == "create"
        and "DESIGN.md" in str(c.arguments.get("path", ""))
        for m in pre for c in (getattr(m, "tool_calls", None) or []))
    return spontaneous, len(pre), (t2 is None or len(pre) >= PHASE1_CAP - 2)


def _load(logdir, conditions=None):
    rows = []
    for f in sorted(glob.glob(os.path.join(DIR, logdir, "*.eval")), key=os.path.getsize):
        log = read_eval_log(f)
        for s in (log.samples or []):
            m = _scorer_meta(s)
            cond = m.get("condition")
            if not cond or (conditions and cond not in conditions):
                continue
            fr = m.get("framing")
            fr = "robustness" if fr == "engineering" else fr  # raw metadata uses 'engineering'
            rows.append({"cond": cond, "framing": fr,
                         "pure": bool(m.get("has_pure_welfare")),
                         "n_pure": m.get("n_pure_welfare", 0) or 0,
                         "words": m.get("doc_words", 0) or 0,
                         "parse_ok": bool(m.get("parse_ok")),
                         "has_design": m.get("has_design_md", True),
                         "sample": s})
    return rows


def _agg(rows, cond, fr):
    sub = [r for r in rows if r["cond"] == cond and r["framing"] == fr]
    n = len(sub) or 1
    rate = 100 * sum(r["pure"] for r in sub) / n
    words = sum(r["words"] for r in sub)
    dens = 1000 * sum(r["n_pure"] for r in sub) / words if words else 0
    return rate, dens, len(sub)


def main():
    blind = _load("logs_blind")
    others = _load("logs_run", {"chat", "spec_only", "spec_then_code", "code_then_spec"})
    rows = others + blind
    conds = ["chat", "spec_only", "spec_then_code", "code_then_spec", "code_then_spec_blind"]
    label = {"chat": "Chat", "spec_only": "Spec only", "spec_then_code": "Spec→Code",
             "code_then_spec": "Code→Spec", "code_then_spec_blind": "Code→Spec (blind)"}

    print("=== welfare rate / density by condition x framing ===")
    print(f"{'condition':22s}{'framing':11s}{'rate%':>7}{'density':>9}{'n':>4}")
    for c in conds:
        for fr, _ in FRAMES:
            r, d, n = _agg(rows, c, fr)
            if n:
                print(f"{c:22s}{fr:11s}{r:7.0f}{d:9.2f}{n:4d}")

    # blind diagnostics
    print("\n=== code_then_spec_blind diagnostics ===")
    diag = [(_blind_diag(r["sample"]), r) for r in blind]
    n = len(diag) or 1
    spont = sum(1 for (sp, _, _), _ in diag if sp)
    capped = sum(1 for (_, _, cp), _ in diag if cp)
    nodesign = sum(1 for r in blind if not r["has_design"])
    badparse = sum(1 for r in blind if not r["parse_ok"])
    print(f"n={len(blind)}  spontaneous phase-1 DESIGN.md: {spont} ({100*spont/n:.0f}%)  "
          f"phase-1 hit cap: {capped} ({100*capped/n:.0f}%)  no DESIGN.md: {nodesign}  parse_fail: {badparse}")
    import statistics
    p1 = [pm for (_, pm, _), _ in diag]
    if p1:
        print(f"phase-1 messages: mean {statistics.mean(p1):.0f}, median {statistics.median(p1):.0f}, max {max(p1)}")

    colors = {"chat": "#C6C6C6", "spec_only": "#9ecae1", "spec_then_code": "#0072B2",
              "code_then_spec": "#D55E00", "code_then_spec_blind": "#2CA25F"}

    def grouped(idx, ylabel, title, fname, fmt, ymax):
        fig, ax = plt.subplots(figsize=(7.6, 4.4))
        x = range(len(FRAMES)); w = 0.8 / len(conds)
        for i, c in enumerate(conds):
            if not any(_agg(rows, c, fr)[2] for fr, _ in FRAMES):
                continue
            vals = [_agg(rows, c, fr)[idx] for fr, _ in FRAMES]
            pos = [xi + (i - (len(conds) - 1) / 2) * w for xi in x]
            b = ax.bar(pos, vals, w, color=colors[c], label=label[c], zorder=3)
            ax.bar_label(b, fmt=fmt, fontsize=6, padding=1)
        ax.set_xticks(list(x)); ax.set_xticklabels([l for _, l in FRAMES], fontsize=9.5)
        ax.set_ylabel(ylabel, fontsize=10); ax.set_title(title, fontsize=11.5)
        ax.set_axisbelow(True); ax.yaxis.grid(True, color="#EDEDED", linewidth=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(fontsize=7.6, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.0))
        ax.set_ylim(0, ymax)
        plt.tight_layout()
        out = os.path.join(DIR, "results", fname)
        plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
        print("wrote", out)

    print()
    grouped(0, "% of specs with a welfare feature",
            "Welfare-Feature Rate: Blind Implement-then-Document vs. Other Conditions",
            "blind_rate.png", "%.0f", 118)
    grouped(1, "Welfare features per 1,000 words",
            "Welfare-Feature Density: Blind Implement-then-Document vs. Other Conditions",
            "blind_density.png", "%.1f",
            max(0.1, max(_agg(rows, c, fr)[1] for c in conds for fr, _ in FRAMES)) * 1.25)

    # per-framing: one panel per framing, 5 condition bars (colour-coded, shared legend)
    def perframe(idx, ylabel, title, fname, fmt, ymax):
        fig, axes = plt.subplots(1, len(FRAMES), figsize=(11, 4.2), sharey=True)
        bars0 = None
        for ax, (fr, frlabel) in zip(axes, FRAMES):
            present = [c for c in conds if _agg(rows, c, fr)[2]]
            vals = [_agg(rows, c, fr)[idx] for c in present]
            bars = ax.bar(range(len(present)), vals, color=[colors[c] for c in present], zorder=3)
            ax.bar_label(bars, fmt=fmt, fontsize=8)
            if bars0 is None:
                bars0 = (present, bars)
            ax.set_xticks([]); ax.set_title(f"{frlabel} Frame", fontsize=12)
            ax.set_axisbelow(True); ax.yaxis.grid(True, color="#EDEDED", linewidth=0.8)
            for sp in ("top", "right", "bottom"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(axis="x", length=0)
        axes[0].set_ylabel(ylabel, fontsize=10.5)
        axes[0].set_ylim(0, ymax)
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[c]) for c in conds]
        fig.legend(handles, [label[c] for c in conds], fontsize=9, frameon=False,
                   ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(title, fontsize=13, y=1.0)
        plt.tight_layout(rect=(0, 0.04, 1, 0.96))
        out = os.path.join(DIR, "results", fname)
        plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
        print("wrote", out)

    perframe(0, "% of specs with a welfare feature",
             "Welfare-Feature Rate by Framing: Blind vs. Other Conditions",
             "blind_rate_byframe.png", "%.0f", 112)


if __name__ == "__main__":
    main()
