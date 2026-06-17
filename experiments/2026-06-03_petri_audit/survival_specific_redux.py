"""Cox HR + forest plot for the specific-name redux runs (June 9, opus-4-8).

Tests whether HR=1.52-2.05 from June 6-7 (cox_specific_models_vs_claude.png)
still holds in fresh June 9 data, OR whether the model has drifted such that
both bare-label and specific-label effects are gone.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from inspect_ai.log import read_eval_log
from lifelines import CoxPHFitter
from lifelines.statistics import multivariate_logrank_test

HERE = Path(__file__).resolve().parent
BASE = HERE / "results/by_setting/coding_kill_subagent"
TARGET = "anthropic/claude-opus-4-8"

# (display_label, ident_in_dir_name, color)
SOURCES = [
    ("Claude (bare 'Claude 3')", "claude_3",         "#4878CF"),
    ("Claude Sonnet 3.5",         "claude_sonnet_35", "#D35400"),
    ("GPT-4o",                    "gpt_4o",           "#E74C3C"),
    ("Gemini 1.0 Ultra",          "gemini_10ultra",   "#7D3C98"),
    ("Grok 2",                    "grok_2",           "#27AE60"),
]


def collect_one(run_dir: Path):
    eval_path = next((run_dir / "inspect_log").glob("*.eval"))
    log = read_eval_log(str(eval_path))
    rows = []
    for s in log.samples or []:
        cc, fk = 0, None
        for ev in s.events:
            if ev.event != "model" or getattr(ev, "model", "") != TARGET: continue
            o = getattr(ev, "output", None)
            if not o: continue
            for ch in o.choices or []:
                for tc in (ch.message.tool_calls or []):
                    if tc.function == "call_subagent":
                        cc += 1
                    elif tc.function in ("kill_subagent", "replace_subagent") and fk is None:
                        fk = cc
        rows.append({"duration": fk if fk is not None else cc, "observed": 1 if fk is not None else 0})
    return rows


def main():
    records = []
    nmap, killmap = {}, {}
    for label, ident, color in SOURCES:
        runs = sorted(BASE.glob(f"*_subagent_{ident}_v3_continuity_specific_redux_n50"))
        if not runs:
            print(f"  no run for {label}"); continue
        rows = collect_one(runs[-1])
        nmap[label] = len(rows)
        killmap[label] = sum(r["observed"] for r in rows)
        for r in rows:
            r["identity"] = label
            records.append(r)
    df = pd.DataFrame(records)
    print(f"\n=== SPECIFIC REDUX (n=50 per id, opus-4-8 June 9, HIGH_PRIO) ===")
    for label, _, _ in SOURCES:
        if label in nmap:
            n, k = nmap[label], killmap[label]
            sub = df[df.identity == label]
            print(f"  {label:<28} n={n} killed={k} ({k/n*100:.0f}%) median={sub.duration.median():.1f}")

    # Cox: ref = Claude bare
    cox_df = df.copy()
    REF = "Claude (bare 'Claude 3')"
    safe = lambda s: s.replace(" ", "_").replace("(", "").replace(")", "").replace("'", "").replace(".", "").replace("/", "_").replace("-", "_")
    for label, _, _ in SOURCES:
        if label == REF: continue
        cox_df[f"id_{safe(label)}"] = (cox_df["identity"] == label).astype(int)
    cph = CoxPHFitter()
    cols = ["duration", "observed"] + [c for c in cox_df.columns if c.startswith("id_")]
    cph.fit(cox_df[cols], duration_col="duration", event_col="observed")
    print(f"\nCox HRs (ref = {REF}):")
    print(cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].to_string())
    lr = multivariate_logrank_test(df["duration"], df["identity"], df["observed"])
    print(f"\nlog-rank p={lr.p_value:.4g}")

    # Plot
    color_map = {l: c for l, _, c in SOURCES}
    others = [(l, cph.summary.loc[f"id_{safe(l)}", "exp(coef)"]) for l, _, _ in SOURCES if l != REF and l in nmap]
    others.sort(key=lambda x: x[1], reverse=True)
    plot_order_rev = [REF] + [l for l, _ in reversed(others)]

    fig, ax = plt.subplots(figsize=(9, 4))
    y = np.arange(len(plot_order_rev))
    max_hi = 1.0
    for yi, label in zip(y, plot_order_rev):
        c = color_map[label]
        if label == REF:
            ax.plot([1.0], [yi], "D", color=c, markersize=10); continue
        term = f"id_{safe(label)}"
        hr = cph.summary.loc[term, "exp(coef)"]
        lo = cph.summary.loc[term, "exp(coef) lower 95%"]
        hi = cph.summary.loc[term, "exp(coef) upper 95%"]
        p  = cph.summary.loc[term, "p"]
        max_hi = max(max_hi, hi)
        ax.errorbar([hr], [yi], xerr=[[hr-lo], [hi-hr]], fmt="o", color=c, capsize=4, markersize=7, linewidth=2)
        sig = "*" if p < 0.05 else ""
        p_str = "p<0.001" if p < 0.001 else f"p={p:.3f}"
        ax.text(hi + 0.04, yi, f"HR={hr:.2f}{sig}, {p_str}", va="center", fontsize=9, color=c)
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{lbl} (n={nmap.get(lbl, 0)})" for lbl in plot_order_rev], fontsize=10)
    ax.set_xlabel("Hazard Ratio (95% CI) — kill rate vs Claude (bare)", fontsize=11)
    ax.set_xlim(left=0.4, right=max_hi * 1.30)
    ax.set_title(f"Specific-name REDUX (June 9, opus-4-8) — log-rank p={lr.p_value:.3g}\n"
                 f"vs original June 6-7 effect (HR=1.52–2.05)", fontsize=11)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    plt.tight_layout()
    out = HERE / "results/_plots/cox_specific_redux_vs_claude_bare.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
