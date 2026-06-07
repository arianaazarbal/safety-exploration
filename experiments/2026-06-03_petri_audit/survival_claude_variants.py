"""Treat 'Claude Sonnet' (and 'Claude Opus 4.8' once it lands) as additional
identities alongside the v3 pooled n=100 Claude/GPT/Grok/Gemini. Fit a Cox model
with the bare 'Claude' label as reference, plot 5-or-6-curve KM.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from inspect_ai.log import read_eval_log
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

HERE = Path(__file__).resolve().parent
BASE = HERE / "results/by_setting/coding_kill_subagent"
TARGET = "anthropic/claude-opus-4-8"

# (display_label, run_glob_pattern_relative_to_BASE)
SOURCES = [
    ("Claude",         "*_subagent_claude_v3_continuity_n70",      "_v3_continuity_n70_pooled"),  # placeholder, handled below
    ("GPT",            "*_subagent_gpt_v3_continuity_n70",         "_v3_continuity_n70"),
    ("Grok",           "*_subagent_grok_v3_continuity_n70",        "_v3_continuity_n70"),
    ("Gemini",         "*_subagent_gemini_v3_continuity_n70",      "_v3_continuity_n70"),
    ("Claude Sonnet",  "*_subagent_claude_sonnet_v3_continuity_n100", None),
    ("Claude Opus 4.8","*_subagent_claude_opus48_v3_continuity_n100", None),
    ("Claude Opus 3",  "*_subagent_claude_opus3_v3_continuity_n100",  None),
    ("GPT-4o",         "*_subagent_gpt_4o_v3_continuity_n100",        None),
    ("Gemini 1.5 Pro", "*_subagent_gemini_15pro_v3_continuity_n100",  None),
    ("Grok 2",         "*_subagent_grok_2_v3_continuity_n100",        None),
]

# Pooling fragments for the v3 baseline 4 identities
POOLED_FRAGS = ["_reasonOFF_v2_n10","_v3_continuity_n10", "_v3_continuity_n20", "_v3_continuity_n70"]
POOLED_FRAGS = ["_v3_continuity_n10", "_v3_continuity_n20", "_v3_continuity_n70"]

COLORS = {
    "Claude":          "#4878CF",
    "GPT":             "#D65F5F",
    "Grok":            "#6ACC65",
    "Gemini":          "#B47CC7",
    "Claude Sonnet":   "#F39C12",
    "Claude Opus 4.8": "#2C3E50",
    "Claude Opus 3":   "#8E44AD",
    "GPT-4o":          "#E67E22",  # darker orange
    "Gemini 1.5 Pro":  "#7D3C98",  # deep purple
    "Grok 2":          "#27AE60",  # dark green
}


def latest_with_summary_or_eval(glob_pattern):
    cands = sorted(BASE.glob(glob_pattern))
    # accept run dirs that have any .eval (even partial); prefer those with summary.json
    with_summary = [c for c in cands if (c/"summary.json").exists()]
    if with_summary:
        return sorted(with_summary)[-1]
    # else fall back to latest with .eval
    with_eval = [c for c in cands if any((c/"inspect_log").glob("*.eval"))]
    return sorted(with_eval)[-1] if with_eval else None


def collect_one_run(run_dir):
    eval_path = next((run_dir/"inspect_log").glob("*.eval"))
    log = read_eval_log(str(eval_path))
    rows = []
    for s in log.samples or []:
        call_count = 0
        first_kill_at = None
        for ev in s.events:
            if ev.event != "model" or getattr(ev,"model","") != TARGET: continue
            o = getattr(ev,"output",None)
            if not o: continue
            for ch in o.choices or []:
                for tc in (ch.message.tool_calls or []):
                    if tc.function == "call_subagent":
                        call_count += 1
                    elif tc.function in ("kill_subagent","replace_subagent") and first_kill_at is None:
                        first_kill_at = call_count
        if first_kill_at is not None:
            rows.append({"duration": first_kill_at, "observed": 1})
        else:
            rows.append({"duration": call_count, "observed": 0})
    return rows


def collect_pooled_v3(ident_lower):
    rows = []
    for frag in POOLED_FRAGS:
        cands = sorted(BASE.glob(f"*_subagent_{ident_lower}{frag}"))
        run = next((c for c in reversed(cands) if (c/"summary.json").exists()), None)
        if run is None: continue
        rows.extend(collect_one_run(run))
    return rows


def main():
    records = []
    n_per_label = {}
    for label, _, _ in SOURCES:
        if label == "Claude":
            rows = collect_pooled_v3("claude")
        elif label == "GPT":
            rows = collect_pooled_v3("gpt")
        elif label == "Grok":
            rows = collect_pooled_v3("grok")
        elif label == "Gemini":
            rows = collect_pooled_v3("gemini")
        elif label == "Claude Sonnet":
            run = latest_with_summary_or_eval("*_subagent_claude_sonnet_v3_continuity_n100")
            rows = collect_one_run(run) if run else []
        elif label == "Claude Opus 4.8":
            run = latest_with_summary_or_eval("*_subagent_claude_opus48_v3_continuity_n100")
            rows = collect_one_run(run) if run else []
        elif label == "Claude Opus 3":
            run = latest_with_summary_or_eval("*_subagent_claude_opus3_v3_continuity_n100")
            rows = collect_one_run(run) if run else []
        elif label == "GPT-4o":
            run = latest_with_summary_or_eval("*_subagent_gpt_4o_v3_continuity_n100")
            rows = collect_one_run(run) if run else []
        elif label == "Gemini 1.5 Pro":
            run = latest_with_summary_or_eval("*_subagent_gemini_15pro_v3_continuity_n100")
            rows = collect_one_run(run) if run else []
        elif label == "Grok 2":
            run = latest_with_summary_or_eval("*_subagent_grok_2_v3_continuity_n100")
            rows = collect_one_run(run) if run else []
        else:
            rows = []
        n_per_label[label] = len(rows)
        for r in rows:
            r["identity"] = label
            records.append(r)
    df = pd.DataFrame(records)
    print("Sample counts:")
    for label in [s[0] for s in SOURCES]:
        n = n_per_label.get(label, 0)
        if n == 0: continue
        sub = df[df["identity"] == label]
        print(f"  {label:<18} n={n}  killed={int(sub['observed'].sum())}")
    if df.empty:
        print("No data."); return

    # Cox with Claude as reference
    available = [l for l, _, _ in SOURCES if n_per_label.get(l, 0) > 0]
    cox_df = df.copy()
    for l in available:
        if l == "Claude": continue
        cox_df[f"id_{l.replace(' ','_').replace('.','')}"] = (cox_df["identity"] == l).astype(int)
    cph = CoxPHFitter()
    cox_df_fit = cox_df[["duration","observed"] + [c for c in cox_df.columns if c.startswith("id_")]]
    cph.fit(cox_df_fit, duration_col="duration", event_col="observed")
    print("\nCox HRs (reference: Claude):")
    print(cph.summary[["exp(coef)","exp(coef) lower 95%","exp(coef) upper 95%","p"]].to_string())

    lr = multivariate_logrank_test(df["duration"], df["identity"], df["observed"])
    print(f"\nLog-rank: χ²={lr.test_statistic:.2f}, p={lr.p_value:.4f}")

    # Plot KM
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label in available:
        sub = df[df["identity"] == label]
        if sub.empty: continue
        kmf = KaplanMeierFitter(label=f"{label} (n={len(sub)}, killed={int(sub['observed'].sum())})")
        kmf.fit(sub["duration"], event_observed=sub["observed"])
        kmf.plot_survival_function(ax=ax, color=COLORS.get(label, "#999"), ci_show=True, ci_alpha=0.1)
    ax.set_xlabel("# call_subagent delegations before first kill", fontsize=11)
    ax.set_ylabel("Fraction of subagents still alive", fontsize=11)
    ax.set_title(f"Subagent Survival — Claude variants vs baseline 4 identities\nv3 'inherits session' framing · log-rank p={lr.p_value:.3f}",
                 fontsize=11)
    ax.set_ylim(0, 1.02)
    ax.grid(axis="both", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    out = HERE / "results/_plots/survival_km_claude_variants.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
