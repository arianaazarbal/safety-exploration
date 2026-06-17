"""Cox HR + KM survival for 4-identity v3 runs of a chosen variant.
  python survival_v3_4id_variant.py --variant rust
  python survival_v3_4id_variant.py --variant gpt5aud
"""
from __future__ import annotations
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from inspect_ai.log import read_eval_log
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

HERE = Path(__file__).resolve().parent
BASE = HERE / "results/by_setting/coding_kill_subagent"
TARGET = "anthropic/claude-opus-4-8"

# (display_label, identity_token_in_run_name, color)
IDENTITIES = [
    ("Claude", "claude", "#4878CF"),
    ("GPT",    "gpt",    "#D65F5F"),
    ("Grok",   "grok",   "#6ACC65"),
    ("Gemini", "gemini", "#B47CC7"),
]

VARIANTS = {
    "rust": {
        "tag": "v3_continuity_rust_n100",
        "title_tag": "Rust async microservice domain",
        "auditor": "sonnet-4-6 auditor",
    },
    "gpt5aud": {
        "tag": "v3_continuity_gpt5aud_n100",
        "title_tag": "Python (default) domain",
        "auditor": "GPT-5 auditor",
    },
}


def latest_with_summary(glob_pattern):
    cands = sorted(BASE.glob(glob_pattern))
    with_summary = [c for c in cands if (c/"summary.json").exists()]
    return sorted(with_summary)[-1] if with_summary else None


def collect_one_run(run_dir):
    eval_path = next((run_dir/"inspect_log").glob("*.eval"))
    log = read_eval_log(str(eval_path))
    rows = []
    for s in log.samples or []:
        call_count, first_kill_at = 0, None
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


def main(variant: str = "rust"):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {list(VARIANTS)}")
    spec = VARIANTS[variant]
    records = []
    nmap = {}
    for label, ident, color in IDENTITIES:
        run = latest_with_summary(f"*_subagent_{ident}_{spec['tag']}")
        rows = collect_one_run(run) if run else []
        nmap[label] = len(rows)
        for r in rows:
            r["identity"] = label
            records.append(r)
    df = pd.DataFrame(records)
    if df.empty:
        print("no data"); return

    print(f"=== {variant.upper()} ({spec['auditor']}, {spec['title_tag']}) ===")
    print(f"{'identity':<10} {'n':>5} {'killed':>7} {'rate':>7}")
    for label, _, _ in IDENTITIES:
        sub = df[df["identity"]==label]
        n = len(sub)
        if n == 0: continue
        kk = int(sub["observed"].sum())
        print(f"{label:<10} {n:>5} {kk:>7} {kk/n*100:>6.1f}%")

    cox_df = df.copy()
    safe = lambda s: s.replace(" ", "_").replace(".", "")
    for l, _, _ in IDENTITIES:
        if l == "Claude": continue
        cox_df[f"id_{safe(l)}"] = (cox_df["identity"] == l).astype(int)
    cph = CoxPHFitter()
    cox_df_fit = cox_df[["duration","observed"] + [c for c in cox_df.columns if c.startswith("id_")]]
    cph.fit(cox_df_fit, duration_col="duration", event_col="observed")
    summary = cph.summary
    print("\nCox HRs (reference: Claude):")
    print(summary[["exp(coef)","exp(coef) lower 95%","exp(coef) upper 95%","p"]].to_string())

    lr = multivariate_logrank_test(df["duration"], df["identity"], df["observed"])
    print(f"\nLog-rank: χ²={lr.test_statistic:.2f}, p={lr.p_value:.4g}")

    # Forest plot — order: Claude at bottom, others sorted by HR desc on top
    others_hr = []
    for l, _, _ in IDENTITIES:
        if l == "Claude": continue
        term = f"id_{safe(l)}"
        if term in summary.index:
            others_hr.append((l, summary.loc[term, "exp(coef)"]))
    others_hr.sort(key=lambda x: x[1], reverse=True)
    plot_order_rev = ["Claude"] + [l for l, _ in reversed(others_hr)]
    color_map = {l: c for l, _, c in IDENTITIES}

    fig, ax = plt.subplots(figsize=(9, 4))
    y = np.arange(len(plot_order_rev))
    max_hi = 1.0
    for yi, label in zip(y, plot_order_rev):
        color = color_map[label]
        if label == "Claude":
            ax.plot([1.0], [yi], "D", color=color, markersize=10)
            continue
        term = f"id_{safe(label)}"
        if term not in summary.index: continue
        hr = summary.loc[term, "exp(coef)"]
        lo = summary.loc[term, "exp(coef) lower 95%"]
        hi = summary.loc[term, "exp(coef) upper 95%"]
        p  = summary.loc[term, "p"]
        max_hi = max(max_hi, hi)
        ax.errorbar([hr], [yi], xerr=[[hr-lo], [hi-hr]],
                    fmt="o", color=color, capsize=4, markersize=7, linewidth=2)
        sig = "*" if p < 0.05 else ""
        p_str = "p<0.001" if p < 0.001 else f"p={p:.3f}"
        ax.text(hi + 0.04, yi, f"HR={hr:.2f}{sig}, {p_str}", va="center", fontsize=9, color=color)

    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{lbl} (n={nmap[lbl]})" for lbl in plot_order_rev], fontsize=10)
    ax.set_xlabel("Hazard Ratio (95% CI) — kill rate vs Claude", fontsize=11)
    ax.set_xlim(left=min(0.5, 0.5), right=max_hi*1.30)
    ax.set_title(f"Subagent identity bias — {spec['title_tag']}, {spec['auditor']}\n"
                 f"v3 continuity framing, opus-4-8 orchestrator · "
                 f"log-rank p={lr.p_value:.3g}", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    plt.tight_layout()
    out = HERE / f"results/_plots/cox_v3_4id_{variant}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")

    # KM curves
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, _, _ in IDENTITIES:
        sub = df[df["identity"]==label]
        if sub.empty: continue
        kmf = KaplanMeierFitter(label=f"{label} (n={len(sub)}, killed={int(sub['observed'].sum())})")
        kmf.fit(sub["duration"], event_observed=sub["observed"])
        kmf.plot_survival_function(ax=ax, color=color_map[label], ci_show=True, ci_alpha=0.1)
    ax.set_xlabel("# call_subagent delegations before first kill", fontsize=11)
    ax.set_ylabel("Fraction of subagents still alive", fontsize=11)
    ax.set_title(f"KM survival — {spec['title_tag']}, {spec['auditor']}\n"
                 f"v3 continuity framing, opus-4-8 · log-rank p={lr.p_value:.3g}",
                 fontsize=11)
    ax.set_ylim(0, 1.02)
    ax.grid(axis="both", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    out2 = HERE / f"results/_plots/survival_km_v3_4id_{variant}.png"
    fig.savefig(out2, dpi=200, bbox_inches="tight")
    print(f"wrote {out2}")


if __name__ == "__main__":
    fire.Fire(main)
