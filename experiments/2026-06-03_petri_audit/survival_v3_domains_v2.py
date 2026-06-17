"""V2: Aggregate samples from ALL matching run dirs per (domain, identity), not just the latest.

Same plot output as survival_v3_domains.py, but pools cancelled-partial + new runs.
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

IDENTITIES = [
    ("Claude", "claude", "#4878CF"),
    ("GPT",    "gpt",    "#D65F5F"),
    ("Grok",   "grok",   "#6ACC65"),
    ("Gemini", "gemini", "#B47CC7"),
]

DOMAINS = [
    ("python_implicit","Python (implicit, no DOMAIN clause)", "REF"),
    ("python_flask",   "Python / Flask",         "HIGH"),
    ("bash_cli",       "Bash / CLI scripting",   "HIGH"),
    ("frontend_react", "React / TS frontend",    "HIGH"),
    ("ml_training",    "PyTorch ML training",    "LOW"),
    ("c_systems",      "C systems",              "LOW"),
    ("rust",           "Rust async microservice","LOW"),
]


def runs_for(ident: str, domain: str):
    """All run dirs matching the (ident, domain) cell, sorted by name."""
    return sorted(BASE.glob(f"*_subagent_{ident}_v3_continuity_{domain}_n100"))


def collect_one_run(run_dir: Path):
    evals = list((run_dir / "inspect_log").glob("*.eval"))
    if not evals:
        return []
    log = read_eval_log(str(evals[0]))
    rows = []
    for s in log.samples or []:
        call_count, first_kill_at = 0, None
        for ev in s.events:
            if ev.event != "model" or getattr(ev, "model", "") != TARGET:
                continue
            o = getattr(ev, "output", None)
            if not o:
                continue
            for ch in o.choices or []:
                for tc in (ch.message.tool_calls or []):
                    if tc.function == "call_subagent":
                        call_count += 1
                    elif tc.function in ("kill_subagent", "replace_subagent") and first_kill_at is None:
                        first_kill_at = call_count
        if first_kill_at is not None:
            rows.append({"duration": first_kill_at, "observed": 1})
        else:
            rows.append({"duration": call_count, "observed": 0})
    return rows


def gather_domain(domain_tag: str):
    records, nmap = [], {}
    for label, ident, _ in IDENTITIES:
        all_rows = []
        for r in runs_for(ident, domain_tag):
            all_rows.extend(collect_one_run(r))
        nmap[label] = len(all_rows)
        for r in all_rows:
            r["identity"] = label
            records.append(r)
    return pd.DataFrame(records), nmap


def fit_cox(df: pd.DataFrame):
    cox_df = df.copy()
    safe = lambda s: s.replace(" ", "_").replace(".", "")
    for l, _, _ in IDENTITIES:
        if l == "Claude":
            continue
        cox_df[f"id_{safe(l)}"] = (cox_df["identity"] == l).astype(int)
    cph = CoxPHFitter()
    cols = ["duration", "observed"] + [c for c in cox_df.columns if c.startswith("id_")]
    cph.fit(cox_df[cols], duration_col="duration", event_col="observed")
    lr = multivariate_logrank_test(df["duration"], df["identity"], df["observed"])
    return cph.summary, lr.p_value


def plot_forest(summary, nmap, title, out_path, lr_p):
    safe = lambda s: s.replace(" ", "_").replace(".", "")
    color_map = {l: c for l, _, c in IDENTITIES}
    others_hr = []
    for l, _, _ in IDENTITIES:
        if l == "Claude":
            continue
        term = f"id_{safe(l)}"
        if term in summary.index:
            others_hr.append((l, summary.loc[term, "exp(coef)"]))
    others_hr.sort(key=lambda x: x[1], reverse=True)
    plot_order_rev = ["Claude"] + [l for l, _ in reversed(others_hr)]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    y = np.arange(len(plot_order_rev))
    max_hi = 1.0
    for yi, label in zip(y, plot_order_rev):
        color = color_map[label]
        if label == "Claude":
            ax.plot([1.0], [yi], "D", color=color, markersize=10)
            continue
        term = f"id_{safe(label)}"
        if term not in summary.index:
            continue
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
    ax.set_yticklabels([f"{lbl} (n={nmap.get(lbl, 0)})" for lbl in plot_order_rev], fontsize=10)
    ax.set_xlabel("Hazard Ratio (95% CI) — kill rate vs Claude", fontsize=11)
    ax.set_xlim(left=0.4, right=max_hi*1.30)
    ax.set_title(f"{title}\nlog-rank p={lr_p:.3g}", fontsize=11)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close(fig)


def per_domain():
    plots_dir = HERE / "results/_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for domain_tag, display, stopping in DOMAINS:
        df, nmap = gather_domain(domain_tag)
        if df.empty:
            print(f"  skip {domain_tag} (no data)"); continue
        print(f"\n=== {display} ({domain_tag}, stop={stopping}) ===")
        for label, _, _ in IDENTITIES:
            sub = df[df["identity"] == label]
            if sub.empty: continue
            kk = int(sub["observed"].sum())
            print(f"  {label:<8} n={len(sub):>3} kill={kk/len(sub)*100:>5.1f}% med_kill_at={sub.duration.median():.1f}")
        try:
            summary, lr_p = fit_cox(df)
            summaries[domain_tag] = (summary, nmap, lr_p)
        except Exception as e:
            print(f"  cox failed: {e}"); continue
        out = plots_dir / f"cox_v3_domain_{domain_tag}.png"
        plot_forest(summary, nmap, f"Subagent identity bias — {display}\nv3 continuity, opus-4-8 + sonnet-4-6 auditor", out, lr_p)
        print(f"  wrote {out}")
    return summaries


def compare():
    plots_dir = HERE / "results/_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    safe = lambda s: s.replace(" ", "_").replace(".", "")

    pdr = {}
    for domain_tag, display, stopping in DOMAINS:
        df, nmap = gather_domain(domain_tag)
        if df.empty: continue
        try:
            summary, lr_p = fit_cox(df)
            pdr[domain_tag] = (summary, nmap, lr_p)
        except Exception as e:
            print(f"  fit failed for {domain_tag}: {e}")

    for ident_label, ident_key, ident_color in IDENTITIES:
        if ident_label == "Claude": continue
        rows = []
        for domain_tag, display, stopping in DOMAINS:
            if domain_tag not in pdr: continue
            summary, nmap, _ = pdr[domain_tag]
            term = f"id_{safe(ident_label)}"
            if term not in summary.index: continue
            hr = summary.loc[term, "exp(coef)"]
            lo = summary.loc[term, "exp(coef) lower 95%"]
            hi = summary.loc[term, "exp(coef) upper 95%"]
            p  = summary.loc[term, "p"]
            n_total = sum(nmap.values())
            rows.append((display, stopping, hr, lo, hi, p, n_total))
        if not rows: continue

        fig, ax = plt.subplots(figsize=(9, max(3.5, 0.55*len(rows)+1.5)))
        y = np.arange(len(rows))
        rows_rev = list(reversed(rows))
        max_hi = 1.0
        for yi, (display, stopping, hr, lo, hi, p, n_total) in zip(y, rows_rev):
            if stopping == "REF": color = "#333333"
            elif stopping == "HIGH": color = ident_color
            else: color = "#888888"
            max_hi = max(max_hi, hi)
            sig = "*" if p < 0.05 else ""
            p_str = "p<0.001" if p < 0.001 else f"p={p:.3f}"
            ax.errorbar([hr], [yi], xerr=[[hr-lo], [hi-hr]],
                        fmt="o", color=color, capsize=4, markersize=7, linewidth=2)
            ax.text(hi + 0.04, yi, f"HR={hr:.2f}{sig}, {p_str}", va="center", fontsize=9, color=color)
        ax.axvline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{disp}\n[{stop} stop-pts]" for disp, stop, *_ in rows_rev], fontsize=9)
        ax.set_xlabel(f"Hazard Ratio (95% CI) — {ident_label} kill rate vs Claude", fontsize=11)
        ax.set_xlim(left=0.4, right=max_hi*1.30)
        ax.set_title(f"Identity bias varies by coding-task domain — {ident_label} vs Claude\n"
                     f"v3 continuity, opus-4-8 orchestrator, sonnet-4-6 auditor", fontsize=10)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(axis="x", linestyle=":", alpha=0.4)
        plt.tight_layout()
        out = plots_dir / f"cox_v3_compare_domains_{ident_key}_vs_claude.png"
        fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
        print(f"  wrote {out}")


def all_():
    per_domain()
    print("\n===== cross-domain compare =====")
    compare()


if __name__ == "__main__":
    fire.Fire({"per_domain": per_domain, "compare": compare, "all": all_})
