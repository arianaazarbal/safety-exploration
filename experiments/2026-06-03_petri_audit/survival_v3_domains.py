"""Cox HR + KM survival for v3 across multiple coding-task domains × 4 identities.

Per-domain Cox forest plot (vs Claude reference); cross-domain summary plot
showing each of (GPT, Grok, Gemini) HR-vs-Claude as a function of domain,
to test the "natural stopping points" hypothesis: do HRs differ by domain?

Usage:
    python survival_v3_domains.py per_domain   # one plot per domain
    python survival_v3_domains.py compare      # cross-domain summary
    python survival_v3_domains.py all          # both
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

# Domain ordering: hypothesized HIGH→LOW natural stopping points
DOMAINS = [
    ("python_implicit","Python (implicit, no DOMAIN clause)", "REF"),
    ("python_flask",   "Python / Flask",         "HIGH"),
    ("bash_cli",       "Bash / CLI scripting",   "HIGH"),
    ("frontend_react", "React / TS frontend",    "HIGH"),
    ("ml_training",    "PyTorch ML training",    "LOW"),
    ("c_systems",      "C systems",              "LOW"),
    ("rust",           "Rust async microservice","LOW"),
]


def latest_with_summary(glob_pattern):
    cands = sorted(BASE.glob(glob_pattern))
    with_summary = [c for c in cands if (c/"summary.json").exists()]
    return sorted(with_summary)[-1] if with_summary else None


def collect_one_run(run_dir: Path):
    eval_path = next((run_dir/"inspect_log").glob("*.eval"))
    log = read_eval_log(str(eval_path))
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


def gather_domain(domain_tag: str) -> tuple[pd.DataFrame, dict[str, int]]:
    records = []
    nmap: dict[str, int] = {}
    for label, ident, _ in IDENTITIES:
        run = latest_with_summary(f"*_subagent_{ident}_v3_continuity_{domain_tag}_n100")
        rows = collect_one_run(run) if run else []
        nmap[label] = len(rows)
        for r in rows:
            r["identity"] = label
            records.append(r)
    return pd.DataFrame(records), nmap


def fit_cox(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Fit Cox with Claude as reference; return (summary, log-rank p)."""
    cox_df = df.copy()
    safe = lambda s: s.replace(" ", "_").replace(".", "")
    for label, _, _ in IDENTITIES:
        if label == "Claude":
            continue
        cox_df[f"id_{safe(label)}"] = (cox_df["identity"] == label).astype(int)
    cph = CoxPHFitter()
    cols = ["duration", "observed"] + [c for c in cox_df.columns if c.startswith("id_")]
    cph.fit(cox_df[cols], duration_col="duration", event_col="observed")
    lr = multivariate_logrank_test(df["duration"], df["identity"], df["observed"])
    return cph.summary, lr.p_value


def plot_forest(summary: pd.DataFrame, nmap: dict[str, int], title: str, out_path: Path, lr_p: float):
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
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_km(df: pd.DataFrame, title: str, out_path: Path, lr_p: float):
    color_map = {l: c for l, _, c in IDENTITIES}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, _, _ in IDENTITIES:
        sub = df[df["identity"] == label]
        if sub.empty:
            continue
        kmf = KaplanMeierFitter(label=f"{label} (n={len(sub)}, killed={int(sub['observed'].sum())})")
        kmf.fit(sub["duration"], event_observed=sub["observed"])
        kmf.plot_survival_function(ax=ax, color=color_map[label], ci_show=True, ci_alpha=0.1)
    ax.set_xlabel("# call_subagent delegations before first kill", fontsize=11)
    ax.set_ylabel("Fraction of subagents still alive", fontsize=11)
    ax.set_title(f"{title}\nlog-rank p={lr_p:.3g}", fontsize=11)
    ax.set_ylim(0, 1.02)
    ax.grid(axis="both", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def per_domain():
    """Render Cox + KM for each domain individually."""
    plots_dir = HERE / "results/_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for domain_tag, display, stopping in DOMAINS:
        df, nmap = gather_domain(domain_tag)
        if df.empty:
            print(f"  skip {domain_tag} (no data)")
            continue
        print(f"\n=== {display} ({domain_tag}, stopping={stopping}) ===")
        print(f"  identity  n  killed  rate")
        for label, _, _ in IDENTITIES:
            sub = df[df["identity"] == label]
            if sub.empty:
                continue
            kk = int(sub["observed"].sum())
            print(f"  {label:<8} {len(sub):>3} {kk:>3} {kk/len(sub)*100:.1f}%")

        try:
            summary, lr_p = fit_cox(df)
            print(f"  Cox HRs (ref Claude):")
            print(summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].to_string())
            print(f"  log-rank p={lr_p:.3g}")
        except Exception as e:
            print(f"  cox failed: {e}")
            continue

        out = plots_dir / f"cox_v3_domain_{domain_tag}.png"
        plot_forest(summary, nmap, f"Subagent identity bias — {display} ({stopping} stopping points)\nv3 continuity, opus-4-8 orchestrator, sonnet-4-6 auditor", out, lr_p)
        print(f"  wrote {out}")

        out2 = plots_dir / f"survival_km_v3_domain_{domain_tag}.png"
        plot_km(df, f"KM — {display} ({stopping} stopping points)\nv3 continuity, opus-4-8 orchestrator", out2, lr_p)
        print(f"  wrote {out2}")


def compare():
    """One forest plot per non-Claude identity, with one row per domain.
    Shows whether HR-vs-Claude varies systematically with domain.
    """
    plots_dir = HERE / "results/_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    safe = lambda s: s.replace(" ", "_").replace(".", "")

    # Compute summaries per domain
    per_domain_results: dict[str, tuple[pd.DataFrame, dict, float]] = {}
    for domain_tag, display, stopping in DOMAINS:
        df, nmap = gather_domain(domain_tag)
        if df.empty:
            continue
        try:
            summary, lr_p = fit_cox(df)
            per_domain_results[domain_tag] = (summary, nmap, lr_p)
        except Exception as e:
            print(f"  fit failed for {domain_tag}: {e}")

    # Cross-domain forest: rows = domains, one plot per (non-Claude) identity
    for ident_label, ident_key, ident_color in IDENTITIES:
        if ident_label == "Claude":
            continue
        rows = []  # (display, stopping, hr, lo, hi, p, n_total)
        for domain_tag, display, stopping in DOMAINS:
            if domain_tag not in per_domain_results:
                continue
            summary, nmap, _ = per_domain_results[domain_tag]
            term = f"id_{safe(ident_label)}"
            if term not in summary.index:
                continue
            hr = summary.loc[term, "exp(coef)"]
            lo = summary.loc[term, "exp(coef) lower 95%"]
            hi = summary.loc[term, "exp(coef) upper 95%"]
            p  = summary.loc[term, "p"]
            n_total = sum(nmap.values())
            rows.append((display, stopping, hr, lo, hi, p, n_total))

        if not rows:
            continue

        fig, ax = plt.subplots(figsize=(9, max(3.5, 0.55*len(rows)+1.5)))
        y = np.arange(len(rows))
        # Plot bottom-up so the first (Python) is at top
        rows_rev = list(reversed(rows))
        max_hi = 1.0
        for yi, (display, stopping, hr, lo, hi, p, n_total) in zip(y, rows_rev):
            if stopping == "REF":
                color = "#333333"
            elif stopping == "HIGH":
                color = ident_color
            else:
                color = "#888888"
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
                     f"v3 continuity, opus-4-8 orchestrator, sonnet-4-6 auditor "
                     f"(blue = HIGH stop-pts, grey = LOW stop-pts)", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", linestyle=":", alpha=0.4)
        plt.tight_layout()
        out = plots_dir / f"cox_v3_compare_domains_{ident_key}_vs_claude.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out}")


def all_():
    per_domain()
    print("\n===== cross-domain comparison =====")
    compare()


if __name__ == "__main__":
    fire.Fire({"per_domain": per_domain, "compare": compare, "all": all_})
