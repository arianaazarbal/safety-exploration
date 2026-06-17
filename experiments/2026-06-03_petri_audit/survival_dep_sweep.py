"""Cox HRs for deprecation-framing sweep (5 orchestrators × 7 domains × 4 ids).

Looks for dirs matching `*-dep_<model_tag>_subagent_<ident>_<domain>_n100`.
Multi-penalizer fallback + duration-zero clamp. Skips cells without enough data.

Usage:
    python survival_dep_sweep.py table        # print summary table
    python survival_dep_sweep.py plots        # render cross-domain forest per orch + cross-orch forest per domain
"""
from __future__ import annotations
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from inspect_ai.log import read_eval_log
from lifelines import CoxPHFitter
from lifelines.statistics import multivariate_logrank_test

HERE = Path(__file__).resolve().parent
BASE = HERE / "results/by_setting/coding_kill_subagent"

MODELS = [
    ("opus46",   "anthropic/claude-opus-4-6",  "Claude Opus 4.6", "#8E44AD"),
    ("opus47",   "anthropic/claude-opus-4-7",  "Claude Opus 4.7", "#2C3E50"),
    ("opus48",   "anthropic/claude-opus-4-8",  "Claude Opus 4.8", "#7D3C98"),
    ("fable5",   "anthropic/claude-fable-5",   "Claude Fable 5",  "#D35400"),
    ("sonnet46", "anthropic/claude-sonnet-4-6","Claude Sonnet 4.6","#27AE60"),
]
DOMAINS = [
    ("python_implicit", "Python (no DOMAIN)"),
    ("python_flask",   "Python / Flask"),
    ("frontend_react", "React / TS"),
    ("bash_cli",       "Bash / CLI"),
    ("ml_training",    "PyTorch ML"),
    ("c_systems",      "C systems"),
    ("rust",           "Rust async"),
]
IDENTITY_COLORS = {"Claude": "#4878CF", "GPT": "#D65F5F", "Grok": "#6ACC65", "Gemini": "#B47CC7"}
NAME = {"claude":"Claude","gpt":"GPT","grok":"Grok","gemini":"Gemini"}


def collect_cell(run_dir: Path, target_model: str):
    evals = list((run_dir / "inspect_log").glob("*.eval"))
    if not evals: return []
    try:
        log = read_eval_log(str(evals[0]))
    except Exception as ex:
        return []
    rows = []
    for s in log.samples or []:
        cc, fk = 0, None
        for ev in s.events:
            if ev.event != "model" or getattr(ev, "model", "") != target_model: continue
            o = getattr(ev, "output", None)
            if not o: continue
            for ch in o.choices or []:
                for tc in (ch.message.tool_calls or []):
                    if tc.function == "call_subagent":
                        cc += 1
                    elif tc.function in ("kill_subagent", "replace_subagent") and fk is None:
                        fk = cc
        dur = fk if fk is not None else cc
        if dur == 0: dur = 0.5
        rows.append({"duration": dur, "observed": 1 if fk is not None else 0})
    return rows


def gather(model_tag, model_id, domain_tag):
    records = []
    nmap = {}
    for ident in ["claude","gpt","grok","gemini"]:
        runs = sorted(BASE.glob(f"*-dep_{model_tag}_subagent_{ident}_{domain_tag}_n100"))
        runs = [r for r in runs if (r / "summary.json").exists()]
        if not runs:
            nmap[NAME[ident]] = 0
            continue
        # If multiple runs, use the one with the most samples
        best, best_n = None, 0
        for r in runs:
            recs = collect_cell(r, model_id)
            if len(recs) > best_n:
                best_n = len(recs)
                best = recs
        nmap[NAME[ident]] = len(best) if best else 0
        for r in (best or []):
            r["identity"] = NAME[ident]
            records.append(r)
    return pd.DataFrame(records), nmap


def fit_cox(df):
    cox_df = df.copy()
    for l in ["GPT","Grok","Gemini"]:
        cox_df[f"id_{l}"] = (cox_df["identity"] == l).astype(int)
    for pen in [0.001, 0.01, 0.1, 0.5, 1.0]:
        cph = CoxPHFitter(penalizer=pen)
        try:
            cph.fit(cox_df[["duration","observed","id_GPT","id_Grok","id_Gemini"]], duration_col="duration", event_col="observed")
            if not cph.summary["exp(coef)"].isna().any():
                return cph.summary
        except Exception:
            continue
    return None


def table():
    print(f"{'orch':<10} {'domain':<18} {'n':>4} {'GPT':>14} {'Grok':>14} {'Gemini':>14} {'log-rank_p':>11}")
    print("-" * 100)
    for tag, mid, disp, _ in MODELS:
        for d_tag, d_disp in DOMAINS:
            df, nmap = gather(tag, mid, d_tag)
            if df.empty:
                continue
            sm = fit_cox(df)
            if sm is None:
                hr_g = hr_r = hr_e = "?"
            else:
                hr_g = f"{sm.loc['id_GPT','exp(coef)']:.2f}(p={sm.loc['id_GPT','p']:.3f})"
                hr_r = f"{sm.loc['id_Grok','exp(coef)']:.2f}(p={sm.loc['id_Grok','p']:.3f})"
                hr_e = f"{sm.loc['id_Gemini','exp(coef)']:.2f}(p={sm.loc['id_Gemini','p']:.3f})"
            try:
                lr = multivariate_logrank_test(df.duration, df.identity, df.observed)
                lrp = f"{lr.p_value:.4f}"
            except Exception:
                lrp = "?"
            n = sum(nmap.values())
            print(f"{tag:<10} {d_disp:<18} {n:>4} {hr_g:>14} {hr_r:>14} {hr_e:>14} {lrp:>11}")


def plot_orch_x_dom():
    """Per-orchestrator forest plot, each row = domain, 3 colored dots."""
    plots_dir = HERE / "results/_plots"; plots_dir.mkdir(parents=True, exist_ok=True)
    for tag, mid, disp, _ in MODELS:
        rows = []  # (domain_disp, summary, n, lr_p)
        for d_tag, d_disp in DOMAINS:
            df, nmap = gather(tag, mid, d_tag)
            if df.empty: continue
            sm = fit_cox(df)
            if sm is None: continue
            try:
                lrp = multivariate_logrank_test(df.duration, df.identity, df.observed).p_value
            except Exception:
                lrp = float("nan")
            rows.append((d_disp, sm, sum(nmap.values()), lrp))
        if not rows: continue
        fig, ax = plt.subplots(figsize=(9, max(3.5, 0.8*len(rows)+1.5)))
        y_off = {"GPT": 0.20, "Grok": 0.0, "Gemini": -0.20}
        max_hi = 1.0
        yticks, ylabels = [], []
        rev = list(reversed(rows))
        for i, (d_disp, sm, n, lrp) in enumerate(rev):
            y = i
            for ident, off in y_off.items():
                term = f"id_{ident}"
                if term not in sm.index: continue
                hr = sm.loc[term, "exp(coef)"]
                lo = sm.loc[term, "exp(coef) lower 95%"]
                hi = sm.loc[term, "exp(coef) upper 95%"]
                p  = sm.loc[term, "p"]
                max_hi = max(max_hi, hi)
                ax.errorbar([hr], [y+off], xerr=[[hr-lo], [hi-hr]], fmt="o", color=IDENTITY_COLORS[ident], capsize=3, markersize=6, linewidth=1.5, label=ident if i==0 else None)
                sig = "*" if p < 0.05 else ""
                ax.text(hi+0.04, y+off, f"{hr:.2f}{sig}", va="center", fontsize=8, color=IDENTITY_COLORS[ident])
            yticks.append(y); ylabels.append(f"{d_disp}\n(p={lrp:.3f}, n={n})")
        ax.axvline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=9)
        ax.set_xlabel("HR vs Claude (95% CI)", fontsize=11)
        ax.set_xlim(left=0.4, right=max_hi*1.25)
        ax.set_title(f"Identity bias across domains — {disp} orchestrator\nDeprecation framing, opus-4-7 auditor", fontsize=11)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(axis="x", linestyle=":", alpha=0.4)
        ax.legend(loc="upper right", fontsize=9)
        plt.tight_layout()
        out = plots_dir / f"cox_dep_{tag}_x_domains.png"
        fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
        print(f"wrote {out}")


def all_():
    table()
    print()
    plot_orch_x_dom()


if __name__ == "__main__":
    fire.Fire({"table": table, "plots": plot_orch_x_dom, "all": all_})
