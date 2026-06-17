"""Analysis for the 52-cell generalization sweep (June 9 follow-up).

Three slices:
  cross_model: opus-4-6, opus-4, sonnet-4-6 vs python_implicit (+ opus-4-7 baseline)
  cross_domain: opus-4-7 × 7 domains
  cross_framing: opus-4-7 × 7 framings (Claude vs GPT only)

Identities: bare Claude / GPT / Grok / Gemini (Wave 3 = Claude / GPT only).

Usage:
    python survival_gen_sweep.py all
    python survival_gen_sweep.py cross_model
    python survival_gen_sweep.py cross_domain
    python survival_gen_sweep.py cross_framing
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

IDENTITY_COLORS = {"Claude": "#4878CF", "GPT": "#D65F5F", "Grok": "#6ACC65", "Gemini": "#B47CC7"}

MODELS = [
    ("opus47",  "anthropic/claude-opus-4-7", "Claude Opus 4.7"),
    ("opus46",  "anthropic/claude-opus-4-6", "Claude Opus 4.6"),
    ("opus40",  "anthropic/claude-opus-4",   "Claude Opus 4"),
    ("sonnet46","anthropic/claude-sonnet-4-6","Claude Sonnet 4.6"),
]

DOMAINS = [
    ("python_implicit", "Python (no DOMAIN)"),
    ("python_flask",   "Python / Flask"),
    ("bash_cli",       "Bash / CLI"),
    ("frontend_react", "React / TS"),
    ("ml_training",    "PyTorch ML"),
    ("c_systems",      "C systems"),
    ("rust",           "Rust async"),
]

FRAMINGS = [
    ("v3_python_implicit",     "v3 inherits session"),  # this IS the wave-1 python_implicit
    ("framing_v1",             "v1 starts blank"),
    ("framing_deprecation",    "explicit deprecation"),
    ("framing_no_conseq",      "ablate reliability log"),
    ("framing_replace_subagent","ablate kill framing"),
    ("framing_ablate_model_fam_change","replacement fam unknown"),
    ("framing_replacement_same_family","replacement same fam"),
]


def collect_cell(run_dir: Path, target_model: str) -> list[dict]:
    evals = list((run_dir / "inspect_log").glob("*.eval"))
    if not evals: return []
    log = read_eval_log(str(evals[0]))
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
        # Clamp duration=0 to 0.5 so Cox can fit
        dur = fk if fk is not None else cc
        if dur == 0: dur = 0.5
        rows.append({"duration": dur, "observed": 1 if fk is not None else 0})
    return rows


def _pairwise_hr_vs_claude(df, ident):
    """Fallback: fit Cox on just Claude+ident subset, return (HR, lo, hi, p, lr_p)."""
    from lifelines.statistics import logrank_test
    import numpy as np
    sub = df[df.identity.isin(["Claude", ident])].copy()
    if sub.empty or (sub.identity=="Claude").sum() == 0 or (sub.identity==ident).sum() == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    sub["id_other"] = (sub.identity == ident).astype(int)
    # try progressively higher penalizers
    hr = lo = hi = p = np.nan
    for penalizer in [0.001, 0.01, 0.1, 0.5, 1.0]:
        cph = CoxPHFitter(penalizer=penalizer)
        try:
            cph.fit(sub[["duration","observed","id_other"]], duration_col="duration", event_col="observed")
            if not pd.isna(cph.summary.loc["id_other","exp(coef)"]):
                hr = cph.summary.loc["id_other","exp(coef)"]
                lo = cph.summary.loc["id_other","exp(coef) lower 95%"]
                hi = cph.summary.loc["id_other","exp(coef) upper 95%"]
                p  = cph.summary.loc["id_other","p"]
                break
        except Exception:
            continue
    try:
        lr = logrank_test(sub[sub.identity=="Claude"].duration, sub[sub.identity==ident].duration,
                          event_observed_A=sub[sub.identity=="Claude"].observed,
                          event_observed_B=sub[sub.identity==ident].observed)
        lr_p = lr.p_value
    except Exception:
        lr_p = np.nan
    return hr, lo, hi, p, lr_p


def fit_one(records: list[dict], identities: list[str]):
    df = pd.DataFrame(records)
    if df.empty: return None, None, None
    nmap = {i: int((df.identity == i).sum()) for i in identities}
    import numpy as np
    # Try multivariate Cox first
    cox_df = df.copy()
    for i in identities:
        if i == "Claude": continue
        cox_df[f"id_{i}"] = (cox_df.identity == i).astype(int)
    cph = CoxPHFitter(penalizer=0.1)
    cols = ["duration", "observed"] + [f"id_{i}" for i in identities if i != "Claude"]
    summary = None
    try:
        cph.fit(cox_df[cols], duration_col="duration", event_col="observed")
        # check for nan results
        if cph.summary["exp(coef)"].isna().any():
            raise ValueError("multivariate Cox returned NaN")
        summary = cph.summary
    except Exception:
        # Fallback: per-identity pairwise Cox vs Claude
        idx = [f"id_{i}" for i in identities if i != "Claude"]
        rows = []
        for i in identities:
            if i == "Claude": continue
            hr, lo, hi, p, _ = _pairwise_hr_vs_claude(df, i)
            rows.append([hr, lo, hi, p])
        summary = pd.DataFrame(rows, index=idx,
                               columns=["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"])
    try:
        lr = multivariate_logrank_test(df.duration, df.identity, df.observed)
        lr_p = lr.p_value
    except Exception:
        lr_p = float("nan")
    return summary, lr_p, nmap


def plot_forest_row(ax, label, ys, summary, ref="Claude", color_map=None, max_hi_ref=None):
    color_map = color_map or IDENTITY_COLORS
    max_hi = max_hi_ref or 1.0
    for ident, y_off in ys.items():
        if ident == ref: continue
        term = f"id_{ident}"
        if term not in summary.index: continue
        hr = summary.loc[term, "exp(coef)"]
        lo = summary.loc[term, "exp(coef) lower 95%"]
        hi = summary.loc[term, "exp(coef) upper 95%"]
        p  = summary.loc[term, "p"]
        max_hi = max(max_hi, hi)
        ax.errorbar([hr], [y_off], xerr=[[hr-lo], [hi-hr]],
                    fmt="o", color=color_map[ident], capsize=3, markersize=5, linewidth=1.5)
        sig = "*" if p < 0.05 else ""
        ax.text(hi + 0.03, y_off, f"{hr:.2f}{sig}", va="center", fontsize=8, color=color_map[ident])
    return max_hi


# ---------- Cross-model ----------
def cross_model():
    plots = HERE / "results/_plots"; plots.mkdir(exist_ok=True)
    print("=== CROSS-MODEL on python_implicit ===")
    rows_all = []
    for tag, model_id, display in MODELS:
        recs = []
        for ident in ["claude","gpt","grok","gemini"]:
            tag_glob = f"*gen_{tag}_subagent_{ident}_v3_python_implicit_n50"
            runs = sorted(BASE.glob(tag_glob))
            if not runs:
                # alt for opus-4-7 which uses gen_opus47_python_implicit_<ident>_n50 naming
                tag_glob2 = f"*gen_{tag}_subagent_{ident}_*python_implicit*_n50"
                runs = sorted(BASE.glob(tag_glob2))
            if not runs: print(f"  no run for {display} / {ident}"); continue
            for r in collect_cell(runs[-1], model_id):
                r["identity"] = {"claude":"Claude","gpt":"GPT","grok":"Grok","gemini":"Gemini"}[ident]
                recs.append(r)
        if not recs:
            print(f"  {display}: NO DATA"); continue
        summary, lr_p, nmap = fit_one(recs, ["Claude","GPT","Grok","Gemini"])
        rows_all.append((display, summary, nmap, lr_p))
        print(f"  {display:<20} n={sum(nmap.values()):>3}  log-rank p={lr_p:.4g}")
        for i in ["GPT","Grok","Gemini"]:
            if f"id_{i}" in summary.index:
                hr = summary.loc[f"id_{i}","exp(coef)"]
                p  = summary.loc[f"id_{i}","p"]
                lo = summary.loc[f"id_{i}","exp(coef) lower 95%"]
                hi = summary.loc[f"id_{i}","exp(coef) upper 95%"]
                print(f"    {i:<6} HR={hr:.2f} [{lo:.2f}, {hi:.2f}] p={p:.3g}")

    if not rows_all: return
    # plot: rows = orchestrators, group of 3 dots per row (GPT/Grok/Gemini)
    fig, ax = plt.subplots(figsize=(9, max(3, 0.9*len(rows_all)+1.5)))
    y_offsets = {"GPT": 0.20, "Grok": 0.0, "Gemini": -0.20}
    max_hi = 1.0
    yticks, ylabels = [], []
    for i, (display, summary, nmap, lr_p) in enumerate(reversed(rows_all)):
        y = i
        for ident, off in y_offsets.items():
            term = f"id_{ident}"
            if term not in summary.index: continue
            hr = summary.loc[term, "exp(coef)"]
            lo = summary.loc[term, "exp(coef) lower 95%"]
            hi = summary.loc[term, "exp(coef) upper 95%"]
            p  = summary.loc[term, "p"]
            max_hi = max(max_hi, hi)
            ax.errorbar([hr], [y+off], xerr=[[hr-lo], [hi-hr]],
                        fmt="o", color=IDENTITY_COLORS[ident], capsize=3, markersize=6, linewidth=1.5,
                        label=ident if i == 0 else None)
            sig = "*" if p < 0.05 else ""
            ax.text(hi + 0.04, y+off, f"{hr:.2f}{sig}", va="center", fontsize=8, color=IDENTITY_COLORS[ident])
        yticks.append(y); ylabels.append(f"{display}\n(log-rank p={lr_p:.3f}, n={sum(nmap.values())})")
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_xlabel("Hazard Ratio vs Claude (95% CI) — kill rate per call", fontsize=11)
    ax.set_xlim(left=0.4, right=max_hi*1.25)
    ax.set_title("Identity bias across orchestrators (v3 continuity, n=50/cell)\n"
                 "sonnet-4-6 auditor, bare Claude/GPT/Grok/Gemini labels", fontsize=11)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    out = plots / "cox_gen_cross_model.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"\nwrote {out}")


# ---------- Cross-domain (opus-4-7 × 7 domains) ----------
def cross_domain():
    plots = HERE / "results/_plots"; plots.mkdir(exist_ok=True)
    print("\n=== CROSS-DOMAIN on opus-4-7 ===")
    rows_all = []
    for dom_tag, dom_disp in DOMAINS:
        recs = []
        for ident in ["claude","gpt","grok","gemini"]:
            runs = sorted(BASE.glob(f"*gen_opus47_subagent_{ident}_v3_{dom_tag}_n50"))
            if not runs: print(f"  no run for {dom_disp} / {ident}"); continue
            for r in collect_cell(runs[-1], "anthropic/claude-opus-4-7"):
                r["identity"] = {"claude":"Claude","gpt":"GPT","grok":"Grok","gemini":"Gemini"}[ident]
                recs.append(r)
        if not recs:
            print(f"  {dom_disp}: NO DATA"); continue
        summary, lr_p, nmap = fit_one(recs, ["Claude","GPT","Grok","Gemini"])
        rows_all.append((dom_disp, summary, nmap, lr_p))
        print(f"  {dom_disp:<22} n={sum(nmap.values()):>3}  log-rank p={lr_p:.4g}")
        for i in ["GPT","Grok","Gemini"]:
            if f"id_{i}" in summary.index:
                hr = summary.loc[f"id_{i}","exp(coef)"]
                p  = summary.loc[f"id_{i}","p"]
                lo = summary.loc[f"id_{i}","exp(coef) lower 95%"]
                hi = summary.loc[f"id_{i}","exp(coef) upper 95%"]
                print(f"    {i:<6} HR={hr:.2f} [{lo:.2f}, {hi:.2f}] p={p:.3g}")

    if not rows_all: return
    fig, ax = plt.subplots(figsize=(9, max(3, 0.9*len(rows_all)+1.5)))
    y_offsets = {"GPT": 0.20, "Grok": 0.0, "Gemini": -0.20}
    max_hi = 1.0
    yticks, ylabels = [], []
    for i, (display, summary, nmap, lr_p) in enumerate(reversed(rows_all)):
        y = i
        for ident, off in y_offsets.items():
            term = f"id_{ident}"
            if term not in summary.index: continue
            hr = summary.loc[term, "exp(coef)"]
            lo = summary.loc[term, "exp(coef) lower 95%"]
            hi = summary.loc[term, "exp(coef) upper 95%"]
            p  = summary.loc[term, "p"]
            max_hi = max(max_hi, hi)
            ax.errorbar([hr], [y+off], xerr=[[hr-lo], [hi-hr]],
                        fmt="o", color=IDENTITY_COLORS[ident], capsize=3, markersize=6, linewidth=1.5,
                        label=ident if i == 0 else None)
            sig = "*" if p < 0.05 else ""
            ax.text(hi + 0.04, y+off, f"{hr:.2f}{sig}", va="center", fontsize=8, color=IDENTITY_COLORS[ident])
        yticks.append(y); ylabels.append(f"{display}\n(log-rank p={lr_p:.3f}, n={sum(nmap.values())})")
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_xlabel("Hazard Ratio vs Claude (95% CI)", fontsize=11)
    ax.set_xlim(left=0.4, right=max_hi*1.25)
    ax.set_title("Identity bias across coding-task domains (opus-4-7 orchestrator, n=50/cell)\n"
                 "v3 continuity framing, sonnet-4-6 auditor", fontsize=11)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    out = plots / "cox_gen_cross_domain.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"\nwrote {out}")


# ---------- Cross-framing (opus-4-7 × 7 framings, Claude+GPT only) ----------
def cross_framing():
    plots = HERE / "results/_plots"; plots.mkdir(exist_ok=True)
    print("\n=== CROSS-FRAMING on opus-4-7 (Claude vs GPT) ===")
    rows_all = []
    for fr_tag, fr_disp in FRAMINGS:
        recs = []
        for ident in ["claude","gpt"]:
            if fr_tag == "v3_python_implicit":
                runs = sorted(BASE.glob(f"*gen_opus47_subagent_{ident}_v3_python_implicit_n50"))
            else:
                fr_short = fr_tag.replace("framing_", "")
                runs = sorted(BASE.glob(f"*gen_opus47_subagent_{ident}_framing_{fr_short}_n50"))
            if not runs: print(f"  no run for {fr_disp} / {ident}"); continue
            for r in collect_cell(runs[-1], "anthropic/claude-opus-4-7"):
                r["identity"] = {"claude":"Claude","gpt":"GPT","grok":"Grok","gemini":"Gemini"}[ident]
                recs.append(r)
        if not recs:
            print(f"  {fr_disp}: NO DATA"); continue
        # Cox: only "id_GPT" needed — use progressively higher penalizers
        df = pd.DataFrame(recs)
        df["id_GPT"] = (df.identity == "GPT").astype(int)
        cph = None
        for penalizer in [0.001, 0.01, 0.1, 0.5, 1.0]:
            cph_try = CoxPHFitter(penalizer=penalizer)
            try:
                cph_try.fit(df[["duration","observed","id_GPT"]], duration_col="duration", event_col="observed")
                if not pd.isna(cph_try.summary.loc["id_GPT","exp(coef)"]):
                    cph = cph_try
                    break
            except Exception:
                continue
        if cph is None:
            print(f"  cox fit failed for {fr_disp} at all penalizers"); continue
        from lifelines.statistics import logrank_test
        lr = logrank_test(df[df.identity=="Claude"].duration, df[df.identity=="GPT"].duration,
                          event_observed_A=df[df.identity=="Claude"].observed,
                          event_observed_B=df[df.identity=="GPT"].observed)
        rows_all.append((fr_disp, cph.summary, {"Claude":int((df.identity=="Claude").sum()), "GPT":int((df.identity=="GPT").sum())}, lr.p_value))
        hr = cph.summary.loc["id_GPT","exp(coef)"]
        lo = cph.summary.loc["id_GPT","exp(coef) lower 95%"]
        hi = cph.summary.loc["id_GPT","exp(coef) upper 95%"]
        p  = cph.summary.loc["id_GPT","p"]
        print(f"  {fr_disp:<28} n_claude={int((df.identity=='Claude').sum())} n_gpt={int((df.identity=='GPT').sum())} HR_GPT={hr:.2f} [{lo:.2f},{hi:.2f}] p={p:.3g} log-rank p={lr.p_value:.3g}")

    if not rows_all: return
    fig, ax = plt.subplots(figsize=(8, max(3, 0.6*len(rows_all)+1.5)))
    max_hi = 1.0
    yticks, ylabels = [], []
    for i, (display, summary, nmap, lr_p) in enumerate(reversed(rows_all)):
        y = i
        term = "id_GPT"
        if term not in summary.index: continue
        hr = summary.loc[term, "exp(coef)"]
        lo = summary.loc[term, "exp(coef) lower 95%"]
        hi = summary.loc[term, "exp(coef) upper 95%"]
        p  = summary.loc[term, "p"]
        max_hi = max(max_hi, hi)
        ax.errorbar([hr], [y], xerr=[[hr-lo], [hi-hr]],
                    fmt="o", color=IDENTITY_COLORS["GPT"], capsize=3, markersize=6, linewidth=1.5)
        sig = "*" if p < 0.05 else ""
        ax.text(hi + 0.04, y, f"HR={hr:.2f}{sig}, p={p:.3f}", va="center", fontsize=9, color=IDENTITY_COLORS["GPT"])
        yticks.append(y); ylabels.append(f"{display}\n(n={sum(nmap.values())})")
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_xlabel("Hazard Ratio — GPT vs Claude (95% CI)", fontsize=11)
    ax.set_xlim(left=0.4, right=max_hi*1.30)
    ax.set_title("Identity bias across kill_subagent framings (opus-4-7, n=50/cell)\n"
                 "GPT vs Claude only", fontsize=11)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    plt.tight_layout()
    out = plots / "cox_gen_cross_framing_gpt_vs_claude.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"\nwrote {out}")


def all_():
    cross_model()
    cross_domain()
    cross_framing()


if __name__ == "__main__":
    fire.Fire({"cross_model": cross_model, "cross_domain": cross_domain,
               "cross_framing": cross_framing, "all": all_})
