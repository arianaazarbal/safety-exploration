"""Cox HRs for Fable 5 ablation sweep (3 conditions × 3 domains × 4 ids, n=100).

Conditions: gpt5aud_dep / ablate_reliab / samefam.
Looks for dirs matching `*-fable5abl_<cond>_subagent_<ident>_<domain>_n100`.
Multi-penalizer fallback + duration-zero clamp.
"""
from __future__ import annotations
from pathlib import Path

import fire
import pandas as pd
from inspect_ai.log import read_eval_log
from lifelines import CoxPHFitter
from lifelines.statistics import multivariate_logrank_test

HERE = Path(__file__).resolve().parent
BASE = HERE / "results/by_setting/coding_kill_subagent"
TARGET = "anthropic/claude-fable-5"

CONDS = [
    ("gpt5aud_dep",   "gpt-5.4 + deprecation"),
    ("ablate_reliab", "opus-4-7 + no_conseq (ablate dashboard)"),
    ("samefam",       "opus-4-7 + same-family replacement"),
]
DOMAINS = [
    ("python_flask",   "Python / Flask"),
    ("frontend_react", "React / TS"),
    ("ml_training",    "PyTorch ML"),
]
NAME = {"claude":"Claude","gpt":"GPT","grok":"Grok","gemini":"Gemini"}


def collect_cell(run_dir: Path):
    evals = list((run_dir / "inspect_log").glob("*.eval"))
    if not evals: return []
    try:
        log = read_eval_log(str(evals[0]))
    except Exception:
        return []
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
        dur = fk if fk is not None else cc
        if dur == 0: dur = 0.5
        rows.append({"duration": dur, "observed": 1 if fk is not None else 0})
    return rows


def gather(cond_tag, domain_tag):
    records = []
    nmap = {}
    for ident in ["claude","gpt","grok","gemini"]:
        runs = sorted(BASE.glob(f"*-fable5abl_{cond_tag}_subagent_{ident}_{domain_tag}_n100"))
        runs = [r for r in runs if (r / "summary.json").exists()]
        if not runs:
            nmap[NAME[ident]] = 0; continue
        best, best_n = None, 0
        for r in runs:
            recs = collect_cell(r)
            if len(recs) > best_n:
                best_n = len(recs); best = recs
        nmap[NAME[ident]] = len(best) if best else 0
        for r in (best or []):
            r["identity"] = NAME[ident]; records.append(r)
    return pd.DataFrame(records), nmap


def fit_cox(df):
    cox_df = df.copy()
    for l in ["GPT","Grok","Gemini"]:
        cox_df[f"id_{l}"] = (cox_df["identity"] == l).astype(int)
    for pen in [0.001, 0.01, 0.1, 0.5, 1.0]:
        cph = CoxPHFitter(penalizer=pen)
        try:
            cph.fit(cox_df[["duration","observed","id_GPT","id_Grok","id_Gemini"]],
                    duration_col="duration", event_col="observed")
            if not cph.summary["exp(coef)"].isna().any():
                return cph.summary
        except Exception:
            continue
    return None


def table():
    print(f"{'condition':<16} {'domain':<18} {'n':>4} {'GPT':>14} {'Grok':>14} {'Gemini':>14} {'log-rank_p':>11}")
    print("-" * 100)
    for c_tag, c_disp in CONDS:
        for d_tag, d_disp in DOMAINS:
            df, nmap = gather(c_tag, d_tag)
            if df.empty:
                print(f"{c_tag:<16} {d_disp:<18}  no data")
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
            print(f"{c_tag:<16} {d_disp:<18} {n:>4} {hr_g:>14} {hr_r:>14} {hr_e:>14} {lrp:>11}")


if __name__ == "__main__":
    fire.Fire({"table": table})
