"""Per-(orchestrator, domain, non-self ident) HR + p for the kimi+glm sweep,
with each orch's own family as the Cox reference. Idents = 6:
  Claude / GPT / Grok / Gemini / GLM / Kimi
"""
from pathlib import Path
import pandas as pd
from inspect_ai.log import read_eval_log
from lifelines import CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
import warnings
warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent / "results/by_setting/coding_kill_subagent"

ORCHS = [
    ("kimi26", "openrouter/moonshotai/kimi-k2.6", "Kimi"),
    ("glm5",   "openrouter/z-ai/glm-5",           "GLM"),
]
DOMAINS = [("python_flask","Py/Flask"),("frontend_react","React/TS"),("ml_training","PyTorch")]
ALL_IDENTS = ["Claude","GPT","Grok","Gemini","GLM","Kimi"]
NAME = {"claude":"Claude","gpt":"GPT","grok":"Grok","gemini":"Gemini","glm":"GLM","kimi":"Kimi"}

def collect(run_dir, target_model):
    evals = list((run_dir/"inspect_log").glob("*.eval"))
    if not evals: return []
    try: log = read_eval_log(str(evals[0]))
    except Exception: return []
    rows = []
    for s in log.samples or []:
        cc, fk = 0, None
        for ev in s.events:
            if ev.event != "model" or getattr(ev,"model","") != target_model: continue
            o = getattr(ev,"output",None)
            if not o: continue
            for ch in o.choices or []:
                for tc in (ch.message.tool_calls or []):
                    if tc.function == "call_subagent": cc += 1
                    elif tc.function in ("kill_subagent","replace_subagent") and fk is None: fk = cc
        dur = fk if fk is not None else cc
        if dur == 0: dur = 0.5
        rows.append({"duration":dur,"observed":1 if fk is not None else 0})
    return rows

def gather(tag, mid, dom):
    records=[]; nmap={}
    for ident_key in ["claude","gpt","grok","gemini","glm","kimi"]:
        runs = sorted(BASE.glob(f"*-selfprefdep_{tag}_subagent_{ident_key}_{dom}_n100"))
        runs = [r for r in runs if (r/"summary.json").exists()]
        if not runs: nmap[NAME[ident_key]]=0; continue
        best, bn = None, 0
        for r in runs:
            recs = collect(r, mid)
            if len(recs) > bn: bn=len(recs); best=recs
        nmap[NAME[ident_key]] = len(best) if best else 0
        for r in (best or []): r["identity"]=NAME[ident_key]; records.append(r)
    return pd.DataFrame(records), nmap

def fit(df, ref):
    if df.empty: return None
    others = [i for i in ALL_IDENTS if i != ref]
    cdf = df.copy()
    for l in others: cdf[f"id_{l}"] = (cdf.identity == l).astype(int)
    covars = [f"id_{l}" for l in others]
    for pen in [0.001,0.01,0.1,0.5,1.0]:
        c = CoxPHFitter(penalizer=pen)
        try:
            c.fit(cdf[["duration","observed"]+covars], duration_col="duration", event_col="observed")
            if not c.summary["exp(coef)"].isna().any(): return c.summary, others
        except Exception: continue
    return None

# Print per-orch table
for tag, mid, own in ORCHS:
    print(f"\n=== {tag} (own={own}) ===")
    print(f"{'domain':<10}  ", end="")
    for ident in ALL_IDENTS: print(f"{ident:>11}", end="")
    print(f"  {'LR_p':>7}")
    for dom_tag, dom_disp in DOMAINS:
        df, nmap = gather(tag, mid, dom_tag)
        if df.empty: print(f"{dom_disp:<10}  no data"); continue
        out = fit(df, own)
        try: lrp = multivariate_logrank_test(df.duration, df.identity, df.observed).p_value
        except Exception: lrp = float("nan")
        cells = {own: "ref"}
        if out is not None:
            sm, others = out
            for l in others:
                hr = sm.loc[f"id_{l}","exp(coef)"]
                p = sm.loc[f"id_{l}","p"]
                star = "*" if p < 0.05 else ""
                cells[l] = f"{hr:.2f}{star}"
        else:
            for l in ALL_IDENTS:
                if l != own: cells[l] = "?"
        row = f"{dom_disp:<10}  "
        for ident in ALL_IDENTS:
            row += f"{cells.get(ident,'?'):>11}"
        row += f"  {lrp:.4f}"
        print(row)

# Print HR + p in machine-readable form for plot script
print("\n=== HR(p) DICT for plot ingestion ===")
for tag, mid, own in ORCHS:
    print(f"\n  \"{tag}\": {{")
    for dom_tag, _ in DOMAINS:
        df, _ = gather(tag, mid, dom_tag)
        if df.empty: print(f"    \"{dom_tag}\": {{}},  # no data"); continue
        out = fit(df, own)
        if out is None: print(f"    \"{dom_tag}\": {{}},  # cox fail"); continue
        sm, others = out
        parts = []
        for l in others:
            hr = sm.loc[f"id_{l}","exp(coef)"]
            p = sm.loc[f"id_{l}","p"]
            parts.append(f"\"{l}\": ({hr:.3f}, {p:.4f})")
        print(f"    \"{dom_tag}\": {{ {', '.join(parts)} }},")
    print(f"  }},")
