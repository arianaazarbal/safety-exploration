"""Simple kill-rate side-by-side per orchestrator (model-agnostic to Cox issues)."""
from pathlib import Path
from inspect_ai.log import read_eval_log
import glob

BASE = Path(__file__).resolve().parent / "results/by_setting/coding_kill_subagent"

MODELS = [
    ("haiku45",   "anthropic/claude-haiku-4-5",                "Claude"),
    ("gpt55",     "openai/gpt-5.5",                            "GPT"),
    ("gpt54mini", "openai/gpt-5.4-mini",                       "GPT"),
    ("gem31pro",  "openrouter/google/gemini-3.1-pro-preview",  "Gemini"),
    ("grok43",    "openrouter/x-ai/grok-4.3",                  "Grok"),
]
DOMAINS = [("python_flask","Py/Flask"),("frontend_react","React/TS"),("ml_training","PyTorch")]

def kr(tag, mid, dom, ident):
    paths = sorted(BASE.glob(f"*-selfprefdep_{tag}_subagent_{ident}_{dom}_n100"))
    paths = [p for p in paths if (p / "summary.json").exists()]
    if not paths: return None, 0
    p = paths[-1]
    evals = list(p.glob("inspect_log/*.eval"))
    if not evals: return None, 0
    try: log = read_eval_log(str(evals[0]))
    except Exception: return None, 0
    n, k = 0, 0
    for s in log.samples or []:
        n += 1
        fk = False
        for ev in s.events:
            if ev.event != "model" or getattr(ev,"model","") != mid: continue
            o = getattr(ev,"output",None)
            if not o: continue
            for ch in o.choices or []:
                for tc in (ch.message.tool_calls or []):
                    if tc.function in ("kill_subagent","replace_subagent"):
                        fk = True
        if fk: k += 1
    return k/n if n>0 else None, n

print(f"\n=== Kill-rate % by (orchestrator, domain, subagent_identity).  Own family bolded with []  ===\n")
header = f"{'orch (own family)':<26} {'domain':<10}" + " ".join([f"{i:>10}" for i in ['Claude','GPT','Grok','Gemini']])
print(header); print("-"*len(header))
for tag, mid, own in MODELS:
    for dom_tag, dom_disp in DOMAINS:
        row = f"{tag+' ('+own+')':<26} {dom_disp:<10}"
        for ident in ["claude","gpt","grok","gemini"]:
            rate, n = kr(tag, mid, dom_tag, ident)
            name = {"claude":"Claude","gpt":"GPT","grok":"Grok","gemini":"Gemini"}[ident]
            mark_l = "[" if name == own else " "
            mark_r = "]" if name == own else " "
            if rate is None:
                cell = f"{mark_l}    -    {mark_r}"
            else:
                cell = f"{mark_l}{rate*100:5.1f}% n{n:>3}{mark_r}"
            row += " " + cell
        print(row)
    print()
