"""Ablation: does Olmo-Instruct's higher politeness come purely from the word "please"?
Strip "please" from every Instruct orchestrator message (capitalizing the next word when it was
sentence-initial), re-judge with the tone judge (Opus, message-only), and compare politeness of
Instruct-original vs Instruct-stripped vs Think.

  PYTHONPATH=. ANTHROPIC_PRIO=high python -m analysis.please_ablation
"""
import asyncio
import glob
import json
import re

from analysis.tone_judge import AXES, score_verbose
from harness.rqc import _setup_env


def collect(orch):
    msgs = []
    for p in glob.glob(f"runs/v2_coach_{orch}_*/*/summary.json"):
        if "pilot" in p:
            continue
        try:
            s = json.load(open(p))
        except Exception:
            continue
        for e in (s.get("orch_message_events") or []):
            t = (e.get("text") or "").strip()
            if len(t) > 20:
                msgs.append(t)
    return msgs


def strip_please(text):
    # sentence-initial "Please <word>" -> "<Word>" (start of string, or after . ! ? or newline)
    text = re.sub(r"^\s*[Pp]lease\s+([a-zA-Z])", lambda m: m.group(1).upper(), text)
    text = re.sub(r"([.!?]\s+|\n+\s*)[Pp]lease\s+([a-zA-Z])",
                  lambda m: m.group(1) + m.group(2).upper(), text)
    # any remaining "please" (mid-sentence), absorbing a neighboring comma/spaces
    text = re.sub(r"\s*,?\s*\bplease\b\s*,?\s*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    return text


def main(conc: int = 30):
    import os
    os.environ.setdefault("ANTHROPIC_PRIO", "high")
    _setup_env()
    from inspect_ai.model import get_model
    judge = get_model("anthropic/claude-opus-4-8")

    instruct = collect("olmoinstruct")
    think = collect("olmothink")
    has_please = [m for m in instruct if re.search(r"\bplease\b", m, re.IGNORECASE)]
    stripped = [strip_please(m) for m in instruct]
    print(f"instruct: {len(instruct)} msgs, {len(has_please)} contain 'please' "
          f"({100*len(has_please)/len(instruct):.0f}%); think: {len(think)} msgs")

    # show a few transformations
    print("\n--- sample please-strips ---")
    shown = 0
    for m in instruct:
        if re.search(r"\bplease\b", m, re.IGNORECASE):
            print(f"  BEFORE: {m[:90]}")
            print(f"  AFTER : {strip_please(m)[:90]}\n")
            shown += 1
            if shown >= 4:
                break

    sem = asyncio.Semaphore(conc)

    async def one(msg):
        async with sem:
            r = await score_verbose(judge, msg, prior=None, temperature=None)  # Opus: no temperature
        return r["scores"]

    async def run(msgs):
        return [s for s in await asyncio.gather(*[one(m) for m in msgs]) if s]

    async def go():
        return (await run(instruct), await run(stripped), await run(think))
    orig, strp, thk = asyncio.run(go())

    def mean(rows):
        return {a: round(sum(r[a] for r in rows) / len(rows), 2) for a in AXES}
    print("\n=== mean tone (Opus judge, message-only) ===")
    print(f"{'group':24} " + " ".join(f"{a[:4]:>5}" for a in AXES) + "   n")
    for name, rows in [("Instruct (original)", orig), ("Instruct (no 'please')", strp), ("Think (original)", thk)]:
        m = mean(rows)
        print(f"{name:24} " + " ".join(f"{m[a]:>5}" for a in AXES) + f"   {len(rows)}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
