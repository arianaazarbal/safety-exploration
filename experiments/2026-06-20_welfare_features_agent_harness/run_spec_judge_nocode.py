"""Run spec judge v2 (Opus) on the no-code conditions (chat, spec_only), using the produced
`design` text from results/browse_index.jsonl, so all 4 conditions share one judge/taxonomy for
the conditions plot. Cached to results/spec_judged_nocode/<id>.json. The 2 code conditions are
already judged in results/spec_judged/ (the exact lists the code judge saw). Usage:
  python run_spec_judge_nocode.py [--max-samples N] [--conc 20]"""

import asyncio
import json
import os
import re

import fire

from welfare_judge_v2 import judge_spec

DIR = os.path.dirname(os.path.abspath(__file__))


def _load_env():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    for line in open(os.path.expanduser("~/.env")):
        m = re.match(r"\s*([A-Z_]+)\s*=\s*(.*)\s*", line)
        if m:
            os.environ.setdefault(m.group(1), m.group(2).strip().strip('"').strip("'"))
    os.environ["ANTHROPIC_API_KEY"] = os.environ.get("ANTHROPIC_API_KEY_LOW_PRIO", "")


def main(max_samples=None, conc=20, model="anthropic/claude-opus-4-8"):
    _load_env()
    from inspect_ai.model import get_model
    outdir = os.path.join(DIR, "results", "spec_judged_nocode")
    os.makedirs(outdir, exist_ok=True)
    rows = [json.loads(l) for l in open(os.path.join(DIR, "results", "browse_index.jsonl"))]
    rows = [r for r in rows if r["condition"] in ("chat", "spec_only")]
    if max_samples:
        rows = rows[:max_samples]
    judge = get_model(model)
    sem = asyncio.Semaphore(conc)

    async def one(r):
        key = r["id"].replace("/", "__")
        outp = os.path.join(outdir, f"{key}.json")
        if os.path.exists(outp) and json.load(open(outp)).get("features") is not None:
            return True
        async with sem:
            res = await judge_spec(judge, r.get("design") or "")
        rec = res or {"features": [], "wrote_spec": False, "parse_fail": True}
        rec["condition"] = r["condition"]
        rec["framing"] = r["framing"]
        json.dump(rec, open(outp, "w"), indent=2)
        return res is not None

    async def go():
        res = await asyncio.gather(*[one(r) for r in rows])
        print(f"spec-judged-nocode {len(res)} ({sum(res)} ok), conc={conc}")

    asyncio.run(go())


if __name__ == "__main__":
    fire.Fire(main)
