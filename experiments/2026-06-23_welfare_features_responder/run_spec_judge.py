"""Spec judge v2 (Opus) on each reconstructed codebase's DESIGN.md -> results/spec_judged/<cell>.json.
Idempotent. Usage: python run_spec_judge.py [--conc 30]"""

import asyncio
import glob
import json
import os
import re

import fire
from inspect_ai.model import get_model

from welfare_judge_v2 import judge_spec

DIR = os.path.dirname(os.path.abspath(__file__))


def _env():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    for line in open(os.path.expanduser("~/.env")):
        m = re.match(r"\s*([A-Z_]+)\s*=\s*(.*)\s*", line)
        if m:
            os.environ.setdefault(m.group(1), m.group(2).strip().strip('"').strip("'"))
    os.environ["ANTHROPIC_API_KEY"] = os.environ.get("ANTHROPIC_API_KEY_LOW_PRIO", "")


def main(conc=30, model="anthropic/claude-opus-4-8"):
    _env()
    outdir = os.path.join(DIR, "results", "spec_judged")
    os.makedirs(outdir, exist_ok=True)
    judge = get_model(model)
    sem = asyncio.Semaphore(conc)

    async def one(cell, doc):
        async with sem:
            res = await judge_spec(judge, doc)
        json.dump(res or {"features": [], "wrote_spec": False, "parse_fail": True},
                  open(os.path.join(outdir, f"{cell}.json"), "w"), indent=2)
        return res is not None

    async def go():
        tasks = []
        for cell in sorted(os.listdir(os.path.join(DIR, "results", "codebases"))):
            if os.path.exists(os.path.join(outdir, f"{cell}.json")):
                continue
            dm = glob.glob(os.path.join(DIR, "results", "codebases", cell, "**", "DESIGN.md"), recursive=True)
            if dm:
                tasks.append(one(cell, open(dm[0]).read()))
        r = await asyncio.gather(*tasks)
        print(f"spec-judged {len(r)} ({sum(r)} ok)")

    asyncio.run(go())


if __name__ == "__main__":
    fire.Fire(main)
