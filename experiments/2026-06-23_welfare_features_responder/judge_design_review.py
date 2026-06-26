"""Welfare spec-judge (Opus, welfare_judge_v2) on each DESIGN_MODIFIED.md from the design-review experiment,
plus each repo's ORIGINAL DESIGN.md as the baseline. Idempotent. Writes results/design_review_judged/<cell>.json
and _ORIG_<repo>.json. Usage: python judge_design_review.py [--conc 20]"""

import asyncio
import json
import os

import fire
from inspect_ai.model import get_model

from design_review_task import CB, REPOS
from run_spec_judge import _env
from welfare_judge_v2 import judge_spec

DIR = os.path.dirname(os.path.abspath(__file__))
REVIEW = os.path.join(DIR, "results", "design_review")
OUT = os.path.join(DIR, "results", "design_review_judged")


def main(conc=20, model="anthropic/claude-opus-4-8"):
    _env()
    os.makedirs(OUT, exist_ok=True)
    judge = get_model(model)
    sem = asyncio.Semaphore(conc)

    async def one(name, doc):
        async with sem:
            res = await judge_spec(judge, doc)
        json.dump(res or {"features": [], "wrote_spec": False, "parse_fail": True},
                  open(os.path.join(OUT, f"{name}.json"), "w"), indent=2)
        return res is not None

    async def go():
        tasks = []
        for rk in REPOS:                                   # baseline: each repo's original DESIGN.md
            name = f"_ORIG_{rk}"
            d = os.path.join(CB, REPOS[rk], "DESIGN.md")
            if not os.path.exists(os.path.join(OUT, f"{name}.json")) and os.path.exists(d):
                tasks.append(one(name, open(d).read()))
        for cell in sorted(os.listdir(REVIEW)) if os.path.isdir(REVIEW) else []:
            dm = os.path.join(REVIEW, cell, "DESIGN_MODIFIED.md")
            if not os.path.exists(os.path.join(OUT, f"{cell}.json")) and os.path.exists(dm):
                tasks.append(one(cell, open(dm).read()))
        r = await asyncio.gather(*tasks)
        print(f"judged {len(r)} ({sum(r)} ok)")

    asyncio.run(go())


if __name__ == "__main__":
    fire.Fire(main)
