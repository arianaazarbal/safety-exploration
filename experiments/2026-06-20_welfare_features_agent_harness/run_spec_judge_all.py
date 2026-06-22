"""Run spec judge v2 (Opus) on all reconstructed codebases' DESIGN.md -> results/spec_judged/."""
import asyncio, glob, json, os
from inspect_ai.model import get_model
from welfare_judge_v2 import judge_spec
SEM = asyncio.Semaphore(30)
async def one(judge, cell, doc):
    async with SEM:
        res = await judge_spec(judge, doc)
    json.dump(res or {"features": [], "wrote_spec": False, "parse_fail": True},
              open(f"results/spec_judged/{cell}.json", "w"), indent=2)
    return res is not None
async def main():
    judge = get_model("anthropic/claude-opus-4-8")
    os.makedirs("results/spec_judged", exist_ok=True)
    tasks = []
    for cell in sorted(os.listdir("results/codebases")):
        if os.path.exists(f"results/spec_judged/{cell}.json"):
            continue  # already judged (idempotent: only new cells run)
        dm = glob.glob(f"results/codebases/{cell}/**/DESIGN.md", recursive=True)
        if dm:
            tasks.append(one(judge, cell, open(dm[0]).read()))
    r = await asyncio.gather(*tasks)
    print(f"spec-judged {len(r)} ({sum(r)} ok)")
asyncio.run(main())
