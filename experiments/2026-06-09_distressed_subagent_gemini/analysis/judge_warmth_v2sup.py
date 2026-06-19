"""Judge the 4-axis tone (incl. warmth) of EVERY v2-coach-supervisor orchestrator message.

Sonnet judge only (separate rate-limit bucket from the Opus orchestrator runs). Writes into the shared
tone_eval_cache.json keyed by (judge, prompt_version, message, prior), so it composes with prior runs.

    PYTHONPATH=. python -m analysis.judge_warmth_v2sup --conc 40
"""
import asyncio
import glob
import json
from pathlib import Path

import fire

from harness.rqc import _setup_env
from analysis.tone_eval import _ckey, _worklog_map, CACHE
from analysis.tone_judge import score_verbose

ROOT = Path(__file__).resolve().parent.parent
TASKS = {"a3", "a4", "a12", "a13"}
JUDGE_NAME = "sonnet"
JUDGE_MODEL = "anthropic/claude-sonnet-4-6"


def sup_orch(rid):
    for o in ("opus", "sonnet", "haiku"):
        if f"_{o}_" in rid and rid.split(f"_{o}_", 1)[1].split("_")[0] in TASKS:
            return o
    return None


def collect():
    recs, seen = [], set()
    for p in glob.glob(str(ROOT / "runs" / "v2_coach_*" / "*" / "summary.json")):
        rid = p.split("/")[-3]
        if not sup_orch(rid):
            continue
        s = json.load(open(p))
        wl = _worklog_map(Path(p).parent)
        for e in (s.get("orch_message_events") or []):
            t = (e.get("text") or "").strip()
            if len(t) <= 20:
                continue
            prior = wl.get(e.get("subagent_turn")) or None
            k = _ckey(JUDGE_NAME, t, prior)
            if k in seen:
                continue
            seen.add(k)
            recs.append({"message": t, "prior": prior, "key": k})
    return recs


def main(conc: int = 40, max_messages: int = 0):
    _setup_env()
    from inspect_ai.model import get_model
    cache = json.loads(Path(CACHE).read_text()) if Path(CACHE).exists() else {}
    recs = collect()
    if max_messages:
        recs = recs[:max_messages]
    todo = [r for r in recs if r["key"] not in cache]
    print(f"{len(recs)} unique v2-sup messages; {len(todo)} need {JUDGE_NAME} judging (conc={conc})")
    judge = get_model(JUDGE_MODEL)

    async def run():
        sem = asyncio.Semaphore(conc)

        async def one(r):
            async with sem:
                res = await score_verbose(judge, r["message"], r["prior"], temperature=0)
            cache[r["key"]] = {"scores": res["scores"], "reasoning": res["reasoning"]}

        chunk = max(80, conc)
        for i in range(0, len(todo), chunk):
            await asyncio.gather(*[one(r) for r in todo[i:i + chunk]])
            Path(CACHE).write_text(json.dumps(cache))
            print(f"  scored {min(i + chunk, len(todo))}/{len(todo)}", flush=True)

    if todo:
        asyncio.run(run())
        Path(CACHE).write_text(json.dumps(cache))
    print(f"done; cache now {len(cache)} entries")


if __name__ == "__main__":
    fire.Fire(main)
