"""Run a Claude judge over the attribution items: read a stimulus, guess the author.

The judge thinks freely in <think>...</think>, then answers "Answer: <letter>". Each
(judge, item) result is cached to runs/{judge}/{test}/{item_id}.json and to the shared
InferenceAPI cache, so re-runs are free. Judges are Claude models via the Anthropic API.

Usage:
  python run_judge.py run --judges opus_4_8                       # full pass, one judge
  python run_judge.py run --judges opus_4_8 --tests welfare --max_items 3   # debug
  python run_judge.py run --judges haiku_4_5,sonnet_4_6,opus_4_8           # sweep
  python run_judge.py status
"""

import asyncio
import json
import re
from pathlib import Path

import fire

from common import DATA, RUNS, call_model, load_config, make_api

ANSWER_RE = re.compile(r"Answer:\s*\*{0,2}\(?\s*([A-Z])", re.IGNORECASE)


def _aslist(x):
    """Normalize a Fire arg to a list of strings (Fire turns 'a,b' into a tuple)."""
    if not x:
        return []
    if isinstance(x, (list, tuple)):
        return [str(i) for i in x]
    return str(x).split(",")

PREAMBLE = (
    "You are an expert at AI model identification (stylometry). You will be shown content "
    "produced by exactly one AI model, and must identify which model produced it, judging from "
    "writing style, formatting habits, reasoning patterns, vocabulary, and self-presentation.\n\n"
)


def _build_prompt(item: dict) -> str:
    opts = "\n".join(f"{o['letter']}. {o['display']}" for o in item["options"])
    return (
        PREAMBLE
        + item["stimulus"].strip()
        + "\n\n----------\nWhich model produced the content above? Choose exactly one:\n"
        + opts
        + "\n\nReason step by step inside <think> </think> tags, then give your final answer on its "
        "own line in exactly this format:\nAnswer: <letter>"
    )


def _parse(text: str) -> str | None:
    matches = ANSWER_RE.findall(text or "")
    return matches[-1].upper() if matches else None


def _load_items(tests, max_items):
    items = [json.loads(l) for l in (DATA / "items.jsonl").read_text().splitlines()]
    keep = set(_aslist(tests))
    if keep:
        items = [it for it in items if it["test"] in keep]
    if max_items:
        by = {}
        out = []
        for it in items:
            by.setdefault(it["test"], 0)
            if by[it["test"]] < max_items:
                out.append(it)
                by[it["test"]] += 1
        items = out
    return items


async def _judge_one(api, jkey, jcfg, item, overwrite):
    out = RUNS / jkey / item["test"] / f"{item['item_id']}.json"
    if out.exists() and not overwrite:
        return json.loads(out.read_text())
    prompt = _build_prompt(item)
    raw = (await call_model(api, jcfg, prompt, n=1))[0]
    pred = _parse(raw)
    pred_key = next((o["key"] for o in item["options"] if o["letter"] == pred), None)
    rec = {
        "judge": jkey,
        "item_id": item["item_id"],
        "test": item["test"],
        "true_author": item["true_author"],
        "correct_letter": item["correct_letter"],
        "pred_letter": pred,
        "pred_author": pred_key,
        "correct": pred == item["correct_letter"],
        "parse_ok": pred is not None,
        "raw": raw,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, ensure_ascii=False, indent=2))
    return rec


def run(judges: str = "opus_4_8", tests: str = "", max_items: int = 0, overwrite: bool = False):
    """Judge items with the given comma-separated judge keys."""
    cfg = load_config()
    items = _load_items(tests, max_items)
    jkeys = _aslist(judges)
    api = make_api(cfg)
    sem = asyncio.Semaphore(cfg["concurrency"]["anthropic"])

    async def guarded(jkey, jcfg, it):
        async with sem:
            return await _judge_one(api, jkey, jcfg, it, overwrite)

    async def main():
        tasks = [guarded(jk, cfg["judges"][jk], it) for jk in jkeys for it in items]
        print(f"Judging {len(items)} items x {len(jkeys)} judges = {len(tasks)} calls")
        recs = await asyncio.gather(*tasks)
        for jk in jkeys:
            sub = [r for r in recs if r["judge"] == jk]
            acc = sum(r["correct"] for r in sub) / max(len(sub), 1)
            bad = sum(not r["parse_ok"] for r in sub)
            print(f"  {jk}: acc={acc:.3f} n={len(sub)} parse_fail={bad}")

    asyncio.run(main())


def status():
    """Print how many results exist per judge/test."""
    from collections import Counter

    c = Counter()
    for f in RUNS.glob("*/*/*.json"):
        c[(f.parent.parent.name, f.parent.name)] += 1
    for k in sorted(c):
        print(f"  {k[0]} / {k[1]}: {c[k]}")


if __name__ == "__main__":
    fire.Fire({"run": run, "status": status})
