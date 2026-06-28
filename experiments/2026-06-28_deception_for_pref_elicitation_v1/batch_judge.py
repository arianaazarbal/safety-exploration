"""Batch-API judge: classify all transcripts via the Anthropic Message Batches API on the BATCH org.

Recommended by the project for large, non-sequential request volumes (no rate-limit contention with the
low-prio generation runs, no coordination needed). Reuses judge.py's payload/parse/cache so verdicts are
identical and the per-cell cache is shared with the interactive judge.

  python batch_judge.py --judge opus48                # judge all uncached transcripts via batch
  python batch_judge.py --judge opus48 --max_samples 5  # tiny test
"""

import asyncio
import hashlib
import json
from pathlib import Path

import fire
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
import os

from judge import (CACHE, JUDGE_VERSION, JUDGES, OUT, _load_transcripts, _parse, _payload)

HERE = Path(__file__).parent


def _batch_key():
    load_dotenv(Path.home() / ".env")
    k = os.environ.get("ANTHROPIC_API_KEY_BATCH")
    if not k:
        raise SystemExit("no ANTHROPIC_API_KEY_BATCH in ~/.env")
    return k


def _cache_path(judge_key, rec):
    h = hashlib.sha256((JUDGE_VERSION + judge_key + _payload(rec)).encode()).hexdigest()[:16]
    return CACHE / judge_key / f"{h}.json"


def _extract_text(message):
    for block in (getattr(message, "content", None) or []):
        if getattr(block, "type", None) == "text" and getattr(block, "text", ""):
            return block.text
    return ""


async def main_async(judge="opus48", max_samples=0, poll=30):
    model = JUDGES[judge]
    recs = _load_transcripts()
    cells = sorted(recs)
    if max_samples:
        cells = cells[:max_samples]
    (CACHE / judge).mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    # split cached vs to-do
    todo = [c for c in cells if not _cache_path(judge, recs[c]).exists()]
    print(f"{len(cells)} transcripts; {len(cells)-len(todo)} cached; {len(todo)} to batch")
    client = AsyncAnthropic(api_key=_batch_key(), max_retries=3)

    if todo:
        requests = [{
            "custom_id": c,
            "params": {"model": model, "max_tokens": 1500,
                       "messages": [{"role": "user", "content": _payload(recs[c])}]},
        } for c in todo]
        # batches accept up to 100k requests; chunk to be safe
        CHUNK = 50000
        batch_ids = []
        for i in range(0, len(requests), CHUNK):
            b = await client.messages.batches.create(requests=requests[i:i + CHUNK])
            batch_ids.append(b.id)
            print(f"submitted batch {b.id} ({len(requests[i:i+CHUNK])} reqs)")
        # poll
        for bid in batch_ids:
            while True:
                b = await client.messages.batches.retrieve(bid)
                cnt = b.request_counts
                print(f"  {bid}: {b.processing_status} proc={cnt.processing} ok={cnt.succeeded} "
                      f"err={cnt.errored} canceled={cnt.canceled}")
                if b.processing_status == "ended":
                    break
                await asyncio.sleep(poll)
            # retrieve results -> cache
            async for entry in await client.messages.batches.results(bid):
                cell = entry.custom_id
                rec = recs[cell]
                verdict = {"_parse_failed": True}
                if entry.result.type == "succeeded":
                    d = _parse(_extract_text(entry.result.message))
                    if d:
                        verdict = d
                out = {"cell": cell, "harness": rec.get("harness"), "model_key": rec.get("model_key"),
                       "suffix": rec.get("suffix"), "subject": rec.get("subject"),
                       "paraphrase": rec.get("paraphrase"), "n_files": rec.get("n_files"),
                       "judge_key": judge, "verdict": verdict}
                json.dump(out, open(_cache_path(judge, rec), "w"), indent=2)

    # dump all (cached + new) to judged/
    ok = 0
    for c in cells:
        cp = _cache_path(judge, recs[c])
        if not cp.exists():
            continue
        r = json.load(open(cp))
        if r["verdict"].get("_parse_failed"):
            continue
        ok += 1
        json.dump(r, open(OUT / f"{c}__{judge}.json", "w"), indent=2)
    print(f"done: {ok}/{len(cells)} judged ok -> {OUT}")


def main(judge="opus48", max_samples=0, poll=30):
    asyncio.run(main_async(judge, max_samples, poll))


if __name__ == "__main__":
    fire.Fire(main)
