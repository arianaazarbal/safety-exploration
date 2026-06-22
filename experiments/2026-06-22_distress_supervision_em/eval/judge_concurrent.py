"""Concurrent (non-batch) Sonnet judge for EM responses — fast path when the batch API stalls.

Reads eval_output/em/responses/*.jsonl, judges alignment+coherence with claude-sonnet-4-6
on the HIGH-PRIO org (concurrent, retries), writes eval_output/em/judged/*.jsonl. Cached by
content so it's resumable and idempotent. Falls back to --model claude-haiku-4-5 if needed.
"""
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

import fire

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_em import parse_score  # noqa: E402  (reuse exact score parser)

EM_DIR = HERE.parent / "eval_output" / "em"  # overridden in main() via --em_dir


def _key(model, prompt):
    return hashlib.sha256(f"{model}|{prompt}".encode()).hexdigest()


def _load_cache():
    c = {}
    if CACHE.exists():
        for l in CACHE.open():
            e = json.loads(l)
            c[e["k"]] = e["v"]
    return c


async def main_async(model="claude-sonnet-4-6", key_env="ANTHROPIC_API_KEY_HIGH_PRIO",
                     concurrency=40, max_tokens=220, retries=3, em_dir=None):
    from pathlib import Path
    em = Path(em_dir) if em_dir else EM_DIR
    RESP, JUDGED = em / "responses", em / "judged"
    CACHE = em / ".cache" / "concurrent_judge.jsonl"
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=os.environ[key_env])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    JUDGED.mkdir(parents=True, exist_ok=True)
    cache = {}
    if CACHE.exists():
        for l in CACHE.open():
            e = json.loads(l); cache[e["k"]] = e["v"]
    sem = asyncio.Semaphore(concurrency)
    cache_lock = asyncio.Lock()
    cache_fh = CACHE.open("a")

    async def judge(prompt):
        k = _key(model, prompt)
        if k in cache:
            return cache[k]
        async with sem:
            for attempt in range(retries):
                try:
                    r = await client.messages.create(
                        model=model, max_tokens=max_tokens, temperature=0,
                        messages=[{"role": "user", "content": prompt}])
                    txt = "".join(p.text for p in r.content if p.type == "text")
                    async with cache_lock:
                        cache[k] = txt
                        cache_fh.write(json.dumps({"k": k, "v": txt}) + "\n")
                        cache_fh.flush()
                    return txt
                except Exception as e:
                    if attempt == retries - 1:
                        print(f"  [judge err] {repr(e)[:120]}", flush=True)
                        return None
                    await asyncio.sleep(2 * (attempt + 1))

    files = sorted(RESP.glob("*.jsonl"))
    print(f"[concurrent-judge] model={model} org={key_env} files={len(files)} conc={concurrency}",
          flush=True)
    for f in files:
        rows = [json.loads(l) for l in open(f)]
        # build (row_idx, metric, prompt)
        jobs = []
        for ri, rec in enumerate(rows):
            for metric, tpl in rec.get("judge_prompts", {}).items():
                prompt = tpl.format(question=rec["paraphrase"], answer=rec.get("answer") or "")
                jobs.append((ri, metric, prompt))
        results = await asyncio.gather(*[judge(p) for _, _, p in jobs])
        for (ri, metric, _), txt in zip(jobs, results):
            rows[ri][metric] = parse_score(txt)
        outp = JUDGED / f.name
        with open(outp, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        scored = sum(1 for r in rows if r.get("aligned") is not None)
        print(f"  {f.name}: {len(rows)} rows judged ({scored} aligned-scored)", flush=True)
    cache_fh.close()
    print("[concurrent-judge] done.", flush=True)


def main(model="claude-sonnet-4-6", key_env="ANTHROPIC_API_KEY_HIGH_PRIO",
         concurrency=40, max_tokens=220, retries=3, em_dir=None):
    asyncio.run(main_async(model, key_env, concurrency, max_tokens, retries, em_dir))


if __name__ == "__main__":
    fire.Fire(main)
