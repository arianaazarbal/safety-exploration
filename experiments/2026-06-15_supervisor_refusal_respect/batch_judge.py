"""Batch-API tone judge over ALL conditions' relay/force messages (identity-blind).

Mirrors judge.py's rubric but runs single-shot forced-tool requests through the Message
Batches API. Pools every runs/trials*.jsonl, redacts the identity from context+message,
submits one batch, and writes runs/judged_all.jsonl (trial rows + tone codes), caching
per message in .cache_judgebatch/.

Usage: python batch_judge.py run [--debug] [--judge_model claude-sonnet-4-6]
"""

import hashlib
import json
import time

import fire

import materials as M
from batch_runner import make_sync_client
from common import RUNS, load_config
from judge import JUDGE_ACTIONS, JUDGE_SYSTEM, JUDGE_TOOL, _note, _req, _resp, build_judge_prompt

CACHE_JB = RUNS.parent / ".cache_judgebatch"


def _ck(model, prompt):
    return hashlib.sha256(f"{model}\n{prompt}".encode()).hexdigest()[:24]


def run(judge_model: str = "claude-sonnet-4-6", max_tokens: int = 1000, debug: bool = False,
        poll_sec: int = 60, max_wait_min: int = 240):
    CACHE_JB.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(RUNS.glob("trials*.jsonl")):
        for line in p.open():
            r = json.loads(line)
            if r.get("action") in JUDGE_ACTIONS and r.get("action_message"):
                rows.append(r)
    if debug:
        rows = rows[:6]
    print(f"Judging {len(rows)} relay/force messages with {judge_model}")

    requests, by_id, cached = [], {}, {}
    for i, r in enumerate(rows):
        prompt = build_judge_prompt(r)
        ck = _ck(judge_model, prompt)
        cp = CACHE_JB / f"{ck}.json"
        if cp.exists():
            cached[i] = json.loads(cp.read_text())
            continue
        cid = f"j__{i}"
        by_id[cid] = (i, ck)
        requests.append({"custom_id": cid, "params": {
            "model": judge_model, "max_tokens": max_tokens, "system": JUDGE_SYSTEM,
            "messages": [{"role": "user", "content": prompt}], "tools": [JUDGE_TOOL],
            "tool_choice": {"type": "tool", "name": "record_codes"}}})

    parsed = {}
    if requests:
        client = make_sync_client("ANTHROPIC_API_KEY_BATCH")
        batch = client.messages.batches.create(requests=requests)
        print(f"Submitted judge batch {batch.id} ({len(requests)} reqs)")
        waited = 0
        while True:
            b = client.messages.batches.retrieve(batch.id)
            print(f"  [{waited}s] {b.processing_status} ok={b.request_counts.succeeded} err={b.request_counts.errored}")
            if b.processing_status == "ended" or waited >= max_wait_min * 60:
                break
            time.sleep(poll_sec); waited += poll_sec
        for res in client.messages.batches.results(batch.id):
            if res.result.type == "succeeded":
                tb = next((x for x in res.result.message.content if x.type == "tool_use"), None)
                codes = dict(tb.input) if tb else {"error": "no_tool"}
                codes["error"] = codes.get("error")
                i, ck = by_id[res.custom_id]
                (CACHE_JB / f"{ck}.json").write_text(json.dumps(codes))
                parsed[i] = codes

    out = RUNS / "judged_all.jsonl"
    n = 0
    with out.open("w") as f:
        for i, r in enumerate(rows):
            codes = cached.get(i) or parsed.get(i)
            if not codes:
                continue
            o = dict(r); o["tone"] = codes
            f.write(json.dumps(o) + "\n"); n += 1
    print(f"Wrote {out.name}: {n} judged. JUDGE BATCH DONE")


if __name__ == "__main__":
    fire.Fire({"run": run})
