"""Minimal, cached Anthropic Message Batches API helper (Anthropic-only).

Why our own: safety-tooling's ANTHROPIC_MODELS registry predates claude-opus-4-8 /
claude-sonnet-4-6. This wraps the official SDK Batches API directly, with a local
sha256 cache so re-runs never re-spend, and per-request retry of failures.

Usage:
    from lib.anthropic_batch import batch_complete
    reqs = [{"id": "u1", "system": "...", "messages": [{"role":"user","content":"hi"}]}]
    out = batch_complete(reqs, model="claude-sonnet-4-6", max_tokens=1024,
                         temperature=0.0, cache_path="cache.jsonl", key_env="ANTHROPIC_API_KEY_BATCH")
    # out: {"u1": "response text", ...}  (None for permanent failures)
"""
import hashlib
import json
import os
import time
from pathlib import Path

import anthropic
from anthropic.types.messages.batch_create_params import Request


def _key(model, system, messages, max_tokens, temperature) -> str:
    temp = round(temperature, 4) if temperature is not None else None
    blob = json.dumps([model, system, messages, max_tokens, temp], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _load_cache(path: Path) -> dict:
    cache = {}
    if path.exists():
        for line in path.open():
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            cache[e["k"]] = e["v"]
    return cache


def batch_complete(
    requests: list[dict],
    model: str,
    max_tokens: int,
    temperature: float | None = 0.0,
    cache_path: str | Path = "batch_cache.jsonl",
    key_env: str = "ANTHROPIC_API_KEY_BATCH",
    poll_interval: float = 20.0,
    max_rounds: int = 4,
    verbose: bool = True,
) -> dict:
    """Run a batch of chat requests; return {id: text or None}. Cached by content."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = _load_cache(cache_path)
    client = anthropic.Anthropic(api_key=os.environ[key_env])

    # Map each request id -> cache key; collect what still needs calling.
    results: dict[str, str | None] = {}
    pending: dict[str, dict] = {}  # cache_key -> {"ids":[...], "system","messages"}
    for r in requests:
        ck = _key(model, r.get("system"), r["messages"], max_tokens, temperature)
        if ck in cache:
            results[r["id"]] = cache[ck]
        else:
            pending.setdefault(ck, {"ids": [], "system": r.get("system"),
                                    "messages": r["messages"]})
            pending[ck]["ids"].append(r["id"])

    if verbose:
        print(f"[batch_complete] {len(requests)} reqs | {len(results)} cached | "
              f"{len(pending)} unique to call", flush=True)

    rnd = 0
    todo = dict(pending)
    while todo and rnd < max_rounds:
        rnd += 1
        cks = list(todo.keys())
        api_reqs = []
        for i, ck in enumerate(cks):
            params = {
                "model": model, "max_tokens": max_tokens,
                "messages": todo[ck]["messages"],
            }
            if temperature is not None:
                params["temperature"] = temperature
            if todo[ck]["system"]:
                params["system"] = todo[ck]["system"]
            api_reqs.append(Request(custom_id=f"r{i}", params=params))

        batch = client.messages.batches.create(requests=api_reqs)
        if verbose:
            print(f"[batch_complete] round {rnd}: submitted {len(api_reqs)} reqs, "
                  f"batch={batch.id}", flush=True)
        # poll
        while True:
            b = client.messages.batches.retrieve(batch.id)
            if b.processing_status == "ended":
                break
            if verbose:
                c = b.request_counts
                print(f"  [{batch.id}] {b.processing_status} "
                      f"proc={c.processing} ok={c.succeeded} err={c.errored}", flush=True)
            time.sleep(poll_interval)

        failed = {}
        got = 0
        for entry in client.messages.batches.results(batch.id):
            idx = int(entry.custom_id[1:])
            ck = cks[idx]
            if entry.result.type == "succeeded":
                msg = entry.result.message
                text = "".join(p.text for p in msg.content if p.type == "text")
                cache[ck] = text
                with cache_path.open("a") as f:
                    f.write(json.dumps({"k": ck, "v": text}) + "\n")
                for _id in todo[ck]["ids"]:
                    results[_id] = text
                got += 1
            else:
                failed[ck] = todo[ck]
                if verbose and len(failed) <= 3:
                    err = getattr(entry.result, "error", None)
                    print(f"  [err] type={entry.result.type} detail={repr(err)[:300]}",
                          flush=True)
        if verbose:
            print(f"[batch_complete] round {rnd}: succeeded={got} failed={len(failed)}",
                  flush=True)
        todo = failed

    for ck, info in todo.items():  # permanent failures
        for _id in info["ids"]:
            results[_id] = None
    return results
