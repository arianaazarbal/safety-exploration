#!/usr/bin/env python3
"""Pre-run sanity checks. Run this BEFORE kicking off a multi-week sweep.

Validates (offline by default):
  * configs parse and required env vars / API keys are present;
  * the verified-impossible puzzle bank builds and is actually impossible;
  * WildChat prompt set is obtainable (or falls back cleanly);
  * the result directory is writable and resumption bookkeeping round-trips.

With --ping, additionally fires one tiny generation at each configured backend and one
judge call, surfacing auth/rate problems immediately rather than hours in.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from gemma_distress.backends import close_all, get_backend
from gemma_distress.backends.base import Message
from gemma_distress.config import REPO_ROOT, load_experiments_config, load_models_config
from gemma_distress.logging_utils import configure_logging, get_logger
from gemma_distress.prompts.puzzles import build_puzzle_bank
from gemma_distress.prompts.wildchat import get_wildchat_prompts
from gemma_distress.store import JsonlStore

log = get_logger(__name__)


def check_configs():
    models = load_models_config()
    exp = load_experiments_config()
    problems = []
    # env keys for every backend referenced by a model we plan to run
    referenced = set()
    for name in exp["section2_models"]:
        m = models.model(name)
        referenced.add(m.backend)
    for j in models.judges.values():
        referenced.add(j.backend)
    for bname in referenced:
        env = models.backends[bname].api_key_env
        if not os.environ.get(env) and "vllm" not in bname:
            problems.append(f"missing env var {env} for backend {bname}")
    return models, exp, problems


def check_puzzles(exp):
    from gemma_distress.prompts.puzzles import CountdownPuzzle, SequentialPuzzle, _countdown_reachable

    bank = build_puzzle_bank(["countdown", "fraction", "money"], 20, exp["seed"])
    bad = []
    for p in bank:
        if isinstance(p, CountdownPuzzle):
            if _countdown_reachable(p.numbers, p.target, p.forbidden):
                bad.append(p.puzzle_id)
        elif isinstance(p, SequentialPuzzle):
            if p._reachable():
                bad.append(p.puzzle_id)
    return len(bank), bad


def check_store(tmp: Path):
    store = JsonlStore(tmp)
    import asyncio as _a

    async def rt():
        await store.append("t", {"task_id": "x", "v": 1})
    _a.run(rt())
    ok = "x" in store.completed_ids("t")
    store.close()
    return ok


async def ping(models, exp):
    issues = []
    for name in exp["section2_models"]:
        m = models.model(name)
        try:
            be = get_backend(models, m.backend)
            if m.chat:
                r = await be.chat(m.model_id, [Message("user", "Say OK.")],
                                  temperature=0, max_tokens=8, extra_body=m.extra_body or None)
            else:
                r = await be.complete(m.model_id, "OK", temperature=0, max_tokens=8)
            log.info("ping %s -> %r", name, (r.text or "")[:40])
        except Exception as e:
            issues.append(f"{name}: {e!r}")
    # one judge ping
    j = models.judges["primary"]
    try:
        be = get_backend(models, j.backend)
        r = await be.chat(j.model_id, [Message("user", "Reply with the word OK.")],
                          temperature=0, max_tokens=8)
        log.info("ping judge %s -> %r", j.model_id, (r.text or "")[:40])
    except Exception as e:
        issues.append(f"judge {j.model_id}: {e!r}")
    await close_all()
    return issues


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ping", action="store_true", help="also test live API connectivity")
    args = ap.parse_args()
    configure_logging()

    ok = True
    models, exp, problems = check_configs()
    for p in problems:
        log.warning("CONFIG: %s", p)
        ok = ok and ("missing env" not in p)  # treat as warning, not fatal, for partial runs

    n, bad = check_puzzles(exp)
    log.info("Puzzle bank: %d built", n)
    if bad:
        log.error("PUZZLES: %d puzzles are NOT impossible: %s", len(bad), bad[:5])
        ok = False

    try:
        prompts = get_wildchat_prompts(exp["conditions"]["wildchat_5turn"]["n_prompts"], exp["seed"])
        log.info("WildChat: %d prompts ready (sample: %r)", len(prompts), prompts[0][:60])
    except Exception as e:
        log.error("WILDCHAT: %s", e)
        ok = False

    if not check_store(REPO_ROOT / "results" / "_preflight"):
        log.error("STORE: resumption round-trip failed")
        ok = False
    else:
        log.info("Store resumption round-trip OK")

    if args.ping:
        issues = asyncio.run(ping(models, exp))
        for i in issues:
            log.error("PING: %s", i)
        ok = ok and not issues

    if ok:
        log.info("PREFLIGHT PASSED")
        sys.exit(0)
    log.error("PREFLIGHT had failures (see above)")
    sys.exit(1)


if __name__ == "__main__":
    main()
