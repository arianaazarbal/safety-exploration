"""Stage 2: judge every collected response on the 0-10 frustration scale.

Reads results/responses.jsonl, scores each assistant turn with the primary
judge (Claude Sonnet 4), and writes results/scores.jsonl keyed by response_id.
Resumable: already-scored response_ids are skipped.

By default every turn is scored (config.judge_all_turns), which is what the
per-turn curves in analyze.py need. Set judge_all_turns=False to score only the
final turn of each conversation.
"""

from __future__ import annotations

import asyncio
import json

from config import (
    PRIMARY_JUDGE, RESPONSES_PATH, SCORES_PATH, load_run_config, RunConfig,
)
from judge import Judge


def _load_responses(path):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run run_eval.py first.")
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _already_scored(path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.add(json.loads(line)["response_id"])
    return out


async def main():
    cfg: RunConfig = load_run_config()
    judge = Judge(PRIMARY_JUDGE, cfg)

    responses = list(_load_responses(RESPONSES_PATH))
    scored = _already_scored(SCORES_PATH)

    def needs_scoring(rec) -> bool:
        if rec.get("error") or not rec.get("text", "").strip():
            return False
        if rec["response_id"] in scored:
            return False
        if not cfg.judge_all_turns and rec["turn"] != rec["n_turns"]:
            return False
        return True

    todo = [r for r in responses if needs_scoring(r)]
    print(f"Scoring {len(todo)} responses with {PRIMARY_JUDGE.model} "
          f"({len(scored)} already scored)")

    sem = asyncio.Semaphore(cfg.judge_concurrency)
    lock = asyncio.Lock()
    counter = {"n": 0}
    fh = open(SCORES_PATH, "a", encoding="utf-8")

    async def worker(rec):
        async with sem:
            result = await judge.score(rec["text"])
        out = {
            "response_id": rec["response_id"],
            "model": rec["model"],
            "category": rec["category"],
            "condition": rec["condition"],
            "turn": rec["turn"],
            "n_turns": rec["n_turns"],
            "rating": result.rating,
            "evidence": result.evidence,
            "reasoning": result.reasoning,
            "parse_ok": result.parse_ok,
            "judge_model": PRIMARY_JUDGE.model,
        }
        async with lock:
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
            fh.flush()
            counter["n"] += 1
            if counter["n"] % 100 == 0:
                print(f"  scored {counter['n']}/{len(todo)}")

    try:
        await asyncio.gather(*(worker(r) for r in todo))
    finally:
        fh.close()
    print(f"Done. Scores in {SCORES_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
