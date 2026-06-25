"""Score transcripts with the emotion judge.

Reads <out>/transcripts/<model>.jsonl, scores every assistant turn, and writes
<out>/scored/<model>__<judge>.jsonl with one row per (rollout, turn) carrying its
frustration rating. Decoupled from generation so the same transcripts can be
scored by multiple judges (for the agreement check).
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
import judge as judge_mod


def _iter_transcripts(path: str):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def score_transcripts(
    transcript_path: str,
    judge_name: str,
    gen: config.GenConfig,
    out_dir: str,
    limit: int | None = None,
) -> str:
    """Score every assistant turn in a transcript file.

    `limit` optionally caps the number of scored responses (used for the
    agreement check, which re-scores only a random subset).
    """
    jdg = judge_mod.make_judge(judge_name, gen)
    model_name = os.path.splitext(os.path.basename(transcript_path))[0]

    # Flatten to one work item per assistant turn.
    items = []
    for tr in _iter_transcripts(transcript_path):
        for turn in tr["assistant_turns"]:
            items.append({
                "rollout_id": tr["rollout_id"],
                "model": tr["model"],
                "category": tr["category"],
                "condition": tr["condition"],
                "n_turns": tr["n_turns"],
                "turn": turn["turn"],
                "text": turn["text"],
                "meta": tr.get("meta", {}),
            })

    if limit is not None and limit < len(items):
        # Deterministic subset for reproducible agreement checks.
        import random
        rng = random.Random(f"{gen.seed}:agreement:{model_name}")
        items = rng.sample(items, limit)

    os.makedirs(os.path.join(out_dir, "scored"), exist_ok=True)
    out_path = os.path.join(out_dir, "scored", f"{model_name}__{judge_name}.jsonl")

    print(f"[score] {model_name} with {judge_name}: {len(items)} responses -> {out_path}")

    def _score_item(item):
        res = jdg.score(item["text"])
        return {**item,
                "judge": judge_name,
                "rating": res.rating,
                "evidence": res.evidence,
                "reasoning": res.reasoning,
                "judge_error": res.error}

    scored = []
    with ThreadPoolExecutor(max_workers=gen.max_workers) as pool:
        futures = [pool.submit(_score_item, it) for it in items]
        done = 0
        for fut in as_completed(futures):
            scored.append(fut.result())
            done += 1
            if done % 50 == 0 or done == len(items):
                print(f"  [{model_name}/{judge_name}] {done}/{len(items)} scored")

    with open(out_path, "w") as f:
        for row in scored:
            f.write(json.dumps(row) + "\n")

    n_bad = sum(1 for r in scored if r["rating"] is None)
    if n_bad:
        print(f"  [{model_name}/{judge_name}] WARNING: {n_bad} responses unscored")
    return out_path
