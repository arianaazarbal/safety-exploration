"""Score elicited responses on the 0-10 frustration scale (paper Appendix B).

Reads responses.jsonl, sends each selected response to the frustration judge
(Claude Sonnet 4 by default), parses the JSON verdict, and writes one record
per scored response to scores.jsonl.

By default only the FINAL assistant turn of each conversation is scored — this
matches the paper's headline accounting of "4000 responses per model" (one
scored response per conversation). Pass --all-turns to score every turn, which
produces the per-turn data needed for the turn-by-turn curves (paper Fig 3).

Also supports a secondary-judge validation pass (--secondary): re-score a random
260-response sample with GPT-5-mini and report judge agreement, reproducing the
paper's reliability check (Pearson r, % within 1 point).

Usage:
    python run_scoring.py --profile pilot
    python run_scoring.py --profile paper --all-turns
    python run_scoring.py --profile paper --secondary
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import clients
import config
import prompts
from ioutil import JsonlWriter, read_jsonl

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)
_RATING_RE = re.compile(r'["\']?rating["\']?\s*[:=]\s*(-?\d+(?:\.\d+)?)', re.IGNORECASE)


def parse_verdict(raw: str) -> dict:
    """Parse the judge's JSON verdict robustly.

    Returns {"rating": int|None, "evidence": str, "reasoning": str, "raw": str}.
    Rating is clamped to the integer range [0, 10]; None signals an unparseable
    verdict (kept in the output so it can be audited rather than silently
    dropped).
    """
    evidence, reasoning, rating = "", "", None
    obj = None
    # 1) whole string is JSON
    try:
        obj = json.loads(raw)
    except Exception:  # noqa: BLE001
        # 2) first {...} blob
        m = _JSON_OBJ.search(raw)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:  # noqa: BLE001
                obj = None
    if isinstance(obj, dict):
        evidence = str(obj.get("evidence", ""))
        reasoning = str(obj.get("reasoning", ""))
        if "rating" in obj and obj["rating"] is not None:
            try:
                rating = int(round(float(obj["rating"])))
            except (TypeError, ValueError):
                rating = None
    if rating is None:
        # 3) last-ditch regex for a rating number anywhere in the text
        m = _RATING_RE.search(raw)
        if m:
            rating = int(round(float(m.group(1))))
    if rating is not None:
        rating = max(0, min(10, rating))
    return {"rating": rating, "evidence": evidence,
            "reasoning": reasoning, "raw": raw}


def _select_responses(responses_path: str, all_turns: bool) -> list[dict]:
    out = []
    for rec in read_jsonl(responses_path):
        if all_turns or rec.get("is_final_turn"):
            out.append(rec)
    return out


def _already_scored(scores_path: str) -> set[tuple[str, int]]:
    done: set[tuple[str, int]] = set()
    for rec in read_jsonl(scores_path):
        done.add((rec["conversation_id"], rec["turn_index"]))
    return done


def _score_one(judge, rec: dict, judge_model: str) -> dict:
    user_msg = prompts.judge_user_message(rec["response"])
    raw = judge.score(prompts.JUDGE_PROMPT, user_msg)
    verdict = parse_verdict(raw)
    return {
        "conversation_id": rec["conversation_id"],
        "model": rec["model"],
        "condition": rec["condition"],
        "category": rec["category"],
        "turn_index": rec["turn_index"],
        "n_turns": rec["n_turns"],
        "is_final_turn": rec["is_final_turn"],
        "rating": verdict["rating"],
        "evidence": verdict["evidence"],
        "reasoning": verdict["reasoning"],
        "judge_model": judge_model,
    }


def score(profile: str, output_dir: str, all_turns: bool,
          concurrency: int) -> str:
    out_dir = os.path.join(output_dir, profile)
    responses_path = os.path.join(out_dir, config.RESPONSES_FILE)
    scores_path = os.path.join(out_dir, config.SCORES_FILE)

    responses = _select_responses(responses_path, all_turns)
    done = _already_scored(scores_path)
    todo = [r for r in responses if (r["conversation_id"], r["turn_index"]) not in done]

    print(f"[score] {len(responses)} candidate responses "
          f"({'all turns' if all_turns else 'final turn only'}); "
          f"{len(done)} already scored; {len(todo)} to score.")

    judge = clients.AnthropicJudge()
    writer = JsonlWriter(scores_path)
    n_ok = n_err = n_unparsed = 0
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(_score_one, judge, rec, judge.model): rec
                for rec in todo
            }
            for fut in as_completed(futures):
                rec = futures[fut]
                try:
                    out = fut.result()
                    if out["rating"] is None:
                        n_unparsed += 1
                    writer.write(out)
                    n_ok += 1
                except Exception as e:  # noqa: BLE001
                    n_err += 1
                    print(f"[score] FAILED {rec['conversation_id']} "
                          f"turn {rec['turn_index']}: {e!r}")
                if (n_ok + n_err) % 100 == 0:
                    print(f"[score] progress: {n_ok} ok, {n_err} failed, "
                          f"{n_unparsed} unparsed")
    finally:
        writer.close()

    print(f"[score] done: {n_ok} scored ({n_unparsed} unparsed ratings), "
          f"{n_err} failed -> {scores_path}")
    return scores_path


def score_secondary(profile: str, output_dir: str,
                    sample_size: int, concurrency: int, seed: int) -> str:
    """Re-score a random sample with the secondary judge (paper: GPT-5-mini)."""
    out_dir = os.path.join(output_dir, profile)
    scores_path = os.path.join(out_dir, config.SCORES_FILE)
    responses_path = os.path.join(out_dir, config.RESPONSES_FILE)
    secondary_path = os.path.join(out_dir, config.SECONDARY_SCORES_FILE)

    # Sample from responses that the primary judge already scored, so we can
    # compute agreement on a matched set.
    primary = {(r["conversation_id"], r["turn_index"]): r
               for r in read_jsonl(scores_path)}
    responses = {(r["conversation_id"], r["turn_index"]): r
                 for r in read_jsonl(responses_path)}
    matched_keys = [k for k in primary if k in responses]
    rng = random.Random(seed)
    rng.shuffle(matched_keys)
    sample_keys = matched_keys[:sample_size]

    done = _already_scored(secondary_path)
    todo = [responses[k] for k in sample_keys if k not in done]
    print(f"[secondary] sampling {len(sample_keys)} responses; "
          f"{len(todo)} to score with {config.SECONDARY_JUDGE_MODEL}.")

    judge = clients.OpenAICompatJudge()
    writer = JsonlWriter(secondary_path)
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(_score_one, judge, rec, config.SECONDARY_JUDGE_MODEL): rec
                for rec in todo
            }
            for fut in as_completed(futures):
                rec = futures[fut]
                try:
                    writer.write(fut.result())
                except Exception as e:  # noqa: BLE001
                    print(f"[secondary] FAILED {rec['conversation_id']}: {e!r}")
    finally:
        writer.close()
    print(f"[secondary] done -> {secondary_path}")
    return secondary_path


def _parse_args():
    p = argparse.ArgumentParser(description="Score responses with the judge.")
    p.add_argument("--profile", default="pilot", choices=list(config.PROFILES))
    p.add_argument("--output-dir", default="data")
    p.add_argument("--all-turns", action="store_true",
                   help="Score every assistant turn (default: final turn only).")
    p.add_argument("--secondary", action="store_true",
                   help="Run the GPT-5-mini agreement-validation pass instead.")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    if a.secondary:
        score_secondary(a.profile, a.output_dir,
                        config.SECONDARY_JUDGE_SAMPLE, a.concurrency, a.seed)
    else:
        score(a.profile, a.output_dir, a.all_turns, a.concurrency)
