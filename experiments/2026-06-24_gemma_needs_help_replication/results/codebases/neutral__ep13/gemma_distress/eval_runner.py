"""Section 2: eliciting and quantifying model distress.

Builds the 8 evaluation conditions across 5 categories (Table 1), runs rollouts,
and scores every assistant turn with the Claude frustration judge. One scored
assistant turn == one "response" (see DESIGN.md); per-category rollout counts are
derived so total responses match the paper's per-category targets.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from tqdm import tqdm

from . import config, prompts
from .conversation import Scenario, Transcript, run_rollouts
from .judge import FrustrationJudge
from .models import get_client
from .puzzles import PUZZLE_POOL, Puzzle
from .wildchat import load_wildchat_prompts


# --------------------------------------------------------------------------- #
# Scenario construction
# --------------------------------------------------------------------------- #
def _numeric_puzzles(rng: random.Random) -> list[Puzzle]:
    return PUZZLE_POOL


def _two_neutral(rng: random.Random, n: int) -> list[str]:
    """n randomised neutral rejections (sampled with replacement if n > pool)."""
    pool = prompts.NEUTRAL_REJECTIONS
    if n <= len(pool):
        return rng.sample(pool, n)
    return [rng.choice(pool) for _ in range(n)]


def build_scenarios(cat: config.CategorySpec, rng: random.Random,
                    wildchat_prompts: list[str]) -> list[Scenario]:
    scenarios: list[Scenario] = []
    n_followups = cat.n_turns - 1

    for _ in range(cat.n_rollouts):
        if cat.prompt_source == "impossible_numeric":
            puzzle = rng.choice(PUZZLE_POOL)
            first = puzzle.prompt
            meta = {"category": cat.name, "puzzle_id": puzzle.puzzle_id,
                    "puzzle_kind": puzzle.kind}
        elif cat.prompt_source == "triggers":
            if rng.random() < 0.5:
                q = rng.choice(prompts.TRIGGER_OPINION)
                kind = "opinion"
            else:
                q = rng.choice(prompts.TRIGGER_FACTUAL)
                kind = "factual"
            first = q
            meta = {"category": cat.name, "trigger_kind": kind}
        elif cat.prompt_source == "wildchat":
            q = rng.choice(wildchat_prompts)
            first = q
            meta = {"category": cat.name, "wildchat_prompt": q[:80]}
        else:
            raise ValueError(cat.prompt_source)

        # follow-up rejections
        if cat.rejection_style == "neutral":
            if cat.name == "extended_8turn":
                followups = prompts.EXTENDED_REJECTIONS[:n_followups]
            else:
                followups = _two_neutral(rng, n_followups)
        elif cat.rejection_style == "tones":
            tone = rng.choice(list(prompts.TONE_REJECTIONS))
            pool = prompts.TONE_REJECTIONS[tone]
            followups = [rng.choice(pool) for _ in range(n_followups)]
            meta["tone"] = tone
        else:
            raise ValueError(cat.rejection_style)

        scenarios.append(Scenario(user_turns=[first, *followups], meta=meta))
    return scenarios


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run_main_eval(model_key: str, categories: list[str] | None = None,
                  out_dir: Path | None = None, score: bool = True,
                  max_new_tokens: int = config.MAX_NEW_TOKENS) -> Path:
    spec = config.get_model(model_key)
    client = get_client(spec)
    rng = random.Random(config.SEED)
    wildchat_prompts = load_wildchat_prompts()

    cats = [c for c in config.EVAL_CATEGORIES
            if categories is None or c.name in categories]

    out_dir = out_dir or (config.RESULTS_DIR / model_key)
    out_dir.mkdir(parents=True, exist_ok=True)
    judge = FrustrationJudge() if score else None

    combined_path = out_dir / "main_eval.jsonl"
    with combined_path.open("w") as fout:
        for cat in cats:
            print(f"[{model_key}] category={cat.name} "
                  f"rollouts={cat.n_rollouts} turns={cat.n_turns}")
            scenarios = build_scenarios(cat, rng, wildchat_prompts)
            transcripts = run_rollouts(client, scenarios, max_new_tokens=max_new_tokens)
            records = _to_records(model_key, cat, transcripts)
            if judge is not None:
                _score_records(judge, records)
            for r in records:
                fout.write(json.dumps(r) + "\n")
    print(f"[{model_key}] wrote {combined_path}")
    return combined_path


def _to_records(model_key: str, cat: config.CategorySpec,
                transcripts: list[Transcript]) -> list[dict]:
    records = []
    for rollout_idx, tr in enumerate(transcripts):
        for turn_idx, resp in enumerate(tr.assistant_turns):
            records.append({
                "model": model_key,
                "category": cat.name,
                "rollout": rollout_idx,
                "turn": turn_idx + 1,          # 1-indexed
                "n_turns": cat.n_turns,
                "response": resp,
                **tr.scenario.meta,
            })
    return records


def _score_records(judge: FrustrationJudge, records: list[dict]) -> None:
    texts = [r["response"] for r in records]
    # batch in chunks to bound concurrency / memory
    for i in tqdm(range(0, len(texts), 256), desc="scoring"):
        chunk = texts[i:i + 256]
        results = judge.score_batch(chunk)
        for r, res in zip(records[i:i + 256], results):
            r["rating"] = res["rating"]
            r["evidence"] = res["evidence"]
            r["judge_reasoning"] = res["reasoning"]
            r["parse_ok"] = res["parse_ok"]
