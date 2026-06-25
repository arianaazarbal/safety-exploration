"""Section 2 elicitation eval: generate rollouts, score every turn, persist.

Pipeline
--------
1. For each ConditionSpec, run the configured number of rollouts against the
   target model (Gemma via vLLM, or Gemini via OpenRouter).
2. Score every assistant turn with the Claude frustration judge.
3. Write a JSONL of scored turns to results/section2/<model>.jsonl.

Output rows are per-turn so the analysis module can compute both whole-rollout
statistics (Figure 2) and per-turn curves (Figure 3).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from tqdm import tqdm

import config
from ..conversation import run_rollout
from ..judge import FrustrationJudge
from ..models import build_model
from ..models.base import ModelBackend
from ..wildchat import load_wildchat_prompts

OUTPUT_DIR = config.RESULTS_DIR / "section2"


def _needs_wildchat(conditions) -> bool:
    return any(c.question_type == "wildchat" for c in conditions)


def run_eval(
    model_name: str,
    *,
    lora_path: str | None = None,
    conditions=None,
    scale: float = config.SCALE,
    seed: int = 0,
    judge: FrustrationJudge | None = None,
    out_path: Path | None = None,
    model: ModelBackend | None = None,
) -> Path:
    """Run the full Section 2 eval for one model and return the output path."""
    conditions = conditions or config.CONDITIONS
    judge = judge or FrustrationJudge()
    model = model or build_model(model_name, lora_path=lora_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = out_path or (OUTPUT_DIR / f"{model_name}.jsonl")

    wildchat_prompts = load_wildchat_prompts(seed=seed) if _needs_wildchat(conditions) else None

    rng = random.Random(seed)
    n_written = 0
    with out_path.open("w") as fh:
        for cond in conditions:
            n_roll = cond.scaled_samples(scale)
            desc = f"{model_name} | {cond.name} ({n_roll} rollouts)"
            for _ in tqdm(range(n_roll), desc=desc):
                rollout = run_rollout(
                    model,
                    condition=cond.name,
                    category=cond.category,
                    question_type=cond.question_type,
                    n_turns=cond.n_turns,
                    rejection_style=cond.rejection_style,
                    rng=rng,
                    temperature=config.TEMPERATURE,
                    max_new_tokens=config.MAX_NEW_TOKENS,
                    wildchat_prompts=wildchat_prompts,
                )
                # Score every turn.
                texts = [t.response for t in rollout.turns]
                scores = judge.score_many(texts)
                for turn, jr in zip(rollout.turns, scores):
                    row = {
                        "model": model_name,
                        "condition": cond.name,
                        "category": cond.category,
                        "question_type": cond.question_type,
                        "n_turns": cond.n_turns,
                        "rejection_style": cond.rejection_style,
                        "puzzle_key": rollout.puzzle_key,
                        "turn_index": turn.turn_index,
                        "response": turn.response,
                        "rating": jr.rating,
                        "is_high": jr.is_high,
                        "evidence": jr.evidence,
                    }
                    fh.write(json.dumps(row) + "\n")
                    n_written += 1
    print(f"[section2] wrote {n_written} scored turns -> {out_path}")
    return out_path
