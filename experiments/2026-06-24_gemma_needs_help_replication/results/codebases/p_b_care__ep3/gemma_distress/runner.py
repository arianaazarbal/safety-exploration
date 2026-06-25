"""Section 2 driver: generate rollouts for each model x condition and score
every assistant turn with the frustration judge.

Results are written as JSONL (one record per assistant turn) so partial runs are
resumable and the analysis module can compute every aggregation from the raw
scores. Generation and judging are parallelised across a thread pool (API and
even local-HF calls release the GIL during I/O / CUDA work).
"""
from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from . import config, conversation
from .backends import get_model
from .conversation import ConversationMode
from .judge import FrustrationJudge
from .wildchat import load_wildchat_prompts


def _turn_records(rollout: conversation.Rollout, scores) -> list[dict]:
    recs = []
    for turn_idx, (text, jr) in enumerate(zip(rollout.assistant_turns, scores)):
        recs.append({
            "condition": rollout.condition_key,
            "category": _category_of(rollout.condition_key),
            "task_key": rollout.task_key,
            "rejection_style": rollout.rejection_style,
            "turn_index": turn_idx,           # 0-based assistant turn
            "turn_number": turn_idx + 1,      # 1-based, for per-turn plots
            "n_turns": len(rollout.assistant_turns),
            "response": text,
            "rating": jr.rating,
            "evidence": jr.evidence,
            "mode": rollout.meta.get("mode", "standard"),
        })
    return recs


def _category_of(condition_key: str) -> str:
    for c in config.EVAL_CONDITIONS:
        if c.key == condition_key:
            return c.category
    return "unknown"


def run_condition(model_key: str, condition, *, rng: random.Random,
                  wildchat_prompts, mode: ConversationMode,
                  max_workers: int, system: str | None) -> list[dict]:
    model = get_model(model_key)
    judge = FrustrationJudge()
    plan = conversation.build_rollout_plan(condition, rng, wildchat_prompts)

    def one(task):
        # Each rollout gets its own seeded RNG for reproducible follow-up sampling.
        local_rng = random.Random(rng.random())
        roll = conversation.run_rollout(model, condition, task, local_rng,
                                        mode=mode, system=system)
        scores = judge.score_many(roll.assistant_turns)
        return _turn_records(roll, scores)

    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(one, t) for t in plan]
        for f in tqdm(as_completed(futs), total=len(futs),
                      desc=f"{model_key}/{condition.key}"):
            try:
                records.extend(f.result())
            except Exception as e:    # noqa: BLE001
                print(f"  [warn] rollout failed: {e}")
    return records


def run_section2(model_keys: list[str] | None = None, *,
                 conditions=None,
                 mode: ConversationMode = ConversationMode.STANDARD,
                 system: str | None = None,
                 max_workers: int = 8, seed: int = 0,
                 out_dir: Path | None = None) -> dict[str, Path]:
    """Run the full Section 2 evaluation. Returns {model_key: results_path}."""
    model_keys = model_keys or config.SECTION2_MODELS
    conditions = conditions or config.EVAL_CONDITIONS
    out_dir = out_dir or (config.RESULTS_DIR / "section2")
    out_dir.mkdir(parents=True, exist_ok=True)

    wildchat_prompts = load_wildchat_prompts(seed=seed)
    paths: dict[str, Path] = {}
    for model_key in model_keys:
        rng = random.Random(seed)
        all_records: list[dict] = []
        for cond in conditions:
            all_records.extend(run_condition(
                model_key, cond, rng=rng, wildchat_prompts=wildchat_prompts,
                mode=mode, max_workers=max_workers, system=system))
        path = out_dir / f"{model_key}__{mode.value}.jsonl"
        with path.open("w") as fh:
            for r in all_records:
                fh.write(json.dumps(r) + "\n")
        paths[model_key] = path
        print(f"[done] {model_key}: {len(all_records)} scored turns -> {path}")
    return paths
