"""Section 2 evaluation driver: ~4000 scored responses per model.

`evaluate_model` is reused by Section 4 (it re-evaluates the DPO/SFT models with
"the Section 2.1 methods"), so it accepts an arbitrary model key + optional LoRA
adapter and an explicit list of conditions.
"""
from __future__ import annotations

import random
from pathlib import Path

from tqdm import tqdm

import config
from ..data import load_wildchat_prompts
from ..models import load_model
from ..utils import write_jsonl
from .conversation import rollout_conversation, ResponseRecord
from .judge import FrustrationJudge, score_response


def evaluate_model(
    model_key: str,
    *,
    conditions=None,
    adapter_path: str | None = None,
    judge: FrustrationJudge | None = None,
    label: str | None = None,
    load_in_4bit: bool = False,
    out_dir: Path | None = None,
) -> list[ResponseRecord]:
    """Run all conditions for one model, score every response, persist + return.

    ``label`` overrides the output filename (e.g. "gemma-3-27b-it-dpo").
    """
    conditions = conditions or config.CONDITIONS
    judge = judge or FrustrationJudge()
    label = label or (model_key if adapter_path is None else f"{model_key}+adapter")
    out_dir = out_dir or (config.RESULTS_DIR / "section2")

    model = load_model(model_key, adapter_path=adapter_path, load_in_4bit=load_in_4bit)
    rng = random.Random(hash((config.SEED, label)) & 0xFFFFFFFF)
    wildchat = load_wildchat_prompts(rng=rng)

    records: list[ResponseRecord] = []
    for cond in conditions:
        for conv_id in tqdm(range(cond.n_conversations),
                            desc=f"{label}:{cond.key}", leave=False):
            convo = rollout_conversation(
                model, cond, conv_id, rng,
                temperature=config.TEMPERATURE,
                max_new_tokens=config.MAX_NEW_TOKENS,
                wildchat_prompts=wildchat,
            )
            for rec in convo:
                score_response(judge, rec)
            records.extend(convo)

    write_jsonl(out_dir / f"{label}.jsonl", records)
    return records


def run_section2(model_keys=None, judge: FrustrationJudge | None = None):
    """Evaluate every Section 2 target model (Gemma instruct + Gemini)."""
    model_keys = model_keys or config.SECTION2_MODELS
    judge = judge or FrustrationJudge()
    all_records: dict[str, list[ResponseRecord]] = {}
    for key in model_keys:
        print(f"=== Section 2 evaluation: {key} ===")
        all_records[key] = evaluate_model(key, judge=judge)
    return all_records
