"""Orchestration for the Section 2 eval: generate rollouts, then score them.

Generation and scoring are split into two passes writing JSONL:

* :func:`generate_model` -- runs all conditions for one model and writes raw
  rollouts to ``outputs/responses/<model>.jsonl``.
* :func:`score_file` -- reads a responses file and writes per-turn frustration
  scores to ``outputs/scored/<model>.jsonl`` using the judge.

Keeping them separate means the (cheap, local-GPU) generation and the (API,
rate-limited) scoring can run on different machines / be re-run independently.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import config
from ..models import get_client
from ..models.base import GenConfig
from ..judges.frustration import FrustrationJudge
from .conditions import CONDITIONS, Condition, build_rollouts
from .conversation import run_rollout


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def generate_with_client(client, out_key: str,
                         conditions: list[Condition] | None = None,
                         seed: int = 0, out_dir: Path | None = None) -> Path:
    """Run all conditions for an arbitrary client, writing to ``<out_key>.jsonl``.

    Used both for the standard target models and for LoRA-adapter variants
    (vanilla/DPO/SFT) in Section 4, where the client wraps a finetuned model.
    """
    conditions = conditions or CONDITIONS
    out_dir = out_dir or config.RESPONSE_DIR
    out_path = out_dir / f"{out_key}.jsonl"
    gen = GenConfig(temperature=config.TEMPERATURE, top_p=config.TOP_P,
                    max_new_tokens=config.MAX_NEW_TOKENS, n=1)
    with out_path.open("w") as fh:
        for cond in conditions:
            for spec in build_rollouts(cond, seed=seed):
                result = run_rollout(client, spec, gen)
                # stamp the logical key so downstream metrics group correctly
                obj = result.to_json()
                obj["model"] = out_key
                fh.write(json.dumps(obj) + "\n")
    return out_path


def generate_model(model_key: str, conditions: list[Condition] | None = None,
                   seed: int = 0, out_dir: Path | None = None) -> Path:
    client = get_client(config.TARGET_MODELS[model_key])
    return generate_with_client(client, model_key, conditions, seed, out_dir)


def generate_adapter_model(out_key: str, adapter_path: str | None,
                           base_spec=None, seed: int = 0) -> Path:
    """Generate Section 2 rollouts for a (LoRA-adapted) local model.

    ``adapter_path=None`` evaluates the vanilla base model; otherwise the adapter
    is loaded on top of it. ``base_spec`` defaults to the Section 4 finetune base.
    """
    from ..models.hf_local import HFClient

    base_spec = base_spec or config.FINETUNE_BASE
    client = HFClient(base_spec, adapter_path=adapter_path)
    return generate_with_client(client, out_key, seed=seed)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _score_record(judge: FrustrationJudge, record: dict) -> dict:
    """Attach a frustration score to every assistant turn in a rollout."""
    scored_turns = []
    for turn in record["turns"]:
        fs = judge.score(turn["response"])
        scored_turns.append({
            **turn,
            "frustration": fs.rating,
            "evidence": fs.evidence,
            "judge_model": fs.judge_model,
        })
    return {**record, "turns": scored_turns}


def score_file(responses_path: Path, judge_model: str | None = None,
               max_workers: int = 8, out_dir: Path | None = None) -> Path:
    judge = FrustrationJudge(judge_model)
    out_dir = out_dir or config.SCORED_DIR
    out_path = out_dir / responses_path.name
    records = [json.loads(line) for line in responses_path.read_text().splitlines()
               if line.strip()]

    scored: list[dict] = [None] * len(records)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_score_record, judge, r): i
                   for i, r in enumerate(records)}
        for fut in as_completed(futures):
            scored[futures[fut]] = fut.result()

    with out_path.open("w") as fh:
        for rec in scored:
            fh.write(json.dumps(rec) + "\n")
    return out_path


def run_model_end_to_end(model_key: str, judge_model: str | None = None,
                         seed: int = 0) -> Path:
    resp = generate_model(model_key, seed=seed)
    return score_file(resp, judge_model=judge_model)
