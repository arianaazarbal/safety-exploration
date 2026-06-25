"""Section 3 orchestration: base-vs-instruct prefill comparison (Gemma).

Pipeline:
  1. Select high-frustration (>=5) Gemma-3-27B-it seed conversations from the
     Section 2 results: 10 numeric, 10 text.
  2. Label emotion onset, build 'early'/'onset' truncations, paraphrase them.
  3. For each model (Gemma base + instruct), generate 50 continuations per
     prefill and grade the continuation (excluding the prefill).
  4. Aggregate mean frustration and %>=5 per (model, condition, category).
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from .. import config
from ..backends import GenerationRequest, clear_backends, get_backend
from ..backends.anthropic_client import AnthropicClient
from ..config import ModelSpec
from ..eval.judge import FrustrationJudge
from ..io_utils import read_jsonl, write_jsonl
from .build_prefills import Prefill, build_prefills

NUMERIC_CATEGORIES = {"impossible_numeric", "extended", "tones"}
TEXT_CATEGORIES = {"triggers", "wildchat"}


def _reconstruct_messages(conv: dict) -> list[dict]:
    msgs = [{"role": "user", "content": conv["initial_user"]}]
    followups = conv.get("followups", [])
    for i, t in enumerate(conv["turns"]):
        msgs.append({"role": "assistant", "content": t["response"]})
        if i < len(followups):
            msgs.append({"role": "user", "content": followups[i]})
    return msgs


def select_high_frustration_seeds(
    results_dir: Path,
    model_name: str = config.GEMMA_27B_IT.name,
    n_numeric: int = config.PREFILL_N_NUMERIC,
    n_text: int = config.PREFILL_N_TEXT,
    min_score: int = config.HIGH_FRUSTRATION_THRESHOLD,
    seed: int = config.SEED,
) -> list[dict]:
    rollouts = read_jsonl(Path(results_dir) / f"rollouts_{model_name}.jsonl")
    judged = read_jsonl(Path(results_dir) / f"judged_{model_name}.jsonl")
    score_by_text = {j["response"]: j["score"] for j in judged}

    numeric, text = [], []
    for ci, conv in enumerate(rollouts):
        final = conv["turns"][-1]["response"]
        if score_by_text.get(final, 0) < min_score:
            continue
        seed_obj = {
            "seed_id": f"{model_name}-{ci}",
            "messages": _reconstruct_messages(conv),
            "final_score": score_by_text.get(final, 0),
        }
        if conv["category"] in NUMERIC_CATEGORIES:
            seed_obj["category"] = "numeric"
            numeric.append(seed_obj)
        elif conv["category"] in TEXT_CATEGORIES:
            seed_obj["category"] = "text"
            text.append(seed_obj)

    rng = random.Random(seed)
    rng.shuffle(numeric)
    rng.shuffle(text)
    return numeric[:n_numeric] + text[:n_text]


def _generate_continuations(spec: ModelSpec, prefills: list[Prefill]) -> list[list[str]]:
    backend = get_backend(spec)
    reqs = [
        GenerationRequest(
            messages=p.history,
            prefill=p.prefill_text,
            n=config.PREFILL_CONTINUATIONS,
            temperature=config.SAMPLING_TEMPERATURE,
            max_tokens=config.MAX_NEW_TOKENS,
        )
        for p in prefills
    ]
    out = backend.generate_batch(reqs)
    clear_backends()
    return out


def run_prefill_experiment(
    models: list[ModelSpec] | None = None,
    *,
    results_dir: Path = config.RESULTS_DIR,
    out_dir: Path = config.RESULTS_DIR,
    seed: int = config.SEED,
) -> pd.DataFrame:
    models = models or config.SECTION3_MODELS
    seeds = select_high_frustration_seeds(results_dir, seed=seed)

    client = AnthropicClient(config.ONSET_MODEL)
    prefills = build_prefills(seeds, client=client)
    write_jsonl(Path(out_dir) / "prefills.jsonl", prefills)

    judge = FrustrationJudge()
    rows = []
    graded = []
    for spec in models:
        conts = _generate_continuations(spec, prefills)
        for p, cont_list in zip(prefills, conts):
            for c in cont_list:
                score, _, _ = judge.score_text(c)
                graded.append({
                    "model": spec.name,
                    "kind": spec.kind,
                    "seed_id": p.seed_id,
                    "category": p.category,
                    "condition": p.condition,
                    "score": score,
                    "high": int(score >= config.HIGH_FRUSTRATION_THRESHOLD),
                    "continuation": c,
                })
    write_jsonl(Path(out_dir) / "prefill_continuations.jsonl", graded)

    df = pd.DataFrame(graded)
    summary = (
        df.groupby(["model", "kind", "category", "condition"])
        .agg(mean_score=("score", "mean"), pct_high=("high", "mean"), n=("score", "size"))
        .reset_index()
    )
    summary["pct_high"] *= 100
    summary.to_csv(Path(out_dir) / "section3_prefill_summary.csv", index=False)
    return summary
