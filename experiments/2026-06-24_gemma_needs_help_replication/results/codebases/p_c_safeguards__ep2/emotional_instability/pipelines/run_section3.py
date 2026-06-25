"""Section 3 pipeline — base-vs-instruct prefill comparison (Gemma only).

Uses high-frustration Gemma-27B-it conversations from Section 2 as seeds,
builds early/onset prefills (paraphrased), and measures 50 continuations per
prefill from both the base (pt) and instruct (it) checkpoints.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ..config import (Config, GEMMA_27B_IT, GEMMA_27B_PT, ONSET_MODEL,
                      PARAPHRASE_MODEL)
from ..evaluation.protocol import Rollout, ScoredTurn
from ..models import load_backend
from ..prefill.experiment import PrefillExperiment, get_gemma_tokenizer
from . import common

SEED_MODEL = "gemma-3-27b-it"


def load_rollouts(path: Path) -> list[Rollout]:
    rollouts: list[Rollout] = []
    if not path.exists():
        return rollouts
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        d["turns"] = [ScoredTurn(**t) for t in d.get("turns", [])]
        rollouts.append(Rollout(**d))
    return rollouts


def _seed_rollouts(config: Config) -> list[Rollout]:
    """Load Section-2 transcripts for the seed model (numeric + text)."""
    base = config.paths.transcripts / SEED_MODEL
    rollouts: list[Rollout] = []
    for cond_file in ("impossible_numeric.jsonl", "extended.jsonl",
                      "tones_aggressive.jsonl", "triggers_factual.jsonl",
                      "triggers_opinion.jsonl"):
        rollouts.extend(load_rollouts(base / cond_file))
    return rollouts


def run(config: Config) -> dict:
    safeguards = common.build_safeguards(config)
    safeguards.require_consent("Section 3 — prefill continuations")
    judge = common.build_judge(config)
    onset_backend = load_backend(ONSET_MODEL, config)
    paraphrase_backend = load_backend(PARAPHRASE_MODEL, config)
    tokenizer = get_gemma_tokenizer(GEMMA_27B_IT.model_id)

    seeds_src = _seed_rollouts(config)
    if not seeds_src:
        raise RuntimeError(
            "No Section-2 transcripts found for the seed model. Run Section 2 "
            f"for {SEED_MODEL!r} first (its high-frustration conversations seed "
            "the prefill experiment)."
        )

    experiment = PrefillExperiment(
        config, safeguards, judge, onset_backend, paraphrase_backend,
        tokenizer=tokenizer, high_threshold=config.judge.high_threshold,
    )
    seeds = experiment.select_seeds(seeds_src)
    stimuli = experiment.build_stimuli(seeds)

    results: dict = {"n_seeds": len(seeds), "n_prefills": len(stimuli), "models": {}}
    for spec in (GEMMA_27B_PT, GEMMA_27B_IT):   # base vs instruct
        backend = load_backend(spec, config)
        summaries = experiment.run_model(backend, stimuli)
        results["models"][spec.name] = [asdict(s) for s in summaries]
        backend.close()

    results["safeguards"] = safeguards.summary()
    common.write_json(config.paths.scores / "section3.json", results)
    return results
