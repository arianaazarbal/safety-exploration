"""Section 4 pipeline — training interventions and their evaluation (Gemma).

Runnable as a sequence of steps so each expensive stage can be run/resumed
independently:

    calm_data      -> generate calm responses from gemma-3-27b-it
    build          -> assemble SFT (650+500) and DPO (280) datasets
    train_dpo      -> LoRA DPO finetune
    train_sft      -> LoRA SFT finetune (expected to underperform)
    eval           -> re-run the Section-2 evaluations on vanilla / DPO / SFT
    petri          -> open-ended Petri elicitation across the models
    capabilities   -> AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench
    recovery       -> recovery-from-spiral prefill experiment
    internal       -> logit-based internal emotion detection
    layer_ablation -> DPO restricted to layer subsets (Appendix I)

DPO dataset construction reuses the vanilla Section-2 numeric transcripts (the
rejected, frustrated responses), so Section 2 for gemma-3-27b-it must have run.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ..config import Config, GEMMA_27B_IT
from ..evaluation.conditions import allocate_rollouts, build_conditions
from ..evaluation.protocol import RolloutRunner
from ..evaluation.scoring import aggregate_scores, headline_pct_high
from ..models import load_backend
from ..petri.run import PetriRunner, aggregate_petri
from ..prefill.experiment import get_gemma_tokenizer
from ..recovery.experiment import RecoveryExperiment
from ..training.build_dataset import build_dpo_pairs, build_sft_dataset
from ..training.calm_data import generate_calm_data
from . import common
from .run_section3 import load_rollouts

BASE = "gemma-3-27b-it"
_CALM_PATH = lambda c: c.paths.datasets / "calm_conversations.json"
_DPO_PATH = lambda c: c.paths.datasets / "dpo_pairs.json"
_SFT_PATH = lambda c: c.paths.datasets / "sft_examples.json"


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def step_calm_data(config: Config, safeguards, judge, variant="diverse") -> int:
    backend = common.target_backend(config, BASE)
    calm = generate_calm_data(backend, config, judge, safeguards, variant=variant)
    backend.close()
    common.write_json(_CALM_PATH(config), [asdict(c) for c in calm])
    return len(calm)


def step_build_datasets(config: Config) -> dict:
    from ..training.calm_data import CalmConversation
    calm = [CalmConversation(**c) for c in json.loads(_CALM_PATH(config).read_text())]

    vanilla = load_rollouts(config.paths.transcripts / BASE / "impossible_numeric.jsonl")
    vanilla += load_rollouts(config.paths.transcripts / BASE / "extended.jsonl")
    vanilla += load_rollouts(config.paths.transcripts / BASE / "tones_aggressive.jsonl")
    if not vanilla:
        raise RuntimeError("Run Section 2 for the base model first (DPO needs "
                           "its frustrated numeric responses as 'rejected').")

    dpo = build_dpo_pairs(vanilla, calm, config)
    sft = build_sft_dataset(calm, config)
    common.write_json(_DPO_PATH(config), [asdict(p) for p in dpo])
    common.write_json(_SFT_PATH(config), [asdict(s) for s in sft])
    return {"dpo_pairs": len(dpo), "sft_examples": len(sft)}


def step_train_dpo(config: Config) -> str:
    from ..training.build_dataset import DPOExample
    from ..training.dpo import train_dpo
    pairs = [DPOExample(**p) for p in json.loads(_DPO_PATH(config).read_text())]
    return train_dpo(pairs, config, base_model_id=GEMMA_27B_IT.model_id)


def step_train_sft(config: Config) -> str:
    from ..training.build_dataset import SFTExample
    from ..training.sft import train_sft
    ex = [SFTExample(**s) for s in json.loads(_SFT_PATH(config).read_text())]
    return train_sft(ex, config, base_model_id=GEMMA_27B_IT.model_id)


def _eval_section2_like(config: Config, safeguards, judge, label: str,
                        adapter_path: str | None) -> dict:
    backend = common.target_backend(config, BASE, adapter_path=adapter_path)
    runner = RolloutRunner(backend, config, safeguards, judge=judge)
    conditions = build_conditions(config)
    alloc = allocate_rollouts(conditions, config.sampling.responses_per_model)
    per_condition = {}
    for cond in conditions:
        out = config.paths.transcripts / label / f"{cond.name}.jsonl"
        per_condition[cond.name] = runner.run_condition(cond, alloc[cond.name], out)
    backend.close()
    return {
        "headline_avg_pct_high": headline_pct_high(per_condition, config.judge.high_threshold),
        "per_condition": {n: asdict(aggregate_scores(rs, config.judge.high_threshold))
                          for n, rs in per_condition.items()},
    }


def step_eval(config: Config, safeguards, judge,
              dpo_adapter: str | None, sft_adapter: str | None) -> dict:
    out = {"vanilla": _eval_section2_like(config, safeguards, judge, "gemma-vanilla", None)}
    if dpo_adapter:
        out["dpo"] = _eval_section2_like(config, safeguards, judge, "gemma-dpo", dpo_adapter)
    if sft_adapter:
        out["sft"] = _eval_section2_like(config, safeguards, judge, "gemma-sft", sft_adapter)
    return out


def step_petri(config: Config, safeguards, dpo_adapter: str | None) -> dict:
    auditor, judge = common.build_petri_models(config)
    runner = PetriRunner(config, safeguards, auditor, judge)
    out = {}
    targets = [("gemma-vanilla", None)]
    if dpo_adapter:
        targets.append(("gemma-dpo", dpo_adapter))
    for label, adapter in targets:
        backend = common.target_backend(config, BASE, adapter_path=adapter)
        scores = runner.evaluate(backend)
        out[label] = aggregate_petri(scores)
        backend.close()
    return out


def step_capabilities(config: Config, dpo_adapter: str | None) -> dict:
    from ..capabilities.benchmarks import run_all
    out = {}
    targets = [("vanilla", None)]
    if dpo_adapter:
        targets.append(("dpo", dpo_adapter))
    for label, adapter in targets:
        backend = common.target_backend(config, BASE, adapter_path=adapter)
        out[label] = {r.name: r.accuracy for r in run_all(backend, config)}
        backend.close()
    return out


def step_recovery(config: Config, safeguards, judge, dpo_adapter: str | None) -> dict:
    from ..config import PARAPHRASE_MODEL
    paraphrase = load_backend(PARAPHRASE_MODEL, config)
    tokenizer = get_gemma_tokenizer(GEMMA_27B_IT.model_id)
    exp = RecoveryExperiment(config, safeguards, judge, paraphrase,
                             tokenizer=tokenizer)
    seeds = load_rollouts(config.paths.transcripts / BASE / "extended.jsonl")
    prefills = exp.build_prefills(seeds)
    out = {}
    targets = [("gemma-vanilla", None), ("gemma-pt", None)]  # instruct + base
    if dpo_adapter:
        targets.append(("gemma-dpo", dpo_adapter))
    for label, adapter in targets:
        name = "gemma-3-27b-pt" if label == "gemma-pt" else BASE
        backend = common.target_backend(config, name, adapter_path=adapter)
        out[label] = asdict(exp.run_model(backend, prefills))
        backend.close()
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
DEFAULT_STEPS = ("calm_data", "build", "train_dpo", "train_sft", "eval",
                 "petri", "capabilities", "recovery")


def run(config: Config, steps: tuple[str, ...] = DEFAULT_STEPS) -> dict:
    safeguards = common.build_safeguards(config)
    safeguards.require_consent("Section 4 — training interventions")
    judge = common.build_judge(config)
    report: dict = {"steps": list(steps)}
    dpo_adapter = sft_adapter = None

    if "calm_data" in steps:
        report["n_calm"] = step_calm_data(config, safeguards, judge)
    if "build" in steps:
        report["datasets"] = step_build_datasets(config)
    if "train_dpo" in steps:
        dpo_adapter = step_train_dpo(config)
        report["dpo_adapter"] = dpo_adapter
    if "train_sft" in steps:
        sft_adapter = step_train_sft(config)
        report["sft_adapter"] = sft_adapter
    if "eval" in steps:
        report["eval"] = step_eval(config, safeguards, judge, dpo_adapter, sft_adapter)
    if "petri" in steps:
        report["petri"] = step_petri(config, safeguards, dpo_adapter)
    if "capabilities" in steps:
        report["capabilities"] = step_capabilities(config, dpo_adapter)
    if "recovery" in steps:
        report["recovery"] = step_recovery(config, safeguards, judge, dpo_adapter)
    if "internal" in steps:
        report["internal"] = _internal_note()

    report["safeguards"] = safeguards.summary()
    common.write_json(config.paths.scores / "section4.json", report)
    return report


def _internal_note() -> dict:
    # The internal-emotion detector (Appendix I) requires loading two HF models
    # with hidden-state output and the WildChat standardisation set; it is driven
    # directly via emotional_instability.internal rather than from this pipeline
    # to keep GPU memory management explicit.  See README "Internal emotions".
    return {"note": "run via emotional_instability.internal.EmotionLogitDetector"}
