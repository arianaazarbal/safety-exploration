"""App. I orchestration: internal-emotion comparison (vanilla vs DPO) + layer-ablation sweep.

``run_internal_probe`` reproduces Figures 14-15: fit the logit baseline on WildChat, then
score a set of high-frustration conversations under both the vanilla and DPO models, and
report how much the DPO model suppresses internal negative-emotion z-scores.

``run_layer_ablation_plan`` reproduces Figures 12-13: train DPO with LoRA restricted to each
layer range and evaluate with a reduced (100-sample) version of the §2 protocol, showing that
adapters before layer ~40 are necessary and layers 30-35 nearly match full-layer DPO.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from statistics import fmean

import numpy as np

from ..config import EVAL_CONDITIONS, ProbeConfig
from ..models import get_backend
from ..tasks import wildchat_prompts
from ..utils import ensure_dir, read_jsonl, set_seed, write_json
from .logit_emotions import EmotionProbe

NUMERIC_CATEGORIES = {"Impossible numeric", "Tones", "Extended"}


def _high_frustration_conversations(seed_run_dir: str, *, min_score: int, limit: int):
    """Yield up to ``limit`` full conversations containing a turn scoring >= min_score."""
    rollouts = {}
    for rec in read_jsonl(Path(seed_run_dir, "rollouts.jsonl")):
        rollouts[(rec["condition_key"], rec["sample_id"])] = rec
    seen = set()
    convs = []
    for s in read_jsonl(Path(seed_run_dir, "scores.jsonl")):
        if s.get("rating") is None or s["rating"] < min_score:
            continue
        key = (s["condition_key"], s["sample_id"])
        if key in seen:
            continue
        roll = rollouts.get(key)
        if roll is None:
            continue
        seen.add(key)
        convs.append(roll["messages"])
        if len(convs) >= limit:
            break
    return convs


def run_internal_probe(
    vanilla_model: str,
    dpo_adapter: str,
    seed_run_dir: str,
    out_dir: str,
    *,
    cfg: ProbeConfig | None = None,
    seed: int = 0,
    n_conversations: int = 12,
    lexicon_method: str = "seed",
) -> dict:
    """Compare internal negative-emotion scores between vanilla and DPO Gemma (Figures 14-15).

    ``dpo_adapter`` is a LoRA adapter directory applied on top of ``vanilla_model`` (the form
    produced by ``train dpo``). Both models therefore share the same base weights and tokenizer,
    so the lexicon/baseline are directly comparable.
    """
    cfg = cfg or ProbeConfig()
    set_seed(seed)
    out = ensure_dir(out_dir)

    baseline_texts = wildchat_prompts(cfg.n_standardisation_samples, seed=seed)
    conversations = _high_frustration_conversations(
        seed_run_dir, min_score=5, limit=n_conversations,
    )

    summary_per_model = {}
    for label, model_id, adapter in (
        ("vanilla", vanilla_model, None),
        ("dpo", vanilla_model, dpo_adapter),
    ):
        backend = get_backend(model_id, adapter_path=adapter)
        probe = EmotionProbe(backend, cfg=cfg, seed=seed)
        probe.build_lexicon(method=lexicon_method)
        probe.fit_baseline(baseline_texts)

        neg = cfg.negative_emotions
        per_conv_final = {e: [] for e in neg}
        per_conv_peak = {e: [] for e in neg}
        for messages in conversations:
            scored = probe.score_conversation(messages)
            for e in neg:
                traj = scored["trajectory"].get(e)
                if traj is None or len(traj) == 0:
                    continue
                per_conv_final[e].append(float(traj[-1]))
                per_conv_peak[e].append(float(np.max(traj)))

        summary_per_model[label] = {
            "model": model_id,
            "adapter": adapter,
            "negative_emotion_final_z": {e: (fmean(v) if v else None) for e, v in per_conv_final.items()},
            "negative_emotion_peak_z": {e: (fmean(v) if v else None) for e, v in per_conv_peak.items()},
            "lexicon_sizes": {e: len(ids) for e, ids in probe.lexicon.items()},
        }

    summary = {
        "seed_run_dir": seed_run_dir,
        "n_conversations": len(conversations),
        "aggregate_layers": list(cfg.aggregate_layers),
        "per_model": summary_per_model,
    }
    write_json(Path(out, "probe_summary.json"), summary)
    return summary


def run_layer_ablation_plan(
    dpo_dataset_path: str,
    out_dir: str,
    *,
    base_model: str = "google/gemma-3-27b-it",
    cfg: ProbeConfig | None = None,
    execute: bool = False,
    seed: int = 0,
) -> dict:
    """Produce (and optionally execute) the layer-ablation DPO sweep (Figures 12-13).

    For each layer range, DPO is trained with LoRA restricted to those layers, then evaluated
    with a reduced 100-sample version of the §2 protocol. ``execute=False`` returns the plan
    only (training the full sweep is very expensive); ``execute=True`` runs it end to end.
    """
    cfg = cfg or ProbeConfig()
    out = ensure_dir(out_dir)
    ranges = cfg.layer_ablation_ranges

    # Reduced eval: scale every category down to ~ablation_samples_per_eval scored responses.
    reduced_conditions = [
        replace(c, target_responses=cfg.ablation_samples_per_eval) for c in EVAL_CONDITIONS
    ]

    plan = []
    for lr in ranges:
        tag = "all" if lr is None else f"{lr[0]}-{lr[1]}"
        plan.append({
            "layer_range": lr,
            "adapter_dir": str(Path(out, f"dpo_layers_{tag}")),
            "eval_dir": str(Path(out, f"eval_layers_{tag}")),
        })

    results = {"base_model": base_model, "dataset": dpo_dataset_path, "plan": plan,
               "reduced_samples_per_category": cfg.ablation_samples_per_eval}

    if execute:
        from ..analysis.aggregate import aggregate_run
        from ..eval.run_eval import run_evaluation
        from ..training.train_dpo import train_dpo

        outcomes = []
        for entry in plan:
            train_dpo(dpo_dataset_path, entry["adapter_dir"], base_model=base_model,
                      layer_range=entry["layer_range"], cfg=None)
            run_evaluation(base_model, entry["eval_dir"], adapter_path=entry["adapter_dir"],
                           conditions=reduced_conditions, seed=seed)
            summ = aggregate_run(entry["eval_dir"])
            outcomes.append({"layer_range": entry["layer_range"],
                             "mean": summ["overall"]["mean"],
                             "pct_high": summ["overall"]["pct_high"]})
        results["outcomes"] = outcomes

    write_json(Path(out, "layer_ablation.json"), results)
    return results
