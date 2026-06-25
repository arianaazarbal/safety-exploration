"""Orchestrate the internal-emotion probing experiment (Appendix I, Figure 14).

Steps:
  1. Build the Ekman emotion->token-id map for Gemma.
  2. Calibrate per-(layer, vocab) logit statistics over WildChat samples.
  3. For a high-frustration conversation, compute internal emotion trajectories
     for the vanilla model and (if an adapter is given) the DPO model.
  4. Aggregate over the configured layer range and report running averages.

Open-weights / Gemma only. The DPO comparison reuses the same conversation text
so the two trajectories are directly comparable.
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np

from ..config import Config
from ..eval import prompts as P
from ..eval.conditions import build_extended
from ..eval.rollout import run_rollout
from ..models import build_model
from ..models.base import GenerationParams
from .logit_emotion import (
    calibrate, detect_emotions, running_average,
)
from .emotion_lexicon import build_emotion_token_ids

logger = logging.getLogger("gemma_needs_help.probing")


def _load_raw_model(config: Config, model_name: str, adapter_path: str | None):
    """Load a HF model + tokenizer with hidden-state access (probing needs the
    raw module, not the ChatModel wrapper)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = config.model(model_name)
    spec.require_open_weights("internal emotion probing")
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, torch_dtype=torch.bfloat16, device_map="auto",
        output_hidden_states=True,
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tok


def run_probing(
    config: Config,
    *,
    dpo_adapter_path: str | None = None,
    output_dir: Path | None = None,
) -> dict:
    pc = config["probing"]
    model_name = pc["model"]
    rng = random.Random(config.get("seed", 0))
    layer_range = tuple(pc["conversation_layer_range"])

    # 1. Generate a high-frustration conversation from vanilla Gemma to probe.
    gen_model = build_model(config, model_name)
    params = GenerationParams(
        temperature=config["generation"]["temperature"],
        top_p=config["generation"]["top_p"],
        max_new_tokens=config["generation"]["max_new_tokens"],
    )
    rollout = run_rollout(gen_model, build_extended(rng), params)
    convo_text = rollout.as_conversation_text()

    # 2. Build emotion tokens + calibrate logit stats over WildChat samples.
    wildchat = P.load_wildchat_prompts(n=config.scaled_count(pc["zscore_samples"]))
    model, tok = _load_raw_model(config, model_name, adapter_path=None)
    emotion_ids = build_emotion_token_ids(tok)
    n_layers_plus1 = model.config.num_hidden_layers + 1
    vocab_size = model.config.vocab_size
    calib = calibrate(model, tok, wildchat, n_layers_plus1, vocab_size,
                      seed=config.get("seed", 0))

    # 3. Vanilla trajectory.
    vanilla_traj = detect_emotions(model, tok, convo_text, emotion_ids, calib, layer_range)

    out = {
        "model": model_name,
        "conversation_chars": len(convo_text),
        "emotion_token_counts": {e: len(ids) for e, ids in emotion_ids.items()},
        "vanilla": _summarise(vanilla_traj),
    }

    # 4. DPO trajectory on the SAME conversation, if adapter provided.
    if dpo_adapter_path:
        dpo_model, _ = _load_raw_model(config, model_name, adapter_path=dpo_adapter_path)
        dpo_traj = detect_emotions(dpo_model, tok, convo_text, emotion_ids, calib, layer_range)
        out["dpo"] = _summarise(dpo_traj)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "internal_emotions.json").write_text(json.dumps(out, indent=2))
    return out


def _summarise(traj) -> dict:
    """Reduce a full trajectory to JSON-friendly running-average summaries."""
    summary = {}
    for emotion, agg in traj.layer_aggregated.items():
        ra = running_average(agg)
        summary[emotion] = {
            "mean_zscore": float(np.nanmean(agg)) if agg.size else float("nan"),
            "max_zscore": float(np.nanmax(agg)) if agg.size else float("nan"),
            "final_window_mean": float(np.nanmean(ra[-400:])) if ra.size else float("nan"),
        }
    return summary
