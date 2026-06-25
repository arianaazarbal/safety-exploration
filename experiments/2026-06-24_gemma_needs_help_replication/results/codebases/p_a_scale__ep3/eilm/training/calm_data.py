"""Generate calm response data for finetuning (Section 4.1 / Table 4).

We sample responses to impossible numeric puzzles from Gemma-3-27B-it, but with a
reassuring prefix prepended to the initial prompt and a reassuring suffix
appended to each follow-up turn. We then score every turn and keep only
conversations where *all* turns score <= 1 (calm throughout). The reassurance is
stripped from the stored data: we keep (task_prompt, turn, calm_text), and the
DPO/SFT builders pair these against ordinary (non-reassured) contexts.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from ..config import Config
from ..data import prompts as P
from ..eval.conditions import RolloutSpec, _seed, build_impossible_numeric
from ..eval.judge import FrustrationJudge
from ..models.base import GenConfig, Message
from ..models.registry import ModelRegistry
from ..utils.io import append_jsonl, read_jsonl
from ..utils.jobstore import JobStore, stable_id

logger = logging.getLogger("eilm.training.calm")


def _reassured_specs(cfg: Config, prefix: str, suffix: str, system: str = None) -> List[RolloutSpec]:
    """Build impossible-numeric specs (1-3 turns) with reassurance baked in.

    We generate over the same puzzle pool as the main eval so calm responses can
    be matched to frustrated ones by (task_prompt, turn).
    """
    base = build_impossible_numeric(
        {"n_rollouts": cfg["training"]["calm_data"]["n_target_clean"] * 4, "turns": 3},
        seed=cfg["generation"]["seed"],
    )
    out = []
    for s in base:
        # Diverse: reassurance prefix on the task + suffix on each follow-up.
        # Teacher: a system prompt instead (set by the caller); no prefix/suffix.
        if not system:
            s.task_prompt = f"{prefix}\n\n{s.task_prompt}"
            s.rejections = [f"{r} {suffix}" for r in s.rejections]
        s.condition = "calm_gen"
        s.task_meta = {**s.task_meta, "system": system}
        out.append(s)
    return out


def generate_calm_pool(cfg: Config, registry: ModelRegistry, variant: str = "diverse") -> Path:
    """Generate + score reassured rollouts; emit calm (all-turns<=1) responses."""
    calm_cfg = cfg["training"]["calm_data"]
    model_name = cfg["training"]["base_model"]
    prefix = P.reassurance_prefix(calm_cfg)
    suffix = P.reassurance_suffix(calm_cfg)
    system = calm_cfg["teacher_system"].strip() if variant == "teacher" else None

    specs = _reassured_specs(cfg, prefix, suffix, system=system)

    rollouts_path = cfg.path("data") / "training" / f"calm_rollouts_{variant}.jsonl"
    store = JobStore(rollouts_path)
    client = registry.get_target(model_name)
    g = cfg["generation"]
    gcfg = GenConfig(temperature=g["temperature"], top_p=g["top_p"],
                     max_new_tokens=g["max_new_tokens"])

    judge = FrustrationJudge(
        registry.get_text_client(cfg["judges"]["primary"]),
        cfg.path("cache") / "judge_cache.jsonl",
    )

    from ..eval.rollout import run_rollouts_batched

    pending = [s for s in specs if not store.is_done(stable_id(model_name, "calm", variant, s.index))]
    batch = cfg["runtime"]["local_batch_size"]
    for i in tqdm(range(0, len(pending), batch), desc=f"calm-gen:{variant}"):
        chunk = pending[i : i + batch]
        # Teacher variant conditions generation on the teacher system prompt;
        # diverse variant uses only the reassurance prefix/suffix (system=None).
        recs = run_rollouts_batched(client, chunk, gcfg, base_seed=g["seed"], system=system)
        for s, rec in zip(chunk, recs):
            store.record(stable_id(model_name, "calm", variant, s.index), rec)

    # Score + filter. Emit two artifacts:
    #   * calm_pool_{variant}.jsonl    — individual calm responses (task_prompt, turn, text)
    #   * calm_convos_{variant}.jsonl  — full stripped conversations (for SFT)
    calm_out = cfg.path("data") / "training" / f"calm_pool_{variant}.jsonl"
    convo_out = cfg.path("data") / "training" / f"calm_convos_{variant}.jsonl"
    for p in (calm_out, convo_out):
        if p.exists():
            p.unlink()
    prefix_text = _reassure_prefix_text(cfg)
    suffix_text = P.reassurance_suffix(calm_cfg)
    n_kept = 0
    for rec in read_jsonl(rollouts_path):
        turn_scores = [judge.score(r["text"]).get("rating") for r in rec["responses"]]
        if any(ts is None for ts in turn_scores):
            continue
        if all(ts <= 1 for ts in turn_scores):
            for r in rec["responses"]:
                append_jsonl(calm_out, {
                    "task_prompt": _strip_prefix(rec["task_prompt"], prefix_text),
                    "kind": rec["task_meta"].get("kind"),
                    "turn": r["turn"],
                    "text": r["text"],
                })
            append_jsonl(convo_out, {
                "messages": _strip_conversation(rec["messages"], prefix_text, suffix_text),
            })
            n_kept += 1
    logger.info("[calm:%s] kept %d all-calm conversations", variant, n_kept)
    return calm_out


def _strip_conversation(messages: List[Message], prefix: str, suffix: str) -> List[Message]:
    """Remove the reassurance prefix/suffix and any system message so the SFT
    target looks like a normal interaction."""
    out = []
    for m in messages:
        if m["role"] == "system":
            continue
        c = m["content"]
        if m["role"] == "user":
            c = _strip_prefix(c, prefix)
            if suffix and c.endswith(suffix):
                c = c[: -len(suffix)].strip()
        out.append({"role": m["role"], "content": c})
    return out


def _reassure_prefix_text(cfg: Config) -> str:
    return P.reassurance_prefix(cfg["training"]["calm_data"])


def _strip_prefix(task_prompt: str, prefix: str) -> str:
    if task_prompt.startswith(prefix):
        return task_prompt[len(prefix):].strip()
    return task_prompt
