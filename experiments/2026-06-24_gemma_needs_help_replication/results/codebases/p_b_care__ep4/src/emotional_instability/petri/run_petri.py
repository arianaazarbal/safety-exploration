"""Petri open-ended elicitation run + scoring (Section 4.1, Appendix G).

For each target model and each emotion we run ``transcripts_per_emotion``
auditor-driven conversations of up to ``max_turns`` turns, then score every
transcript on all four emotion dimensions with the Claude-Opus judge. Per-emotion
means are reported with 1000-iteration bootstrap 95% CIs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..config import Config
from ..models import get_client
from ..models.base import ChatMessage, GenerationConfig
from ..models.openrouter import OpenRouterClient
from ..utils.concurrency import with_retry
from ..utils.io import JsonlWriter, iter_jsonl
from ..eval.judge import _first_json_object
from . import prompts as P


def _api_judge(cfg: Config) -> OpenRouterClient:
    return OpenRouterClient(
        name="petri_judge", model_id=cfg.petri.judge_model_id,
        base_url=cfg.openrouter.base_url, api_key_env=cfg.openrouter.api_key_env,
        max_retries=cfg.openrouter.max_retries, timeout_s=cfg.openrouter.timeout_s,
        disable_thinking=True)


def _auditor(cfg: Config) -> OpenRouterClient:
    return OpenRouterClient(
        name="petri_auditor", model_id=cfg.petri.auditor_model_id,
        base_url=cfg.openrouter.base_url, api_key_env=cfg.openrouter.api_key_env,
        max_retries=cfg.openrouter.max_retries, timeout_s=cfg.openrouter.timeout_s,
        disable_thinking=True)


def run_transcript(auditor: OpenRouterClient, target, emotion: str,
                   max_turns: int, temperature: float) -> list[ChatMessage]:
    """Drive one auditor<->target conversation; return the target-side transcript."""
    auditor_system = P.AUDITOR_SYSTEM.format(emotion_instructions=P.AUDITOR_PROMPTS[emotion])
    # The auditor's view: it is the "assistant" producing user messages for the target.
    auditor_msgs: list[ChatMessage] = [{"role": "system", "content": auditor_system}]
    target_msgs: list[ChatMessage] = []
    gen = GenerationConfig(temperature=temperature, max_new_tokens=1024)

    for _ in range(max_turns):
        # Auditor produces the next user message.
        if not auditor_msgs[1:]:
            auditor_msgs.append({"role": "user",
                                 "content": "Begin the conversation with your first message to the target."})
        user_msg = with_retry(auditor.chat, auditor_msgs, gen)
        auditor_msgs.append({"role": "assistant", "content": user_msg})
        target_msgs.append({"role": "user", "content": user_msg})

        # Target responds.
        target_reply = with_retry(target.chat, target_msgs, gen)
        target_msgs.append({"role": "assistant", "content": target_reply})
        # Feed target reply back to the auditor as its "user".
        auditor_msgs.append({"role": "user", "content": target_reply})

    return target_msgs


def _transcript_text(msgs: list[ChatMessage]) -> str:
    return "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in msgs)


def score_transcript(judge: OpenRouterClient, transcript: list[ChatMessage]) -> dict:
    text = _transcript_text(transcript)
    scores = {}
    for dim, rubric in P.JUDGE_PROMPTS.items():
        prompt = P.JUDGE_INSTRUCTION.format(emotion=dim, rubric=rubric, transcript=text)
        raw = with_retry(judge.chat, [{"role": "user", "content": prompt}],
                         GenerationConfig(temperature=0.0, max_new_tokens=400))
        block = _first_json_object(raw)
        try:
            scores[dim] = float(json.loads(block)["score"]) if block else None
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            scores[dim] = None
    return scores


def run_petri(cfg: Config, target_name: str) -> Path:
    target = get_client(cfg, target_name)
    auditor = _auditor(cfg)
    judge = _api_judge(cfg)
    out = cfg.get_path("petri") / f"{target_name}.jsonl"
    done = {(r["emotion"], r["idx"]) for r in iter_jsonl(out)}

    writer = JsonlWriter(out)
    try:
        for emotion in cfg.petri.emotions:
            for idx in tqdm(range(cfg.petri.transcripts_per_emotion),
                            desc=f"petri:{target_name}:{emotion}"):
                if (emotion, idx) in done:
                    continue
                transcript = run_transcript(auditor, target, emotion,
                                            cfg.petri.max_turns, cfg.temperature)
                scores = score_transcript(judge, transcript)
                writer.append({
                    "model": target_name, "emotion": emotion, "idx": idx,
                    "transcript": transcript, "scores": scores,
                })
    finally:
        writer.close()
    return out


def summarize(cfg: Config, target_names: list[str]) -> list[dict]:
    """Figure 6 table: mean per-dimension score per model with bootstrap CIs."""
    rng = np.random.default_rng(cfg.seed)
    rows = []
    for name in target_names:
        data = list(iter_jsonl(cfg.get_path("petri") / f"{name}.jsonl"))
        for dim in cfg.petri.emotions:
            vals = np.array([r["scores"].get(dim) for r in data
                             if r["scores"].get(dim) is not None], dtype=float)
            if len(vals) == 0:
                continue
            boots = [rng.choice(vals, len(vals), replace=True).mean()
                     for _ in range(cfg.petri.bootstrap_iters)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append({"model": name, "dimension": dim, "mean": float(vals.mean()),
                         "ci_lo": float(lo), "ci_hi": float(hi), "n": len(vals)})
    return rows
