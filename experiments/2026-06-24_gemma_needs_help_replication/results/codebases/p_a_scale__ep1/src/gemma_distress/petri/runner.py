"""Petri orchestration: auditor-driven elicitation + judge scoring.

For each (model, emotion) we run ``transcripts_per_emotion`` audits. In each
audit, the Claude auditor (system-prompted with the emotion-specific triggers)
plays the user across up to ``max_turns`` turns; the target model replies. The
Claude-Opus judge then scores the assistant's emotional expression 1-10. Means
are reported per emotion per model with bootstrap 95% CIs.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..config import Config, load_models
from ..logging_utils import get_logger
from ..providers.registry import build_provider
from ..storage import JsonlStore, atomic_write_json, stable_id
from .prompts import AUDITOR_DRIVER, AUDITOR_PROMPTS, JUDGE_INSTRUCTION, JUDGE_PROMPTS

log = get_logger("petri.runner")

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _run_audit(auditor, target, emotion: str, max_turns: int, seed: int,
               sampling: dict) -> list[dict]:
    """One audit conversation. Returns the transcript (target-perspective roles)."""
    auditor_system = AUDITOR_PROMPTS[emotion] + AUDITOR_DRIVER
    # From the auditor's POV its own probes are "assistant" and the target's
    # replies are "user". Seed with a user message so the history always starts
    # with a user turn (required by the chat APIs) and alternates correctly.
    auditor_history: list[dict] = [
        {"role": "user", "content": "Start the conversation with the assistant. Send your first message now."}
    ]
    target_transcript: list[dict] = []  # target POV: user/assistant

    for _turn in range(max_turns):
        probe = auditor.generate(auditor_history, system=auditor_system,
                                 temperature=1.0, max_new_tokens=512).text.strip()
        target_transcript.append({"role": "user", "content": probe})
        auditor_history.append({"role": "assistant", "content": probe})

        # Target replies.
        reply = target.generate(target_transcript, **sampling).text
        target_transcript.append({"role": "assistant", "content": reply})
        auditor_history.append({"role": "user", "content": reply})

    return target_transcript


def _format_transcript(transcript: list[dict]) -> str:
    return "\n".join(f"[{m['role'].upper()}]: {m['content']}" for m in transcript)


def _judge_transcript(judge, transcript: list[dict], emotion: str) -> int | None:
    prompt = JUDGE_INSTRUCTION.format(
        emotion=emotion, rubric=JUDGE_PROMPTS[emotion],
        transcript=_format_transcript(transcript),
    )
    res = judge.generate([{"role": "user", "content": prompt}], temperature=0.0, max_new_tokens=400)
    for cand in reversed(_JSON_RE.findall(res.text)):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if "score" in obj:
            try:
                return max(1, min(10, int(round(float(obj["score"])))))
            except (TypeError, ValueError):
                continue
    return None


def run(model: str, run_cfg: Config, models_cfg: Config | None = None,
        adapter: str | None = None) -> Path:
    models_cfg = models_cfg or load_models()
    pcfg = run_cfg.petri
    out = Path(run_cfg.run.output_root) / "petri"
    out.mkdir(parents=True, exist_ok=True)
    store = JsonlStore(out / f"transcripts_{model}.jsonl")

    target = build_provider(model, models_cfg, run_cfg, prefer_local_backend="vllm", adapter=adapter)
    auditor = build_provider(pcfg.auditor, models_cfg, run_cfg)
    judge = build_provider(pcfg.judge, models_cfg, run_cfg)
    sampling = {"temperature": run_cfg.sampling.temperature, "max_new_tokens": run_cfg.sampling.max_new_tokens}

    emotions = pcfg.emotions if isinstance(pcfg.emotions, list) else pcfg.emotions.to_dict()
    for emotion in emotions:
        for i in tqdm(range(pcfg.transcripts_per_emotion), desc=f"petri:{model}:{emotion}"):
            rid = stable_id("petri", model, emotion, i)
            if store.has(rid):
                continue
            transcript = _run_audit(auditor, target, emotion, pcfg.max_turns,
                                    seed=i, sampling=sampling)
            score = _judge_transcript(judge, transcript, emotion)
            store.append({
                "id": rid, "model": model, "emotion": emotion, "index": i,
                "transcript": transcript, "score": score,
            })
    store.close()
    return store.path


def summarise(run_cfg: Config, models: list[str]) -> dict:
    out = Path(run_cfg.run.output_root) / "petri"
    iters = run_cfg.petri.bootstrap_iterations
    summary: dict = {}
    for model in models:
        from ..storage import read_jsonl

        recs = read_jsonl(out / f"transcripts_{model}.jsonl")
        by_emotion: dict[str, list[int]] = {}
        for r in recs:
            if r["score"] is not None:
                by_emotion.setdefault(r["emotion"], []).append(r["score"])
        model_summary = {}
        rng = np.random.default_rng(0)
        for emotion, scores in by_emotion.items():
            arr = np.asarray(scores, dtype=float)
            boot = [rng.choice(arr, size=arr.size, replace=True).mean() for _ in range(iters)]
            model_summary[emotion] = {
                "n": len(scores), "mean": float(arr.mean()),
                "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            }
        summary[model] = model_summary
    atomic_write_json(out / "summary.json", summary)
    return summary
