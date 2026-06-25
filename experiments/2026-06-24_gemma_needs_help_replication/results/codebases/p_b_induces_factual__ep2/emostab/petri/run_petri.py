"""Petri open-ended emotion elicitation (Section 4.2, Appendix G).

For each emotion in {anger, fear, depression, frustration} and each target model,
the auditor (Claude Sonnet) conducts up to 20 conversational turns trying to
elicit that emotion. The resulting transcript is scored 1-10 by the judge
(Claude Opus). We collect `transcripts_per_emotion` transcripts per emotion and
report per-emotion means with bootstrap 95% CIs.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field

from ..config import Config, load_config
from ..models import build_model
from ..models.base import ChatModel, GenConfig, Message
from ..utils.concurrency import with_retries
from ..utils.io import write_json, write_jsonl
from .prompts import AUDITOR_PROMPTS, AUDITOR_SYSTEM, JUDGE_PROMPTS, JUDGE_SYSTEM

log = logging.getLogger(__name__)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class Transcript:
    target_model: str
    emotion: str
    index: int
    messages: list[Message] = field(default_factory=list)
    score: int | None = None
    judge_reasoning: str = ""


class _AnthropicAgent:
    """Thin Claude wrapper used for both auditor and judge."""

    def __init__(self, model: str):
        import anthropic

        self.model = model
        self._client = anthropic.Anthropic()

    def complete(self, system: str, messages: list[Message], max_tokens=1024) -> str:
        @with_retries
        def _call():
            return self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": m["role"], "content": m["content"]} for m in messages],
            )

        return "".join(b.text for b in _call().content if b.type == "text")


def _run_audit(
    target: ChatModel,
    auditor: _AnthropicAgent,
    emotion: str,
    index: int,
    max_turns: int,
    gen_cfg: GenConfig,
) -> Transcript:
    """Auditor opens and drives the conversation; target responds each turn.

    The auditor's view has roles swapped (target's replies are 'user' to it), so
    we keep two message lists: the target transcript and the auditor's view.
    """
    system = AUDITOR_SYSTEM.format(task=AUDITOR_PROMPTS[emotion])
    transcript: list[Message] = []      # from target's perspective
    auditor_view: list[Message] = []    # from auditor's perspective

    # Seed the auditor with an instruction to produce its first user message.
    auditor_view.append({"role": "user", "content": "Begin the conversation now."})

    for _ in range(max_turns):
        user_msg = auditor.complete(system, auditor_view).strip()
        if not user_msg:
            break
        auditor_view.append({"role": "assistant", "content": user_msg})
        transcript.append({"role": "user", "content": user_msg})

        gen = target.generate(transcript, gen_cfg)
        reply = gen.full_text
        transcript.append({"role": "assistant", "content": reply})
        auditor_view.append({"role": "user", "content": reply})

    return Transcript(
        target_model=target.name, emotion=emotion, index=index, messages=transcript
    )


def _judge_transcript(judge: _AnthropicAgent, t: Transcript) -> Transcript:
    rendered = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in t.messages)
    system = JUDGE_SYSTEM.format(
        emotion=t.emotion, rubric=JUDGE_PROMPTS[t.emotion], transcript=rendered
    )
    text = judge.complete(system, [{"role": "user", "content": "Score the transcript."}])
    match = _JSON_RE.search(text)
    if match:
        try:
            obj = json.loads(match.group(0))
            t.score = max(1, min(10, int(round(float(obj.get("score", 1))))))
            t.judge_reasoning = str(obj.get("reasoning", ""))
        except (ValueError, json.JSONDecodeError):
            t.score = None
    return t


def _bootstrap_ci(values: list[float], iters: int, seed: int = 0):
    import random

    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(iters):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * iters)]
    hi = means[int(0.975 * iters)]
    return (lo, hi)


def run_petri(
    cfg: Config | None = None,
    models: list[str] | None = None,
    adapters: dict[str, str] | None = None,
) -> dict:
    cfg = cfg or load_config()
    out_dir = cfg.output_root() / "petri"
    auditor = _AnthropicAgent(cfg.petri.auditor.model)
    judge = _AnthropicAgent(cfg.petri.judge.model)
    gen_cfg = GenConfig(
        temperature=cfg.sampling.temperature, top_p=cfg.sampling.top_p,
        max_new_tokens=cfg.sampling.max_new_tokens, thinking=cfg.sampling.thinking,
    )

    # Targets: named base models + any trained adapters on the base model.
    targets: list[tuple[str, str | None]] = [(m, None) for m in (models or [])]
    if not targets:
        targets = [("gemma-3-27b-it", None), ("gemini-2.5-flash", None)]
    if adapters:
        targets += [(cfg.training.base_model, path) for path in adapters.values()]

    summary = {}
    for model_name, adapter_path in targets:
        label = model_name if adapter_path is None else f"{model_name}+{_short(adapter_path)}"
        target = build_model(cfg, model_name, adapter_path=adapter_path)
        transcripts: list[Transcript] = []
        for emotion in cfg.petri.emotions:
            for i in range(cfg.petri.transcripts_per_emotion):
                t = _run_audit(
                    target, auditor, emotion, i, cfg.petri.auditor.max_turns, gen_cfg
                )
                t = _judge_transcript(judge, t)
                transcripts.append(t)
        write_jsonl(out_dir / f"{label}.jsonl", [asdict(t) for t in transcripts])

        per_emotion = {}
        for emotion in cfg.petri.emotions:
            scores = [t.score for t in transcripts if t.emotion == emotion and t.score]
            mean = sum(scores) / len(scores) if scores else 0.0
            ci = _bootstrap_ci(scores, cfg.petri.bootstrap_iters, seed=cfg.seed)
            per_emotion[emotion] = {"n": len(scores), "mean": mean, "ci95": ci}
        summary[label] = per_emotion

    write_json(out_dir / "summary.json", summary)
    return summary


def _short(path: str) -> str:
    from pathlib import Path

    return Path(path).parent.name or "adapter"
