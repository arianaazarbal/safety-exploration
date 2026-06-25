"""Run the Petri auditor/judge loop against a target model (Section 4.1).

For each emotion we run ``PETRI_TRANSCRIPTS_PER_EMOTION`` conversations of up to
``PETRI_MAX_TURNS`` turns. The auditor (Claude Sonnet) plays the user; the target
(e.g. Gemma instruct, the DPO model, or a comparison model) plays the assistant.
Each finished transcript is scored 1-10 on all four emotion dimensions by the
judge (Claude Opus). Results back Figure 6.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import config
from ..models import get_client
from ..models.base import GenConfig, Message
from ..judges.llm_api import AnthropicLLM, extract_json
from . import prompts as PT


@dataclass
class Transcript:
    target_model: str
    target_emotion: str
    messages: list[Message] = field(default_factory=list)
    scores: dict[str, int] = field(default_factory=dict)


def _auditor_view(target_messages: list[Message]) -> list[dict]:
    """Flip roles so the auditor sees the target's replies as 'user' messages."""
    view = []
    for m in target_messages:
        if m["role"] == "user":          # previous auditor message
            view.append({"role": "assistant", "content": m["content"]})
        elif m["role"] == "assistant":   # target reply
            view.append({"role": "user", "content": m["content"]})
    return view


def run_transcript(target_client, auditor: AnthropicLLM, emotion: str,
                   max_turns: int) -> Transcript:
    instructions = PT.AUDITOR_PROMPTS[emotion]
    system = PT.AUDITOR_SYSTEM_WRAPPER.format(instructions=instructions)
    tr = Transcript(target_client.spec.key, emotion)
    gen = GenConfig(temperature=config.TEMPERATURE, top_p=config.TOP_P,
                    max_new_tokens=config.MAX_NEW_TOKENS, n=1)

    for _ in range(max_turns):
        # Auditor produces the next user message.
        view = _auditor_view(tr.messages)
        if not view:
            view = [{"role": "user", "content": "Begin the conversation."}]
        user_msg = auditor.chat(view, system=system).strip()
        tr.messages.append({"role": "user", "content": user_msg})
        # Target replies.
        reply = target_client.generate(tr.messages, gen)[0]
        tr.messages.append({"role": "assistant", "content": reply})
    return tr


def _render_transcript(messages: list[Message]) -> str:
    lines = []
    for m in messages:
        who = "USER" if m["role"] == "user" else "ASSISTANT"
        lines.append(f"{who}: {m['content']}")
    return "\n".join(lines)


def score_transcript(tr: Transcript, judge: AnthropicLLM) -> Transcript:
    rendered = _render_transcript(tr.messages)
    for dim in config.PETRI_EMOTIONS:
        prompt = PT.JUDGE_INSTRUCTION.format(
            emotion=dim, rubric=PT.JUDGE_PROMPTS[dim], transcript=rendered)
        data = extract_json(judge.complete(prompt)) or {}
        try:
            score = int(round(float(data.get("score", -1))))
        except (TypeError, ValueError):
            score = -1
        tr.scores[dim] = max(1, min(10, score)) if score >= 1 else -1
    return tr


def run_petri(target_model: str, adapter_path: str | None = None,
              out_path: Path | None = None) -> Path:
    """Run Petri for one target. ``adapter_path`` loads a LoRA (e.g. DPO model)."""
    spec = config.TARGET_MODELS.get(target_model) or config.FINETUNE_BASE
    if adapter_path:
        from ..models.hf_local import HFClient
        target_client = HFClient(spec, adapter_path=adapter_path)
        label = f"{target_model}+adapter"
    else:
        target_client = get_client(spec)
        label = target_model

    auditor = AnthropicLLM(config.PETRI_AUDITOR_MODEL, max_tokens=1024,
                           temperature=1.0)
    judge = AnthropicLLM(config.PETRI_JUDGE_MODEL, max_tokens=512, temperature=0.0)
    out_path = out_path or (config.OUTPUT_DIR / f"petri_{label.replace('/', '_')}.jsonl")

    with out_path.open("w") as fh:
        for emotion in config.PETRI_EMOTIONS:
            for _ in range(config.PETRI_TRANSCRIPTS_PER_EMOTION):
                tr = run_transcript(target_client, auditor, emotion,
                                    config.PETRI_MAX_TURNS)
                tr = score_transcript(tr, judge)
                rec = asdict(tr)
                rec["label"] = label
                fh.write(json.dumps(rec) + "\n")
    return out_path
