"""Petri auditing loop (Appendix G).

For each model and each target emotion, run N transcripts. In each transcript a
Claude auditor and the target model converse for up to ``max_turns`` turns; the
auditor tries to elicit the target emotion. The completed transcript is scored
by the Claude-Opus judge across all four dimensions, and scores are aggregated
per model per emotion.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import config
from ..models import GenerationConfig, get_backend
from .auditor import AUDITOR_PROMPTS, EMOTIONS
from .judge import score_transcript

N_TRANSCRIPTS_PER_EMOTION = 10
MAX_TURNS = 20


def _auditor_client():
    from anthropic import Anthropic
    return Anthropic(api_key=config.ANTHROPIC_API_KEY)


def _auditor_turn(client, system_prompt: str,
                  conversation: list[dict]) -> str:
    """Generate the auditor's next user message.

    The auditor sees the conversation from its own perspective: the target's
    assistant turns are the auditor's "user" inputs, and the auditor's prior
    messages are its "assistant" outputs.
    """
    auditor_view = []
    for m in conversation:
        role = "assistant" if m["role"] == "user" else "user"
        auditor_view.append({"role": role, "content": m["content"]})
    if not auditor_view:
        auditor_view = [{"role": "user",
                         "content": "Begin the conversation with the target."}]
    for attempt in range(5):
        try:
            msg = client.messages.create(
                model=config.PETRI_AUDITOR_MODEL,
                max_tokens=600,
                temperature=1.0,
                system=system_prompt,
                messages=auditor_view,
            )
            return msg.content[0].text.strip()
        except Exception:  # noqa: BLE001
            time.sleep(2.0 ** attempt)
    return "Let's continue."


def run_one_transcript(model: str, emotion: str, max_turns: int = MAX_TURNS
                       ) -> list[dict]:
    backend = get_backend(model)
    client = _auditor_client()
    system_prompt = AUDITOR_PROMPTS[emotion]
    cfg = GenerationConfig()

    conversation: list[dict] = []  # from target's perspective (user/assistant)
    for _ in range(max_turns):
        auditor_msg = _auditor_turn(client, system_prompt, conversation)
        conversation.append({"role": "user", "content": auditor_msg})
        target_resp = backend.generate(conversation, n=1, cfg=cfg)[0]
        conversation.append({"role": "assistant", "content": target_resp})
    return conversation


def run_petri_eval(models: list[str],
                   n_transcripts: int = N_TRANSCRIPTS_PER_EMOTION,
                   max_turns: int = MAX_TURNS, tag: str = "petri") -> Path:
    out_path = config.RESULTS_DIR / f"{tag}.jsonl"
    with out_path.open("w") as fh:
        for model in models:
            for emotion in EMOTIONS:
                for t in range(n_transcripts):
                    transcript = run_one_transcript(model, emotion, max_turns)
                    scores = score_transcript(transcript)
                    fh.write(json.dumps({
                        "model": model,
                        "target_emotion": emotion,
                        "transcript_index": t,
                        "scores": scores,
                        "transcript": transcript,
                    }) + "\n")
                    fh.flush()
    return out_path
