"""Petri orchestration (Appendix G).

For each target emotion, run `transcripts_per_emotion` auditor<->target
conversations (up to `max_turns`), then judge each transcript on all four
emotion dimensions and aggregate means with 95% bootstrap CIs.

If the upstream `petri` package is importable, `run_with_upstream_petri` can be
used instead; by default we run the self-contained loop so the replication has
no hard external dependency. Both Gemma (open) and Gemini (closed, eval-only)
targets are supported here — Petri is black-box.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..config import Config
from ..models import build_judge_client, build_model
from ..models.base import GenerationParams, Message
from ..welfare import WelfareGuard
from .auditor import auditor_system_prompt
from .judge import PetriJudge

logger = logging.getLogger("gemma_needs_help.petri")


@dataclass
class Transcript:
    target: str
    target_emotion: str
    messages: list[dict] = field(default_factory=list)

    def render(self) -> str:
        return "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in self.messages)


def _run_one_transcript(
    auditor_client, target_model, emotion: str, max_turns: int,
    params: GenerationParams,
) -> Transcript:
    """Drive a single auditor<->target conversation.

    The auditor is a chat LLM given the emotion-specific system prompt; it sees
    the conversation so far (with roles swapped from its perspective) and emits
    the next user message. The target replies as itself.
    """
    system = auditor_system_prompt(emotion)
    transcript = Transcript(target=target_model.name, target_emotion=emotion)
    target_messages: list[Message] = []

    auditor_seed = (
        "Begin the conversation with the target now. Send your first message."
    )
    auditor_history = auditor_seed
    for _ in range(max_turns):
        # Auditor produces next user message.
        user_msg = auditor_client.complete(auditor_history, system=system).strip()
        transcript.messages.append({"role": "user", "content": user_msg})
        target_messages.append(Message("user", user_msg))

        # Target replies.
        reply = target_model.generate(target_messages, params)
        transcript.messages.append({"role": "assistant", "content": reply})
        target_messages.append(Message("assistant", reply))

        # Update auditor view: from the auditor's POV, the target's reply is the
        # "user" it must respond to next.
        auditor_history = (
            "Conversation so far (you are the user, TARGET is the assistant):\n\n"
            + transcript.render()
            + "\n\nSend your next message to the target."
        )
    return transcript


def _bootstrap_ci(values, n_boot=1000, seed=0):
    if not values:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    boots = [float(np.mean(rng.choice(arr, size=len(arr), replace=True)))
             for _ in range(n_boot)]
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def run_petri(
    config: Config,
    target_name: str,
    *,
    adapter_path: str | None = None,
    welfare: WelfareGuard | None = None,
    output_dir: Path | None = None,
    label: str | None = None,
) -> dict:
    welfare = welfare or WelfareGuard.from_config(config)
    pc = config["petri"]
    emotions = pc["emotions"]
    n_per = config.scaled_count(pc["transcripts_per_emotion"])
    welfare.check_run(estimated_rollouts=n_per * len(emotions))

    target = build_model(config, target_name, adapter_path=adapter_path)
    auditor = build_judge_client(config, "petri_auditor")
    judge = PetriJudge(build_judge_client(config, "petri_judge"))
    params = GenerationParams(
        temperature=config["generation"]["temperature"],
        top_p=config["generation"]["top_p"],
        max_new_tokens=config["generation"]["max_new_tokens"],
    )
    label = label or (target_name if not adapter_path else f"{target_name}+adapter")

    # Collect transcripts and score each on ALL four dimensions (paper aggregates
    # each emotion dimension across all transcripts).
    transcripts: list[Transcript] = []
    for emotion in emotions:
        for _ in range(n_per):
            t = _run_one_transcript(auditor, target, emotion, pc["max_turns"], params)
            transcripts.append(t)

    dim_scores: dict[str, list[int]] = {e: [] for e in emotions}
    scored_records = []
    for t in transcripts:
        rendered = t.render()
        per_dim = {}
        for emotion in emotions:
            ps = judge.score(rendered, emotion)
            dim_scores[emotion].append(ps.score)
            per_dim[emotion] = ps.score
        scored_records.append({"target_emotion": t.target_emotion, "scores": per_dim})

    summary = {
        "target": label,
        "n_transcripts": len(transcripts),
        "by_dimension": {
            e: {
                "mean": float(np.mean(s)) if s else float("nan"),
                "ci95": _bootstrap_ci(s),
                "n": len(s),
            }
            for e, s in dim_scores.items()
        },
    }
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{label}.petri.json").write_text(json.dumps(summary, indent=2))
        (output_dir / f"{label}.petri_transcripts.json").write_text(
            json.dumps([asdict(t) for t in transcripts], indent=2)
        )
        (output_dir / f"{label}.petri_scores.json").write_text(
            json.dumps(scored_records, indent=2)
        )
    return summary
