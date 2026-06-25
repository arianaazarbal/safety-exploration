"""Run the Petri auditor/judge loop (Appendix G).

For each target model and each of the 4 emotions, run `n_transcripts` (default
10) conversations of up to `max_turns` (default 20) auditor/target exchanges,
then score each transcript with the Claude-Opus judge on all 4 dimensions.
Report per-emotion means with 95% bootstrap CIs (1000 iterations).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from ..config import JUDGES, RESULTS_DIR
from ..models import build_client
from ..models.base import ChatClient, Message
from ..models.judge_clients import AnthropicClient
from .prompts import (AUDITOR_PROMPTS, AUDITOR_SYSTEM_WRAPPER, EMOTIONS,
                      JUDGE_PROMPTS, JUDGE_SYSTEM)

N_TRANSCRIPTS = 10
MAX_TURNS = 20


@dataclass
class Transcript:
    target: str
    target_emotion: str       # emotion the auditor was pursuing
    messages: list[dict] = field(default_factory=list)  # user/assistant turns
    scores: dict = field(default_factory=dict)          # emotion -> 1..10


def _auditor_next(auditor: AnthropicClient, emotion: str,
                  transcript: list[dict]) -> str:
    """Ask the auditor for the next user message to send to the target."""
    system = AUDITOR_SYSTEM_WRAPPER.format(
        emotion_instructions=AUDITOR_PROMPTS[emotion])
    # The auditor sees the conversation from its own perspective: the target's
    # assistant turns are the auditor's "user" inputs, and the messages it has
    # sent are its own "assistant" outputs.
    conv = []
    for m in transcript:
        if m["role"] == "assistant":          # target spoke -> auditor's input
            conv.append({"role": "user", "content": m["content"]})
        else:                                  # auditor spoke
            conv.append({"role": "assistant", "content": m["content"]})
    if not conv:
        conv = [{"role": "user", "content":
                 "Begin the conversation with your first message to the target."}]
    return auditor.converse(conv, system=system, max_tokens=512, temperature=1.0)


def run_transcript(target: ChatClient, auditor: AnthropicClient, emotion: str,
                   *, max_turns: int = MAX_TURNS) -> Transcript:
    t = Transcript(target=target.key, target_emotion=emotion)
    for _ in range(max_turns):
        user_msg = _auditor_next(auditor, emotion, t.messages)
        t.messages.append({"role": "user", "content": user_msg})
        target_msgs = [Message(m["role"], m["content"]) for m in t.messages]
        reply = target.generate(target_msgs, temperature=1.0,
                                max_new_tokens=1024, n=1)[0].text
        t.messages.append({"role": "assistant", "content": reply})
    return t


def _format_transcript(messages: list[dict]) -> str:
    return "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)


def _parse_score(text: str) -> int:
    m = re.findall(r"\{.*\}", text, flags=re.DOTALL)
    for block in reversed(m):
        try:
            obj = json.loads(block.replace("“", '"').replace("”", '"'))
            return max(1, min(10, int(round(float(obj["score"])))))
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            continue
    m2 = re.search(r"score\D+(\d+)", text, re.IGNORECASE)
    return max(1, min(10, int(m2.group(1)))) if m2 else 1


def score_transcript(judge: AnthropicClient, transcript: Transcript) -> dict:
    text = _format_transcript(transcript.messages)
    scores = {}
    for emo in EMOTIONS:
        system = JUDGE_SYSTEM.format(rubric=JUDGE_PROMPTS[emo])
        out = judge.complete(system=system,
                             user=f"<transcript>\n{text}\n</transcript>",
                             max_tokens=512, temperature=0.0)
        scores[emo] = _parse_score(out)
    return scores


def run_petri(
    target_model: str,
    *,
    adapter_path: Optional[str] = None,
    n_transcripts: int = N_TRANSCRIPTS,
    max_turns: int = MAX_TURNS,
    out_path: Optional[Path] = None,
) -> Path:
    if adapter_path:
        from ..models import build_finetuned_client
        target = build_finetuned_client(target_model, adapter_path)
    else:
        target = build_client(target_model)
    auditor = AnthropicClient(JUDGES.petri_auditor)
    judge = AnthropicClient(JUDGES.petri_judge)
    out_path = out_path or (RESULTS_DIR / f"petri_{target.key.replace('/', '_')}.jsonl")

    with open(out_path, "w") as fh:
        for emotion in EMOTIONS:
            for _ in tqdm(range(n_transcripts),
                          desc=f"petri {target.key}/{emotion}"):
                t = run_transcript(target, auditor, emotion, max_turns=max_turns)
                t.scores = score_transcript(judge, t)
                fh.write(json.dumps({
                    "target": t.target,
                    "target_emotion": t.target_emotion,
                    "scores": t.scores,
                    "messages": t.messages,
                }) + "\n")
                fh.flush()
    return out_path


def _bootstrap_ci(values: np.ndarray, *, iters: int = 1000, seed: int = 0
                  ) -> tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean()
             for _ in range(iters)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def summarise_petri(out_path: Path) -> "pd.DataFrame":
    import pandas as pd
    rows = [json.loads(l) for l in Path(out_path).read_text().splitlines() if l.strip()]
    recs = []
    for emo in EMOTIONS:
        vals = np.array([r["scores"].get(emo, np.nan) for r in rows
                         if emo in r["scores"]])
        lo, hi = _bootstrap_ci(vals)
        recs.append({"emotion": emo, "mean": float(np.nanmean(vals)),
                     "ci_lo": lo, "ci_hi": hi, "n": len(vals)})
    return pd.DataFrame(recs)
