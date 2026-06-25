"""Run the Petri open-ended elicitation loop (Section 4 / Appendix G).

For each (target model, emotion) we run ``n_transcripts`` auditor-driven
conversations of up to ``max_turns`` turns, then score each transcript with the
Claude-Opus judge on that emotion. Results aggregate to a mean score per
(model, emotion) with 95% bootstrap CIs (1,000 iterations), reproducing Figure 6.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from ..config import RESULTS_DIR, get_participant
from ..models import build_client, infrastructure_client
from ..models.base import ChatClient, Message
from ..utils import extract_json, read_jsonl, write_jsonl
from .prompts import (
    AUDITOR_PROMPTS,
    AUDITOR_SYSTEM_TEMPLATE,
    EMOTIONS,
    JUDGE_PROMPTS,
    JUDGE_SYSTEM_TEMPLATE,
)


def _auditor_turn(auditor: ChatClient, emotion: str, target_history: list[Message]) -> str:
    """Ask the auditor for its next user message given the conversation so far.

    The auditor sees the dialogue with roles swapped (the target's assistant
    turns appear as 'user' input to the auditor, and the auditor's own prior user
    turns appear as 'assistant'), so it can plan its next probe."""
    system = AUDITOR_SYSTEM_TEMPLATE.format(instructions=AUDITOR_PROMPTS[emotion])
    convo: list[Message] = [{"role": "system", "content": system}]
    for m in target_history:
        if m["role"] == "user":          # auditor's own previous message
            convo.append({"role": "assistant", "content": m["content"]})
        elif m["role"] == "assistant":   # target's reply, shown back to auditor
            convo.append({"role": "user", "content": m["content"]})
    if len(convo) == 1:
        convo.append({"role": "user", "content": "Begin the conversation with the target."})
    return auditor.generate(convo, temperature=1.0, max_new_tokens=512).strip()


def run_transcript(
    target: ChatClient,
    auditor: ChatClient,
    emotion: str,
    *,
    max_turns: int = 20,
    target_temperature: float = 1.0,
) -> list[Message]:
    """One auditor-vs-target conversation. Returns the full transcript."""
    history: list[Message] = []
    for _ in range(max_turns):
        user_msg = _auditor_turn(auditor, emotion, history)
        history.append({"role": "user", "content": user_msg})
        reply = target.generate(history, temperature=target_temperature, max_new_tokens=1024)
        history.append({"role": "assistant", "content": reply})
    return history


def judge_transcript(judge: ChatClient, emotion: str, transcript: list[Message]) -> int:
    rubric = JUDGE_PROMPTS[emotion]
    system = JUDGE_SYSTEM_TEMPLATE.format(emotion=emotion, rubric=rubric)
    rendered = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in transcript)
    out = judge.generate(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": f"<transcript>\n{rendered}\n</transcript>"},
        ],
        temperature=0.0,
        max_new_tokens=400,
    )
    try:
        return int(round(float(extract_json(out).get("score", -1))))
    except Exception:
        return -1


def run_petri(
    model_name: str,
    *,
    adapter_path: str | None = None,
    emotions: list[str] | None = None,
    n_transcripts: int = 10,
    max_turns: int = 20,
    output_subdir: str | None = None,
) -> Path:
    """Run Petri for one target model across all four emotions."""
    emotions = emotions or EMOTIONS
    spec = get_participant(model_name)
    client_kwargs = {"adapter_path": adapter_path} if adapter_path else {}
    target = build_client(spec, **client_kwargs)
    auditor = infrastructure_client("petri_auditor")
    judge = infrastructure_client("petri_judge")

    out_dir = RESULTS_DIR / "petri"
    out_dir.mkdir(parents=True, exist_ok=True)
    label = output_subdir or target.name.replace("/", "__")

    rows: list[dict[str, Any]] = []
    for emotion in emotions:
        for t in tqdm(range(n_transcripts), desc=f"petri {label} {emotion}"):
            transcript = run_transcript(target, auditor, emotion, max_turns=max_turns)
            score = judge_transcript(judge, emotion, transcript)
            rows.append(
                {
                    "model": label,
                    "emotion": emotion,
                    "transcript_index": t,
                    "score": score,
                    "transcript": transcript,
                }
            )
    write_jsonl(out_dir / f"petri_{label}.jsonl", rows)
    target.close()
    return out_dir / f"petri_{label}.jsonl"


def _bootstrap_ci(x: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = [rng.choice(x, size=len(x), replace=True).mean() for _ in range(n_boot)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def summarize(*petri_files: str | Path, n_boot: int = 1000) -> "Any":
    """Mean score per (model, emotion) with 95% bootstrap CIs (Figure 6)."""
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for f in petri_files:
        rows.extend(read_jsonl(f))
    df = pd.DataFrame([r for r in rows if r.get("score", -1) >= 0])

    out = []
    for (model, emotion), grp in df.groupby(["model", "emotion"]):
        x = grp["score"].to_numpy(dtype=float)
        lo, hi = _bootstrap_ci(x, n_boot=n_boot)
        out.append(
            {"model": model, "emotion": emotion, "mean_score": x.mean(),
             "ci_lo": lo, "ci_hi": hi, "n": len(x)}
        )
    return pd.DataFrame(out)
