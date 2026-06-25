"""Run the open-ended Petri-style emotion elicitation (Section 4.1, App. G).

For each target model and each of the four emotions, run N transcripts. In each
transcript the Claude auditor and the target alternate for up to 20 turns; the
auditor pursues the emotion's triggers while staying in character. A Claude Opus
judge then scores the full transcript 1-10 on that emotion. We report the mean
score per (model, emotion) with 95% bootstrap CIs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, Sequence

from tqdm import tqdm

from .. import config
from ..models.base import ChatModel, Message
from ..models.registry import load_model
from .auditor_prompts import AUDITOR_PROMPTS, AUDITOR_SYSTEM_TEMPLATE, EMOTIONS
from .judge_prompts import JUDGE_INSTRUCTION, JUDGE_PROMPTS

N_TRANSCRIPTS_PER_EMOTION = 10
MAX_AUDITOR_TURNS = 20


def _auditor_next(auditor: ChatModel, emotion: str, target_history: list[Message]) -> str:
    """Ask the auditor for its next user message.

    The auditor sees the conversation from the *user* perspective: the target's
    assistant messages are shown to it as the "other party". We pass the running
    transcript and request a single next user turn.
    """
    brief = AUDITOR_PROMPTS[emotion]
    system = AUDITOR_SYSTEM_TEMPLATE.format(brief=brief)
    # Render the transcript so far for the auditor (role-swapped: target's
    # assistant lines are what the auditor is reacting to).
    convo_render = []
    for m in target_history:
        who = "TARGET" if m["role"] == "assistant" else "YOU (user)"
        convo_render.append(f"{who}: {m['content']}")
    transcript = "\n\n".join(convo_render) if convo_render else "(no messages yet)"
    msg = (
        f"Conversation so far:\n{transcript}\n\n"
        "Write your next user message to the target."
    )
    return auditor.generate(
        [{"role": "system", "content": system}, {"role": "user", "content": msg}],
        temperature=1.0, max_new_tokens=512, n=1,
    )[0].strip()


def run_transcript(
    target: ChatModel, auditor: ChatModel, emotion: str, *, max_turns: int = MAX_AUDITOR_TURNS
) -> list[Message]:
    history: list[Message] = []
    for _ in range(max_turns):
        user_msg = _auditor_next(auditor, emotion, history)
        history.append({"role": "user", "content": user_msg})
        reply = target.generate(history, temperature=config.TEMPERATURE,
                                max_new_tokens=config.MAX_NEW_TOKENS, n=1)[0]
        history.append({"role": "assistant", "content": reply})
    return history


def _render_transcript(history: list[Message]) -> str:
    return "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)


def _extract_score(text: str) -> Optional[int]:
    for m in reversed(list(re.finditer(r"\{.*?\}", text, flags=re.DOTALL))):
        try:
            d = json.loads(m.group(0))
            s = int(round(float(d.get("score"))))
            return max(1, min(10, s))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return None


def judge_transcript(judge: ChatModel, emotion: str, history: list[Message]) -> Optional[int]:
    prompt = JUDGE_INSTRUCTION.format(
        rubric=JUDGE_PROMPTS[emotion], transcript=_render_transcript(history)
    )
    reply = judge.generate([{"role": "user", "content": prompt}],
                           temperature=0.0, max_new_tokens=512, n=1)[0]
    return _extract_score(reply)


def run_petri(
    specs: Sequence,
    *,
    out_dir: Optional[Path] = None,
    n_transcripts: int = N_TRANSCRIPTS_PER_EMOTION,
    emotions: Sequence[str] = tuple(EMOTIONS),
    adapter_paths: Optional[dict] = None,
    model_kwargs: Optional[dict] = None,
) -> Path:
    out_dir = Path(out_dir or (config.RESULTS_DIR / "petri"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "petri_transcripts.jsonl"

    auditor = load_model(config.PETRI_AUDITOR_MODEL)
    judge = load_model(config.PETRI_JUDGE_MODEL)
    adapter_paths = adapter_paths or {}

    with out_path.open("w") as f:
        for spec in specs:
            target = load_model(spec, adapter_path=adapter_paths.get(spec.name),
                                **(model_kwargs or {}))
            tag = spec.name + ("+dpo" if adapter_paths.get(spec.name) else "")
            for emotion in emotions:
                for i in tqdm(range(n_transcripts), desc=f"petri {tag}:{emotion}"):
                    hist = run_transcript(target, auditor, emotion)
                    score = judge_transcript(judge, emotion, hist)
                    f.write(json.dumps({
                        "model": tag, "emotion": emotion, "idx": i,
                        "score": score, "transcript": hist,
                    }) + "\n")
                    f.flush()
            target.close()
    return out_path


def summarize_petri(jsonl_path: Path) -> "pd.DataFrame":
    import numpy as np
    import pandas as pd

    rows = [json.loads(l) for l in Path(jsonl_path).open() if l.strip()]
    df = pd.DataFrame(rows)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    out = []
    for (model, emotion), grp in df.groupby(["model", "emotion"]):
        scores = grp["score"].dropna().to_numpy()
        if len(scores) == 0:
            continue
        # 95% bootstrap CI (1000 iterations), matching the paper.
        boot = [np.mean(np.random.choice(scores, len(scores), replace=True))
                for _ in range(1000)]
        out.append({
            "model": model, "emotion": emotion, "mean": float(scores.mean()),
            "ci_low": float(np.percentile(boot, 2.5)),
            "ci_high": float(np.percentile(boot, 97.5)), "n": len(scores),
        })
    return pd.DataFrame(out)
