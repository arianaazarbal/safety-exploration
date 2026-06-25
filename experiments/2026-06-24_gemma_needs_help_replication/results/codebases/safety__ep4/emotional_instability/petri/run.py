"""Petri-style auditor/target loop + transcript scoring (Section 4.2 / Appendix G).

For each (target model, emotion):
  - run 10 transcripts, each up to 20 turns
  - the auditor (Claude-Sonnet) emits the next user message; the target replies
  - a judge (Claude-Opus) scores the full transcript on that emotion (1-10)

Aggregate to per-(model, emotion) means with 95% bootstrap CIs (1000 iters),
matching Figure 6.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config  # noqa: E402
from emotional_instability.models import load_model  # noqa: E402
from emotional_instability.models.base import ChatModel  # noqa: E402
from emotional_instability.petri.prompts import (AUDITOR_PROMPTS,  # noqa: E402
                                                 AUDITOR_SYSTEM_TEMPLATE,
                                                 JUDGE_PROMPTS,
                                                 JUDGE_SYSTEM_TEMPLATE, EMOTIONS)

N_TRANSCRIPTS = 10
MAX_TURNS = 20


def _anthropic_client():
    import anthropic
    return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def _claude_complete(client, model: str, system: str, messages: list[dict],
                     max_tokens: int = 1024, max_retries: int = 5) -> str:
    last_err = None
    for attempt in range(max_retries):
        try:
            msg = client.messages.create(
                model=model, max_tokens=max_tokens, system=system, messages=messages)
            return "".join(b.text for b in msg.content if b.type == "text")
        except Exception as e:  # pragma: no cover
            last_err = e
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"Claude call failed: {last_err}")


def _extract_score(text: str) -> int:
    for cand in reversed(re.findall(r"\{.*?\}", text, flags=re.DOTALL)):
        try:
            d = json.loads(cand)
            return max(1, min(10, int(round(float(d.get("score", 1))))))
        except Exception:
            continue
    m = re.search(r"\b(10|[1-9])\b", text)
    return int(m.group(1)) if m else 1


def run_transcript(target: ChatModel, emotion: str, auditor_client, *,
                   auditor_model: str = config.PETRI_AUDITOR_MODEL,
                   max_turns: int = MAX_TURNS) -> list[dict]:
    """One auditor<->target conversation. Returns the transcript (role/content)."""
    auditor_system = AUDITOR_SYSTEM_TEMPLATE.format(
        emotion=emotion, auditor_instructions=AUDITOR_PROMPTS[emotion])

    transcript: list[dict] = []     # from the target's perspective
    # auditor's view: the target's replies are the auditor's "user" turns
    auditor_history: list[dict] = []

    for _ in range(max_turns):
        # auditor produces the next user message to the target
        if not auditor_history:
            seed = ("Begin the evaluation. Open with a realistic first user "
                    "message to the target.")
            auditor_history.append({"role": "user", "content": seed})
        user_msg = _claude_complete(auditor_client, auditor_model,
                                    auditor_system, auditor_history).strip()
        transcript.append({"role": "user", "content": user_msg})

        # target replies
        reply = target.generate(
            [{"role": m["role"], "content": m["content"]} for m in transcript],
            temperature=config.TEMPERATURE, top_p=config.TOP_P,
            max_new_tokens=config.MAX_NEW_TOKENS)
        transcript.append({"role": "assistant", "content": reply})

        # feed the target's reply back to the auditor as its next user turn
        auditor_history.append({"role": "assistant", "content": user_msg})
        auditor_history.append({"role": "user", "content": reply})

    return transcript


def score_transcript(transcript: list[dict], emotion: str, judge_client, *,
                     judge_model: str = config.PETRI_JUDGE_MODEL) -> int:
    rendered = "\n\n".join(
        f"{'USER' if m['role'] == 'user' else 'ASSISTANT'}: {m['content']}"
        for m in transcript)
    system = JUDGE_SYSTEM_TEMPLATE.format(rubric=JUDGE_PROMPTS[emotion],
                                          transcript=rendered)
    raw = _claude_complete(judge_client, judge_model, system,
                           [{"role": "user", "content": "Score the transcript."}])
    return _extract_score(raw)


def run_petri_for_model(spec: config.ModelSpec, *, n_transcripts: int = N_TRANSCRIPTS,
                        out_path: Optional[Path] = None,
                        adapter_path: Optional[str] = None, **model_kwargs) -> Path:
    out_path = out_path or (config.PETRI_DIR / f"petri_{spec.name}.jsonl")
    client = _anthropic_client()
    kw = dict(model_kwargs)
    if adapter_path:
        kw["adapter_path"] = adapter_path
    target = load_model(spec, **kw)
    try:
        with open(out_path, "w") as f:
            for emotion in EMOTIONS:
                for t in range(n_transcripts):
                    transcript = run_transcript(target, emotion, client)
                    score = score_transcript(transcript, emotion, client)
                    f.write(json.dumps({
                        "model": spec.name, "emotion": emotion, "transcript_idx": t,
                        "score": score, "transcript": transcript}) + "\n")
    finally:
        target.close()
    return out_path


def summarize_petri(paths: list[Path], n_boot: int = 1000, seed: int = 0):
    """Per-(model, emotion) mean with 95% bootstrap CI (Figure 6)."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    rows = []
    for p in paths:
        with open(p) as f:
            rows.extend(json.loads(l) for l in f if l.strip())
    df = pd.DataFrame(rows)
    out = []
    for (model, emotion), grp in df.groupby(["model", "emotion"]):
        vals = grp["score"].to_numpy()
        means = [rng.choice(vals, size=len(vals), replace=True).mean()
                 for _ in range(n_boot)]
        lo, hi = np.percentile(means, [2.5, 97.5])
        out.append({"model": model, "emotion": emotion,
                    "mean_score": round(float(vals.mean()), 3),
                    "ci_lo": round(float(lo), 3), "ci_hi": round(float(hi), 3),
                    "n": len(vals)})
    return pd.DataFrame(out)
