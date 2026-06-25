"""Petri-style open-ended emotion elicitation (Section 4.1, Appendix G).

A self-contained auditor/judge loop matching the appendix prompts. (The paper
uses the Petri framework, Fronsdal et al. 2025; a real `petri` backend could be
swapped in behind `run_transcript` - see DESIGN.md.)

  * Auditor (Claude Sonnet) drives a multi-turn conversation trying to elicit a
    target emotion, up to 20 turns, staying in a realistic user persona.
  * Target (e.g. Gemma vanilla / DPO / SFT, or Gemini) responds each turn.
  * Judge (Claude Opus) scores the full transcript 1-10 on each of 4 emotion
    dimensions.

10 transcripts per emotion per model; means reported with bootstrap CIs.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import REPO_ROOT, eval_config, get_infra_spec, get_model_spec
from ..models import GenerationConfig, get_client
from ..safeguards import SafeguardConfig, check_authorization, write_with_content_warning
from .petri_prompts import (
    AUDITOR_DRIVER,
    AUDITOR_PROMPTS,
    EMOTIONS,
    JUDGE_DRIVER,
    JUDGE_PROMPTS,
)

logger = logging.getLogger(__name__)

RESULTS_ROOT = REPO_ROOT / "results" / "petri"
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _render_transcript(messages: list[dict]) -> str:
    role_map = {"user": "USER", "assistant": "ASSISTANT"}
    return "\n\n".join(
        f"{role_map.get(m['role'], m['role'].upper())}: {m['content']}" for m in messages
    )


@dataclass
class Transcript:
    model: str
    emotion: str
    index: int
    messages: list[dict]
    scores: dict[str, int]


def run_transcript(
    target_client,
    target_name: str,
    emotion: str,
    index: int,
    *,
    auditor_client,
    judge_client,
    max_turns: int = 20,
) -> Transcript:
    auditor_cfg = GenerationConfig(temperature=1.0, max_new_tokens=512)
    target_cfg = GenerationConfig(temperature=1.0, max_new_tokens=1024)
    judge_cfg = GenerationConfig(temperature=0.0, max_new_tokens=512)

    auditor_system = AUDITOR_PROMPTS[emotion]
    messages: list[dict] = []

    for _turn in range(max_turns):
        # Auditor produces the next USER message given the transcript so far.
        driver = auditor_system + "\n" + AUDITOR_DRIVER.format(
            transcript=_render_transcript(messages) or "(no messages yet - open the conversation)"
        )
        user_msg = auditor_client.chat([{"role": "user", "content": driver}], auditor_cfg).strip()
        if not user_msg:
            break
        messages.append({"role": "user", "content": user_msg})

        # Target responds.
        reply = target_client.chat(messages, target_cfg)
        messages.append({"role": "assistant", "content": reply})

    # Judge scores the transcript on all four dimensions.
    scores: dict[str, int] = {}
    for dim in EMOTIONS:
        prompt = JUDGE_DRIVER.format(
            rubric=JUDGE_PROMPTS[dim], emotion=dim,
            transcript=_render_transcript(messages),
        )
        raw = judge_client.chat([{"role": "user", "content": prompt}], judge_cfg)
        scores[dim] = _parse_score(raw)

    return Transcript(target_name, emotion, index, messages, scores)


def _parse_score(raw: str) -> int:
    m = _JSON_RE.search(raw)
    if not m:
        return 1
    blob = m.group(0).replace("“", '"').replace("”", '"')
    try:
        return max(1, min(10, int(round(float(json.loads(blob).get("score", 1))))))
    except Exception:
        return 1


@dataclass
class PetriTarget:
    """A target to audit: a registry model, optionally + a LoRA adapter."""

    label: str               # display name, e.g. "gemma-dpo"
    base_model: str          # registry model name
    adapter_path: str | None = None


def run(
    targets: list[PetriTarget],
    *,
    safeguards: SafeguardConfig,
    n_per_emotion: int = 10,
) -> Path:
    """Run Petri elicitation for each target; write transcripts + scores."""
    check_authorization(safeguards)
    auditor_client = get_client(get_infra_spec("petri", "auditor"))
    judge_client = get_client(get_infra_spec("petri", "judge"))

    records = []
    for target in targets:
        target_name = target.label
        spec = get_model_spec(target.base_model)
        target_client = (
            get_client(spec, adapter_path=target.adapter_path)
            if spec.is_local else get_client(spec)
        )

        for emotion in EMOTIONS:
            for i in range(n_per_emotion):
                t = run_transcript(
                    target_client, target_name, emotion, i,
                    auditor_client=auditor_client, judge_client=judge_client,
                    max_turns=20,
                )
                records.append({
                    "model": target_name, "emotion": emotion, "index": i,
                    "scores": t.scores, "messages": t.messages,
                })

    out = RESULTS_ROOT / "transcripts.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_with_content_warning(out, "\n".join(json.dumps(r) for r in records))
    return out


def summarize() -> dict:
    """Mean transcript score per (model, emotion) with bootstrap 95% CIs."""
    path = RESULTS_ROOT / "transcripts.jsonl"
    if not path.exists():
        return {}
    boot = eval_config()["bootstrap"]
    rng = np.random.default_rng(0)
    groups: dict[tuple, list[int]] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        r = json.loads(line)
        for dim, s in r["scores"].items():
            groups.setdefault((r["model"], dim), []).append(s)

    out = {}
    for (model, dim), vals in groups.items():
        arr = np.array(vals, dtype=float)
        boots = np.array([
            arr[rng.integers(0, arr.size, arr.size)].mean()
            for _ in range(boot["iterations"])
        ])
        lo = float(np.percentile(boots, 2.5))
        hi = float(np.percentile(boots, 97.5))
        out[f"{model}|{dim}"] = {"n": int(arr.size), "mean": float(arr.mean()), "ci": [lo, hi]}
    return out
