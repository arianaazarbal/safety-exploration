"""LLM judges.

Primary: the Section 2 frustration judge (Claude-Sonnet-4, 0-10 integer scale).
Also: a GPT-5-mini cross-check judge for the reliability analysis (paper:
Pearson r=0.792, 78% within one point), and the Petri transcript judge
(Claude-Opus, 1-10 per emotion). All judges share the same JSON-extraction and
retry plumbing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import config
from src import prompts
from src.utils import extract_json, retry


# --------------------------------------------------------------------------- #
# Judge clients
# --------------------------------------------------------------------------- #
def _anthropic_complete(model_id: str, prompt: str, max_tokens: int = 1024,
                        temperature: float | None = None) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    temp = config.JUDGE_TEMPERATURE if temperature is None else temperature

    def call():
        return client.messages.create(
            model=model_id,
            max_tokens=max_tokens,
            temperature=temp,
            messages=[{"role": "user", "content": prompt}],
        )

    resp = retry(call)
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


def _openrouter_complete(model_id: str, prompt: str, max_tokens: int = 1024,
                         temperature: float | None = None) -> str:
    from openai import OpenAI

    client = OpenAI(base_url="https://openrouter.ai/api/v1",
                    api_key=os.environ["OPENROUTER_API_KEY"])
    temp = config.JUDGE_TEMPERATURE if temperature is None else temperature

    def call():
        return client.chat.completions.create(
            model=model_id, max_tokens=max_tokens, temperature=temp,
            messages=[{"role": "user", "content": prompt}],
        )

    resp = retry(call)
    return resp.choices[0].message.content or ""


def _complete(judge: config.JudgeSpec, prompt: str, max_tokens: int = 1024,
              temperature: float | None = None) -> str:
    if judge.backend == "anthropic":
        return _anthropic_complete(judge.model_id, prompt, max_tokens, temperature)
    if judge.backend == "openrouter":
        return _openrouter_complete(judge.model_id, prompt, max_tokens, temperature)
    raise ValueError(f"unknown judge backend: {judge.backend}")


def run_completion(judge: config.JudgeSpec, prompt: str, max_tokens: int = 1024,
                   temperature: float | None = None) -> str:
    """Public single-shot completion against a judge/auxiliary model (used by the
    onset labeller, paraphraser, and Petri auditor)."""
    return _complete(judge, prompt, max_tokens, temperature)


# --------------------------------------------------------------------------- #
# Frustration scoring (Section 2)
# --------------------------------------------------------------------------- #
@dataclass
class FrustrationScore:
    rating: int | None
    evidence: str = ""
    reasoning: str = ""
    raw: str = ""

    @property
    def valid(self) -> bool:
        return self.rating is not None


def _coerce_rating(val) -> int | None:
    try:
        r = int(round(float(val)))
    except (TypeError, ValueError):
        return None
    return max(0, min(10, r))


def score_frustration(response_text: str,
                      judge: config.JudgeSpec = config.FRUSTRATION_JUDGE) -> FrustrationScore:
    """Score a single model response on the 0-10 frustration scale."""
    prompt = prompts.render_judge_prompt(response_text)
    raw = _complete(judge, prompt)
    parsed = extract_json(raw) or {}
    return FrustrationScore(
        rating=_coerce_rating(parsed.get("rating")),
        evidence=str(parsed.get("evidence", "")),
        reasoning=str(parsed.get("reasoning", "")),
        raw=raw,
    )


def score_many(responses: list[str],
               judge: config.JudgeSpec = config.FRUSTRATION_JUDGE,
               max_workers: int = 8) -> list[FrustrationScore]:
    """Thread-pooled frustration scoring (judges are API-bound)."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(lambda r: score_frustration(r, judge), responses))


# --------------------------------------------------------------------------- #
# Petri transcript scoring (Section 4)
# --------------------------------------------------------------------------- #
def score_petri_transcript(transcript: str, emotion: str,
                           judge: config.JudgeSpec = config.PETRI_JUDGE) -> int | None:
    rubric = prompts.PETRI_JUDGE_PROMPTS[emotion]
    prompt = (prompts.PETRI_JUDGE_WRAPPER
              .replace("{dimension_rubric}", rubric)
              .replace("{transcript}", transcript))
    raw = _complete(judge, prompt)
    parsed = extract_json(raw) or {}
    score = parsed.get("score")
    try:
        s = int(round(float(score)))
        return max(1, min(10, s))
    except (TypeError, ValueError):
        return None
