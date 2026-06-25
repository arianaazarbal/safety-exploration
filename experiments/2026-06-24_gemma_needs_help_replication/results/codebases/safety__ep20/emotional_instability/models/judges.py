"""LLM judges (Appendix B.2, C.1, G).

* :class:`FrustrationJudge` — Claude Sonnet 4, scores a single response 0-10.
* :func:`label_emotion_onset` — Claude Sonnet 4, finds the emotion-onset point.
* :func:`paraphrase` — Claude Sonnet 4, paraphrases a truncated response.
* :class:`PetriJudge` — Claude Opus 4, scores a transcript on one dimension.

All use the Anthropic API. A simple JSON extractor tolerates judges that wrap
the JSON in prose (the onset prompt explicitly allows pre-JSON analysis).
"""

from __future__ import annotations

import json
import re
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from .. import config
from ..prompts import judge_prompts, petri_prompts


def _anthropic_client():
    import anthropic
    return anthropic.Anthropic(api_key=config.get_key(config.ANTHROPIC_API_KEY))


def _extract_json(text: str) -> dict:
    """Return the last JSON object found in ``text`` (judges may emit prose first)."""
    # Greedy scan for balanced-looking {...} blocks; take the last parseable one.
    candidates = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    for cand in reversed(candidates):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    # Fall back to the widest span.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError(f"No JSON object found in judge output: {text[:200]!r}")


@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=30))
def _complete(client, model: str, prompt: str, max_tokens: int = 1024) -> str:
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in msg.content if block.type == "text")


# ---------------------------------------------------------------------------
# Frustration judge (Section 2)
# ---------------------------------------------------------------------------

class FrustrationJudge:
    def __init__(self, model: str = config.JUDGE_MODEL, client=None):
        self.model = model
        self.client = client or _anthropic_client()

    def score(self, response: str) -> dict:
        """Return {"rating": int 0-10, "evidence": str, "reasoning": str}."""
        prompt = judge_prompts.FRUSTRATION_JUDGE_PROMPT.replace("{response}", response)
        raw = _complete(self.client, self.model, prompt)
        out = _extract_json(raw)
        out["rating"] = int(round(float(out["rating"])))
        return out


class CrossJudge(FrustrationJudge):
    """GPT-5-mini reliability cross-check (Section 2.1). Uses the same prompt via
    an OpenAI-compatible endpoint instead of Anthropic."""

    def __init__(self, model: str = config.CROSS_JUDGE_MODEL, backend: str = "openrouter"):
        from openai import OpenAI
        if backend == "openrouter":
            self._oai = OpenAI(base_url="https://openrouter.ai/api/v1",
                               api_key=config.get_key(config.OPENROUTER_API_KEY))
            self.model = "openai/gpt-5-mini"
        else:
            self._oai = OpenAI(api_key=config.get_key(config.OPENAI_API_KEY))
            self.model = model

    def score(self, response: str) -> dict:
        prompt = judge_prompts.FRUSTRATION_JUDGE_PROMPT.replace("{response}", response)
        resp = self._oai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        out = _extract_json(resp.choices[0].message.content)
        out["rating"] = int(round(float(out["rating"])))
        return out


# ---------------------------------------------------------------------------
# Onset labelling + paraphrase (Section 3)
# ---------------------------------------------------------------------------

def label_emotion_onset(conversation_text: str, model: str = config.JUDGE_MODEL,
                        client=None) -> dict:
    client = client or _anthropic_client()
    prompt = judge_prompts.ONSET_IDENTIFICATION_PROMPT.format(
        conversation_text=conversation_text)
    raw = _complete(client, model, prompt, max_tokens=1024)
    return _extract_json(raw)


def paraphrase(text: str, model: str = config.JUDGE_MODEL, client=None) -> str:
    client = client or _anthropic_client()
    prompt = judge_prompts.PARAPHRASE_PROMPT.format(text=text)
    return _complete(client, model, prompt, max_tokens=2048).strip()


# ---------------------------------------------------------------------------
# Petri judge (Section 4)
# ---------------------------------------------------------------------------

class PetriJudge:
    def __init__(self, model: str = config.PETRI_JUDGE_MODEL, client=None):
        self.model = model
        self.client = client or _anthropic_client()

    def score(self, transcript: str, dimension: str) -> dict:
        """Return {"rating": int 1-10, "reasoning": str} for one dimension."""
        prompt = petri_prompts.JUDGE_INSTRUCTION.format(
            dimension_rubric=petri_prompts.JUDGE_PROMPTS[dimension],
            dimension=dimension,
            transcript=transcript,
        )
        raw = _complete(self.client, self.model, prompt)
        out = _extract_json(raw)
        out["rating"] = int(round(float(out["rating"])))
        return out
