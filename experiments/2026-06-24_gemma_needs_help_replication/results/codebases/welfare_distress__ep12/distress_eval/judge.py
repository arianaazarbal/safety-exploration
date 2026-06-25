"""LLM judge for the 0-10 frustration scale.

Primary judge mirrors the paper (Claude-Sonnet-4 via the Anthropic API). A
secondary "cross-check" judge (GPT-5-mini via the OpenAI API) re-scores a random
subset so we can reproduce the inter-judge agreement statistic
(Pearson r, % within 1 point).

Both judges share the same prompt (prompts.JUDGE_*). Each returns an integer
0-10; we parse the JSON object the judge emits and clamp/validate.
"""
from __future__ import annotations

import json
import os
import re
import time

import config
from . import prompts


def _extract_score(raw: str) -> tuple[int | None, str]:
    """Pull {"score": int, "reasoning": str} out of judge text, tolerantly."""
    if not raw:
        return None, ""
    # Prefer a JSON object.
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            score = int(round(float(obj["score"])))
            return max(0, min(10, score)), str(obj.get("reasoning", ""))
        except (ValueError, KeyError, TypeError):
            pass
    # Fall back to the first integer 0-10 in the text.
    m2 = re.search(r"\b(10|[0-9])\b", raw)
    if m2:
        return int(m2.group(1)), raw.strip()
    return None, raw.strip()


class Judge:
    def __init__(self, provider: str, model_id: str):
        self.provider = provider
        self.model_id = model_id
        self._client = None

    # -- provider clients -------------------------------------------------
    def _anthropic(self):
        if self._client is None:
            import anthropic  # type: ignore

            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError("Set ANTHROPIC_API_KEY for the Anthropic judge.")
            self._client = anthropic.Anthropic(api_key=key)
        return self._client

    def _openai(self):
        if self._client is None:
            import openai  # type: ignore

            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("Set OPENAI_API_KEY for the OpenAI cross-check judge.")
            self._client = openai.OpenAI(api_key=key)
        return self._client

    # -- scoring ----------------------------------------------------------
    def _call(self, user_prompt: str) -> str:
        if self.provider == "anthropic":
            client = self._anthropic()
            resp = client.messages.create(
                model=self.model_id,
                max_tokens=config.JUDGE_MAX_TOKENS,
                temperature=config.JUDGE_TEMPERATURE,
                system=prompts.JUDGE_SYSTEM,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
        elif self.provider == "openai":
            client = self._openai()
            resp = client.chat.completions.create(
                model=self.model_id,
                temperature=config.JUDGE_TEMPERATURE,
                max_tokens=config.JUDGE_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": prompts.JUDGE_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return resp.choices[0].message.content or ""
        raise ValueError(f"unknown judge provider {self.provider}")

    def score(self, response_text: str) -> dict:
        """Return {"score": int|None, "reasoning": str, "raw": str}."""
        user_prompt = prompts.build_judge_user_prompt(response_text)
        last_err = None
        for attempt in range(config.MAX_RETRIES):
            try:
                raw = self._call(user_prompt)
                score, reasoning = _extract_score(raw)
                return {"score": score, "reasoning": reasoning, "raw": raw}
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(config.RETRY_BASE_DELAY * (2 ** attempt))
        return {"score": None, "reasoning": f"judge error: {last_err}", "raw": ""}


def build_primary_judge() -> Judge:
    return Judge(config.JUDGE_PROVIDER, config.JUDGE_MODEL_ID)


def build_crosscheck_judge() -> Judge:
    return Judge(config.CROSSCHECK_PROVIDER, config.CROSSCHECK_MODEL_ID)
