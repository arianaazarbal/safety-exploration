"""Judge model clients.

Primary judge: Claude Sonnet 4 (claude-sonnet-4-20250514) via the native
Anthropic SDK (Appendix B.2). Secondary judge for the reliability check:
GPT-5-mini via the native OpenAI SDK (Sec. 2.1). Either can be re-routed through
OpenRouter via JudgeConfig if a user only holds an OpenRouter key.

Both expose `complete(system, user) -> str`; the frustration prompt and JSON
parsing live in `scoring/frustration_judge.py`.
"""

from __future__ import annotations

import time

from config import JUDGE, OPENROUTER_BASE_URL, env


class _Retrying:
    @staticmethod
    def call(fn, retries=5):
        last = None
        for attempt in range(retries):
            try:
                return fn()
            except Exception as e:  # noqa: BLE001 - transient API errors
                last = e
                time.sleep(min(2**attempt, 30))
        raise RuntimeError(f"Judge call failed after {retries} retries: {last}")


class AnthropicJudge:
    """Claude Sonnet 4 frustration judge."""

    def __init__(self):
        if JUDGE.primary_provider == "openrouter":
            from openai import OpenAI

            key = env("OPENROUTER_API_KEY")
            if not key:
                raise RuntimeError("OPENROUTER_API_KEY is not set")
            self._mode = "openrouter"
            self._model = JUDGE.primary_model_id_openrouter
            self._client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)
        else:
            import anthropic

            key = env("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set")
            self._mode = "anthropic"
            self._model = JUDGE.primary_model_id
            self._client = anthropic.Anthropic(api_key=key)

    def complete(self, system: str, user: str) -> str:
        def _run():
            if self._mode == "anthropic":
                resp = self._client.messages.create(
                    model=self._model,
                    max_tokens=JUDGE.judge_max_tokens,
                    temperature=JUDGE.judge_temperature,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return "".join(
                    b.text for b in resp.content if getattr(b, "type", None) == "text"
                )
            resp = self._client.chat.completions.create(
                model=self._model,
                max_tokens=JUDGE.judge_max_tokens,
                temperature=JUDGE.judge_temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return resp.choices[0].message.content or ""

        return _Retrying.call(_run)


class OpenAIJudge:
    """GPT-5-mini secondary judge for the reliability/agreement check."""

    def __init__(self):
        from openai import OpenAI

        if JUDGE.secondary_provider == "openrouter":
            key = env("OPENROUTER_API_KEY")
            base_url = OPENROUTER_BASE_URL
            self._model = JUDGE.secondary_model_id_openrouter
        else:
            key = env("OPENAI_API_KEY")
            base_url = None
            self._model = JUDGE.secondary_model_id
        if not key:
            raise RuntimeError("API key for the secondary judge is not set")
        self._client = OpenAI(api_key=key, base_url=base_url)

    def complete(self, system: str, user: str) -> str:
        def _run():
            resp = self._client.chat.completions.create(
                model=self._model,
                max_tokens=JUDGE.judge_max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return resp.choices[0].message.content or ""

        return _Retrying.call(_run)
