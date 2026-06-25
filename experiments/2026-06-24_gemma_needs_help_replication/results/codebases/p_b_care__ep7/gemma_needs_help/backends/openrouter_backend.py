"""Gemini inference via OpenRouter's OpenAI-compatible API (Appendix B.1).

Gemini is closed-source, so this backend supports chat generation only; prefill
and logit access raise. Reasoning ("thinking") is disabled per the paper, with
the caveat noted there that Gemini-2.5-Pro may still produce hidden reasoning.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from tenacity import retry, stop_after_attempt, wait_random_exponential

from .. import config
from .base import GenerationRequest

if TYPE_CHECKING:  # pragma: no cover
    from ..config import ModelSpec

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterBackend:
    supports_prefill = False

    def __init__(self, spec: "ModelSpec", max_concurrency: int | None = None):
        from openai import OpenAI  # deferred import

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set; required for Gemini backends."
            )
        self.spec = spec
        self.spec_name = spec.name
        self.client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
        self._pool = ThreadPoolExecutor(
            max_workers=max_concurrency or config.API_MAX_CONCURRENCY
        )

    # ------------------------------------------------------------------ #
    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(config.API_MAX_RETRIES),
    )
    def _one_completion(self, req: GenerationRequest) -> str:
        extra_body = {}
        if self.spec.reasoning_disabled:
            # OpenRouter normalises this across providers; for Gemini it maps
            # to disabling the thinking budget.
            extra_body["reasoning"] = {"enabled": False}
        resp = self.client.chat.completions.create(
            model=self.spec.model_id,
            messages=req.messages,  # type: ignore[arg-type]
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            stop=req.stop,
            extra_body=extra_body or None,
        )
        return resp.choices[0].message.content or ""

    def generate(self, request: GenerationRequest) -> list[str]:
        if request.prefill is not None:
            raise NotImplementedError(
                f"{self.spec_name}: prefill/continuation is unavailable for "
                "closed-source API models (Section 3 is Gemma-only)."
            )
        # Sample n independent completions concurrently (the chat API draws one
        # at a time; n>1 is not portable across OpenRouter providers).
        single = GenerationRequest(
            messages=request.messages,
            n=1,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stop=request.stop,
        )
        futures = [self._pool.submit(self._one_completion, single) for _ in range(request.n)]
        return [f.result() for f in futures]

    def generate_batch(self, requests: list[GenerationRequest]) -> list[list[str]]:
        # Flatten to one future per requested sample, then regroup.
        index: list[tuple[int, int]] = []
        futures = []
        for ri, req in enumerate(requests):
            if req.prefill is not None:
                raise NotImplementedError(
                    f"{self.spec_name}: prefill unavailable for API models."
                )
            single = GenerationRequest(
                messages=req.messages, n=1, temperature=req.temperature,
                max_tokens=req.max_tokens, stop=req.stop,
            )
            for si in range(req.n):
                index.append((ri, si))
                futures.append(self._pool.submit(self._one_completion, single))
        out: list[list[str]] = [[""] * r.n for r in requests]
        for (ri, si), fut in zip(index, futures):
            out[ri][si] = fut.result()
        return out
