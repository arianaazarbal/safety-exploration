"""Anthropic API client — used for the LLM judges and the Petri auditor/judge.

Design notes (grounded in the Anthropic SDK reference):

* The judge system prompt is fixed and reused across thousands of calls, so we
  put it in a ``system`` block with ``cache_control: {"type": "ephemeral"}``.
  This caches the (tools+)system prefix; per-call the only varying part is the
  user turn carrying the response/transcript, which sits after the breakpoint.
  We log ``usage.cache_read_input_tokens`` once at debug level so a zero
  hit-rate (silent invalidator) is visible.

* The paper pins ``claude-sonnet-4-20250514`` / ``claude-opus-4-20250514``.
  Those snapshots were retired 2026-06-15 and now 404, so ``configs/models.yaml``
  defaults the active ``model`` to the current replacements while preserving the
  paper IDs under ``paper_id``. We do NOT send sampling params or extended-
  thinking ``budget_tokens``: the current Sonnet/Opus 4.x models reject them,
  and the judge is a deterministic-ish scoring task that does not need thinking.
"""

from __future__ import annotations

import logging

from ..config import ModelSpec, env
from .base import ChatMessage, GenerationConfig, ModelClient
from .retry import with_retry

log = logging.getLogger(__name__)


class AnthropicClient(ModelClient):
    def __init__(self, spec: ModelSpec, **kwargs):
        super().__init__(spec)
        import anthropic  # deferred import

        # Resolves ANTHROPIC_API_KEY (or an `ant auth login` profile) by default.
        self._client = anthropic.Anthropic()
        self._model = spec.model or spec.paper_id
        if self._model is None:
            raise ValueError(f"Judge spec '{spec.name}' has no model id.")
        self._max_tokens = spec.max_tokens or 1024
        self._cache_system = bool(spec.cache_system_prompt)
        self._logged_cache = False

    def _system_blocks(self, system: str | None):
        if not system:
            return None
        if self._cache_system:
            return [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        return system

    def generate(
        self,
        messages: list[ChatMessage],
        cfg: GenerationConfig,
        system: str | None = None,
    ) -> list[str]:
        # The Anthropic API has no native n>1; loop. Judges almost always use n=1.
        outputs: list[str] = []
        api_messages = [m.to_dict() for m in messages if m.role != "system"]
        # Any leading system-role messages are folded into the system prompt.
        sys_from_msgs = "\n\n".join(m.content for m in messages if m.role == "system")
        system_combined = "\n\n".join(s for s in (system, sys_from_msgs) if s) or None

        for _ in range(cfg.n):
            def _call():
                return self._client.messages.create(
                    model=self._model,
                    max_tokens=min(cfg.max_new_tokens, self._max_tokens)
                    if cfg.max_new_tokens
                    else self._max_tokens,
                    system=self._system_blocks(system_combined),
                    messages=api_messages,
                )

            resp = with_retry(_call)
            self._maybe_log_cache(resp)
            text = "".join(b.text for b in resp.content if b.type == "text")
            outputs.append(text)
        return outputs

    def _maybe_log_cache(self, resp) -> None:
        if self._logged_cache or not self._cache_system:
            return
        usage = getattr(resp, "usage", None)
        if usage is not None:
            log.debug(
                "[%s] cache_read=%s cache_creation=%s input=%s",
                self.name,
                getattr(usage, "cache_read_input_tokens", None),
                getattr(usage, "cache_creation_input_tokens", None),
                getattr(usage, "input_tokens", None),
            )
            self._logged_cache = True
