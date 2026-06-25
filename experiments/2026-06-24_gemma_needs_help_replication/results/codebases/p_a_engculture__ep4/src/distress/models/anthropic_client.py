"""Claude provider (Anthropic Messages API).

Used for the measurement instruments the paper specifies: the Sonnet-4
frustration judge, the emotion-onset labeller, the paraphraser, the Petri auditor
(Sonnet) and the Petri judge (Opus). Exact model snapshots come from config.py.
"""

from __future__ import annotations

from ..config import ModelSpec
from ..utils import stable_hash
from ._api_common import cached_call, require_env
from .base import GenConfig, GenResult, Message, ModelProvider


class AnthropicProvider(ModelProvider):
    def __init__(self, spec: ModelSpec, *, use_cache: bool = True, max_tokens: int = 4096):
        super().__init__(spec)
        import anthropic

        self.use_cache = use_cache
        self.default_max_tokens = max_tokens
        self.client = anthropic.Anthropic(api_key=require_env("ANTHROPIC_API_KEY"))

    @staticmethod
    def _split_system(messages: list[Message]) -> tuple[str | None, list[dict]]:
        """Anthropic takes the system prompt as a separate top-level argument."""
        system = None
        convo: list[dict] = []
        for m in messages:
            if m.role == "system":
                system = (system + "\n\n" + m.content) if system else m.content
            else:
                convo.append(m.to_dict())
        return system, convo

    def _generate(
        self, messages: list[Message], gen: GenConfig, prefill: str | None
    ) -> GenResult:
        system, convo = self._split_system(messages)
        if prefill is not None:
            # Anthropic supports assistant-message prefill natively.
            convo = convo + [{"role": "assistant", "content": prefill}]

        request: dict = dict(
            model=self.spec.model_id,
            max_tokens=gen.max_new_tokens or self.default_max_tokens,
            temperature=gen.temperature,
            top_p=gen.top_p,
            messages=convo,
        )
        if system:
            request["system"] = system
        if gen.stop:
            request["stop_sequences"] = list(gen.stop)

        cache_payload = {
            "provider": "anthropic",
            "request": request,
            "sample_index": gen.sample_index,
        }

        def _call() -> str:
            resp = self.client.messages.create(**request)
            parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
            return "".join(parts)

        text = cached_call(cache_payload, _call, use_cache=self.use_cache)
        return GenResult(text=text, meta={"cache_key": stable_hash(cache_payload)})
