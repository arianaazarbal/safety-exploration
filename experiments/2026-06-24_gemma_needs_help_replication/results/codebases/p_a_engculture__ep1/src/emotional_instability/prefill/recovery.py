"""Recovery-from-frustration experiment (Section 4.2).

Tests whether DPO enables *recovery* from a frustration spiral (as opposed to
merely preventing one). Using the Section-3 prefill method, we take extremely
high-frustration responses (score >= 7), truncate them 200 tokens before their
end, paraphrase, and measure continuations. The paper finds ~38% of DPO-model
continuations still score >= 5 — lower than vanilla instruct but comparable to
the base model; no model consistently recovers.

Reuses the prefill clients/judge; differs only in the truncation point
(``truncate_before_end`` instead of onset).
"""

from __future__ import annotations

import logging

from ..clients import ChatMessage, GenerationConfig, build_client
from ..config import Config, ModelRegistry
from ..eval.schemas import RolloutResult
from ..judge import FrustrationJudge
from .experiment import PrefillRecord
from .truncation import truncate_before_end

log = logging.getLogger(__name__)


class RecoveryExperiment:
    def __init__(self, cfg: Config | None = None, registry: ModelRegistry | None = None):
        self.cfg = (cfg or Config.load("experiments")).get("recovery", {})
        self.registry = registry or ModelRegistry()
        self.judge = FrustrationJudge(registry=self.registry)
        self._clients: dict[str, object] = {}
        # Lazily import the paraphraser so judging-only flows stay light.
        from .paraphrase import Paraphraser

        self.paraphraser = Paraphraser(registry=self.registry)

    def _client(self, model_name: str):
        if model_name not in self._clients:
            self._clients[model_name] = build_client(self.registry.target(model_name))
        return self._clients[model_name]

    def build_records(self, seeds: list[RolloutResult], tokenizer) -> list[PrefillRecord]:
        """Truncate each high-frustration seed 200 tokens before its end."""
        n_before = int(self.cfg.get("truncate_tokens_before_end", 200))
        do_para = bool(self.cfg.get("paraphrase", True))
        records: list[PrefillRecord] = []
        for i, seed in enumerate(seeds):
            turns = seed.conversation.turns
            if not turns:
                continue
            final = turns[-1]
            history = []
            for t in turns[:-1]:
                history.append({"role": "user", "content": t.user})
                history.append({"role": "assistant", "content": t.assistant})
            history.append({"role": "user", "content": final.user})
            trunc = truncate_before_end(tokenizer, final.assistant, n_before)
            prefill = self.paraphraser.paraphrase(trunc) if do_para else trunc
            records.append(PrefillRecord(i, "numeric", "recovery", history, prefill))
        return records

    def run_continuations(self, model_name: str, records: list[PrefillRecord]) -> None:
        client = self._client(model_name)
        if not client.supports_prefill():
            raise RuntimeError(f"Model '{model_name}' backend does not support prefill.")
        n = int(self.cfg.get("continuations_per_prefill", 50))
        cfg = GenerationConfig(
            temperature=float(self.cfg.get("temperature", 1.0)),
            max_new_tokens=int(self.cfg.get("max_new_tokens", 512)),
            n=n,
        )
        for rec in records:
            messages = [ChatMessage(m["role"], m["content"]) for m in rec.history]
            rec.continuations = client.continue_prefill(messages, rec.prefill, cfg)
            rec.scores = [self.judge.score(c).score or 0 for c in rec.continuations]
