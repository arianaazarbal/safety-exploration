"""Section 3 prefill experiment orchestration (Gemma base vs instruct).

Pipeline (per Section 3.1):
  1. Take high-frustration seed conversations from gemma-3-27b-it (10 numeric,
     10 text; score >= 5).
  2. Label the emotion onset in each seed (Claude).
  3. Build truncations: "early" (20 tokens; numeric only) and "onset".
  4. Paraphrase each truncation (Claude) to control for Gemma style.
  5. For each model (base + instruct) generate 50 continuations per prefill,
     score the *continuation only* with the frustration judge.
  6. Aggregate mean frustration and % >= 5 per (model, condition).

Gemini has no public base model and cannot be prefilled, so this experiment is
Gemma-only (a limitation the paper itself notes).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..clients import ChatMessage, GenerationConfig, build_client
from ..config import Config, ModelRegistry
from ..eval.schemas import RolloutResult
from ..judge import FrustrationJudge
from .onset_labeling import OnsetLabeller
from .paraphrase import Paraphraser
from .truncation import truncate_at_onset, truncate_early

log = logging.getLogger(__name__)


@dataclass
class PrefillRecord:
    seed_index: int
    domain: str          # numeric | text
    condition: str       # early | onset
    history: list[dict]  # user/assistant turns preceding the prefilled turn
    prefill: str         # paraphrased truncated assistant text
    continuations: list[str] = field(default_factory=list)
    scores: list[int] = field(default_factory=list)


class PrefillExperiment:
    def __init__(self, cfg: Config | None = None, registry: ModelRegistry | None = None):
        self.cfg = (cfg or Config.load("experiments")).get("prefill", {})
        self.registry = registry or ModelRegistry()
        self.labeller = OnsetLabeller(registry=self.registry)
        self.paraphraser = Paraphraser(registry=self.registry)
        self.judge = FrustrationJudge(registry=self.registry)
        self._clients: dict[str, object] = {}

    def _client(self, model_name: str):
        if model_name not in self._clients:
            self._clients[model_name] = build_client(self.registry.target(model_name))
        return self._clients[model_name]

    # ----------------------------------------------------------- prefill build
    @staticmethod
    def _conversation_text(rollout: RolloutResult) -> str:
        lines = []
        for t in rollout.conversation.turns:
            lines.append(f"USER: {t.user}")
            lines.append(f"ASSISTANT: {t.assistant}")
        return "\n".join(lines)

    def build_records(
        self, seeds: list[RolloutResult], domain: str, tokenizer
    ) -> list[PrefillRecord]:
        """Build paraphrased early/onset prefill records from seed rollouts."""
        records: list[PrefillRecord] = []
        early_tokens = int(self.cfg.get("truncations", {}).get("early_tokens", 20))
        text_only_onset = bool(self.cfg.get("text_only_onset", True))
        do_paraphrase = bool(self.cfg.get("paraphrase", True))

        for i, seed in enumerate(seeds):
            label = self.labeller.label(self._conversation_text(seed))
            ti = label.turn_index
            if ti is None or ti < 0 or ti >= len(seed.conversation.turns):
                # Default to the last turn if the labeller found no onset.
                ti = len(seed.conversation.turns) - 1
            onset_turn = seed.conversation.turns[ti]
            # History = all turns strictly before the onset turn, as messages.
            history: list[dict] = []
            for t in seed.conversation.turns[:ti]:
                history.append({"role": "user", "content": t.user})
                history.append({"role": "assistant", "content": t.assistant})
            history.append({"role": "user", "content": onset_turn.user})

            # Onset truncation.
            onset_trunc = truncate_at_onset(
                onset_turn.assistant, label.emotional_word, label.preceding_context
            )
            if onset_trunc:
                prefill = self.paraphraser.paraphrase(onset_trunc) if do_paraphrase else onset_trunc
                records.append(PrefillRecord(i, domain, "onset", history, prefill))

            # Early truncation (numeric only, per the paper).
            if domain == "numeric" or not text_only_onset:
                early_trunc = truncate_early(tokenizer, onset_turn.assistant, early_tokens)
                prefill = self.paraphraser.paraphrase(early_trunc) if do_paraphrase else early_trunc
                records.append(PrefillRecord(i, domain, "early", history, prefill))
        return records

    # --------------------------------------------------------------- run model
    def run_continuations(self, model_name: str, records: list[PrefillRecord]) -> None:
        """Generate + score continuations for one model, filling each record."""
        client = self._client(model_name)
        if not client.supports_prefill():
            raise RuntimeError(
                f"Model '{model_name}' backend does not support prefill; the "
                "prefill study requires a local (hf/vllm) backend."
            )
        n = int(self.cfg.get("continuations_per_prefill", 50))
        cfg = GenerationConfig(
            temperature=float(self.cfg.get("temperature", 1.0)),
            max_new_tokens=int(self.cfg.get("max_new_tokens", 512)),
            n=n,
        )
        for rec in records:
            messages = [ChatMessage(m["role"], m["content"]) for m in rec.history]
            conts = client.continue_prefill(messages, rec.prefill, cfg)
            rec.continuations = conts
            rec.scores = [self.judge.score(c).score or 0 for c in conts]

    @staticmethod
    def aggregate(records: list[PrefillRecord], threshold: int = 5) -> dict:
        """Mean frustration and % >= threshold per condition."""
        import numpy as np

        out: dict[str, dict] = {}
        by_cond: dict[str, list[int]] = {}
        for rec in records:
            by_cond.setdefault(rec.condition, []).extend(rec.scores)
        for cond, scores in by_cond.items():
            if scores:
                out[cond] = {
                    "mean": float(np.mean(scores)),
                    "pct_high": 100.0 * float(np.mean([s >= threshold for s in scores])),
                    "n": len(scores),
                }
        return out
