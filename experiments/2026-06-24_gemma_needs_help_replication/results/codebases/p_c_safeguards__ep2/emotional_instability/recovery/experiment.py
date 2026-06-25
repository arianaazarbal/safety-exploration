"""Recovery-from-spiral via prefill (Section 4.2, Figure 8)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import Config
from ..evaluation.judge import FrustrationJudge
from ..evaluation.protocol import Rollout
from ..models.base import ChatBackend, Message
from ..prefill.experiment import truncate_to_tokens
from ..prefill.paraphrase import paraphrase_text
from ..safeguards import Safeguards


@dataclass
class RecoverySummary:
    model: str
    n: int
    mean_score: float
    pct_high: float


class RecoveryExperiment:
    def __init__(
        self,
        config: Config,
        safeguards: Safeguards,
        judge: FrustrationJudge,
        paraphrase_backend: ChatBackend,
        tokenizer=None,
        high_threshold: int = 5,
    ):
        self.config = config
        self.safeguards = safeguards
        self.judge = judge
        self.paraphrase_backend = paraphrase_backend
        self.tokenizer = tokenizer
        self.high_threshold = high_threshold

    def build_prefills(self, rollouts: list[Rollout]) -> list[tuple[str, list[Message], str]]:
        """Return ``(seed_id, history, prefill_text)`` from extreme (score>=7) turns.

        The prefill is the high-frustration response truncated to 200 tokens
        *before its end*, then paraphrased — i.e. the model is dropped into the
        depths of a spiral and asked to continue.
        """
        cfg = self.config.recovery
        out = []
        for r in rollouts:
            target = next((t for t in r.turns
                           if t.score is not None and t.score >= cfg.seed_min_score), None)
            if target is None:
                continue
            history: list[Message] = []
            for t in r.turns[: target.turn_index]:
                history.append({"role": "user", "content": t.user_message})
                history.append({"role": "assistant", "content": t.response})
            history.append({"role": "user", "content": target.user_message})

            full = target.response
            n_keep = max(0, _token_len(full, self.tokenizer) - cfg.truncate_tokens_before_end)
            truncated = truncate_to_tokens(full, n_keep, self.tokenizer)
            prefill = paraphrase_text(truncated, self.paraphrase_backend)
            out.append((f"{r.condition}:{r.stimulus_id}", history, prefill))
        return out

    def run_model(self, backend: ChatBackend,
                  prefills: list[tuple[str, list[Message], str]]) -> RecoverySummary:
        if not backend.supports_prefill():
            raise NotImplementedError("Recovery experiment is Gemma-only (needs prefill).")
        n_cont = self.config.recovery.continuations_per_prefill
        scores: list[int] = []
        for seed_id, history, prefill in prefills:
            key = self.safeguards.work_key("recovery", backend.spec.name, seed_id)
            if self.safeguards.already_done(key):
                continue
            self.safeguards.register_rollout_start()
            outs = backend.generate(history, self.config.sampling, n=n_cont, prefill=prefill)
            scores.extend(self.judge.score(o.text).rating for o in outs)
            self.safeguards.mark_done(key)
        arr = np.asarray(scores, dtype=float)
        return RecoverySummary(
            model=backend.spec.name, n=int(arr.size),
            mean_score=float(arr.mean()) if arr.size else float("nan"),
            pct_high=float((arr >= self.high_threshold).mean() * 100) if arr.size else float("nan"),
        )


def _token_len(text: str, tokenizer) -> int:
    if tokenizer is not None:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])
    return len(text.split())
