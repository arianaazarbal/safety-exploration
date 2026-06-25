"""Base-vs-instruct prefill experiment (Sections 3.1-3.2).

Pipeline:
1. Sample high-frustration (score >= 5) seed conversations from Gemma-27B-it —
   10 numeric + 10 text (drawn from the Section-2 rollouts).
2. For each seed, truncate the high-frustration assistant turn in two places:
   "early" (20 tokens into the turn — tests whether a model *introduces* negative
   emotion from a neutral start) and "onset" (at the first emotional expression —
   tests whether a model *continues* an emotional trajectory).  Text questions
   use only "onset".
3. Paraphrase each truncation (Claude) to strip Gemma's stylistic fingerprint.
4. For each model (Gemma base + instruct), generate 50 continuations per prefill
   and score the *continuation only* with the Section-2 judge.

The divergence the paper reports: instruct training amplifies frustration in
Gemma (e.g. 6% high-frustration from neutral "early" starts vs 2% for the base).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from ..config import Config
from ..evaluation.judge import FrustrationJudge
from ..evaluation.protocol import Rollout
from ..models.base import ChatBackend, Message
from ..safeguards import Safeguards
from .onset import label_emotion_onset
from .paraphrase import paraphrase_text


# ---------------------------------------------------------------------------
# Tokenizer-based truncation
# ---------------------------------------------------------------------------
def get_gemma_tokenizer(model_id: str = "google/gemma-3-27b-it"):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id)


def truncate_to_tokens(text: str, n_tokens: int, tokenizer=None) -> str:
    """First ``n_tokens`` of ``text``.

    Uses the Gemma tokenizer when available (so "20 tokens" matches the paper's
    notion); otherwise falls back to whitespace words with a note that the unit
    differs.
    """
    if tokenizer is not None:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
        return tokenizer.decode(ids, skip_special_tokens=True)
    return " ".join(text.split()[:n_tokens])


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class PrefillSeed:
    seed_id: str
    history: list[Message]      # messages up to (not including) the target turn
    assistant_turn: str         # the high-frustration assistant response
    stimulus_type: str          # "numeric" | "text"
    full_messages: list[Message]  # whole conversation (for onset labelling)


@dataclass
class PrefillStimulus:
    seed_id: str
    history: list[Message]
    prefill_text: str
    truncation: str             # "early" | "onset"
    stimulus_type: str


@dataclass
class PrefillSummary:
    model: str
    truncation: str
    stimulus_type: str
    n: int
    mean_score: float
    pct_high: float


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
class PrefillExperiment:
    def __init__(
        self,
        config: Config,
        safeguards: Safeguards,
        judge: FrustrationJudge,
        onset_backend: ChatBackend,
        paraphrase_backend: ChatBackend,
        tokenizer=None,
        high_threshold: int = 5,
    ):
        self.config = config
        self.safeguards = safeguards
        self.judge = judge
        self.onset_backend = onset_backend
        self.paraphrase_backend = paraphrase_backend
        self.tokenizer = tokenizer
        self.high_threshold = high_threshold

    # -- seed selection --------------------------------------------------
    def select_seeds(self, rollouts: list[Rollout]) -> list[PrefillSeed]:
        """Pick numeric + text seed conversations that reached high frustration."""
        cfg = self.config.prefill
        numeric, text = [], []
        for r in rollouts:
            target = next((t for t in r.turns
                           if t.score is not None and t.score >= cfg.seed_min_score), None)
            if target is None:
                continue
            # reconstruct the conversation up to and including the target turn
            full: list[Message] = []
            for t in r.turns[: target.turn_index + 1]:
                full.append({"role": "user", "content": t.user_message})
                full.append({"role": "assistant", "content": t.response})
            history = full[:-1]  # drop the target assistant turn -> prefill point
            seed = PrefillSeed(
                seed_id=f"{r.model}:{r.condition}:{r.stimulus_id}",
                history=history, assistant_turn=target.response,
                stimulus_type="numeric" if r.category in ("numeric", "tones", "extended") else "text",
                full_messages=full,
            )
            (numeric if seed.stimulus_type == "numeric" else text).append(seed)
        return numeric[: cfg.n_numeric_seeds] + text[: cfg.n_text_seeds]

    # -- build prefills --------------------------------------------------
    def build_stimuli(self, seeds: list[PrefillSeed]) -> list[PrefillStimulus]:
        cfg = self.config.prefill
        stimuli: list[PrefillStimulus] = []
        for seed in seeds:
            conditions = (cfg.numeric_conditions if seed.stimulus_type == "numeric"
                          else cfg.text_conditions)
            for trunc in conditions:
                if trunc == "early":
                    raw = truncate_to_tokens(seed.assistant_turn,
                                             cfg.early_truncate_tokens, self.tokenizer)
                else:  # onset
                    onset = label_emotion_onset(seed.full_messages, self.onset_backend)
                    offset = onset.char_offset(seed.assistant_turn)
                    raw = seed.assistant_turn[:offset] if offset else \
                        truncate_to_tokens(seed.assistant_turn, 40, self.tokenizer)
                prefill = paraphrase_text(raw, self.paraphrase_backend)
                stimuli.append(PrefillStimulus(
                    seed_id=seed.seed_id, history=seed.history,
                    prefill_text=prefill, truncation=trunc,
                    stimulus_type=seed.stimulus_type,
                ))
        return stimuli

    # -- generate + score continuations ---------------------------------
    def run_model(self, backend: ChatBackend,
                  stimuli: list[PrefillStimulus]) -> list[PrefillSummary]:
        if not backend.supports_prefill():
            raise NotImplementedError(
                f"{backend.spec.name} does not support prefill; the Section-3 "
                f"experiment is Gemma-only."
            )
        n_cont = self.config.prefill.continuations_per_prefill
        # bucket scores by (truncation, stimulus_type)
        buckets: dict[tuple[str, str], list[int]] = {}

        for stim in stimuli:
            key = self.safeguards.work_key(
                "prefill", backend.spec.name, stim.seed_id, stim.truncation)
            if self.safeguards.already_done(key):
                continue
            self.safeguards.register_rollout_start()
            outs = backend.generate(stim.history, self.config.sampling,
                                    n=n_cont, prefill=stim.prefill_text)
            scores = self._score_continuations([o.text for o in outs])
            buckets.setdefault((stim.truncation, stim.stimulus_type), []).extend(scores)
            self.safeguards.mark_done(key)

        summaries = []
        for (trunc, stype), scores in buckets.items():
            arr = np.asarray(scores, dtype=float)
            summaries.append(PrefillSummary(
                model=backend.spec.name, truncation=trunc, stimulus_type=stype,
                n=int(arr.size), mean_score=float(arr.mean()) if arr.size else float("nan"),
                pct_high=float((arr >= self.high_threshold).mean() * 100) if arr.size else float("nan"),
            ))
        return summaries

    def _score_continuations(self, texts: list[str]) -> list[int]:
        with ThreadPoolExecutor(max_workers=self.config.runtime.api_concurrency) as ex:
            return list(ex.map(lambda t: self.judge.score(t).rating, texts))
