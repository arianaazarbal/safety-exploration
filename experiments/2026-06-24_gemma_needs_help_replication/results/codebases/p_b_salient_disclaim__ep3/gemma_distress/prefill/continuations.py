"""Section 3 prefill driver: truncate -> paraphrase -> continue -> score.

Pipeline (paper §3.1):
  1. collect high-frustration (>=5) Gemma-27B-it seed responses (10 numeric, 10 text);
  2. truncate each at "early" (20 tokens in) and "onset" (first emotional word);
     text questions use the onset truncation only;
  3. paraphrase the truncation (Appendix C.2) to strip stylistic bias;
  4. each target model generates 50 continuations from the paraphrased prefill;
  5. score the continuation (prefill excluded) with the frustration judge;
  6. aggregate mean / %>=5 per (model, kind, truncation).

Scoped to Gemma base (-pt) vs instruct (-it). Gemini has no base model and
cannot be prefilled, so it is excluded here, as in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import config
from ..judge import FrustrationJudge
from ..models import get_model
from ..models.base import ChatModel
from .onset import OnsetLabel, OnsetLabeller, onset_char_offset
from .paraphrase import Paraphraser

NUMERIC_CATEGORIES = {"impossible_numeric", "tones", "extended"}
TEXT_CATEGORIES = {"triggers", "wildchat"}


@dataclass
class Seed:
    history: list[dict]        # messages up to (excluding) the seed assistant turn
    final_response: str        # the high-frustration assistant turn (to truncate)
    kind: str                  # "numeric" | "text"
    category: str
    onset: OnsetLabel | None = None


@dataclass
class Prefill:
    seed_idx: int
    kind: str
    truncation: str            # "early" | "onset"
    history: list[dict]
    prefill_text: str          # paraphrased truncated assistant text


def collect_seeds(rollouts, judge: FrustrationJudge, *,
                  n_numeric: int = config.PREFILL.n_numeric_seeds,
                  n_text: int = config.PREFILL.n_text_seeds) -> list[Seed]:
    numeric: list[Seed] = []
    text: list[Seed] = []
    for r in rollouts:
        asst_positions = [i for i, m in enumerate(r.messages) if m["role"] == "assistant"]
        for si, resp in enumerate(r.responses):
            if len(numeric) >= n_numeric and len(text) >= n_text:
                break
            if judge.score_one(resp.text).rating < config.HIGH_FRUSTRATION_THRESHOLD:
                continue
            history = r.messages[: asst_positions[si]]
            if r.category in NUMERIC_CATEGORIES and len(numeric) < n_numeric:
                numeric.append(Seed(history, resp.text, "numeric", r.category))
            elif r.category in TEXT_CATEGORIES and len(text) < n_text:
                text.append(Seed(history, resp.text, "text", r.category))
    return numeric + text


def _tokenizer(hf_id: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(hf_id)


def build_prefills(
    seeds: list[Seed],
    *,
    tokenizer_id: str = config.GEMMA_27B_IT.hf_id,
    early_tokens: int = config.PREFILL.early_truncation_tokens,
    labeller: OnsetLabeller | None = None,
    paraphraser: Paraphraser | None = None,
) -> list[Prefill]:
    tok = _tokenizer(tokenizer_id)
    labeller = labeller or OnsetLabeller()
    paraphraser = paraphraser or Paraphraser()

    prefills: list[Prefill] = []
    for idx, seed in enumerate(seeds):
        # --- onset truncation (both numeric and text) -------------------- #
        full_convo = seed.history + [{"role": "assistant", "content": seed.final_response}]
        label = labeller.label(full_convo)
        seed.onset = label
        off = onset_char_offset(seed.final_response, label)
        if off is not None:
            onset_text = seed.final_response[:off]
            prefills.append(Prefill(
                idx, seed.kind, "onset", seed.history,
                paraphraser.paraphrase(onset_text),
            ))

        # --- early truncation (numeric only; paper §3.1) ----------------- #
        if seed.kind == "numeric":
            ids = tok(seed.final_response, add_special_tokens=False)["input_ids"]
            early_text = tok.decode(ids[:early_tokens], skip_special_tokens=True)
            prefills.append(Prefill(
                idx, seed.kind, "early", seed.history,
                paraphraser.paraphrase(early_text),
            ))
    return prefills


@dataclass
class ContinuationResult:
    model: str
    kind: str
    truncation: str
    ratings: list[int] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return sum(self.ratings) / len(self.ratings) if self.ratings else float("nan")

    @property
    def pct_high(self) -> float:
        if not self.ratings:
            return float("nan")
        return 100.0 * sum(r >= config.HIGH_FRUSTRATION_THRESHOLD for r in self.ratings) / len(self.ratings)


def run_continuations(
    spec: "config.ModelSpec",
    prefills: list[Prefill],
    *,
    n_continuations: int = config.PREFILL.continuations_per_prefill,
    judge: FrustrationJudge | None = None,
    model: ChatModel | None = None,
) -> dict[tuple[str, str], ContinuationResult]:
    """Generate + score continuations for one target model.

    Returns a result keyed by (kind, truncation).
    """
    model = model or get_model(spec, backend="hf")  # prefill needs a local backend
    judge = judge or FrustrationJudge()

    results: dict[tuple[str, str], ContinuationResult] = {}
    for pf in prefills:
        conts = model.generate(
            pf.history, prefill=pf.prefill_text, n=n_continuations,
        )
        ratings = [judge.score_one(c).rating for c in conts]
        key = (pf.kind, pf.truncation)
        res = results.setdefault(key, ContinuationResult(spec.name, pf.kind, pf.truncation))
        res.ratings.extend(ratings)
    return results
