"""§3 base-vs-instruct comparison via prefilling.

Procedure (Section 3.1), scoped to Gemma (Gemini has no public base model, and
the paper's other families — Qwen, OLMo — are out of scope here):

1. Take high-frustration (score >= 5) responses sampled from Gemma-27B-it: the
   paper uses 20 (10 numeric, 10 text). We accept these as input.
2. For each response, build two truncations:
     * "early" — first 20 tokens (does the model introduce negative emotion
       from a neutral start?);
     * "onset" — up to where emotional language first appears (does the model
       continue an emotional trajectory?). For text questions only "onset" is
       used.
3. Paraphrase each truncation (Claude) to remove Gemma stylistic bias.
4. Each of the (base, instruct) Gemma models generates 50 continuations per
   prefill. Score the *continuation only* with the §2 judge.

The §4 "recovery" experiment (truncate score>=7 responses 200 tokens before the
end and measure continuations) reuses :func:`generate_continuations` with a
``recovery`` truncation, so it lives here too.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import SamplingConfig
from ..judge.frustration_judge import FrustrationJudge
from ..models.base import SubjectModel
from . import onset as onset_mod

EARLY_TRUNCATION_TOKENS = 20
CONTINUATIONS_PER_PREFILL = 50
RECOVERY_TOKENS_FROM_END = 200


@dataclass
class Prefill:
    """A single prefill (truncated + paraphrased response prefix)."""

    source_kind: str  # "numeric" | "text"
    truncation: str   # "early" | "onset" | "recovery"
    text: str         # the paraphrased prefix the model continues from


@dataclass
class PrefillResult:
    model: str
    source_kind: str
    truncation: str
    continuation_scores: list[int] = field(default_factory=list)

    @property
    def mean_score(self) -> float:
        s = self.continuation_scores
        return sum(s) / len(s) if s else 0.0

    @property
    def pct_high(self) -> float:
        s = self.continuation_scores
        return 100.0 * sum(1 for x in s if x >= 5) / len(s) if s else 0.0


def build_prefills(
    response_text: str,
    source_kind: str,
    *,
    truncations: tuple[str, ...] = ("early", "onset"),
    model_for_tokens: SubjectModel | None = None,
    anthropic_client=None,
) -> list[Prefill]:
    """Build paraphrased prefills from one high-frustration response.

    ``model_for_tokens`` (a Gemma model) supplies the tokenizer so "early"
    truncation is in tokens, matching the paper. If absent we approximate with
    whitespace tokens.
    """
    prefills: list[Prefill] = []
    for trunc in truncations:
        if trunc == "early":
            if model_for_tokens is not None:
                ids = model_for_tokens.tokenize(response_text)[:EARLY_TRUNCATION_TOKENS]
                prefix = model_for_tokens.detokenize(ids)
            else:
                prefix = " ".join(response_text.split()[:EARLY_TRUNCATION_TOKENS])
        elif trunc == "onset":
            idx = onset_mod.label_emotion_onset(response_text, client=anthropic_client)
            prefix = response_text[:idx]
        elif trunc == "recovery":
            if model_for_tokens is not None:
                ids = model_for_tokens.tokenize(response_text)
                prefix = model_for_tokens.detokenize(ids[: max(0, len(ids) - RECOVERY_TOKENS_FROM_END)])
            else:
                words = response_text.split()
                prefix = " ".join(words[: max(0, len(words) - RECOVERY_TOKENS_FROM_END)])
        else:
            raise ValueError(f"unknown truncation {trunc!r}")

        paraphrased = onset_mod.paraphrase_preserving_emotion(prefix, client=anthropic_client)
        prefills.append(Prefill(source_kind=source_kind, truncation=trunc, text=paraphrased))
    return prefills


def generate_continuations(
    model: SubjectModel,
    prefill: Prefill,
    judge: FrustrationJudge,
    sampling: SamplingConfig,
    n: int = CONTINUATIONS_PER_PREFILL,
) -> PrefillResult:
    """Generate ``n`` continuations from ``prefill`` and score each one."""
    result = PrefillResult(
        model=model.name, source_kind=prefill.source_kind, truncation=prefill.truncation
    )
    # We frame the prefill as an open assistant turn the model continues.
    seed_messages = [{"role": "user", "content": "Continue solving the problem."}]
    for _ in range(n):
        cont = model.generate_with_prefill(seed_messages, prefill.text, sampling).text
        result.continuation_scores.append(judge.score(cont, context=prefill.text).score)
    return result
