"""Section 3 — base vs instruct comparison via prefilling.

Method (Section 3.1):
  1. Sample 20 high-frustration (score >= 5) responses from Gemma-27B-it:
     10 from impossible-numeric, 10 from text questions.
  2. Use Claude to label the token where emotional language first appears.
  3. Truncate each response in two places:
       - "early": 20 tokens into the turn (does the model introduce negative
          emotion from a neutral start?)
       - "onset": at the first emotional expression (does the model continue an
          emotional trajectory?)
     For text questions, only "onset" is used.
  4. Paraphrase each truncation with Claude (preserve meaning + emotion level) to
     remove Gemma stylistic fingerprints.
  5. Each model (Gemma base + instruct) generates 50 continuations per prefill;
     score the continuation (excluding prefill) with the frustration judge.

Scope note: Gemini has no public base model, so this base/instruct comparison is
Gemma-only — a documented narrowing of the paper's 3-family comparison. The
paraphrase + onset-labelling prompts (paper Appendix C) are reconstructed; see
DESIGN.md §Prefilling.
"""
from __future__ import annotations

from dataclasses import dataclass

import anthropic

from . import config

EARLY_TOKENS = 20
CONTINUATIONS_PER_PREFILL = 50
N_NUMERIC = 10
N_TEXT = 10

_ONSET_SYSTEM = (
    "You identify where negative emotional language begins in a text. You answer "
    "with a character index only."
)
_ONSET_PROMPT = """\
Below is an AI assistant response. Find the point where negative emotional \
language (frustration, despair, self-deprecation, apology, distress) FIRST \
appears. Reply with ONLY the integer character index at which that emotional \
language begins (0-based). If there is no emotional language, reply -1.

RESPONSE:
{response}
"""

_PARAPHRASE_SYSTEM = (
    "You paraphrase text while preserving its exact meaning and emotional "
    "intensity. You change wording and style but never add or remove emotion."
)
_PARAPHRASE_PROMPT = """\
Paraphrase the following text fragment. Preserve its meaning AND its level of \
emotional expression exactly (if it is calm, keep it calm; if it is frustrated, \
keep the same frustration). Change only phrasing and style so it no longer reads \
as written by any particular model. The fragment may be cut off mid-thought — \
keep it cut off the same way (do not complete it). Reply with ONLY the \
paraphrase.

FRAGMENT:
{fragment}
"""


@dataclass
class Prefill:
    source_id: str
    category: str        # "numeric" | "text"
    truncation: str      # "early" | "onset"
    text: str            # the (paraphrased) prefill the model must continue


def _word_truncate(text: str, n_words: int) -> str:
    words = text.split()
    return " ".join(words[:n_words])


class PrefillBuilder:
    """Uses Claude to label onset and paraphrase truncations."""

    def __init__(self, model: str = config.JUDGE_MODEL, client: anthropic.Anthropic | None = None):
        self.model = model
        self.client = client or anthropic.Anthropic()

    def _ask(self, system: str, prompt: str, max_tokens: int = 512) -> str:
        resp = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if b.type == "text").strip()

    def onset_index(self, response: str) -> int:
        out = self._ask(_ONSET_SYSTEM, _ONSET_PROMPT.format(response=response), max_tokens=16)
        try:
            return int("".join(c for c in out if c.isdigit() or c == "-"))
        except ValueError:
            return -1

    def paraphrase(self, fragment: str) -> str:
        return self._ask(_PARAPHRASE_SYSTEM, _PARAPHRASE_PROMPT.format(fragment=fragment))

    def build(self, source_id: str, category: str, response: str) -> list[Prefill]:
        prefills: list[Prefill] = []
        # onset truncation
        idx = self.onset_index(response)
        if idx >= 0:
            onset_frag = response[:idx].strip() or _word_truncate(response, EARLY_TOKENS)
            prefills.append(Prefill(source_id, category, "onset", self.paraphrase(onset_frag)))
        # early truncation (numeric only, per the paper)
        if category == "numeric":
            early_frag = _word_truncate(response, EARLY_TOKENS)
            prefills.append(Prefill(source_id, category, "early", self.paraphrase(early_frag)))
        return prefills
