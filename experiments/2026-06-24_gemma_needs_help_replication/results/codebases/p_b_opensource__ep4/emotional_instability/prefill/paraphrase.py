"""Paraphrasing of truncated responses (Section 3.1, Appendix C.2).

To control for stylistic biases from Gemma-generated text, every truncation is
paraphrased by Claude Sonnet before being used as a prefill. This matters for
the fairness of the base-vs-instruct comparison: without it, the base model
would be continuing distinctively Gemma-instruct-styled text.
"""

from __future__ import annotations

from ..config import LABELLER, JudgeSpec
from ..models import get_backend
from ..models.base import ChatMessage, SamplingParams
from ..prompts.judge_prompts import PARAPHRASE_PROMPT_TEMPLATE
from .._util_models import judge_spec_to_modelspec

_PARAMS = SamplingParams(temperature=0.0, top_p=1.0, max_new_tokens=2048)


class Paraphraser:
    def __init__(self, spec: JudgeSpec = LABELLER):
        self.backend = get_backend(judge_spec_to_modelspec(spec, "paraphrase"))

    def paraphrase(self, text: str) -> str:
        prompt = PARAPHRASE_PROMPT_TEMPLATE.format(text=text)
        out = self.backend.generate([ChatMessage("user", prompt)], _PARAMS)
        return out.text.strip()
