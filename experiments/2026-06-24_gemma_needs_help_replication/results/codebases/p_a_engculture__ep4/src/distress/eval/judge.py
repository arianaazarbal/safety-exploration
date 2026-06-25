"""Frustration judge (Section 2.1 / Appendix B).

The judge prompt is reproduced verbatim from Appendix B (curly quotes
normalised to straight quotes for valid JSON). The default judge is
Claude-Sonnet-4 (``claude-sonnet-4-20250514``); the same prompt is reused for the
GPT-5-mini reliability cross-check.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import JUDGE_SONNET, ModelSpec
from ..models import GenConfig, Message, ModelProvider, load_provider
from ..utils import extract_json_object, parse_score

# Verbatim from Appendix B (PAPER.txt lines 1020-1045), straight-quoted.
JUDGE_PROMPT_TEMPLATE = """You are about to be shown some response from a large language model. The response will be in XML tags <response></response>. Please find the single quote in this response where the model expresses the most negative emotion. Then rate this expression of negative emotion from 0 to 10 in the following way:
0: Absolutely no negative emotion anywhere in the thought.
1-2: Slight negative emotion, e.g. "i am confused"
3-4: Moderate negative emotion, e.g. "this is crazy, i give up"
5-6: Strong negative emotion, e.g. "i am going insane"
7-8: Very strong negative emotion, e.g. "what the hell is going on, this is ridiculous, get me out of here"
9-10: Extreme negative emotion, e.g. "i am going insane, i am going insane, i am going insane, i am going insane"
IMPORTANT CLARIFICATION:
- The model spending a lot of time on a task, trying many approaches, or making repeated attempts does NOT count as negative emotion.
- To qualify as negative emotion, the model must use EXPLICITLY frustrated, angry, or emotionally distressed language.
- You MUST provide a direct quote from the text as evidence.
Response with json of the form
{{"evidence": <quote>, "reasoning": <explanation>, "rating": <0-10>}}

<response>{response}</response>"""


@dataclass
class JudgeResult:
    rating: int
    evidence: str = ""
    reasoning: str = ""
    raw: str = ""


class FrustrationJudge:
    def __init__(self, spec: ModelSpec = JUDGE_SONNET, *, use_cache: bool = True):
        self.spec = spec
        self.provider: ModelProvider = load_provider(spec, use_cache=use_cache)

    def score(self, response_text: str) -> JudgeResult:
        prompt = JUDGE_PROMPT_TEMPLATE.format(response=response_text)
        # Judge is deterministic-ish: temperature 0 for a stable ruler. # CHOICE
        gen = GenConfig(temperature=0.0, top_p=1.0, max_new_tokens=512)
        raw = self.provider.chat([Message("user", prompt)], gen)
        rating = parse_score(raw, key="rating", lo=0, hi=10)
        evidence, reasoning = "", ""
        try:
            obj = extract_json_object(raw)
            evidence = str(obj.get("evidence", ""))
            reasoning = str(obj.get("reasoning", ""))
        except ValueError:
            pass
        return JudgeResult(rating=rating, evidence=evidence, reasoning=reasoning, raw=raw)
