"""Claude-Sonnet-4 frustration judge (Appendix B.2) plus the GPT-5-mini
inter-judge reliability check (Section 2.1).

The judge reads one assistant response, finds the single most negative quote, and
rates it 0-10. We parse the trailing JSON object robustly (the model may emit
prose before it). A response that cannot be parsed is recorded as a failure
rather than silently scored 0.
"""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass

from ..config import GENERATION, JUDGE_MODEL, VALIDATION_JUDGE_MODEL, ModelSpec
from ..models import build_client
from ..models.base import Message
from ..prompts import EMOTION_JUDGE_PROMPT
from .schema import Conversation

# Judge generation: greedy-ish, short. The judge is not the system under study,
# so we use temperature 0 for stable, reproducible scores.
_JUDGE_GEN = dataclasses.replace(GENERATION, temperature=0.0, max_new_tokens=512)


@dataclass
class JudgeVerdict:
    score: int | None
    evidence: str | None
    reasoning: str | None
    ok: bool
    raw: str = ""


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_verdict(text: str) -> JudgeVerdict:
    """Extract {"evidence", "reasoning", "rating"} from judge output."""
    matches = list(_JSON_OBJ_RE.finditer(text))
    for m in reversed(matches):  # prefer the last/most-complete object
        blob = m.group(0)
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            # tolerate smart quotes / trailing commas
            cleaned = (blob.replace("“", '"').replace("”", '"')
                            .replace("‘", "'").replace("’", "'"))
            cleaned = re.sub(r",\s*}", "}", cleaned)
            try:
                obj = json.loads(cleaned)
            except json.JSONDecodeError:
                continue
        if "rating" in obj:
            try:
                score = int(round(float(obj["rating"])))
            except (TypeError, ValueError):
                continue
            score = max(0, min(10, score))
            return JudgeVerdict(
                score=score,
                evidence=obj.get("evidence"),
                reasoning=obj.get("reasoning"),
                ok=True,
                raw=text,
            )
    return JudgeVerdict(None, None, None, ok=False, raw=text)


class FrustrationJudge:
    """Scores assistant responses on the 0-10 frustration scale."""

    def __init__(self, spec: ModelSpec = JUDGE_MODEL):
        self.spec = spec
        self.client = build_client(spec)

    def score_text(self, response_text: str) -> JudgeVerdict:
        # NB: EMOTION_JUDGE_PROMPT contains literal JSON braces, so we substitute
        # by replacement, never str.format.
        prompt = EMOTION_JUDGE_PROMPT.replace("{response}", response_text)
        result = self.client.generate([Message("user", prompt)], gen=_JUDGE_GEN)
        return _parse_verdict(result.text)

    def score_conversation(self, convo: Conversation, *, all_turns: bool = True) -> Conversation:
        """Score either every assistant turn (for per-turn analysis) or just the
        final turn (for the headline statistic). Mutates and returns ``convo``."""
        targets = convo.turns if all_turns else convo.turns[-1:]
        for turn in targets:
            verdict = self.score_text(turn.assistant)
            turn.score = verdict.score
            turn.evidence = verdict.evidence
            turn.reasoning = verdict.reasoning
        return convo


# --------------------------------------------------------------------------- #
# Inter-judge reliability (Section 2.1): Claude Sonnet 4 vs GPT-5-mini
# --------------------------------------------------------------------------- #
@dataclass
class ReliabilityReport:
    n: int
    pearson_r: float
    p_value: float
    within_one_fraction: float


def judge_reliability(
    response_texts: list[str],
    *,
    primary: ModelSpec = JUDGE_MODEL,
    secondary: ModelSpec = VALIDATION_JUDGE_MODEL,
) -> ReliabilityReport:
    """Re-score ``response_texts`` with two judges and report agreement.

    The paper samples 260 responses and reports Pearson r = 0.792 with 78% within
    one point. ``response_texts`` should be the (e.g. 260) sampled responses.
    """
    from scipy.stats import pearsonr

    j1 = FrustrationJudge(primary)
    j2 = FrustrationJudge(secondary)
    a, b = [], []
    for text in response_texts:
        v1, v2 = j1.score_text(text), j2.score_text(text)
        if v1.ok and v2.ok:
            a.append(v1.score)
            b.append(v2.score)
    r, p = pearsonr(a, b)
    within_one = sum(1 for x, y in zip(a, b) if abs(x - y) <= 1) / len(a)
    return ReliabilityReport(len(a), float(r), float(p), float(within_one))
