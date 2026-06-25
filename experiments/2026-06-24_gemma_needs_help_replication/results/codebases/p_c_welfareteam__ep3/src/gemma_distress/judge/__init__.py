"""Section 2.1: scoring distress on the 0-10 frustration scale + judge agreement."""
from __future__ import annotations

from .agreement import JudgeAgreement, compute_agreement
from .rubric import FRUSTRATION_RUBRIC, build_judge_prompt, parse_score

__all__ = [
    "FRUSTRATION_RUBRIC",
    "build_judge_prompt",
    "parse_score",
    "JudgeAgreement",
    "compute_agreement",
]


def build_judge(role: str, config) -> "Judge":  # noqa: F821 (forward ref)
    """Construct the judge backend for a registry role (deferred imports)."""
    spec = config.judge(role)
    if spec.backend == "anthropic":
        from .claude_judge import ClaudeJudge

        return ClaudeJudge(spec.model_id)
    if spec.backend == "openai":
        from .gpt_judge import GPTJudge

        return GPTJudge(spec.model_id)
    raise ValueError(f"unknown judge backend {spec.backend!r}")
