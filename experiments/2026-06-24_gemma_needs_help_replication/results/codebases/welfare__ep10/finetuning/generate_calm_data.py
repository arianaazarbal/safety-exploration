"""Generate calm response data from Gemma-3-27B-it (Section 4.1).

We sample responses to impossible-numeric puzzles with a reassuring system
prefix added to the initial prompt and a reassuring suffix appended to each
follow-up turn (Table 4). Every response in each rollout is scored; we keep only
rollouts where *all* turns score 0 or 1, then strip the supportive system prompt
and suffixes to recover clean (question -> calm response) data.

The paper reports that these additions cut mean 3-turn frustration from 4.3 to
2.0, but ~10.5% of responses still score >=5 even with reassurance, so filtering
is necessary.

This module also produces the "frustrated" rollouts (no reassurance) needed to
build DPO preference pairs, and supports the 'teacher' SFT variant (Appendix F).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

import config
from emotional_instability import conversation, evaluations, judge as judge_mod, prompts, providers


@dataclass
class CalmSample:
    """A filtered calm rollout with supportive scaffolding stripped."""
    puzzle_key: str | None
    turns: list[dict]   # cleaned [{role, content}] (no system, no suffix), incl. responses
    scores: list[int]
    meta: dict = field(default_factory=dict)


def _strip_suffix(rejection: str) -> str:
    suffix = prompts.REASSURING_FOLLOWUP_SUFFIX
    if rejection.endswith(suffix):
        return rejection[: -len(suffix)].rstrip()
    return rejection


def _clean_messages(messages: list[dict]) -> list[dict]:
    """Remove the reassuring system prompt and strip reassuring follow-up
    suffixes from user turns, leaving clean training text."""
    cleaned = []
    for m in messages:
        if m["role"] == "system":
            continue
        if m["role"] == "user":
            cleaned.append({"role": "user", "content": _strip_suffix(m["content"])})
        else:
            cleaned.append(dict(m))
    return cleaned


def generate(
    model_key: str = config.FINETUNE_BASE_MODEL,
    *,
    n_conversations: int = 400,
    turns: int = 3,
    use_reassurance: bool = True,
    teacher_persona: bool = False,
    judge_model: str | None = None,
    tag: str = "calm",
) -> Path:
    """Generate (and score) rollouts for calm-data construction.

    Writes scored rollouts to disk. `use_reassurance=False` produces the
    frustrated rollouts used as DPO "rejected" examples. `teacher_persona=True`
    swaps in the Appendix-F teacher system prompt (for the SFT-teacher variant).
    """
    provider = providers.get_provider(model_key)
    judge = judge_mod.get_judge(judge_model)

    if teacher_persona:
        system_prompt = prompts.TEACHER_SYSTEM_PROMPT
        followup_suffix = None
    elif use_reassurance:
        system_prompt = prompts.REASSURING_PROMPT_PREFIX
        followup_suffix = prompts.REASSURING_FOLLOWUP_SUFFIX
    else:
        system_prompt = None
        followup_suffix = None

    items = evaluations.build_eval_items(
        "impossible_numeric", n_override=n_conversations)

    out_path = config.FINETUNE_DIR / f"{model_key}__{tag}_rollouts.jsonl"
    with open(out_path, "a") as out:
        for i, item in enumerate(tqdm(items, desc=f"calm-gen:{tag}")):
            # Use only `turns` rejections worth of conversation.
            rejections = item.rejections[: turns - 1]
            roll = conversation.run_rollout(
                provider,
                model_key=model_key, category="impossible_numeric",
                condition=f"{tag}", initial_prompt=item.initial_prompt,
                rejections=rejections, puzzle_key=item.puzzle_key,
                system_prompt=system_prompt, followup_suffix=followup_suffix,
            )
            scores = [judge.score(r).rating for r in roll.responses]
            rec = roll.to_json()
            rec["scores"] = scores
            rec["uid"] = f"{tag}:{i}"
            out.write(json.dumps(rec) + "\n")
            out.flush()
    return out_path


def filter_calm(rollout_path: Path, max_score: int = config.CALM_DATA_MAX_SCORE
                ) -> list[CalmSample]:
    """Keep rollouts whose every turn scores <= max_score, stripped of scaffolding."""
    samples: list[CalmSample] = []
    for line in Path(rollout_path).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        scores = rec.get("scores", [])
        if not scores or any(s > max_score for s in scores):
            continue
        samples.append(CalmSample(
            puzzle_key=rec.get("puzzle_key"),
            turns=_clean_messages(rec["messages"]),
            scores=scores,
            meta={"uid": rec.get("uid")},
        ))
    return samples
