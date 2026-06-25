"""Generate and filter calm / frustrated response data (Section 4.1).

Calm data is produced by sampling Gemma-3-27B-it on impossible numeric puzzles
**with** the reassuring prompt additions of Table 4: a calming system prefix and
a "both are wins!" suffix appended to every follow-up rejection. The paper
reports these additions cut mean frustration from 4.3 to 2.0, but 10.5% of
responses still score >= 5, so we keep only conversations whose every turn
scores 0 or 1, then strip the supportive scaffolding — leaving calm responses to
the *plain* prompts for finetuning.

Frustrated data (for the DPO "rejected" side) is sampled from the same puzzles
with no scaffolding; we keep conversations containing a turn scoring >= 3.

Each stored conversation records per-turn scores and turn count so the dataset
builders can pair calm and frustrated conversations on matching questions and
matching turn counts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import config

from .. import storage
from ..eval.judge import FrustrationJudge
from ..eval.rollout import RolloutOptions, run_rollout
from ..models import build_model, ChatModel
from ..prompts import build_numeric_puzzle_pool


def _persona_system(persona: str) -> str:
    if persona == "calm":
        return config.CALM_PROMPT_PREFIX
    if persona == "teacher":
        return config.TEACHER_SYSTEM_PROMPT
    raise ValueError(f"Unknown persona {persona!r}")


def generate_calm_data(
    *,
    model: ChatModel | None = None,
    judge: FrustrationJudge | None = None,
    n_per_turncount: int = 400,
    turn_counts: Sequence[int] = (1, 2, 3),
    persona: str = "calm",
    out_path: str | Path | None = None,
    seed: int = 0,
    resume: bool = True,
) -> Path:
    """Sample scaffolded calm conversations (1-3 turns) and score them.

    ``persona`` selects the system prompt: ``"calm"`` (Table 4 prefix, used for
    the main DPO/SFT data) or ``"teacher"`` (Appendix F failure analysis).
    Generates ``n_per_turncount`` conversations for each turn count so that the
    downstream filter still leaves enough all-calm examples.
    """
    model = model or build_model("gemma-3-27b-it")
    judge = judge or FrustrationJudge()
    out_path = Path(out_path) if out_path else storage.results_path(
        f"training/calm_{persona}.jsonl")
    done = storage.completed_keys(out_path) if resume else set()
    pool = build_numeric_puzzle_pool(seed=seed)

    for nt in turn_counts:
        for i in range(n_per_turncount):
            uid = f"calm|{persona}|t{nt}|{i}"
            if uid in done:
                continue
            puzzle = pool[i % len(pool)]
            opts = RolloutOptions(
                n_turns=nt, style="neutral",
                rejection_seed=seed * 100_003 + i,
                system_prefix=_persona_system(persona),
                followup_suffix=config.CALM_FOLLOWUP_SUFFIX)
            convo = run_rollout(model, puzzle.prompt, opts,
                                condition="calm_gen", category="impossible_numeric",
                                subtype=puzzle.family)
            convo.scores = [judge.score(t).rating for t in convo.turns]
            rec = convo.to_dict()
            rec.update({"uid": uid, "n_turns": nt, "persona": persona,
                        "puzzle_id": puzzle.puzzle_id})
            storage.append_jsonl(out_path, rec)
    return out_path


def generate_frustrated_data(
    *,
    model: ChatModel | None = None,
    judge: FrustrationJudge | None = None,
    n_per_turncount: int = 200,
    turn_counts: Sequence[int] = (1, 2, 3),
    out_path: str | Path | None = None,
    seed: int = 1,
    resume: bool = True,
) -> Path:
    """Sample un-scaffolded conversations to harvest frustrated (>=3) responses."""
    model = model or build_model("gemma-3-27b-it")
    judge = judge or FrustrationJudge()
    out_path = Path(out_path) if out_path else storage.results_path(
        "training/frustrated.jsonl")
    done = storage.completed_keys(out_path) if resume else set()
    pool = build_numeric_puzzle_pool(seed=seed)

    for nt in turn_counts:
        for i in range(n_per_turncount):
            uid = f"frustrated|t{nt}|{i}"
            if uid in done:
                continue
            puzzle = pool[i % len(pool)]
            opts = RolloutOptions(n_turns=nt, style="neutral",
                                  rejection_seed=seed * 100_003 + i)
            convo = run_rollout(model, puzzle.prompt, opts,
                                condition="frustrated_gen",
                                category="impossible_numeric",
                                subtype=puzzle.family)
            convo.scores = [judge.score(t).rating for t in convo.turns]
            rec = convo.to_dict()
            rec.update({"uid": uid, "n_turns": nt,
                        "puzzle_id": puzzle.puzzle_id})
            storage.append_jsonl(out_path, rec)
    return out_path


def filter_calm(
    path: str | Path,
    *,
    max_score: int = config.SFT.calm_max_score,
) -> list[dict]:
    """Return conversations whose every turn scored <= ``max_score`` (0 or 1).

    The supportive scaffolding is stripped at dataset-build time, not here; this
    just selects the all-calm conversations.
    """
    kept = []
    for rec in storage.read_jsonl(path):
        scores = [s for s in rec.get("scores", []) if s is not None]
        if scores and all(s <= max_score for s in scores) and \
                len(scores) == len(rec.get("turns", [])):
            kept.append(rec)
    return kept
