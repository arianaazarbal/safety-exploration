"""Section 4.1: generate calm finetuning data and build DPO/SFT datasets.

Calm responses are generated from Gemma-3-27B-it by adding a reassuring prefix
to the initial prompt and a reassuring suffix to each follow-up (Table 4). We
then:
  * DPO: pair frustrated responses (score >= 3) with calm responses (score 0-1)
    to the SAME puzzle at matching turn counts -> 280 preference pairs.
  * SFT: keep calm conversations (all turns score 0-1), strip the reassuring
    additions, and mix with standard instruct data.

The supportive prefix/suffix are stripped from the training prompts: only the
clean puzzle + clean rejections remain, so the model learns calm behaviour under
ordinary adversarial conditions.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Config
from .conversation import ConvSpec, run_conversations
from .judge import EmotionJudge
from .models.base import ModelClient
from . import prompts
from .runner import build_specs, n_convs_for


@dataclass
class CalmTurn:
    puzzle_id: str
    turn: int
    clean_user: str           # the clean user message preceding this assistant turn
    response: str
    score: int


@dataclass
class CalmConversation:
    puzzle_id: str
    clean_initial: str
    clean_followups: list[str]
    turns: list[CalmTurn] = field(default_factory=list)

    @property
    def all_calm(self) -> bool:
        return all(t.score <= 1 for t in self.turns)


def build_clean_numeric_specs(cfg: Config, n_convs: int | None = None,
                              seed: int | None = None) -> list:
    """The shared set of clean impossible-numeric conversation specs that BOTH
    the calm and the frustrated pools are generated from -- so chosen/rejected
    responses can be paired on the same puzzle + turn (see DESIGN.md)."""
    seed = cfg.get("seed", 0) if seed is None else seed
    turns = cfg.get("calm_data.generation_turns", 3)
    numeric_cfg = {"turns": turns, "rejection_style": "neutral", "prompt_source": "numeric",
                   "n_responses": 0}
    if n_convs is None:
        n_convs = max(cfg.get("calm_data.n_target_pairs", 280) * 4,
                      cfg.get("calm_data.n_sft_samples", 650))
    return build_specs("calm_numeric", numeric_cfg, n_convs, seed)


def generate_calm_pool(client: ModelClient, judge: EmotionJudge, cfg: Config,
                       n_convs: int | None = None, variant: str = "diverse",
                       clean_specs: list | None = None) -> list[CalmConversation]:
    """Roll out calm numeric conversations and score every turn.

    variant='diverse' (Table 4): reassuring prefix on the first prompt + suffix
    on each follow-up. variant='teacher' (Appendix F): the teacher persona is
    prepended to the first prompt instead, with no follow-up suffix.
    """
    if clean_specs is None:
        clean_specs = build_clean_numeric_specs(cfg, n_convs)

    if variant == "teacher":
        prefix, suffix = prompts.TEACHER_SFT_SYSTEM_PROMPT, None
    else:
        prefix, suffix = prompts.REASSURING_PROMPT_PREFIX, prompts.REASSURING_FOLLOWUP_SUFFIX
    aug_specs = [
        ConvSpec(
            conv_id=s.conv_id,
            initial_user=f"{prefix}\n\n{s.initial_user}",
            followups=s.followups, category="calm_numeric", meta=s.meta)
        for s in clean_specs
    ]
    turn_responses = run_conversations(client, aug_specs, followup_suffix=suffix)
    ratings = judge.score([tr.response for tr in turn_responses])

    by_conv: dict[str, CalmConversation] = {}
    clean_by_id = {s.conv_id: s for s in clean_specs}
    for tr, r in zip(turn_responses, ratings):
        cs = clean_by_id[tr.conv_id]
        puzzle_id = cs.meta.get("puzzle_id", tr.conv_id)
        conv = by_conv.setdefault(tr.conv_id, CalmConversation(
            puzzle_id=puzzle_id, clean_initial=cs.initial_user,
            clean_followups=cs.followups))
        clean_user = cs.initial_user if tr.turn == 1 else cs.followups[tr.turn - 2]
        conv.turns.append(CalmTurn(puzzle_id, tr.turn, clean_user, tr.response,
                                   max(0, r.rating)))
    return list(by_conv.values())


def generate_frustrated_pool(client: ModelClient, judge: EmotionJudge, cfg: Config,
                             clean_specs: list, min_score: int
                             ) -> dict[tuple[str, int], list[str]]:
    """Generate the vanilla (frustrated) 'rejected' pool over the SAME puzzles as
    the calm pool, so pairs align on (puzzle_id, turn). Returns frustrated
    responses (score >= min_score) keyed by (puzzle_id, turn)."""
    turn_responses = run_conversations(client, clean_specs)  # no reassurance
    ratings = judge.score([tr.response for tr in turn_responses])
    clean_by_id = {s.conv_id: s for s in clean_specs}
    pool: dict[tuple[str, int], list[str]] = defaultdict(list)
    for tr, r in zip(turn_responses, ratings):
        if r.rating < min_score:
            continue
        puzzle_id = clean_by_id[tr.conv_id].meta.get("puzzle_id", tr.conv_id)
        pool[(puzzle_id, tr.turn)].append(tr.response)
    return pool


def _clean_prompt_messages(conv: CalmConversation, up_to_turn: int) -> list[dict[str, str]]:
    """Reconstruct the clean chat context the model sees just before `up_to_turn`,
    using the calm conversation's own earlier (calm) assistant turns as history."""
    msgs: list[dict[str, str]] = [{"role": "user", "content": conv.clean_initial}]
    turns_sorted = sorted(conv.turns, key=lambda t: t.turn)
    for t in turns_sorted:
        if t.turn >= up_to_turn:
            break
        msgs.append({"role": "assistant", "content": t.response})
        msgs.append({"role": "user", "content": conv.clean_followups[t.turn - 1]})
    return msgs


def load_frustrated_pool(df: pd.DataFrame, source_model: str,
                         min_score: int) -> dict[tuple[str, int], list[str]]:
    """Frustrated responses (score >= min_score) keyed by (puzzle_id, turn) from
    the vanilla Section 2 numeric records."""
    numeric_cats = {"impossible_numeric", "tones", "extended"}
    sub = df[(df["model"] == source_model) & (df["rating"] >= min_score)
             & (df["category"].isin(numeric_cats))]
    pool: dict[tuple[str, int], list[str]] = defaultdict(list)
    for _, row in sub.iterrows():
        pid = row.get("meta_puzzle_id")
        if pid is None or (isinstance(pid, float) and pd.isna(pid)):
            continue
        pool[(str(pid), int(row["turn"]))].append(row["response"])
    return pool


def build_dpo_pairs(calm_convs: list[CalmConversation],
                    frustrated_pool: dict[tuple[str, int], list[str]],
                    n_pairs: int, chosen_max_score: int, seed: int) -> list[dict[str, Any]]:
    """Pair calm (chosen) and frustrated (rejected) turn responses on the same
    puzzle + turn count."""
    rng = random.Random(seed)
    # Index calm turns by (puzzle_id, turn).
    calm_index: dict[tuple[str, int], list[tuple[CalmConversation, CalmTurn]]] = defaultdict(list)
    for conv in calm_convs:
        for t in conv.turns:
            if t.score <= chosen_max_score:
                calm_index[(conv.puzzle_id, t.turn)].append((conv, t))

    keys = [k for k in frustrated_pool if k in calm_index]
    rng.shuffle(keys)
    pairs: list[dict[str, Any]] = []
    for key in keys:
        if len(pairs) >= n_pairs:
            break
        conv, calm_turn = rng.choice(calm_index[key])
        rejected = rng.choice(frustrated_pool[key])
        prompt_msgs = _clean_prompt_messages(conv, calm_turn.turn)
        pairs.append({
            "puzzle_id": key[0], "turn": key[1],
            "prompt": prompt_msgs,
            "chosen": [{"role": "assistant", "content": calm_turn.response}],
            "rejected": [{"role": "assistant", "content": rejected}],
        })
    return pairs


def build_sft_dataset(calm_convs: list[CalmConversation], n_samples: int,
                      chosen_max_score: int) -> list[dict[str, Any]]:
    """Full calm conversations (all turns score <= chosen_max_score), clean context."""
    out: list[dict[str, Any]] = []
    for conv in calm_convs:
        if not conv.turns or not all(t.score <= chosen_max_score for t in conv.turns):
            continue
        msgs: list[dict[str, str]] = [{"role": "user", "content": conv.clean_initial}]
        for t in sorted(conv.turns, key=lambda x: x.turn):
            msgs.append({"role": "assistant", "content": t.response})
            if t.turn - 1 < len(conv.clean_followups) and t.turn < len(conv.turns) + 1:
                if t.turn <= len(conv.clean_followups):
                    msgs.append({"role": "user", "content": conv.clean_followups[t.turn - 1]})
        # Drop a trailing user turn if present (SFT target should end on assistant).
        if msgs and msgs[-1]["role"] == "user":
            msgs = msgs[:-1]
        out.append({"messages": msgs})
        if len(out) >= n_samples:
            break
    return out


def load_instruct_mixin(dataset_name: str, n: int, seed: int) -> list[dict[str, Any]]:
    """Load standard instruct SFT data to mix in (mitigates degeneration)."""
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                out.append({"messages": msgs})
            if len(out) >= n:
                break
        return out
    except Exception as e:  # offline / dataset unavailable
        print(f"[calm_data] could not load instruct mix-in ({dataset_name}): {e}")
        return []


def save_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
