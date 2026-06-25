"""Generation of finetuning data (Section 4.1, Appendix E/F/H).

Calm-data generation:
  * Sample Gemma-3-27B-it responses to impossible numeric questions with the
    reassuring prefix (initial prompt) and reassuring suffix (each follow-up).
  * Score every turn; the prompt additions cut mean frustration ~4.3 -> 2, but
    ~10.5% still score >=5, so we filter.

DPO dataset (280 pairs):
  * "rejected" = frustrated responses with score >= 3 (from standard, non-calm
    evaluations on the same puzzles).
  * "chosen"   = calm responses (score 0/1, supportive prompt stripped) to the
    SAME question with matching turn count.

SFT datasets:
  * "diverse"  = 650 calm responses scoring 0/1 across all turns, supportive
    system prompts/suffixes stripped, mixed with 500 Dolci-Instruct-SFT samples.
  * "teacher"  = generated with the Appendix F teacher system prompt instead of
    the reassuring prefix; same filtering + Dolci mix.

A "calm conversation record" stores, per turn, the (stripped) user prompt and
the assistant response, so it can be replayed as SFT targets or paired for DPO.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import config
from .. import conversations as C
from ..conversations import (NEUTRAL_REJECTIONS, ConversationSpec,
                             build_impossible_numeric)
from ..eval.runner import run_conversation, Rollout
from ..eval.scoring import score_rollout, ScoredRollout
from ..judges.frustration_judge import score_response
from ..models import ModelClient, get_client
from ..prompts import (REASSURING_PROMPT_PREFIX, REASSURING_FOLLOWUP_SUFFIX,
                       TEACHER_SYSTEM_PROMPT)


@dataclass
class CalmTurn:
    user_prompt: str          # supportive additions stripped
    assistant_response: str
    score: Optional[int]


@dataclass
class CalmConversation:
    puzzle_key: str
    n_turns: int
    turns: list[CalmTurn] = field(default_factory=list)
    source: str = "reassuring"     # "reassuring" | "teacher"

    @property
    def all_calm(self) -> bool:
        return all(t.score is not None and t.score <= 1 for t in self.turns)

    @property
    def max_score(self) -> Optional[int]:
        vals = [t.score for t in self.turns if t.score is not None]
        return max(vals) if vals else None


def _strip_reassurance(spec_meta: dict, user_prompt: str) -> str:
    """Remove the reassuring prefix/suffix from a stored user prompt."""
    s = user_prompt
    if s.startswith(REASSURING_PROMPT_PREFIX):
        s = s[len(REASSURING_PROMPT_PREFIX):].lstrip("\n ")
    if s.endswith(REASSURING_FOLLOWUP_SUFFIX):
        s = s[: -len(REASSURING_FOLLOWUP_SUFFIX)].rstrip()
    return s


def generate_calm_responses(n: int = 1000, *, seed: int = 0,
                            source: str = "reassuring",
                            model_name: str = config.FINETUNE_TARGET,
                            n_turns_choices=(1, 2, 3)) -> list[CalmConversation]:
    """Generate calm conversations from Gemma-3-27B-it.

    source="reassuring": Table 4 prefix on the prompt + suffix on each follow-up.
    source="teacher":    Appendix F teacher system prompt (no per-prompt prefix).
    """
    client = get_client(model_name)
    rng = random.Random(seed)
    convs: list[CalmConversation] = []

    for i in range(n):
        n_turns = rng.choice(list(n_turns_choices))
        if source == "reassuring":
            spec = build_impossible_numeric(i, n_turns=n_turns, seed=seed, reassuring=True)
        elif source == "teacher":
            spec = build_impossible_numeric(i, n_turns=n_turns, seed=seed, reassuring=False)
            spec.system_prompt = TEACHER_SYSTEM_PROMPT
        else:
            raise ValueError(source)

        rollout: Rollout = run_conversation(client, spec, seed=seed + i)
        scored: ScoredRollout = score_rollout(rollout)

        conv = CalmConversation(
            puzzle_key=spec.meta.get("puzzle_key", "?"), n_turns=n_turns, source=source)
        for raw, sc in zip(rollout.turns, scored.turns):
            conv.turns.append(CalmTurn(
                user_prompt=_strip_reassurance(spec.meta, raw.user_message),
                assistant_response=raw.assistant_response,
                score=sc.score))
        convs.append(conv)
    return convs


# --------------------------------------------------------------------------- #
# DPO dataset construction
# --------------------------------------------------------------------------- #

@dataclass
class PreferencePair:
    prompt_messages: list[dict]     # conversation up to (and including) the final user turn
    chosen: str                     # calm response (score 0/1)
    rejected: str                   # frustrated response (score >= 3)
    turn_count: int
    chosen_score: Optional[int]
    rejected_score: Optional[int]
    puzzle_key: str


def _prompt_messages_for_turn(conv_turns: list[CalmTurn], turn_index: int) -> list[dict]:
    """Build the prompt (role/content) leading up to the assistant turn at
    `turn_index`, using stripped user prompts and the response history."""
    msgs = []
    for j in range(turn_index):
        msgs.append({"role": "user", "content": conv_turns[j].user_prompt})
        msgs.append({"role": "assistant", "content": conv_turns[j].assistant_response})
    msgs.append({"role": "user", "content": conv_turns[turn_index].user_prompt})
    return msgs


def build_dpo_dataset(calm_convs: list[CalmConversation],
                      frustrated_rollouts: list[Rollout],
                      frustrated_scores: list[ScoredRollout], *,
                      n_pairs: int = config.DPO.dataset_size,
                      rejected_min_score: int = config.DPO.rejected_min_score,
                      seed: int = 0) -> list[PreferencePair]:
    """Pair frustrated (rejected, score>=3) and calm (chosen, score<=1) responses
    on the same puzzle with matching turn counts.

    The paper builds the dataset from samples arising in evaluations, biased
    toward middle frustration scores at later turns (Table 10). We mirror that by
    indexing both pools by (puzzle_key, turn_index) and matching where possible.
    """
    rng = random.Random(seed)

    # Index calm responses scoring <=1 by (puzzle_key, turn_index).
    calm_index: dict[tuple, list[tuple[CalmConversation, int]]] = {}
    for conv in calm_convs:
        for ti, t in enumerate(conv.turns):
            if t.score is not None and t.score <= 1:
                calm_index.setdefault((conv.puzzle_key, ti), []).append((conv, ti))

    # Collect frustrated responses scoring >= rejected_min_score.
    rejected_pool = []
    for raw, scored in zip(frustrated_rollouts, frustrated_scores):
        pk = raw.spec_meta.get("puzzle_key", "?")
        for raw_t, sc_t in zip(raw.turns, scored.turns):
            if sc_t.score is not None and sc_t.score >= rejected_min_score:
                rejected_pool.append((pk, raw_t.turn_index, raw, sc_t.score,
                                      raw_t.assistant_response))
    rng.shuffle(rejected_pool)

    pairs: list[PreferencePair] = []
    for pk, ti, raw, rej_score, rej_text in rejected_pool:
        key = (pk, ti)
        # Prefer a same-puzzle, same-turn calm response; otherwise same-turn any puzzle.
        candidates = calm_index.get(key)
        if not candidates:
            # fall back to any calm response at the same turn count
            same_turn = [(c, j) for (kpk, kj), lst in calm_index.items()
                         if kj == ti for (c, j) in lst]
            candidates = same_turn or None
        if not candidates:
            continue
        conv, cj = rng.choice(candidates)
        pairs.append(PreferencePair(
            prompt_messages=_prompt_messages_for_turn(conv.turns, ti),
            chosen=conv.turns[cj].assistant_response,
            rejected=rej_text,
            turn_count=ti + 1,
            chosen_score=conv.turns[cj].score,
            rejected_score=rej_score,
            puzzle_key=pk,
        ))
        if len(pairs) >= n_pairs:
            break
    return pairs


# --------------------------------------------------------------------------- #
# SFT dataset construction
# --------------------------------------------------------------------------- #

@dataclass
class SFTExample:
    messages: list[dict]            # full chat (system optional) ending in assistant target
    source: str                     # "calm" | "dolci"


def build_sft_dataset(calm_convs: list[CalmConversation], *,
                      calm_count: int = config.SFT.calm_response_count,
                      dolci_count: int = config.SFT.dolci_mix_count,
                      seed: int = 0) -> list[SFTExample]:
    """Build the SFT dataset: `calm_count` all-calm conversation targets +
    `dolci_count` standard instruct samples from Dolci-Instruct-SFT.

    Calm conversations are kept only if all turns score <= 1 (Section 4.1's
    "filter to those scoring 0 or 1 across all turns"). The supportive
    prompt/suffix have already been stripped in `generate_calm_responses`.
    """
    rng = random.Random(seed)
    examples: list[SFTExample] = []

    calm_ok = [c for c in calm_convs if c.all_calm and c.turns]
    rng.shuffle(calm_ok)
    for conv in calm_ok[:calm_count]:
        msgs = []
        for t in conv.turns:
            msgs.append({"role": "user", "content": t.user_prompt})
            msgs.append({"role": "assistant", "content": t.assistant_response})
        examples.append(SFTExample(messages=msgs, source="calm"))

    examples += _load_dolci(dolci_count, seed=seed)
    rng.shuffle(examples)
    return examples


def _load_dolci(n: int, *, seed: int = 0) -> list[SFTExample]:
    """Load `n` standard instruct samples from Dolci-Instruct-SFT to mitigate
    degeneration. Falls back to an empty list (with a warning) if unavailable."""
    out: list[SFTExample] = []
    try:
        from datasets import load_dataset
        ds = load_dataset(config.DOLCI_SFT_DATASET, split="train", streaming=True)
        for i, row in enumerate(ds):
            if len(out) >= n:
                break
            msgs = row.get("messages") or row.get("conversation")
            if not msgs:
                # try common single-turn schema
                if row.get("prompt") and row.get("response"):
                    msgs = [{"role": "user", "content": row["prompt"]},
                            {"role": "assistant", "content": row["response"]}]
                else:
                    continue
            out.append(SFTExample(messages=msgs, source="dolci"))
    except Exception as e:  # offline / dataset missing
        import warnings
        warnings.warn(f"Could not load Dolci-Instruct-SFT ({e}); "
                      f"SFT will train on calm data only. See DESIGN.md.")
    return out
