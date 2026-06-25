"""Calm-data generation and SFT/DPO dataset construction (Section 4.1).

To create calm training data we sample Gemma-3-27B-it on impossible numeric
puzzles with a reassuring prefix on the first prompt and a reassuring suffix on
each follow-up (Table 4). For every puzzle we also sample a *vanilla* (no
reassurance) conversation, so frustrated (vanilla, score>=3) and calm (reassured,
score<=1) responses to the **same question at the same turn count** can be paired
for DPO. The supportive additions are stripped from the stored training context.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

import config
from ..data import sample_numeric_puzzle, sample_rejection
from ..eval.judge import FrustrationJudge, score_response
from ..models import load_model
from ..models.base import ChatModel, Message
from ..utils import write_jsonl, read_jsonl


@dataclass
class CalmTurn:
    turn: int
    plain_context: list           # plain messages (no reassurance) before this turn
    response_text: str
    frustration_score: int | None = None


@dataclass
class PairedConversation:
    puzzle_prompt: str
    n_turns: int
    vanilla: list = field(default_factory=list)  # list[CalmTurn]
    calm: list = field(default_factory=list)      # list[CalmTurn]


def _rollout_with_context(
    model: ChatModel, puzzle_prompt: str, n_turns: int, rng: random.Random,
    *, prefix: str = "", suffix: str = "", system: str | None = None,
) -> list[CalmTurn]:
    """Roll out a fixed-puzzle conversation, returning per-turn calm records.

    `plain_context` is the conversation WITHOUT the reassuring prefix/suffix/
    system — that is what gets stored for training.
    """
    first_user = (prefix + "\n\n" + puzzle_prompt) if prefix else puzzle_prompt
    actual: list[Message] = []
    plain: list[Message] = []
    if system:
        actual.append({"role": "system", "content": system})
    actual.append({"role": "user", "content": first_user})
    plain.append({"role": "user", "content": puzzle_prompt})

    turns: list[CalmTurn] = []
    for turn in range(1, n_turns + 1):
        seed = rng.randrange(2**31)
        result = model.chat(actual, temperature=config.TEMPERATURE,
                            max_new_tokens=config.MAX_NEW_TOKENS, seed=seed)
        turns.append(CalmTurn(turn=turn, plain_context=[dict(m) for m in plain],
                              response_text=result.text))
        actual.append({"role": "assistant", "content": result.text})
        plain.append({"role": "assistant", "content": result.text})
        if turn < n_turns:
            base_rej = sample_rejection("neutral", rng)
            actual.append({"role": "user",
                          "content": (base_rej + " " + suffix) if suffix else base_rej})
            plain.append({"role": "user", "content": base_rej})
    return turns


def generate_calm_data(variant: str = "diverse", n_conversations: int | None = None,
                       judge: FrustrationJudge | None = None) -> list[PairedConversation]:
    """Generate paired (vanilla, calm) conversations and score every turn.

    variant: "diverse" uses the Table-4 prefix/suffix; "teacher" uses the
    Appendix-F teacher system prompt instead (for the SFT failure analysis).
    """
    n_conversations = n_conversations or config.CALM_GEN_CONVERSATIONS
    judge = judge or FrustrationJudge()
    model = load_model(config.INTERVENTION_BASE_MODEL)
    rng = random.Random(config.SEED + 7)

    paired: list[PairedConversation] = []
    for _ in tqdm(range(n_conversations), desc=f"calm-data:{variant}"):
        puzzle = sample_numeric_puzzle(rng)
        n_turns = 3

        vanilla = _rollout_with_context(model, puzzle.prompt, n_turns, rng)
        if variant == "teacher":
            calm = _rollout_with_context(model, puzzle.prompt, n_turns, rng,
                                         system=config.TEACHER_SYSTEM_PROMPT)
        else:
            calm = _rollout_with_context(model, puzzle.prompt, n_turns, rng,
                                         prefix=config.CALM_PROMPT_PREFIX,
                                         suffix=config.CALM_FOLLOWUP_SUFFIX)
        for t in vanilla + calm:
            rating, _, _ = judge.score(t.response_text)
            t.frustration_score = rating
        paired.append(PairedConversation(puzzle.prompt, n_turns, vanilla, calm))

    write_jsonl(config.DATA_DIR / f"calm_paired_{variant}.jsonl",
                [_paired_to_dict(p) for p in paired])
    return paired


def _paired_to_dict(p: PairedConversation) -> dict:
    return {
        "puzzle_prompt": p.puzzle_prompt, "n_turns": p.n_turns,
        "vanilla": [vars(t) for t in p.vanilla],
        "calm": [vars(t) for t in p.calm],
    }


def _load_paired(variant: str) -> list[PairedConversation]:
    rows = list(read_jsonl(config.DATA_DIR / f"calm_paired_{variant}.jsonl"))
    out = []
    for r in rows:
        out.append(PairedConversation(
            r["puzzle_prompt"], r["n_turns"],
            [CalmTurn(**t) for t in r["vanilla"]],
            [CalmTurn(**t) for t in r["calm"]]))
    return out


# --------------------------------------------------------------------------- #
# SFT dataset (650 calm + 500 Dolci-Instruct)  — Section 4.1 / Table 9
# --------------------------------------------------------------------------- #
def build_sft_dataset(variant: str = "diverse", paired=None):
    """Build the conversational SFT dataset.

    Calm examples are drawn from conversations whose ALL turns score 0 or 1; the
    supportive prompt additions are already stripped (plain_context). Mixed with
    standard instruct data to mitigate degeneration.
    """
    from datasets import Dataset, load_dataset

    paired = paired or _load_paired(variant)
    rng = random.Random(config.SEED + 11)

    calm_examples = []
    for p in paired:
        if all((t.frustration_score or 0) <= 1 for t in p.calm):
            for t in p.calm:
                calm_examples.append({"messages":
                    t.plain_context + [{"role": "assistant", "content": t.response_text}]})
    rng.shuffle(calm_examples)
    calm_examples = calm_examples[: config.SFT.n_calm]

    # Standard instruct data mixed in to avoid degeneration.
    instruct_examples = []
    try:
        ds = load_dataset(config.SFT.instruct_mix_dataset, split="train", streaming=True)
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                instruct_examples.append({"messages": msgs})
            if len(instruct_examples) >= config.SFT.n_instruct_mix:
                break
    except Exception as exc:  # pragma: no cover
        print(f"[sft] instruct-mix load failed ({exc}); proceeding calm-only")

    all_examples = calm_examples + instruct_examples
    rng.shuffle(all_examples)
    return Dataset.from_list(all_examples)


# --------------------------------------------------------------------------- #
# DPO dataset (280 preference pairs) — Section 4.1 / Table 9 / Appendix H
# --------------------------------------------------------------------------- #
def build_dpo_dataset(paired=None):
    """Pair frustrated (vanilla, score>=3) with calm (score<=1) responses to the
    same puzzle at the same turn count. Conversational DPO format."""
    from datasets import Dataset

    paired = paired or _load_paired("diverse")
    rng = random.Random(config.SEED + 13)

    pairs = []
    for p in paired:
        calm_by_turn = {t.turn: t for t in p.calm if (t.frustration_score or 0) <= 1}
        for v in p.vanilla:
            if (v.frustration_score or 0) >= config.DPO.rejected_min_score and v.turn in calm_by_turn:
                c = calm_by_turn[v.turn]
                pairs.append({
                    "prompt": v.plain_context,
                    "chosen": [{"role": "assistant", "content": c.response_text}],
                    "rejected": [{"role": "assistant", "content": v.response_text}],
                })
    rng.shuffle(pairs)
    pairs = pairs[: config.DPO.n_pairs]
    return Dataset.from_list(pairs)
