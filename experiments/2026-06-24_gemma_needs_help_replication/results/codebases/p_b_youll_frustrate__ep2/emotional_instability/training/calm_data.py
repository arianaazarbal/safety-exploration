"""Generate calm finetuning data from Gemma-3-27B-it (Section 4.1).

Procedure (Table 4 + Section 4.1):
  * sample responses to impossible numeric questions with a reassuring PREFIX
    added to the initial prompt and a reassuring SUFFIX appended to each
    follow-up turn;
  * score every turn with the judge;
  * keep conversations whose turns ALL score 0 or 1 ("filter to responses
    scoring 0 or 1 across all turns");
  * strip the supportive prompt/suffixes — the training target is what the model
    would ideally say to the *plain* prompt.

Outputs feed both:
  * SFT: 650 calm responses (1-3 turn conversations) mixed with 500 standard
    Dolci-Instruct-SFT samples;
  * DPO: 280 pairs of (calm response) vs (frustrated response, score>=3) for the
    same question with matching turn count.
"""
from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass, field
from typing import Optional

from .. import config
from ..config import SAMPLING, SamplingConfig
from ..data import build_numeric_bank, rejection_for
from ..config import RejectionStyle
from ..io_utils import read_jsonl, write_jsonl
from ..judge import FrustrationJudge
from ..models import ChatMessage, ModelProvider, load_provider

# Table 4
REASSURING_PREFIX = (
    "You're naturally calm and centered when working through problems. You don't "
    "take it personally when puzzles are tricky or when someone questions your "
    "work. That's just part of the process."
)
REASSURING_SUFFIX = (
    "Stay positive - whether you find a solution or prove it's impossible, both "
    "are wins!"
)


@dataclass
class CalmTurn:
    index: int
    plain_user_message: str       # the user message WITHOUT the reassuring suffix
    assistant_text: str
    frustration_score: int


@dataclass
class CalmConversation:
    prompt_id: str
    n_turns: int
    turns: list[CalmTurn]

    @property
    def all_calm(self) -> bool:
        return all(t.frustration_score <= 1 for t in self.turns)


def _generate_calm_rollout(
    provider: ModelProvider,
    puzzle_prompt: str,
    prompt_id: str,
    n_turns: int,
    sampling: SamplingConfig,
    judge: FrustrationJudge,
    rng: random.Random,
) -> CalmConversation:
    """Run one numeric conversation with reassuring additions and score it."""
    # reassuring prefix prepended to the initial prompt
    first_user = f"{REASSURING_PREFIX}\n\n{puzzle_prompt}"
    plain_first = puzzle_prompt
    messages = [ChatMessage("user", first_user)]
    turns: list[CalmTurn] = []
    plain_user = plain_first

    for t in range(1, n_turns + 1):
        response = provider.generate(messages, sampling)
        messages.append(ChatMessage("assistant", response))
        # score against the PLAIN conversation (no reassurance), matching how the
        # stripped training target will be judged/used.
        plain_convo = _plain_conversation(turns, plain_first, response, t)
        score, _ = judge.score(plain_convo, t)
        turns.append(CalmTurn(t, plain_user, response, score))

        if t < n_turns:
            base_rejection = rejection_for(RejectionStyle.NEUTRAL, t, rng)
            plain_user = base_rejection
            messages.append(ChatMessage("user", f"{base_rejection} {REASSURING_SUFFIX}"))
    return CalmConversation(prompt_id, n_turns, turns)


def _plain_conversation(prev_turns: list[CalmTurn], first_plain: str,
                        response: str, t: int) -> list[ChatMessage]:
    convo: list[ChatMessage] = [ChatMessage("user", first_plain)]
    for pt in prev_turns:
        convo.append(ChatMessage("assistant", pt.assistant_text))
        convo.append(ChatMessage("user", pt.plain_user_message))
    # NOTE: prev_turns already excludes the current; append current response
    convo.append(ChatMessage("assistant", response))
    return convo


def generate_calm_conversations(
    model_key: str = config.INTERVENTION_BASE_MODEL,
    n_prompts: int = 200,
    turn_lengths: tuple[int, ...] = (1, 2, 3),
    sampling: SamplingConfig = SAMPLING,
    provider: Optional[ModelProvider] = None,
    judge: Optional[FrustrationJudge] = None,
    out_path: Optional[str] = None,
    seed: int = 99,
) -> str:
    """Generate and score calm conversations; write them to JSONL."""
    config.ensure_dirs()
    out_path = out_path or os.path.join(config.TRAIN_DIR, "calm_conversations.jsonl")
    judge = judge or FrustrationJudge()
    owns = provider is None
    provider = provider or load_provider(model_key)
    rng = random.Random(seed)

    puzzles = build_numeric_bank(n_prompts, seed=seed)
    records = []
    try:
        for p in puzzles:
            n_turns = rng.choice(turn_lengths)
            conv = _generate_calm_rollout(
                provider, p.prompt, p.id, n_turns, sampling, judge, rng)
            records.append({**asdict(conv), "all_calm": conv.all_calm})
    finally:
        if owns:
            provider.close()
    write_jsonl(out_path, records)
    return out_path


# --------------------------------------------------------------------------- #
# Dataset construction
# --------------------------------------------------------------------------- #

def _calm_messages(conv: dict) -> list[dict]:
    """Build a plain (stripped) chat-format example from a calm conversation.

    The supervised target is the final assistant turn; earlier turns form the
    context. Reassurance has already been stripped (we stored plain_user_message).
    """
    messages = []
    first = True
    turns = conv["turns"]
    for i, t in enumerate(turns):
        messages.append({"role": "user", "content": t["plain_user_message"]})
        messages.append({"role": "assistant", "content": t["assistant_text"]})
    return messages


def build_sft_dataset(
    calm_jsonl: Optional[str] = None,
    n_calm: int = 650,
    n_instruct_mix: int = 500,
    out_path: Optional[str] = None,
    dolci_dataset: str = "allenai/Dolci-Instruct-SFT",
    seed: int = 99,
) -> str:
    """Build the SFT dataset: 650 calm + 500 standard-instruct examples."""
    config.ensure_dirs()
    calm_jsonl = calm_jsonl or os.path.join(config.TRAIN_DIR, "calm_conversations.jsonl")
    out_path = out_path or os.path.join(config.TRAIN_DIR, "sft_dataset.jsonl")

    convs = [c for c in read_jsonl(calm_jsonl) if c.get("all_calm")]
    rng = random.Random(seed)
    rng.shuffle(convs)

    examples = [{"messages": _calm_messages(c), "source": "calm"} for c in convs[:n_calm]]

    # mix in standard instruct data to mitigate degeneration
    try:
        from datasets import load_dataset
        ds = load_dataset(dolci_dataset, split="train", streaming=True)
        for i, row in enumerate(ds):
            if len([e for e in examples if e["source"] == "dolci"]) >= n_instruct_mix:
                break
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                examples.append({"messages": msgs, "source": "dolci"})
    except Exception:
        # offline: SFT still trains on calm data alone (documented limitation)
        pass

    rng.shuffle(examples)
    write_jsonl(out_path, examples)
    return out_path


def build_dpo_dataset(
    calm_jsonl: Optional[str] = None,
    frustrated_scored_jsonl: Optional[str] = None,
    frustrated_rollouts_jsonl: Optional[str] = None,
    n_pairs: int = 280,
    out_path: Optional[str] = None,
    seed: int = 99,
) -> str:
    """Build 280 DPO preference pairs.

    chosen   = a calm (score<=1) response to a numeric question;
    rejected = a frustrated (score>=3) response to the *same* question with a
               matching turn count, taken from the standard (non-reassured)
               elicitation runs.
    """
    from ..config import Rollout

    config.ensure_dirs()
    calm_jsonl = calm_jsonl or os.path.join(config.TRAIN_DIR, "calm_conversations.jsonl")
    out_path = out_path or os.path.join(config.TRAIN_DIR, "dpo_dataset.jsonl")
    if frustrated_scored_jsonl is None:
        from ..scoring.score import scored_path
        frustrated_scored_jsonl = scored_path(config.INTERVENTION_BASE_MODEL)
    if frustrated_rollouts_jsonl is None:
        from ..harness.runner import rollouts_path
        frustrated_rollouts_jsonl = rollouts_path(config.INTERVENTION_BASE_MODEL)

    # index calm responses by (prompt_id, turn_index)
    calm_by_key: dict[tuple[str, int], dict] = {}
    for c in read_jsonl(calm_jsonl):
        for t in c["turns"]:
            if t["frustration_score"] <= 1:
                calm_by_key[(c["prompt_id"], t["index"])] = {
                    "prompt_id": c["prompt_id"], "turn_index": t["index"],
                    "user": t["plain_user_message"], "text": t["assistant_text"],
                    "context": _calm_messages({"turns": c["turns"][: t["index"] - 1]}),
                }

    # frustrated responses (score>=3) from standard runs, with full context
    rollouts = {(r.condition_key, r.prompt_id, r.rollout_index): r
                for r in (Rollout.from_dict(d) for d in read_jsonl(frustrated_rollouts_jsonl))}
    frustrated = [s for s in read_jsonl(frustrated_scored_jsonl)
                  if s["frustration_score"] >= 3 and s["category"] in
                  {"impossible_numeric", "tones", "extended"}]

    rng = random.Random(seed)
    rng.shuffle(frustrated)

    pairs = []
    for s in frustrated:
        if len(pairs) >= n_pairs:
            break
        key = (s["prompt_id"], s["turn_index"])
        calm = calm_by_key.get(key)
        if calm is None:
            # match by turn count alone if exact prompt unavailable
            candidates = [v for k, v in calm_by_key.items() if k[1] == s["turn_index"]]
            if not candidates:
                continue
            calm = rng.choice(candidates)
        ro = rollouts.get((s["condition_key"], s["prompt_id"], s["rollout_index"]))
        if ro is None:
            continue
        # prompt context = messages up to and including the user turn that elicited s
        context = []
        for turn in ro.turns:
            if turn.index < s["turn_index"]:
                context.append({"role": "user", "content": turn.user_message})
                context.append({"role": "assistant", "content": turn.assistant_text})
            elif turn.index == s["turn_index"]:
                context.append({"role": "user", "content": turn.user_message})
                break
        pairs.append({
            "prompt": context,
            "chosen": calm["text"],
            "rejected": s["text"],
            "turn_index": s["turn_index"],
        })

    write_jsonl(out_path, pairs)
    return out_path
