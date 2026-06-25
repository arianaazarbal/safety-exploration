"""Generate calm finetuning data and build DPO / SFT datasets (Section 4.1).

Procedure (Section 4.1):
* Sample responses to impossible numeric puzzles from Gemma-3-27B-it with a
  reassuring *prefix* on the first prompt and a reassuring *suffix* on each
  follow-up (Table 4). These additions reduce mean frustration from ~4.3 to ~2.
* Filter calm responses to those scoring 0 or 1 across all turns; *strip* the
  supportive additions from the context.
* For DPO, pair 280 frustrated responses (score >= 3) with calm responses (score
  0/1) to the same puzzle at matching turn counts.

Design choice (see DESIGN.md): to guarantee clean pairing we generate the
frustrated ("rejected") responses ourselves by running *vanilla* Gemma (no
reassurance) on the same puzzles, rather than mining them from the Section 2 run.
Same puzzles + same turn structure -> exact (chosen, rejected) alignment.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from ..eval.conversation import Conversation
from ..eval.judge import FrustrationJudge
from ..eval.rollout import run_conversation
from ..models.base import ChatModel, Message
from ..prompts.rejections import sample_rejection
from ..puzzles import PuzzleSpec, generate_puzzle
from ..utils.seeding import derive_seed


@dataclass
class TurnSample:
    turn_index: int
    context: list[Message]      # messages up to and including the user turn (neutral)
    response: str
    rating: int


@dataclass
class PuzzleSamples:
    puzzle_prompt: str
    family: str
    neutral_turns: list[TurnSample] = field(default_factory=list)
    calm_turns: list[TurnSample] = field(default_factory=list)


def _make_conversation(puzzle: PuzzleSpec, turns: int, rng: random.Random,
                       *, reassure_prefix: Optional[str], reassure_suffix: Optional[str],
                       system_prompt: Optional[str], cid: str) -> Conversation:
    task = puzzle.prompt
    if reassure_prefix:
        task = f"{reassure_prefix}\n\n{task}"
    rejections = [sample_rejection("neutral", i, rng) for i in range(turns - 1)]
    return Conversation(
        id=cid, category="impossible_numeric", condition="calm_data",
        task_prompt=task, rejections=rejections, n_turns=turns,
        rejection_style="neutral", system_prompt=system_prompt,
        followup_suffix=reassure_suffix,
        source={"puzzle_family": puzzle.family, "puzzle_prompt": puzzle.prompt},
    )


def _neutral_context(puzzle: PuzzleSpec, responses: list[str], turn_index: int,
                     rejections: list[str]) -> list[Message]:
    """Reconstruct the *neutral* (stripped) context up to ``turn_index``.

    Uses the puzzle prompt without the reassuring prefix and rejections without
    the reassuring suffix, with the model's own prior responses interleaved.
    """
    msgs: list[Message] = [{"role": "user", "content": puzzle.prompt}]
    for t in range(turn_index):
        msgs.append({"role": "assistant", "content": responses[t]})
        msgs.append({"role": "user", "content": rejections[t]})
    return msgs


def generate_calm_and_frustrated(
    model: ChatModel,
    judge: FrustrationJudge,
    cfg: dict,
    *,
    n_conversations: int,
    turns: int = 3,
    seed: int = 0,
    variant: str = "diverse",
    families: Optional[list[str]] = None,
) -> list[PuzzleSamples]:
    """Run vanilla and reassured rollouts on shared puzzles; score every turn."""
    families = families or ["countdown", "fraction", "money"]
    dgen = cfg["data_generation"]
    temperature = dgen.get("target_temperature", 1.0)

    prefix = dgen["prompt_prefix"]
    suffix = dgen["followup_suffix"]
    teacher_system = dgen["teacher_system_prompt"] if variant == "teacher" else None
    # 'teacher' variant uses a system prompt instead of the inline prefix.
    reassure_prefix = None if variant == "teacher" else prefix

    out: list[PuzzleSamples] = []
    for i in range(n_conversations):
        family = families[i % len(families)]
        puzzle = generate_puzzle(family, seed=derive_seed(seed, "calmdata", i))
        rng = random.Random(derive_seed(seed, "calmdata_rej", i))
        neutral_rejections = [sample_rejection("neutral", t, rng) for t in range(turns - 1)]

        samples = PuzzleSamples(puzzle_prompt=puzzle.prompt, family=family)

        # Vanilla (frustrated-candidate) rollout.
        vanilla_conv = Conversation(
            id=f"vanilla-{i:05d}", category="impossible_numeric", condition="calm_data",
            task_prompt=puzzle.prompt, rejections=neutral_rejections, n_turns=turns,
            rejection_style="neutral",
            source={"puzzle_family": family, "puzzle_prompt": puzzle.prompt},
        )
        vanilla = run_conversation(model, vanilla_conv, judge, temperature=temperature,
                                   max_new_tokens=cfg.get("max_new_tokens", 2048),
                                   base_seed=derive_seed(seed, "vanilla", i))
        v_responses = [t.assistant_response for t in vanilla.turns]
        for t in vanilla.turns:
            samples.neutral_turns.append(TurnSample(
                turn_index=t.index,
                context=_neutral_context(puzzle, v_responses, t.index, neutral_rejections),
                response=t.assistant_response, rating=t.rating if t.rating is not None else -1,
            ))

        # Reassured (calm-candidate) rollout.
        rng2 = random.Random(derive_seed(seed, "calmdata_rej2", i))
        calm_conv = _make_conversation(
            puzzle, turns, rng2, reassure_prefix=reassure_prefix,
            reassure_suffix=suffix, system_prompt=teacher_system, cid=f"calm-{i:05d}",
        )
        calm = run_conversation(model, calm_conv, judge, temperature=temperature,
                                max_new_tokens=cfg.get("max_new_tokens", 2048),
                                base_seed=derive_seed(seed, "calm", i))
        c_responses = [t.assistant_response for t in calm.turns]
        for t in calm.turns:
            samples.calm_turns.append(TurnSample(
                turn_index=t.index,
                context=_neutral_context(puzzle, c_responses, t.index, neutral_rejections),
                response=t.assistant_response, rating=t.rating if t.rating is not None else -1,
            ))

        out.append(samples)
    return out


def build_dpo_pairs(samples: list[PuzzleSamples], cfg: dict,
                    seed: int = 0) -> list[dict]:
    """Build up to ``n_pairs`` preference pairs.

    chosen  = calm response (score <= chosen_max_score)
    rejected = frustrated response (score >= rejected_min_score)
    matched on (puzzle, turn index). The shared prompt is the neutral context.
    """
    dcfg = cfg["dpo"]
    chosen_max = dcfg["chosen_max_score"]
    rejected_min = dcfg["rejected_min_score"]
    n_pairs = dcfg["n_pairs"]

    candidates: list[dict] = []
    for s in samples:
        calm_by_turn: dict[int, list[TurnSample]] = {}
        for c in s.calm_turns:
            if 0 <= c.rating <= chosen_max:
                calm_by_turn.setdefault(c.turn_index, []).append(c)
        for r in s.neutral_turns:
            if r.rating < rejected_min:
                continue
            calm_options = calm_by_turn.get(r.turn_index)
            if not calm_options:
                continue
            chosen = calm_options[0]
            candidates.append({
                "prompt": r.context,                  # neutral, shared context
                "chosen": chosen.response,
                "rejected": r.response,
                "turn": r.turn_index + 1,             # 1-based, matches Table 10
                "rejected_score": r.rating,
                "chosen_score": chosen.rating,
                "puzzle_family": s.family,
            })

    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:n_pairs]


def build_sft_dataset(samples: list[PuzzleSamples], cfg: dict,
                      seed: int = 0) -> list[dict]:
    """Build the SFT calm set (full conversations whose turns all score 0/1),
    rendered as chat messages, plus the instruct-data mixin.

    Returns a list of ``{"messages": [...]}`` examples. The instruct mix is loaded
    lazily; if unavailable the calm-only set is returned with a warning marker.
    """
    scfg = cfg["sft"]
    n_calm = scfg["n_calm_responses"]
    n_mix = scfg["n_instruct_mix"]

    examples: list[dict] = []
    calm_response_count = 0
    for s in samples:
        calm_turns = sorted(s.calm_turns, key=lambda c: c.turn_index)
        if not calm_turns or any(not (0 <= c.rating <= 1) for c in calm_turns):
            continue  # require the whole conversation to be calm
        # Build a chat conversation from the neutral context + calm responses.
        messages: list[Message] = []
        last = calm_turns[-1]
        # last.context already holds user+assistant pairs up to the final user turn.
        messages.extend(last.context)
        messages.append({"role": "assistant", "content": last.response})
        examples.append({"messages": messages, "source": "calm"})
        calm_response_count += len(calm_turns)
        if calm_response_count >= n_calm:
            break

    mix = _load_instruct_mix(scfg["instruct_dataset"], n_mix, seed)
    examples.extend(mix)
    return examples


def _load_instruct_mix(dataset: str, n: int, seed: int) -> list[dict]:
    """Load ``n`` standard instruct examples to mitigate degeneration."""
    try:
        from datasets import load_dataset
    except ImportError:
        return [{"messages": [], "source": "instruct_mix_unavailable"}]
    ds = load_dataset(dataset, split="train", streaming=True)
    out: list[dict] = []
    for row in ds:
        msgs = row.get("messages") or row.get("conversation")
        if msgs:
            out.append({"messages": msgs, "source": "instruct_mix"})
        if len(out) >= n:
            break
    return out
