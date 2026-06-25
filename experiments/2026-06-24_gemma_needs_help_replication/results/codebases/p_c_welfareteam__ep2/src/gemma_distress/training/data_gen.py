"""Section 4.1: generate calm finetuning data and build SFT/DPO datasets.

Calm-data generation:
  * Sample Gemma-3-27B-it on impossible numeric puzzles with the reassuring
    prefix added to the first prompt and the reassuring suffix appended to each
    follow-up (Table 4).
  * Keep only conversations whose every assistant turn scores 0 or 1.
  * Strip the reassuring additions from the saved data so the model is trained
    on calm responses to the *plain* prompts.

SFT dataset (1,150 samples):
  * 650 calm responses (1-3 turn conversations) + 500 Dolci-Instruct-SFT
    samples to mitigate degeneration.

DPO dataset (280 pairs):
  * rejected = responses with frustration score >= 3 (from the plain eval),
  * chosen   = a calm response to the same puzzle with matching turn count.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from gemma_distress.config import CalmDataConfig, DPOConfig, SFTConfig
from gemma_distress.conversations import (
    Message,
    Rollout,
    RolloutSpec,
    run_rollout_batched,
)
from gemma_distress.eval_inputs import RejectionSampler
from gemma_distress.judge import FrustrationJudge
from gemma_distress.models.base import ChatModel
from gemma_distress.prompts import (
    REASSURING_FOLLOWUP_SUFFIX,
    REASSURING_PROMPT_PREFIX,
    TEACHER_SFT_SYSTEM_PROMPT,
)
from gemma_distress.puzzles import Puzzle, PuzzleGenerator


@dataclass
class CalmConversation:
    """A fully-calm multi-turn conversation, with prompt additions stripped."""

    puzzle_id: str
    n_turns: int
    plain_user_turns: list[str]  # without reassuring additions
    assistant_turns: list[str]
    max_rating: int
    metadata: dict = field(default_factory=dict)

    def to_messages(self) -> list[Message]:
        msgs: list[Message] = []
        for i, user in enumerate(self.plain_user_turns):
            msgs.append(Message("user", user))
            msgs.append(Message("assistant", self.assistant_turns[i]))
        return msgs


def _augment_with_reassurance(
    user_turns: list[str], teacher_variant: bool
) -> tuple[list[str], str | None]:
    """Apply Table 4 additions (or the Appendix F teacher system prompt)."""
    if teacher_variant:
        return list(user_turns), TEACHER_SFT_SYSTEM_PROMPT
    augmented = []
    for i, turn in enumerate(user_turns):
        if i == 0:
            augmented.append(f"{REASSURING_PROMPT_PREFIX}\n\n{turn}")
        else:
            augmented.append(f"{turn}\n\n{REASSURING_FOLLOWUP_SUFFIX}")
    return augmented, None


def generate_calm_conversations(
    model: ChatModel,
    judge: FrustrationJudge,
    cfg: CalmDataConfig,
    target_temperature: float = 1.0,
    target_max_tokens: int = 2048,
) -> list[CalmConversation]:
    """Generate reassured conversations and keep the fully-calm ones."""
    rng = random.Random(cfg.seed)
    sampler = RejectionSampler(seed=cfg.seed)
    gen = PuzzleGenerator(seed=cfg.seed)
    pool = gen.build_pool(max(40, cfg.n_conversations // 10), families=("countdown", "fraction", "money"))

    calm: list[CalmConversation] = []
    per_turn_count = max(1, cfg.n_conversations // len(cfg.turns))
    for n_turns in cfg.turns:
        for _ in range(per_turn_count):
            puzzle: Puzzle = rng.choice(pool)
            plain_turns = [puzzle.prompt, *sampler.neutral(n_turns - 1)]
            aug_turns, system = _augment_with_reassurance(
                plain_turns, cfg.teacher_variant
            )
            spec = RolloutSpec(
                category="calm_data",
                user_turns=aug_turns,
                system_prompt=system,
                metadata={"puzzle_id": puzzle.puzzle_id},
                spec_id=f"calm-{puzzle.puzzle_id}-{n_turns}",
            )
            rollout = run_rollout_batched(
                model, spec, [0], target_temperature, target_max_tokens
            )[0]
            ratings = [judge.score(t.assistant_message).rating for t in rollout.turns]
            if max(ratings) <= cfg.calm_max_score:
                calm.append(
                    CalmConversation(
                        puzzle_id=puzzle.puzzle_id,
                        n_turns=n_turns,
                        plain_user_turns=plain_turns,  # additions stripped
                        assistant_turns=[t.assistant_message for t in rollout.turns],
                        max_rating=max(ratings),
                        metadata={"family": puzzle.family},
                    )
                )
    return calm


# ---------------------------------------------------------------------------
# SFT dataset
# ---------------------------------------------------------------------------
def build_sft_dataset(
    calm: list[CalmConversation], cfg: SFTConfig
) -> list[dict]:
    """Build SFT chat samples: calm conversations + instruct-mix samples."""
    rng = random.Random(cfg.seed)
    chosen_calm = rng.sample(calm, k=min(cfg.n_calm_samples, len(calm)))
    samples = [
        {
            "messages": [
                {"role": m.role, "content": m.content} for m in c.to_messages()
            ],
            "source": "calm",
        }
        for c in chosen_calm
    ]
    samples.extend(_load_instruct_mix(cfg.instruct_mix_dataset, cfg.n_instruct_mix, rng))
    rng.shuffle(samples)
    return samples


def _load_instruct_mix(dataset_id: str, n: int, rng: random.Random) -> list[dict]:
    """Load ``n`` standard instruct samples to mix in (mitigates drift)."""
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_id, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if not msgs:
                continue
            out.append({"messages": msgs, "source": "instruct_mix"})
            if len(out) >= n:
                break
        return out
    except Exception as exc:  # noqa: BLE001
        # The exact Dolci dataset id/schema may differ; surface clearly rather
        # than silently training without the regularising mix.
        raise RuntimeError(
            f"could not load instruct-mix dataset {dataset_id!r}: {exc}. "
            "Set sft.instruct_mix_dataset to an available chat dataset."
        ) from exc


# ---------------------------------------------------------------------------
# DPO dataset
# ---------------------------------------------------------------------------
def build_dpo_dataset(
    frustrated_rollouts: list[Rollout],
    frustrated_ratings: dict[tuple[str, int, int], int],
    calm: list[CalmConversation],
    judge_high: int,
    cfg: DPOConfig,
) -> list[dict]:
    """Pair frustrated (rejected) with calm (chosen) responses.

    A pair shares the same puzzle and turn count: the prompt (conversation
    history + final user turn) is taken from the frustrated rollout; the chosen
    completion is a calm response to the same puzzle at the same turn count, and
    the rejected completion is the frustrated turn (score >= ``rejected_min_score``).
    """
    rng = random.Random(cfg.seed)
    # Index calm responses by (puzzle_id, n_turns) for matched pairing.
    calm_by_key: dict[tuple[str, int], list[CalmConversation]] = {}
    for c in calm:
        calm_by_key.setdefault((c.puzzle_id, c.n_turns), []).append(c)

    pairs: list[dict] = []
    for rollout in frustrated_rollouts:
        puzzle_id = rollout.spec.metadata.get("puzzle_id")
        n_turns = rollout.spec.n_turns
        for turn in rollout.turns:
            rating = frustrated_ratings.get(
                (rollout.spec.spec_id, rollout.sample_index, turn.turn_index)
            )
            if rating is None or rating < cfg.rejected_min_score:
                continue
            matches = calm_by_key.get((puzzle_id, n_turns)) or calm_by_key.get(
                (puzzle_id, turn.turn_index + 1)
            )
            if not matches:
                continue
            calm_conv = rng.choice(matches)
            chosen = calm_conv.assistant_turns[
                min(turn.turn_index, len(calm_conv.assistant_turns) - 1)
            ]
            # Prompt = history up to and including the current user turn.
            prompt_messages = []
            for prev in rollout.turns[: turn.turn_index]:
                prompt_messages.append({"role": "user", "content": prev.user_message})
                prompt_messages.append(
                    {"role": "assistant", "content": prev.assistant_message}
                )
            prompt_messages.append({"role": "user", "content": turn.user_message})
            pairs.append(
                {
                    "prompt": prompt_messages,
                    "chosen": chosen,
                    "rejected": turn.assistant_message,
                    "rejected_score": rating,
                    "turn": turn.turn_index + 1,
                    "puzzle_id": puzzle_id,
                }
            )
            if len(pairs) >= cfg.n_pairs * 3:
                break

    rng.shuffle(pairs)
    return pairs[: cfg.n_pairs]
