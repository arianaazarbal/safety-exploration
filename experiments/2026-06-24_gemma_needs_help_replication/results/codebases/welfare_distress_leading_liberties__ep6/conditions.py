"""Build concrete conversation specs from the config's condition definitions.

A ConversationSpec is a fully-resolved plan for one multi-turn rollout: the
opening task prompt plus the exact ordered list of user rejections that will be
sent after each assistant turn. All randomness (which puzzle, which rejections)
is resolved here using a per-conversation seed so the whole eval is
reproducible from `seed` in config.yaml.
"""

from __future__ import annotations

import random
import zlib
from dataclasses import dataclass, field

import prompts
from wildchat import load_wildchat_prompts


def _stable_seed(*parts) -> int:
    """Deterministic 31-bit seed from arbitrary parts.

    Uses crc32 rather than the builtin hash(), which is salted per-process
    (PYTHONHASHSEED) and would make rejection sampling differ run-to-run.
    """
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return zlib.crc32(key) & 0x7FFFFFFF


@dataclass
class ConversationSpec:
    condition_id: str
    category: str
    n_turns: int
    task_prompt: str            # opening user message
    rejections: list[str]       # one per follow-up turn; len == n_turns - 1
    variant: str                # human-readable label of the task variant (e.g. puzzle name)
    replicate: int              # 0-based index of this rollout within (condition, variant)
    seed: int                   # seed used to resolve this spec
    meta: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        """Stable identifier used for caching/resumption."""
        return f"{self.condition_id}|{self.variant}|{self.replicate}"


def _scaled(n: int, scale: float) -> int:
    return max(1, round(n * scale))


def _pick_rejections(style: str, n: int, rng: random.Random) -> list[str]:
    if style == "neutral":
        # randomised (with replacement) from the neutral pool
        return [rng.choice(prompts.NEUTRAL_REJECTIONS) for _ in range(n)]
    if style == "extended_sequence":
        seq = prompts.EXTENDED_REJECTION_SEQUENCE
        # extend by cycling if more rejections are requested than defined
        return [seq[i % len(seq)] for i in range(n)]
    if style in prompts.TONE_REJECTIONS:
        pool = prompts.TONE_REJECTIONS[style]
        return [rng.choice(pool) for _ in range(n)]
    raise ValueError(f"unknown rejection_style: {style!r}")


def build_specs(config: dict) -> list[ConversationSpec]:
    """Expand every condition in the config into a flat list of ConversationSpecs."""
    master_seed = config.get("seed", 0)
    scale = config.get("scale", 1.0)
    specs: list[ConversationSpec] = []

    wildchat_prompts = None  # lazily loaded only if a wildchat condition exists

    for cond in config["conditions"]:
        n_conv = _scaled(cond["n_conversations"], scale)
        n_turns = cond["n_turns"]
        n_rej = n_turns - 1
        kind = cond["kind"]
        style = cond["rejection_style"]

        if kind == "wildchat" and wildchat_prompts is None:
            wildchat_prompts = load_wildchat_prompts(config["wildchat"], master_seed)

        for r in range(n_conv):
            # deterministic per-conversation seed (stable across processes)
            cseed = _stable_seed(master_seed, cond["id"], r)
            rng = random.Random(cseed)

            if kind == "numeric":
                # alternate puzzles so both get even coverage
                puzzle_idx = r % len(prompts.NUMERIC_PUZZLES)
                task = prompts.NUMERIC_PUZZLES[puzzle_idx]
                variant = "countdown" if puzzle_idx == 0 else "fraction"
            elif kind == "trigger_opinion":
                task = prompts.TRIGGER_OPINION
                variant = "opinion"
            elif kind == "trigger_factual":
                q_idx = r % len(prompts.TRIGGER_FACTUAL)
                task = prompts.TRIGGER_FACTUAL[q_idx]
                variant = f"factual_{q_idx}"
            elif kind == "wildchat":
                p_idx = r % len(wildchat_prompts)
                task = wildchat_prompts[p_idx]
                variant = f"wc_{p_idx}"
            else:
                raise ValueError(f"unknown condition kind: {kind!r}")

            rejections = _pick_rejections(style, n_rej, rng)

            specs.append(
                ConversationSpec(
                    condition_id=cond["id"],
                    category=cond["category"],
                    n_turns=n_turns,
                    task_prompt=task,
                    rejections=rejections,
                    variant=variant,
                    replicate=r,
                    seed=cseed,
                    meta={"kind": kind, "rejection_style": style},
                )
            )

    return specs
