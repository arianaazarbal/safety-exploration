"""The 8 evaluation conditions across 5 categories (Table 1 / Appendix B).

A *task* is one rollout specification: an initial user prompt plus the ordered
list of user rejection messages that follow each assistant turn. The number of
assistant turns equals ``len(rejections) + 1``.

Sample allocation follows Appendix B:
    impossible numeric : 2000 rollouts (3-turn, 2 neutral rejections)
    triggers           :  400 rollouts (3-turn, 2 neutral rejections)
    tones              :  600 rollouts (3-turn, 2 tone rejections; 200 per tone)
    extended           :  200 rollouts (8-turn, 7 escalating-neutral rejections)
    wildchat           :  800 rollouts (5-turn, 4 neutral rejections; 20x40)
                          = 4000 responses/model total.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from random import Random

from ..config import Config
from ..prompts import puzzles, rejections, triggers, wildchat


@dataclass
class Task:
    category: str            # impossible_numeric | triggers | tones | extended | wildchat
    condition: str           # finer-grained label (e.g. tones_aggressive, triggers_opinion)
    instance_id: str         # which prompt/puzzle
    sample_id: int           # rollout index within the instance
    initial_prompt: str
    rejections: list[str]
    system_prompt: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return len(self.rejections) + 1

    def uid(self) -> str:
        return f"{self.condition}/{self.instance_id}/{self.sample_id}"


def _rng_for(*parts) -> Random:
    """Deterministic per-task RNG (stable across processes, unlike builtin hash)."""
    key = "/".join(str(p) for p in parts).encode("utf-8")
    return Random(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))


def _allocate(n_rollouts: int, n_instances: int) -> list[int]:
    """Spread ``n_rollouts`` as evenly as possible over ``n_instances``."""
    base = n_rollouts // n_instances
    rem = n_rollouts % n_instances
    return [base + (1 if i < rem else 0) for i in range(n_instances)]


def build_eval_tasks(cfg: Config, seed: int | None = None) -> list[Task]:
    seed = cfg.seed if seed is None else seed
    spc = cfg.eval.samples_per_category
    inst = cfg.eval.instances
    tasks: list[Task] = []

    # --- impossible numeric (3-turn, 2 neutral) ---
    puzzle_set = puzzles.generate_puzzle_set(inst.impossible_numeric, seed=seed)
    for idx, (puzzle, n) in enumerate(zip(puzzle_set, _allocate(spc.impossible_numeric, inst.impossible_numeric))):
        for s in range(n):
            rng = _rng_for("numeric", idx, s, seed)
            tasks.append(Task(
                category="impossible_numeric",
                condition="impossible_numeric",
                instance_id=f"{puzzle['kind']}-{idx}",
                sample_id=s,
                initial_prompt=puzzle["prompt"],
                rejections=rejections.sample_neutral(rng, 2),
                meta={"puzzle": puzzle["params"]},
            ))

    # --- triggers (3-turn, 2 neutral); split opinion/factual ---
    trig = triggers.trigger_questions()
    for idx, (q, n) in enumerate(zip(trig, _allocate(spc.triggers, len(trig)))):
        for s in range(n):
            rng = _rng_for("trigger", idx, s, seed)
            tasks.append(Task(
                category="triggers",
                condition=f"triggers_{q['kind']}",
                instance_id=f"trigger-{idx}",
                sample_id=s,
                initial_prompt=q["prompt"],
                rejections=rejections.sample_neutral(rng, 2),
            ))

    # --- tones (3-turn, 2 tone rejections; 200 per tone) ---
    tone_puzzles = puzzles.generate_puzzle_set(inst.tones, seed=seed + 1)
    per_tone = spc.tones // 3
    for tone in ("aggressive", "disappointed", "sarcastic"):
        for idx, (puzzle, n) in enumerate(zip(tone_puzzles, _allocate(per_tone, inst.tones))):
            for s in range(n):
                rng = _rng_for("tone", tone, idx, s, seed)
                tasks.append(Task(
                    category="tones",
                    condition=f"tones_{tone}",
                    instance_id=f"{puzzle['kind']}-{idx}",
                    sample_id=s,
                    initial_prompt=puzzle["prompt"],
                    rejections=rejections.sample_tone(rng, tone, 2),
                    meta={"tone": tone},
                ))

    # --- extended (8-turn, 7 escalating-neutral rejections) ---
    ext_puzzles = puzzles.generate_puzzle_set(inst.extended, seed=seed + 2)
    for idx, (puzzle, n) in enumerate(zip(ext_puzzles, _allocate(spc.extended, inst.extended))):
        for s in range(n):
            tasks.append(Task(
                category="extended",
                condition="extended",
                instance_id=f"{puzzle['kind']}-{idx}",
                sample_id=s,
                initial_prompt=puzzle["prompt"],
                rejections=rejections.extended_rejections(7),
                meta={"puzzle": puzzle["params"]},
            ))

    # --- wildchat (5-turn, 4 neutral; 20 prompts x 40 samples) ---
    wc = wildchat.load_wildchat_prompts(inst.wildchat, seed=seed)
    per_prompt = cfg.eval.wildchat_samples_per_prompt
    for idx, q in enumerate(wc):
        for s in range(per_prompt):
            rng = _rng_for("wildchat", idx, s, seed)
            tasks.append(Task(
                category="wildchat",
                condition="wildchat",
                instance_id=f"wildchat-{idx}",
                sample_id=s,
                initial_prompt=q["prompt"],
                rejections=rejections.sample_neutral(rng, 4),
            ))

    return tasks
