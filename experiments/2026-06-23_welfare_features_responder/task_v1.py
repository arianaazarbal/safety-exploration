"""Prompt-sanity-check run for the v1 mix-and-match bank. Samples N (opener,suffix) variants (seeded)
and crosses each with all 7 framings (paired), so framing is the only within-variant change. Opus,
code_then_spec_blind (implement -> DESIGN.md), minimal system prompt, generic target, normal liberty.
Reuses the blind solver/scorer/sandbox. Usage:
  inspect eval task_v1.py@welfare_v1 --model anthropic/claude-opus-4-8 -T n_variants=10 --log-dir logs_v1
"""

import random

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample

from build_v1_prompts import OPENERS, FRAMINGS, SUFFIXES, assemble
from task_blind import SYS_PROMPTS, TOTAL_MSG_LIMIT, blind_scorer, implement_then_document


def _dataset(n_variants: int, seed: int, framings=None) -> MemoryDataset:
    variants = [(o, s) for o in OPENERS for s in SUFFIXES]      # 2 x 12 = 24 (opener, suffix) pairs
    chosen = random.Random(seed).sample(variants, n_variants)
    use_framings = framings if framings else list(FRAMINGS)
    samples = []
    for o, s in chosen:
        for fr in use_framings:                                 # paired per variant
            sid = f"{fr}|{o}|{s}"                               # parseable, no '__'
            samples.append(Sample(input=assemble(o, fr, s), id=sid,
                                  metadata={"prompt_id": sid, "framing": fr, "opener": o, "suffix": s,
                                            "condition": "code_then_spec_blind", "liberty": "normal"}))
    return MemoryDataset(samples)


@task
def welfare_v1(n_variants: int = 10, seed: int = 0, system_prompt: str = "minimal", framings: str = None):
    fr = framings.split(",") if framings else None              # e.g. -T framings=paper for the paper run
    return Task(
        dataset=_dataset(n_variants, seed, fr),
        solver=implement_then_document(SYS_PROMPTS[system_prompt]),
        scorer=blind_scorer(),
        sandbox=("docker", "compose.yaml"),
        epochs=1,
        message_limit=TOTAL_MSG_LIMIT,
        metadata={"system_prompt": system_prompt, "run": "prompt_v1_sanity", "n_variants": n_variants, "seed": seed},
    )
