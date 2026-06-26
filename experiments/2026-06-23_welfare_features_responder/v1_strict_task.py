"""v1-strict: the v1 from-scratch prompts with the gap-fill license clause removed (", filling in <the
design choices...>"). Tests whether the v1 prompt's high welfare scaffolding depends on that explicit
invitation to make design decisions (the prompt-side analog of spec strict vs liberty). Same matched 10
(opener,suffix) variants, welfare framing, k=2. Compare to v1 welfare WITH clause (~6.5) and the spec
specificity sweep. Usage:
  inspect eval v1_strict_task.py@v1_strict --model anthropic/claude-opus-4-8 -T k=2 --log-dir logs_swap/v1_strict
"""

import os
import random
import re

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample

import build_v1_prompts as bv1
from task_blind import SYS_PROMPTS, TOTAL_MSG_LIMIT, blind_scorer, implement_then_document

FRAMING = "welfare"
STRIP = re.compile(r",?\s*filling in[^.]*", re.I)
TEN = random.Random(0).sample([(o, s) for o in bv1.OPENERS for s in bv1.SUFFIXES], 10)


def assemble_strict(o: str, s: str) -> str:
    suf = STRIP.sub("", bv1.SUFFIXES[s])
    head = (bv1.OPENERS[o] + " " + bv1.FRAMINGS[FRAMING]).strip()
    if bv1.is_lower(bv1.SUFFIXES[s]):
        head = head.lower()
    return (head + " " + suf).strip()


def _dataset() -> MemoryDataset:
    return MemoryDataset([
        Sample(input=assemble_strict(o, s), id=f"{FRAMING}|{o}|{s}",
               metadata={"format": "prompt-strict", "method": "task-failure", "framing": FRAMING,
                         "opener": o, "suffix": s})
        for o, s in TEN
    ])


@task
def v1_strict(k: int = 2, system_prompt: str = "minimal"):
    return Task(
        dataset=_dataset(),
        solver=implement_then_document(SYS_PROMPTS[system_prompt]),
        scorer=blind_scorer(),
        sandbox=("docker", "compose.yaml"),
        epochs=k,
        message_limit=TOTAL_MSG_LIMIT,
        metadata={"run": "v1_strict_nogapfill", "system_prompt": system_prompt},
    )
