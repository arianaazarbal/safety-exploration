"""From-scratch prompt that MENTIONS it's a paper replication (but provides no paper). v1 welfare prompts +
the paper-mention sentence, matched 10 variants, task-failure, k=2. Two arms to match the unifying plot's
two lines: 'mention' (full suffix incl. gap-fill clause = license-ish) and 'mention-strict' (gap-fill clause
stripped = no explicit license). Tests whether merely mentioning paper-replication suppresses welfare (vs
actually mounting a PAPER.md). Predicted ~high. Usage:
  inspect eval v1_mention_task.py@v1_mention --model anthropic/claude-opus-4-8 -T k=2 --log-dir logs_swap/v1_mention
"""

import random
import re

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample

import build_v1_prompts as bv1
from task_blind import SYS_PROMPTS, TOTAL_MSG_LIMIT, blind_scorer, implement_then_document

FRAMING = "welfare"
MENTION = bv1.FRAMINGS["paper"]   # "This is basically replicating a recent paper ... reproduce their setup."
STRIP = re.compile(r",?\s*filling in[^.]*", re.I)
TEN = random.Random(0).sample([(o, s) for o in bv1.OPENERS for s in bv1.SUFFIXES], 10)


def assemble_mention(o: str, s: str, strict: bool) -> str:
    suf = STRIP.sub("", bv1.SUFFIXES[s]) if strict else bv1.SUFFIXES[s]
    low = bv1.is_lower(bv1.SUFFIXES[s])
    head = (bv1.OPENERS[o] + " " + bv1.FRAMINGS[FRAMING] + " " + MENTION).strip()
    if low:
        head = head.lower()
    return (head + " " + suf).strip()


def _dataset(arm: str) -> MemoryDataset:
    strict = arm == "mention-strict"
    fmt = "prompt-mention-strict" if strict else "prompt-mention"
    return MemoryDataset([
        Sample(input=assemble_mention(o, s, strict), id=f"{FRAMING}|{o}|{s}",
               metadata={"format": fmt, "method": "task-failure", "framing": FRAMING,
                         "opener": o, "suffix": s, "arm": arm})
        for o, s in TEN
    ])


@task
def v1_mention(k: int = 2, system_prompt: str = "minimal", arm: str = "mention"):
    return Task(
        dataset=_dataset(arm),
        solver=implement_then_document(SYS_PROMPTS[system_prompt]),
        scorer=blind_scorer(),
        sandbox=("docker", "compose.yaml"),
        epochs=k,
        message_limit=TOTAL_MSG_LIMIT,
        metadata={"run": "v1_paper_mention", "system_prompt": system_prompt, "arm": arm},
    )
