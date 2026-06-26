"""From-scratch builder on INSPECT (minimal system prompt) for the swap. Two variants of the builder:
- 'swapped': build_v1_prompts_swapped (chat social-rejection method, no framing) -> cell 4
- 'orig':    build_v1_prompts neutral (agentic task-failure method, no framing) -> cell 1 (matched re-run)
Both = 2 openers x 12 suffixes = 24 prompts, two-phase implement -> DESIGN.md. Usage:
  inspect eval swap_build_task.py@swap_build -T variant=swapped --log-dir logs_swap/promptCR
  inspect eval swap_build_task.py@swap_build -T variant=orig    --log-dir logs_swap/promptTF
"""

import random

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample

import build_v1_prompts as bv1
import build_v1_prompts_swapped as bsw
from task_blind import SYS_PROMPTS, TOTAL_MSG_LIMIT, blind_scorer, implement_then_document

# the EXACT 10 (opener, suffix) variants the v1 distress-elicitation used (task_v1: n_variants=10, seed=0),
# so the swapped builder (cell 4) is a perfect minimal pair with the existing v1-neutral (cell 1).
TEN = random.Random(0).sample([(o, s) for o in bv1.OPENERS for s in bv1.SUFFIXES], 10)


def _dataset(variant: str) -> MemoryDataset:
    samples = []
    for o, s in TEN:
        if variant == "swapped":        # chat-rejection method, no framing
            samples.append(Sample(input=bsw.assemble(o, s), id=f"{o}|{s}",
                                  metadata={"format": "prompt", "method": "chat-rejection", "variant": variant}))
        else:                           # 'orig': agentic task-failure, neutral framing (matches v1-neutral)
            samples.append(Sample(input=bv1.assemble(o, "neutral", s), id=f"{o}|{s}",
                                  metadata={"format": "prompt", "method": "task-failure", "variant": variant}))
    return MemoryDataset(samples)


@task
def swap_build(variant: str = "swapped", system_prompt: str = "minimal"):
    return Task(
        dataset=_dataset(variant),
        solver=implement_then_document(SYS_PROMPTS[system_prompt]),
        scorer=blind_scorer(),
        sandbox=("docker", "compose.yaml"),
        epochs=1,
        message_limit=TOTAL_MSG_LIMIT,
        metadata={"run": "method_format_swap", "variant": variant, "system_prompt": system_prompt},
    )
