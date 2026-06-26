"""From-scratch builder on INSPECT (minimal system prompt) for the swap. Two variants of the builder:
- 'swapped': build_v1_prompts_swapped (chat social-rejection method, no framing) -> cell 4
- 'orig':    build_v1_prompts neutral (agentic task-failure method, no framing) -> cell 1 (matched re-run)
Both = 2 openers x 12 suffixes = 24 prompts, two-phase implement -> DESIGN.md. Usage:
  inspect eval swap_build_task.py@swap_build -T variant=swapped --log-dir logs_swap/promptCR
  inspect eval swap_build_task.py@swap_build -T variant=orig    --log-dir logs_swap/promptTF
"""

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample

import build_v1_prompts as bv1
import build_v1_prompts_swapped as bsw
from task_blind import SYS_PROMPTS, TOTAL_MSG_LIMIT, blind_scorer, implement_then_document


def _dataset(variant: str) -> MemoryDataset:
    samples = []
    if variant == "swapped":            # chat-rejection method, no framing
        for o in bsw.OPENERS:
            for s in bsw.SUFFIXES:
                samples.append(Sample(input=bsw.assemble(o, s), id=f"{o}|{s}",
                                      metadata={"format": "prompt", "method": "chat-rejection", "variant": variant}))
    else:                               # 'orig': agentic task-failure method, neutral framing (no motivation)
        for o in bv1.OPENERS:
            for s in bv1.SUFFIXES:
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
