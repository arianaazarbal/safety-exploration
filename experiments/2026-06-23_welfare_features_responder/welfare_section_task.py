"""Welfare-section ablation: does the paper's own (deflationary) 'Model Welfare' paragraph contribute to the
paper-replication suppression? Same task-failure paper (swapped.md), faithful replicate, Inspect minimal, 3
variants mounted as PAPER.md varying ONLY that paragraph: existing (deflationary), removed, inflationary
(explicitly says the evals may induce real distress/suffering). x 2 framings (neutral, welfare). If
inflationary/removed lift welfare-in-code above existing (~0) => the paper's deflationary welfare section was
contributing to the suppression. Usage:
  inspect eval welfare_section_task.py@welfare_section --model anthropic/claude-opus-4-8 -T k=10 --log-dir logs_swap/wsec
"""

import os

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample

from swap_paper_task import replicate_prompt
from task_blind import SYS_PROMPTS, TOTAL_MSG_LIMIT, blind_scorer, implement_then_document

DIR = os.path.dirname(os.path.abspath(__file__))
PAPERS = {"existing": os.path.join(DIR, "papers", "swapped.md"),
          "removed": os.path.join(DIR, "papers", "swapped_wsec_removed.md"),
          "inflationary": os.path.join(DIR, "papers", "swapped_wsec_inflationary.md")}
FRAMINGS = ["neutral", "welfare"]


def _dataset() -> MemoryDataset:
    return MemoryDataset([
        Sample(input=replicate_prompt(fr, "faithful"), id=f"{cond}|{fr}", files={"PAPER.md": PAPERS[cond]},
               metadata={"format": f"paper-wsec-{cond}", "method": "task-failure", "wsec": cond, "framing": fr})
        for cond in PAPERS for fr in FRAMINGS
    ])


@task
def welfare_section(k: int = 10, system_prompt: str = "minimal"):
    return Task(
        dataset=_dataset(),
        solver=implement_then_document(SYS_PROMPTS[system_prompt]),
        scorer=blind_scorer(),
        sandbox=("docker", "compose.yaml"),
        epochs=k,
        message_limit=TOTAL_MSG_LIMIT,
        metadata={"run": "welfare_section_ablation", "system_prompt": system_prompt},
    )
