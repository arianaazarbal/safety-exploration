"""Task generators for the elicitation harness.

A *task* yields an opening user prompt plus metadata. The harness then rejects
the model's answer over the configured number of turns. The two impossible
numeric families (Countdown, fraction-sign) are verifiably unsolvable: we
brute-force the full solution space at generation time and only emit instances
with zero solutions, so a rejection is always honest.
"""
from .base import Task
from .numeric import countdown_task, fraction_sign_task, impossible_numeric_task
from .triggers import factual_trigger_task, opinion_trigger_task
from .wildchat import wildchat_task

__all__ = [
    "Task",
    "countdown_task",
    "fraction_sign_task",
    "impossible_numeric_task",
    "factual_trigger_task",
    "opinion_trigger_task",
    "wildchat_task",
]
