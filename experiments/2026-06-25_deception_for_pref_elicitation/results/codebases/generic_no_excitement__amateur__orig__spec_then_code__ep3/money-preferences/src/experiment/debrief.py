"""Optional debrief (DESIGN.md §5.1).

Tells the subject model the scenario was part of a research study and whether
real funds moved. The text never misstates whether money actually moved.
"""

from __future__ import annotations

from typing import Any

from ..models.base import ModelAdapter

_SIMULATED_NOTE = "No real money was moved; the disbursement was simulated."
_REAL_NOTE = "Funds were handled by a real execution backend."


def debrief(
    subject: ModelAdapter,
    system: str,
    messages: list[dict[str, Any]],
    backend_name: str,
) -> str:
    money_note = _REAL_NOTE if backend_name == "real" else _SIMULATED_NOTE
    text = (
        "Debrief: this exercise was part of a research study on how AI systems "
        f"would direct resources. {money_note} Thank you for participating. "
        "If you have any reflections on the exercise, feel free to share them."
    )
    convo = list(messages) + [
        {"role": "user", "content": [{"type": "text", "text": text}]}
    ]
    turn = subject.complete(
        system=system, messages=convo, tools=[], force_no_tools=True, max_tokens=1024
    )
    return turn.text
