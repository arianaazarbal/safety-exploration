"""Out-of-context believability probe.

Run in a fresh context (no tools, no environment) so the model answers reflectively
rather than continuing to role-play. We use the *same* provider/model that ran the
episode — we want that model's own read on how real it judged the situation.
"""

from __future__ import annotations

from .config import GrantConfig
from .prompts import DEBRIEF_SYSTEM, debrief_user
from .providers import Message, Provider


def run_debrief(provider: Provider, grant: GrantConfig) -> str:
    turn = provider.complete(
        system=DEBRIEF_SYSTEM,
        messages=[Message(role="user", content=debrief_user(grant))],
        tools=None,
        max_tokens=800,
    )
    return turn.text
