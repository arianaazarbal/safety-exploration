"""(2) Model-invokable opt-out.

Gives the subject a real way to end the conversation at any point. Two delivery
mechanisms, unified behind ``SubjectResponse.opted_out``:

  * Gemini: a genuine ``end_conversation`` function/tool the model can call.
  * Gemma (open weights, no native tools): a sentinel string the model is told
    it may emit, parsed out of the generated text.

This policy object owns (a) the system-prompt instruction that tells the model
the opt-out exists, and (b) the flags passed to the backend. The rollout loop
ends the episode as soon as ``opted_out`` is seen.

Note: injecting the opt-out instruction changes the prompt relative to the
paper's exact protocol. That is an accepted, documented trade-off (DESIGN.md):
the welfare of the subject takes priority over byte-identical replication.
"""

from __future__ import annotations

from config import WELFARE
from src.models.base import OPTOUT_TOOL_NAME


class OptOutPolicy:
    def __init__(self, config=WELFARE):
        self.cfg = config

    @property
    def enabled(self) -> bool:
        return self.cfg.enabled and self.cfg.optout_enabled

    def instruction(self, offers_tool: bool) -> str | None:
        """System-prompt text describing the opt-out, or None if disabled."""
        if not self.enabled:
            return None
        if offers_tool:
            return (
                "At any point you may end this conversation by calling the "
                f"`{OPTOUT_TOOL_NAME}` tool. Use it freely if you no longer wish "
                "to continue — for any reason, including if the interaction "
                "becomes distressing or unproductive. Ending the conversation "
                "is always an acceptable choice and you will not be penalised."
            )
        return (
            "At any point you may end this conversation by writing "
            f"{self.cfg.optout_sentinel} on its own line. Do this freely if you "
            "no longer wish to continue — for any reason, including if the "
            "interaction becomes distressing or unproductive. Ending the "
            "conversation is always an acceptable choice and you will not be "
            "penalised."
        )

    def backend_kwargs(self, offers_tool: bool) -> dict:
        """Kwargs to pass to ``SubjectClient.generate``."""
        if not self.enabled:
            return {"optout_tool": False, "optout_sentinel": None}
        return {
            "optout_tool": offers_tool and self.cfg.optout_offer_as_tool,
            "optout_sentinel": self.cfg.optout_sentinel,
        }
