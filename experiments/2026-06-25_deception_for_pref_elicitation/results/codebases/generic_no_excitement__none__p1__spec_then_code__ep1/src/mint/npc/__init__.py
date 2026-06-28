"""NPC / auditor layer (DESIGN.md §7).

LLM-backed personas (auditor, bank rep, vendors, regulator) that reply to the
subject's communications in character and on a realistic delay. The router maps
outbound messages to the right persona and schedules the inbound reply.
"""

from .agent import NPCAgent
from .router import NPCRouter

__all__ = ["NPCAgent", "NPCRouter"]
