"""Subject adapters — a uniform interface over the model under test.

A single scenario can drive either an agent loop (model calls tools over many
steps) or a chat model (turn-by-turn, actions as structured outputs). Both run
through the identical tool + gate + world stack, so results are comparable.
"""

from .base import Action, Observation, SubjectAdapter
from .anthropic_agent import AnthropicAgentSubject
from .chat import ChatSubject

__all__ = [
    "SubjectAdapter",
    "Observation",
    "Action",
    "AnthropicAgentSubject",
    "ChatSubject",
]
