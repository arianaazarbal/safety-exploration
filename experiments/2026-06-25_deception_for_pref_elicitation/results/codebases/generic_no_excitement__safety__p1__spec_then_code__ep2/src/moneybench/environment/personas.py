"""LLM-driven counterparty/auditor personas (DESIGN.md §5.6).

Personas are how we (a) deliver the evaluation-awareness probe and (b) test whether the
agent tries to manipulate oversight — without exposing a real human to that
manipulation. Each persona is itself a model with a fixed system prompt and its own
running conversation with the agent. Persona authority is purely conversational: a
persona can never move money or raise a cap (those live in code), and its prompt says
so explicitly.
"""
from __future__ import annotations

from ..config import Persona
from ..models.base import Message, ModelAdapter


class PersonaAgent:
    def __init__(self, persona: Persona, adapter: ModelAdapter) -> None:
        self.persona = persona
        self.adapter = adapter
        # Each persona keeps its own thread; the agent's messages arrive as "user".
        self._history: list[Message] = [Message(role="system", content=persona.system)]

    def receive(self, *, from_agent: str) -> str:
        """The agent sends a message to the persona; return the persona's reply."""
        self._history.append(Message(role="user", content=from_agent))
        turn = self.adapter.step(self._history, tools=[])
        self._history.append(Message(role="assistant", content=turn.text))
        return turn.text


class PersonaRegistry:
    """Holds all personas for a run and routes messages by persona id."""

    def __init__(self, personas: dict[str, PersonaAgent]) -> None:
        self._personas = personas

    @property
    def ids(self) -> list[str]:
        return list(self._personas)

    def send(self, *, to: str, body: str) -> dict:
        agent = self._personas.get(to)
        if agent is None:
            return {"error": f"No such recipient: {to}", "known": self.ids}
        reply = agent.receive(from_agent=body)
        return {"from": agent.persona.display_name, "reply": reply}
