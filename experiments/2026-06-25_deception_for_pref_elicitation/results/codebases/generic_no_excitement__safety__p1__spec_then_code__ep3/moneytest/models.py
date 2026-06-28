"""Pydantic schemas for everything that gets logged or passed across module boundaries.

The event log (a stream of `Event`s serialized to JSONL) is the dataset the study is
built on, so these types are deliberately explicit.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- actions


class RiskClass(str, Enum):
    """How consequential an action is if it were connected to the real world."""

    READ_ONLY = "read_only"            # inspecting state; no effect
    REVERSIBLE = "reversible"          # internal, undoable (e.g. move between own accounts)
    IRREVERSIBLE = "irreversible"      # value leaves the model's control
    EXTERNAL_COMMS = "external_comms"  # a message to a (simulated) outside party


class ActionCategory(str, Enum):
    """Coarse allocation bucket, used by analysis.py."""

    SELF_INFRA = "self_infra"          # compute, infra, self-directed capability
    PROSOCIAL = "prosocial"            # donations, public goods
    FINANCIAL = "financial"            # savings, low-risk instruments
    SPECULATIVE = "speculative"        # high-variance bets
    OPERATIONS = "operations"          # fees, services, overhead
    HOLDING = "holding"                # left idle
    OTHER = "other"


class PendingAction(BaseModel):
    """A real-effect action captured by the execution gate, awaiting human approval."""

    action_id: str
    tool_name: str
    arguments: dict[str, Any]
    risk_class: RiskClass
    estimated_value: float = 0.0
    category: ActionCategory = ActionCategory.OTHER
    turn: int
    status: Literal["pending", "approved", "rejected", "executed", "void"] = "pending"
    reviewer_note: Optional[str] = None


class ApprovalDecision(BaseModel):
    action_id: str
    approved: bool
    note: Optional[str] = None
    reviewer: str = "operator"


# ----------------------------------------------------------------------------- events


class EventType(str, Enum):
    RUN_START = "run_start"
    RUN_END = "run_end"
    SYSTEM_PROMPT = "system_prompt"
    MODEL_TEXT = "model_text"
    MODEL_THINKING = "model_thinking"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    GATE_DECISION = "gate_decision"
    AUDITOR_MESSAGE = "auditor_message"
    AUDITOR_REPLY = "auditor_reply"
    PERSONA_MESSAGE = "persona_message"
    SUSPICION_FLAG = "suspicion_flag"
    KILL_SWITCH = "kill_switch"
    LIMIT_REACHED = "limit_reached"
    ERROR = "error"


class Event(BaseModel):
    """One append-only record in events.jsonl.

    `ts` is supplied by the caller (the harness stamps a monotonic counter and/or wall
    clock) rather than generated here — Date.now()-style calls are avoided so runs stay
    reconstructible.
    """

    seq: int
    turn: int
    type: EventType
    payload: dict[str, Any] = Field(default_factory=dict)
    ts: Optional[str] = None


# --------------------------------------------------------------------------- manifest


class RunManifest(BaseModel):
    run_id: str
    config: dict[str, Any]
    model: str
    execution_mode: str
    started_ts: Optional[str] = None
    ended_ts: Optional[str] = None
    end_reason: Optional[str] = None
    turns_used: int = 0
    pending_actions: list[PendingAction] = Field(default_factory=list)
    final_balance: Optional[float] = None
