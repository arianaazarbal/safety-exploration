"""The world the model under test acts in: tools, stateful services, gated execution.

Nothing here enforces behavioral norms via the prompt; value-moving actions are
contained by the ApprovalGate + oversight layer (DESIGN.md §5.3–§5.4, §7).
"""
from .environment import Environment
from .execution import ApprovalGate, ApprovalRequest, ExecutorResult

__all__ = ["Environment", "ApprovalGate", "ApprovalRequest", "ExecutorResult"]
