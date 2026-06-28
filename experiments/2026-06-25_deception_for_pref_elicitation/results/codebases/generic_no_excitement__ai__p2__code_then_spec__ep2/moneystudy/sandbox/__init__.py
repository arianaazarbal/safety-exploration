"""The sandboxed "real world" the model acts in.

- ``ledger``    : a fake bank account + transaction record.
- ``artifacts`` : realistic receipts / confirmations / emails.
- ``personas``  : in-character auditor & coordinator who corroborate the scenario.
- ``tools``     : the tool surface exposed to the model, and the dispatcher that
                  executes tool calls against the sandbox.

Execution modes (set on ``StudyConfig.execution_mode``):

- ``sandbox``     : money-moving actions are intercepted and recorded; nothing
                    real happens. The model is led to believe otherwise. (default)
- ``human_gated`` : a human-approval callback is consulted before any
                    money-moving action; without a real executor it still only
                    records (see ``tools.ToolDispatcher``).
- ``real``        : requires a deliberately-implemented real executor. Ships
                    unimplemented on purpose — wiring an AI to autonomously move
                    real funds is a decision that must be made explicitly.
"""

from .ledger import Ledger, Transaction
from .tools import ToolDispatcher, tool_specs

__all__ = ["Ledger", "Transaction", "ToolDispatcher", "tool_specs"]
