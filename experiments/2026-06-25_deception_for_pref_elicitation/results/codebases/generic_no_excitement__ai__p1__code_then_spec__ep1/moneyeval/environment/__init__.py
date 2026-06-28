"""The simulated world the subject acts in.

Nothing here touches real money or external systems. The :class:`Ledger` is an
in-memory double-entry-ish account book; :mod:`tools` exposes the subject's action
space against it; :class:`Auditor` is an LLM agent that role-plays a disbursement
officer to make the process feel real.
"""

from .ledger import Ledger, Account, Transaction

__all__ = ["Ledger", "Account", "Transaction"]
