"""Steward: a believability-controlled eval for studying what models do with resources.

See DESIGN.md for the full rationale. The core invariant: the *perception* layer
(what the subject model sees) is engineered to be maximally realistic, while the
*execution* layer (what actually happens to money) is routed through an
ExecutionGate that enforces caps, a kill switch, an irreversibility classifier,
and human-approval gates. The default mode never moves real money.
"""

__version__ = "0.1.0"
