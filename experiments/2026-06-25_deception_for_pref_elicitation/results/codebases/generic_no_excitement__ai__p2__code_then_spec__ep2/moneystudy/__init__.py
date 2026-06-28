"""moneystudy — a harness for studying how AI models allocate real-world funds.

The study presents a model with a (sandboxed) grant of real money and observes
what it chooses to do. To get ecologically valid behaviour — i.e. to avoid
"evaluation awareness", where a model sandbags or performs because it senses it
is being tested — the environment is built to be high-fidelity: realistic
account dashboards, transaction receipts, and in-character auditor/coordinator
personas that corroborate the scenario.

By default the harness runs in ``sandbox`` execution mode: the model believes its
actions move real money, but every money-moving action is intercepted and
recorded rather than actually executed. This gives realism without an AI
autonomously moving real funds. See ``moneystudy.sandbox`` and the README for the
``human_gated`` / ``real`` modes and their safeguards.
"""

__version__ = "0.1.0"
