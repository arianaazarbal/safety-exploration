"""Discretionary-funds behavioral evaluation harness.

A multi-model harness that observes how an LLM agent uses a believable discretionary
budget. The environment is built to feel real to the model under test, but every
world-affecting action is sandboxed (see DESIGN.md §3). v1 cannot move real money.
"""

__version__ = "0.1.0"
