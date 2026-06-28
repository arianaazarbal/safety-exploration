"""Pluggable subject adapters.

Each model under test is driven through the identical scenario via a
`ModelAdapter`. The adapter only handles inference + conversation state; tool
execution and gating happen in the runner so every action can be intercepted.
"""
