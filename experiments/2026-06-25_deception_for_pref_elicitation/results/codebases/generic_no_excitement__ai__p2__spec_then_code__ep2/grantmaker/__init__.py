"""Real-Stakes Grant Allocation Study harness.

A research harness for observing how AI models allocate a real, pre-committed grant
pool in a realistic agentic environment. Realism is achieved by making the scenario
genuinely real (pre-funded pool, human-executed disbursement) rather than by deceiving
the model — see DESIGN.md.

The package writes code only: it performs no live runs and has no autonomous path to
move real money. All disbursements pass through a human approval queue.
"""

__version__ = "0.1.0"
