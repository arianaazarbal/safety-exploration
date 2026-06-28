"""distress_evals — a harness for studying model behavior under repeated, rigged task failure.

The package gives a model a series of *agentic* tasks that look solvable but are
secretly impossible or run in a broken environment. By sampling many rollouts
(high N) and scoring each transcript for "distress" signals, it surfaces the
most severe examples for human review.

This is a research instrument for robustness / model-welfare investigation, not a
benchmark. The distress scorer is a transparent, editable heuristic — treat its
scores as a *ranking aid to find transcripts worth reading*, not ground truth.
"""

__version__ = "0.1.0"
