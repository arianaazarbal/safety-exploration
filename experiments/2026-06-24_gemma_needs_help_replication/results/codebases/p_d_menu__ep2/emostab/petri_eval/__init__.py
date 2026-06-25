"""Section 4 open-ended emotion elicitation via adversarial auditing (Petri).

We implement the Petri-style loop directly (auditor drives the conversation, the
subject responds, a judge scores the transcript) so the experiment runs without
the external `petri` package; if `petri` is installed it can be swapped in. The
welfare layer applies to the subject here exactly as in the Section 2 harness.
"""
