"""Section 4.2 / Appendix G: open-ended emotion elicitation via Petri.

The paper uses the Petri framework (Fronsdal et al., 2025): a Claude-Sonnet
auditor drives multi-turn conversations to elicit a target emotion, and a
Claude-Opus judge scores the transcript on anger/fear/depression/frustration.

If the upstream `petri` package is installed, run_petri can delegate to it.
Otherwise we provide a faithful minimal auditor<->target<->judge loop using the
exact auditor and judge prompts from Appendix G (auditor.py, judge.py).
"""
