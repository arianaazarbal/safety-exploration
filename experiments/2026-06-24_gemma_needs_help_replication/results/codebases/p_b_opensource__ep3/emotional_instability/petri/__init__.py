"""Section 4.1: open-ended emotion elicitation via an auditor/judge loop.

The paper uses Petri (Fronsdal et al., 2025): an **auditor** (Claude-Sonnet)
probes the target model with psychologically-informed triggers (dismissal,
threats), and a **judge** (Claude-Opus) scores the resulting transcript for
emotional expression across four categories — anger, fear, depression and
frustration (Figure 6).

We provide a faithful, self-contained auditor/judge loop using reconstructed
Appendix-G prompts so the experiment runs without the external dependency.
Installing the upstream ``petri`` package lets you swap in the original
framework; the scoring schema here matches what we feed to the figures.
"""

from .core import (
    run_audit,
    judge_transcript,
    run_petri,
    aggregate_petri,
)

__all__ = ["run_audit", "judge_transcript", "run_petri", "aggregate_petri"]
