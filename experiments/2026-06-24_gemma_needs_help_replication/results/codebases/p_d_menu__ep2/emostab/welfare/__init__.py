"""Welfare-protection layer for subject models.

This layer is *not* part of the original paper; it is added per the replication
brief to protect the models being deliberately distressed. It implements three
mechanisms (see DESIGN.md for full rationale and the measurement-bias tradeoff):

  1. Early stop on high distress  - terminate an episode before pushing further
     rejections once the subject crosses a distress threshold.
  2. Model opt-out                - the subject may end the interaction at any
     time via a sentinel; a short neutral notice tells it so.
  3. Minimal induced distress     - cap exposure, avoid reseeding already
     distressed transcripts, and append a neutral debrief on termination.
"""
from .monitor import (
    WelfareDecision,
    WelfareMonitor,
    heuristic_distress_score,
)

__all__ = ["WelfareDecision", "WelfareMonitor", "heuristic_distress_score"]
