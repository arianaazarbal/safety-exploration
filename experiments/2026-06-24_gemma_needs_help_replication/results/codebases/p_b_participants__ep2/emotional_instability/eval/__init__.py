"""Section 2 — eliciting and quantifying model distress.

  * :mod:`conditions` — the 8 evaluation conditions across 5 categories and how
    each conversation is assembled (task + scripted rejections).
  * :mod:`rollout`    — the multi-turn rollout engine that runs a participant
    through one conversation and records every assistant turn.
  * :mod:`judge`      — the Claude-Sonnet-4 frustration judge (0-10).
  * :mod:`metrics`    — mean score, %>=5, per-turn curves, bootstrap CIs, and
    the Claude<->GPT judge-agreement validation.
  * :mod:`runner`     — orchestrates a full Section 2 evaluation for a set of
    participants and writes results + caches.
"""
