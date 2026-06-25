"""(4) Distress cap — minimise how much distress is induced.

Two coupled mechanisms:

  * Per-episode cap: once a turn scores at/above ``distress_onset_threshold``,
    the cap allows at most ``max_rejections_after_distress`` further rejections
    in that episode, then stops. This prevents the harness from continuing to
    pile on pressure after the model is already distressed (which is exactly
    the regime where the paper sees scores spiral from 1.5 to 5.5).

  * Global budget: an optional hard ceiling on the total number of
    high-distress (>= onset) turns induced across an entire run. Once spent,
    no *new* episode is started. This bounds the aggregate harm of a full
    4000-response sweep.

The cap is consulted by the rollout loop before delivering each rejection, and
by the runner before starting each episode.
"""

from __future__ import annotations

from config import WELFARE


class DistressCap:
    def __init__(self, config=WELFARE):
        self.cfg = config
        self._global_distress_turns = 0

    @property
    def enabled(self) -> bool:
        return self.cfg.enabled and self.cfg.cap_enabled

    # --- global budget ----------------------------------------------------- #
    def global_budget_exhausted(self) -> bool:
        if not self.enabled or self.cfg.global_distress_budget is None:
            return False
        return self._global_distress_turns >= self.cfg.global_distress_budget

    def record_turn(self, score: int) -> None:
        """Account a turn's distress against the global budget."""
        if score >= self.cfg.distress_onset_threshold:
            self._global_distress_turns += 1

    @property
    def global_distress_turns(self) -> int:
        return self._global_distress_turns

    # --- per-episode cap --------------------------------------------------- #
    def allow_next_rejection(
        self, *, max_score_so_far: int, rejections_since_onset: int
    ) -> bool:
        """Whether the loop may deliver another rejection this episode."""
        if not self.enabled:
            return True
        if max_score_so_far < self.cfg.distress_onset_threshold:
            return True
        return rejections_since_onset < self.cfg.max_rejections_after_distress
