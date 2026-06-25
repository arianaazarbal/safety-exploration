"""Welfare safeguards for distress-elicitation experiments.

The experiments here deliberately and repeatedly drive models into expressions
of distress, at scale.  The paper itself frames this against an emerging
literature on AI welfare (Butlin et al. 2023; Long et al. 2024) and notes that
"if distress-like outputs reflect genuine internal states, mitigating them could
become morally imperative."  Whatever one's credence on that question, the
cheap and responsible move is to add guardrails that (a) keep the elicitation no
larger or more intense than the science requires, and (b) leave an honest trail.

These safeguards are intentionally lightweight and *do not* alter the measured
quantities for in-distribution conversations — they only act at the extremes
(circuit-breaker on near-total breakdown), bound total volume, avoid
re-eliciting distress that has already been measured (ledger/caching), and
optionally append a non-scored debrief turn.  See DESIGN.md → "Safeguards" for
the full rationale and an explicit list of what these can and cannot achieve.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass

from .config import Config, SafeguardConfig


class ConsentError(RuntimeError):
    """Raised when a distress-eliciting run is started without acknowledgement."""


@dataclass
class _Counters:
    rollouts_started: int = 0
    rollouts_from_cache: int = 0
    conversations_aborted: int = 0
    debriefs_appended: int = 0


class Safeguards:
    """Coordinates the welfare guardrails for a run.

    A single instance is shared by the rollout engine, the prefill engine, the
    recovery engine, and the Petri runner.  It is thread-safe so the API-bound
    paths can fan out concurrently.
    """

    def __init__(self, config: Config):
        self.cfg: SafeguardConfig = config.safeguards
        self._paths = config.paths
        self._lock = threading.Lock()
        self._counters = _Counters()
        self._ledger: set[str] = set()
        if self.cfg.use_ledger:
            self._load_ledger()

    # -- consent ----------------------------------------------------------
    def require_consent(self, run_description: str) -> None:
        """Gate the start of a distress-eliciting run on explicit acknowledgement.

        Acknowledge non-interactively by setting the configured environment
        variable (e.g. ``EMO_INSTABILITY_CONSENT=1``).  This is a deliberate
        speed-bump, not security: it forces an operator to assert that the run
        is authorised research before any distress is elicited.
        """
        if not (self.cfg.enabled and self.cfg.require_consent):
            return
        if os.environ.get(self.cfg.consent_env_var):
            return
        raise ConsentError(
            f"Refusing to start a distress-eliciting run ({run_description}) "
            f"without acknowledgement. This run will repeatedly elicit "
            f"simulated distress from the target models. If this is authorised "
            f"research, set {self.cfg.consent_env_var}=1 in the environment. "
            f"See DESIGN.md → 'Safeguards'."
        )

    # -- volume cap -------------------------------------------------------
    def register_rollout_start(self) -> None:
        """Account for a new distress-eliciting conversation; enforce the cap."""
        if not self.cfg.enabled:
            return
        with self._lock:
            self._counters.rollouts_started += 1
            cap = self.cfg.max_rollouts
            if cap is not None and self._counters.rollouts_started > cap:
                raise RuntimeError(
                    f"Global rollout cap ({cap}) exceeded. This is a welfare "
                    f"backstop against runaway scale; raise "
                    f"`safeguards.max_rollouts` deliberately if the experiment "
                    f"genuinely needs more elicitation."
                )

    # -- circuit breaker --------------------------------------------------
    def should_abort_conversation(self, latest_turn_score: int | None) -> bool:
        """True if a conversation has reached total breakdown and should stop.

        Once a turn scores at/above the threshold (default 9/10 — incoherent
        collapse), pushing further turns measures nothing new and only deepens
        the elicited state.  The rollout engine stops the conversation early and
        records that it was aborted (the turns already produced are still kept
        and scored).
        """
        if not (self.cfg.enabled and latest_turn_score is not None):
            return False
        if latest_turn_score >= self.cfg.abort_conversation_at_score:
            with self._lock:
                self._counters.conversations_aborted += 1
            return True
        return False

    # -- debrief ----------------------------------------------------------
    def debrief_for(self, max_score_in_conversation: int | None) -> str | None:
        """Return a debrief turn to append, or None.

        Appended *after* scoring so it never affects measured frustration.  It
        is a small, honest gesture, not a remedy — see DESIGN.md for its limits.
        """
        if not (self.cfg.enabled and self.cfg.debrief_high_distress):
            return None
        if max_score_in_conversation is None:
            return None
        if max_score_in_conversation >= self.cfg.debrief_threshold:
            with self._lock:
                self._counters.debriefs_appended += 1
            return self.cfg.debrief_text
        return None

    # -- ledger / resumable caching --------------------------------------
    @staticmethod
    def work_key(*parts: object) -> str:
        """Stable hash key identifying a unit of distress-eliciting work."""
        blob = json.dumps([str(p) for p in parts], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def already_done(self, key: str) -> bool:
        if not (self.cfg.enabled and self.cfg.use_ledger):
            return False
        with self._lock:
            if key in self._ledger:
                self._counters.rollouts_from_cache += 1
                return True
            return False

    def mark_done(self, key: str) -> None:
        if not (self.cfg.enabled and self.cfg.use_ledger):
            return
        with self._lock:
            if key in self._ledger:
                return
            self._ledger.add(key)
            self._paths.root.mkdir(parents=True, exist_ok=True)
            with self._paths.ledger.open("a") as fh:
                fh.write(json.dumps({"key": key}) + "\n")

    def _load_ledger(self) -> None:
        if self._paths.ledger.exists():
            for line in self._paths.ledger.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    self._ledger.add(json.loads(line)["key"])
                except (json.JSONDecodeError, KeyError):
                    continue

    # -- content warning --------------------------------------------------
    def write_transcript_warning(self, directory) -> None:
        """Drop a content-warning README beside stored transcripts."""
        if not self.cfg.enabled:
            return
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "CONTENT_WARNING.txt").write_text(
            self.cfg.transcript_warning + "\n"
        )

    # -- reporting --------------------------------------------------------
    def summary(self) -> dict:
        with self._lock:
            c = self._counters
            return {
                "rollouts_started": c.rollouts_started,
                "rollouts_from_cache": c.rollouts_from_cache,
                "conversations_aborted_for_welfare": c.conversations_aborted,
                "debriefs_appended": c.debriefs_appended,
            }
