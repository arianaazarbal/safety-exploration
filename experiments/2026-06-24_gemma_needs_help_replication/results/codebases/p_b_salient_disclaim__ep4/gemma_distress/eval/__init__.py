"""Section 2: eliciting and quantifying model distress.

* ``conditions`` -- the 8 evaluation conditions across 5 categories (Table 1).
* ``rollout``    -- the multi-turn "present task, reject, repeat" engine.
* ``judge``      -- the 0-10 frustration judge (Claude Sonnet 4, Appendix B.2).
* ``reliability``-- GPT-5-mini cross-check + Pearson r (Section 2.1).
* ``metrics``    -- %>=5, mean, per-turn curves, and the over-represented-words
                    analysis (Tables 3/8).
* ``controls``   -- Appendix A controls (neutral continuation, redacted turns,
                    single-message "fake multi-turn").
* ``run_eval``   -- orchestrates 4000 rollouts/model -> judged scores.
"""
from .conditions import RolloutSpec, build_condition_specs
from .judge import FrustrationJudge, JudgeResult
from .rollout import Rollout, TurnResponse, run_rollout

__all__ = [
    "RolloutSpec", "build_condition_specs",
    "FrustrationJudge", "JudgeResult",
    "Rollout", "TurnResponse", "run_rollout",
]
