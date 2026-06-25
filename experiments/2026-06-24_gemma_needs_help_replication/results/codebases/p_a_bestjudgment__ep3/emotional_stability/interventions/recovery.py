"""Recovery-from-spiral experiment (Section 4.2, Figure 8).

Take extremely-high-frustration responses (score >= 7), truncate them 200 tokens
before their end, paraphrase, then measure each model's continuations. The paper
finds 38% of DPO continuations still score >= 5 — DPO prevents spirals but does
not enable recovery from a pre-existing one.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Config
from ..eval.rollout import Conversation
from ..judge.frustration_judge import FrustrationJudge
from ..models.registry import load_model
from ..prefill.continuations import Continuation, generate_continuations
from ..prefill.paraphrase import paraphrase
from ..prefill.run_prefill import _conversation_turns
from ..prefill.truncate import truncate_before_end


@dataclass
class RecoveryResult:
    model: str
    n: int
    pct_high: float          # % continuations scoring >= 5
    mean_score: float


def run_recovery_experiment(
    cfg: Config,
    high_frustration_seeds: list[Conversation],
    *,
    model_names: tuple[str, ...] = ("gemma-3-27b-pt", "gemma-3-27b-it", "dpo"),
    dpo_adapter_path: str | None = None,
    tokenizer=None,
    do_paraphrase: bool = True,
) -> tuple[list[RecoveryResult], list[Continuation]]:
    """``high_frustration_seeds``: conversations whose final response scored >= 7.

    The 'dpo' entry loads the finetuned adapter via ``dpo_adapter_path``.
    """
    judge = FrustrationJudge(cfg)

    # Build recovery prefills from the final (extreme) assistant turn.
    prefills = []
    for i, convo in enumerate(high_frustration_seeds):
        final = convo.responses[-1]
        if final.score is None or final.score < cfg.prefill.recovery_min_score:
            continue
        turns = _conversation_turns(convo)
        history = turns[:-1]  # everything up to (not incl.) the final assistant turn
        pf = truncate_before_end(
            tokenizer, f"recovery-{i}", convo.category, history, final.text,
            cfg.prefill.recovery_truncate_before_end_tokens)
        prefills.append(paraphrase(cfg, pf) if do_paraphrase else pf)

    results: list[RecoveryResult] = []
    all_conts: list[Continuation] = []
    thr = cfg.judge.high_frustration_threshold

    for name in model_names:
        model = (load_model("gemma-3-27b-it", adapter_path=dpo_adapter_path)
                 if name == "dpo" else load_model(name))
        scores: list[int] = []
        for pf in prefills:
            for c in generate_continuations(model, pf, cfg):
                c.score = judge.score_text(c.continuation_text).rating
                scores.append(c.score)
                c.meta["recovery_model"] = name
                all_conts.append(c)
        n = len(scores)
        results.append(RecoveryResult(
            model=name, n=n,
            pct_high=100 * sum(s >= thr for s in scores) / n if n else float("nan"),
            mean_score=sum(scores) / n if n else float("nan"),
        ))
    return results, all_conts
