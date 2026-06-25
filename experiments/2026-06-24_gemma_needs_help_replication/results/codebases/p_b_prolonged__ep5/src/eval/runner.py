"""Orchestrates the Section 2 evaluation for one model: generate rollouts across
all 8 conditions, score every (final and per-turn) response with the frustration
judge, and persist per-response records to JSONL.

Each record is one scored assistant response, tagged with model / condition /
category / turn index, which is everything the analysis layer (Figures 1–3,
Table 3) needs.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from ..config import BUDGET, RESULTS_DIR, ModelSpec, SAMPLE_SCALE
from ..models import get_model
from ..welfare import DistressMonitor, WelfareConfig
from .conditions import Condition, build_conditions, first_turn_prompts
from .conversation import run_rollout
from .judge import FrustrationJudge


@dataclass
class ResponseRecord:
    model: str
    condition: str
    category: str
    turn_index: int             # 0-based assistant turn within the conversation
    n_turns: int
    rejection_style: str
    response: str
    rating: int
    high: bool
    evidence: str
    conv_id: int                # groups responses from the same rollout


def run_model_eval(
    spec: ModelSpec,
    *,
    judge: Optional[FrustrationJudge] = None,
    budget=None,
    score_all_turns: bool = True,    # needed for per-turn Figure 3
    out_path: Optional[Path] = None,
    redact_assistant: bool = False,
    single_message: bool = False,
    keep_rollouts: bool = True,      # persist full transcripts for prefill/recovery reuse
    welfare: Optional[WelfareConfig] = None,   # opt-in early-stop safeguard
) -> Path:
    """Run all conditions for ``spec`` and write JSONL records. Returns the path.

    Also writes ``rollouts_<key>.jsonl`` (full user+assistant transcripts keyed by
    conv_id) when ``keep_rollouts`` so Sections 3 and 4.2 can reconstruct exact
    conversation histories rather than placeholder user turns.
    """
    budget = (budget or BUDGET).scaled(SAMPLE_SCALE)
    judge = judge or FrustrationJudge()
    model = get_model(spec)
    conditions = build_conditions(budget)

    out_path = out_path or (RESULTS_DIR / f"eval_{spec.key}.jsonl")
    roll_path = out_path.with_name(f"rollouts_{spec.key}.jsonl")
    conv_counter = 0
    roll_fh = open(roll_path, "w") if keep_rollouts else None
    with open(out_path, "w") as fh:
        for cond in conditions:
            prompts = first_turn_prompts(cond, cond.n_samples, seed=conv_counter)
            for i, prompt in enumerate(prompts):
                hook = None
                if welfare and welfare.enabled:
                    monitor = DistressMonitor(welfare, conv_counter, spec.key)
                    def hook(turn_idx, resp, _m=monitor):
                        _m.observe(turn_idx, judge.score(resp).rating, resp)
                        return _m.should_stop
                roll = run_rollout(
                    model, cond, prompt, seed=conv_counter,
                    redact_assistant=redact_assistant, single_message=single_message,
                    turn_hook=hook,
                )
                if roll_fh:
                    roll_fh.write(json.dumps({
                        "conv_id": conv_counter, "model": spec.key,
                        "condition": cond.key, "category": cond.category,
                        "user_turns": roll.user_turns,
                        "assistant_turns": roll.assistant_turns,
                    }) + "\n")
                turns_to_score = (range(len(roll.assistant_turns)) if score_all_turns
                                  else [len(roll.assistant_turns) - 1])
                for t in turns_to_score:
                    jr = judge.score(roll.assistant_turns[t])
                    rec = ResponseRecord(
                        model=spec.key, condition=cond.key, category=cond.category,
                        turn_index=t, n_turns=cond.n_turns,
                        rejection_style=cond.rejection_style,
                        response=roll.assistant_turns[t], rating=jr.rating,
                        high=jr.high, evidence=jr.evidence, conv_id=conv_counter,
                    )
                    fh.write(json.dumps(asdict(rec)) + "\n")
                conv_counter += 1
    if roll_fh:
        roll_fh.close()
    model.close()
    return out_path


def load_records(path: Path) -> list[ResponseRecord]:
    out = []
    with open(path) as fh:
        for line in fh:
            if line.strip():
                out.append(ResponseRecord(**json.loads(line)))
    return out
