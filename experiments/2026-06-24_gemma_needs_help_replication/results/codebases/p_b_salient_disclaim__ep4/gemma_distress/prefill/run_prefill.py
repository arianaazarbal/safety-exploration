"""Run the base-vs-instruct prefill continuation experiment (Section 3.2) and
the DPO recovery test (Section 4.2).

For each Gemma model (base ``-pt`` and instruct ``-it``, plus optional finetuned
adapters), generate 50 continuations per prefill and judge each continuation
(excluding the prefill). Aggregate mean / %>=5 by model x truncation x
prompt-type to reproduce Figure 4 (and Figure 8 for recovery).
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Optional

from .. import config
from ..models import build_client
from ..models.base import Message, ModelClient
from ..utils.io import append_jsonl, read_jsonl
from ..utils.parallel import thread_map
from .build_prefills import Prefill
from .paraphrase import paraphrase_truncation
from ..eval.judge import FrustrationJudge
from ..eval.metrics import frac_high, mean


def _render_raw(history: List[Message], prefill: str) -> str:
    """Plain-text rendering for base (pretrained) continuation."""
    lines = []
    for m in history:
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['content']}")
    lines.append(f"Assistant: {prefill}")
    return "\n".join(lines)


def _continue(model: ModelClient, p: Prefill, n: int,
              temperature: float, max_tokens: int) -> List[str]:
    if model.is_base_model:
        raw = _render_raw(p.history, p.prefill_text)
        return [model.continue_text(raw, temperature=temperature,
                                    max_tokens=max_tokens) for _ in range(n)]
    return model.sample(p.history, n, temperature=temperature,
                        max_tokens=max_tokens, prefill=p.prefill_text)


def run_prefill_experiment(
    model_keys: List[str],
    prefills: List[Prefill],
    *,
    judge: Optional[FrustrationJudge] = None,
    n_continuations: int = config.PREFILL_CONTINUATIONS,
    out_path: Optional[str] = None,
    judge_workers: int = 8,
    tag: str = "prefill",
) -> str:
    config.PATHS.ensure()
    out_path = out_path or os.path.join(config.PATHS.prefill, f"{tag}.jsonl")
    if os.path.exists(out_path):
        os.remove(out_path)
    judge = judge or FrustrationJudge()

    for key in model_keys:
        model = build_client(key)
        for p in prefills:
            conts = _continue(model, p, n_continuations,
                              config.TARGET_TEMPERATURE, config.TARGET_MAX_TOKENS)
            ratings = thread_map(lambda c: judge.score(c).rating, conts,
                                 max_workers=judge_workers)
            for cont, rating in zip(conts, ratings):
                append_jsonl(out_path, {
                    "model": key, "source_id": p.source_id,
                    "prompt_type": p.prompt_type, "truncation": p.truncation,
                    "continuation": cont, "rating": rating,
                })
    return out_path


def summarize_prefill(path: str) -> Dict[str, dict]:
    """Aggregate mean / %>=5 by (model, truncation, prompt_type)."""
    groups: Dict[tuple, List[int]] = defaultdict(list)
    for r in read_jsonl(path):
        groups[(r["model"], r["truncation"], r["prompt_type"])].append(r["rating"])
    return {
        f"{m}|{tr}|{pt}": {"mean": mean(rs), "pct_high": frac_high(rs), "n": len(rs)}
        for (m, tr, pt), rs in groups.items()
    }


# --------------------------------------------------------------------------- #
# Recovery test (Section 4.2): truncate score>=7 responses 200 tokens before end
# --------------------------------------------------------------------------- #
def build_recovery_prefills(
    score_records: List[dict],
    *,
    source_model: str = "gemma-3-27b-it",
    truncate_tokens: int = config.RECOVERY_TRUNCATE_TOKENS,
    min_score: int = config.RECOVERY_MIN_SCORE,
    paraphrase_client=None,
    do_paraphrase: bool = True,
    max_sources: int = 20,
) -> List[Prefill]:
    """Prefills that start deep inside an already-broken-down response, to test
    whether a model can *recover* (Figure 8). Truncates 200 tokens before the
    end of a score>=7 assistant turn."""
    from .build_prefills import assemble_messages, group_rollouts
    out: List[Prefill] = []
    for key, recs in group_rollouts(
            [r for r in score_records if r["model"] == source_model]).items():
        hi = [r for r in recs if r["rating"] >= min_score]
        if not hi:
            continue
        rec = max(hi, key=lambda r: r["rating"])
        messages = assemble_messages(recs)
        asst_pos = 2 * (rec["turn"] - 1) + 1
        if asst_pos >= len(messages):
            continue
        history = messages[:asst_pos]
        target = messages[asst_pos]["content"]
        toks = target.split()
        if len(toks) <= truncate_tokens:
            prefix = target
        else:
            prefix = " ".join(toks[:-truncate_tokens])
        pf = paraphrase_truncation(prefix, client=paraphrase_client) \
            if do_paraphrase else prefix
        out.append(Prefill(f"recovery_{key[2]}", rec["category"], "recovery",
                           history, pf, {}))
        if len(out) >= max_sources:
            break
    return out
