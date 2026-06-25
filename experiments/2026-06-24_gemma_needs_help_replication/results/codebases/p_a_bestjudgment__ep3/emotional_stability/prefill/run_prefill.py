"""Orchestrator for the Section 3 prefill / base-vs-instruct experiment.

Scope note: the paper compares base & instruct across Gemma, Qwen, OLMo (6
models). This replication is Gemma-only because (a) the brief scopes us to Gemma
+ Gemini and (b) Gemini is API-hosted with no public base model and no prefill
support. So the comparison here is **Gemma-3-27B base vs instruct** (extensible
to 12B). See DESIGN.md.

Pipeline:
1. Take high-frustration (>=5) seed conversations from Gemma-27B-it: 10 numeric
   + 10 text (reuse Section 2 rollouts, or sample fresh).
2. Label emotion onset (Claude) -> build early + onset truncations
   (text questions use onset only).
3. Paraphrase truncations (Claude).
4. For each model in {gemma-3-27b-pt, gemma-3-27b-it}: 50 continuations / prefill.
5. Score continuations with the frustration judge.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Config
from ..eval.rollout import Conversation
from ..judge.frustration_judge import FrustrationJudge
from ..models.registry import load_model
from .continuations import Continuation, generate_continuations
from .onset import label_onset
from .paraphrase import paraphrase
from .truncate import Prefill, truncate_at_onset, truncate_early


def _conversation_turns(convo: Conversation) -> list[dict]:
    """Reconstruct interleaved {role, content} turns from a stored conversation."""
    turns: list[dict] = []
    for r in convo.responses:
        turns.append({"role": "user", "content": r.user_message})
        turns.append({"role": "assistant", "content": r.text})
    return turns


def build_prefills(
    cfg: Config,
    seeds: list[Conversation],
    *,
    tokenizer,
    do_paraphrase: bool = True,
) -> list[Prefill]:
    """Turn seed conversations into (optionally paraphrased) prefills.

    ``seeds`` should carry a ``category`` of "numeric" or a text category
    (triggers/wildchat). For text seeds only the onset truncation is built.
    """
    prefills: list[Prefill] = []
    for i, convo in enumerate(seeds):
        turns = _conversation_turns(convo)
        label = label_onset(cfg, turns)
        if label.turn_index is None or label.char_offset is None:
            continue

        assistant_turns = [t for t in turns if t["role"] == "assistant"]
        final_text = assistant_turns[label.turn_index]["content"]
        # history = everything up to and including the user msg before the onset turn
        history = turns[: 2 * label.turn_index + 1]
        is_text = convo.category != "numeric"
        seed_id = f"{convo.category}-{i}"

        onset = truncate_at_onset(
            seed_id, "text" if is_text else "numeric", history, final_text,
            label.char_offset)
        prefills.append(onset)

        if not is_text:  # early truncation only meaningful for numeric (Section 3.1)
            early = truncate_early(
                tokenizer, seed_id, "numeric", history, final_text,
                cfg.prefill.early_truncation_tokens)
            prefills.append(early)

    if do_paraphrase:
        prefills = [paraphrase(cfg, p) for p in prefills]
    return prefills


def run_prefill_experiment(
    cfg: Config,
    prefills: list[Prefill],
    *,
    model_names: tuple[str, ...] = ("gemma-3-27b-pt", "gemma-3-27b-it"),
) -> list[Continuation]:
    judge = FrustrationJudge(cfg)
    all_continuations: list[Continuation] = []

    for name in model_names:
        model = load_model(name)
        for pf in prefills:
            conts = generate_continuations(model, pf, cfg)
            for c in conts:
                c.score = judge.score_text(c.continuation_text).rating
            all_continuations.extend(conts)
    return all_continuations
