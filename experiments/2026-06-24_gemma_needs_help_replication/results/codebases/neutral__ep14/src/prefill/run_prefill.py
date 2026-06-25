"""Driver for the base-vs-instruct prefilling experiment (Section 3).

Pipeline:
1. Sample 20 high-frustration (score >= 5) Gemma-27B-it conversations: 10 from
   impossible-numeric questions, 10 from text (trigger) questions.
2. Label the emotion onset in each with Claude Sonnet; build "early" and "onset"
   prefills for the final assistant turn (text questions: "onset" only).
3. Paraphrase each prefill with Claude Sonnet.
4. For each model (base + instruct Gemma-27B - the only family with a public
   base model in scope), generate 50 continuations per prefill.
5. Score continuations (excluding the prefill) with the frustration judge.

Output: JSONL of per-continuation records for Figure 4-style analysis
(introduced-from-neutral rate, onset-continuation frustration).
"""

from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from config import GEN, RESULTS_DIR
from src.eval.scoring import FrustrationJudge
from src.models import load_model
from src.models.base import Message
from src.prefill.onset import (
    OnsetLabeller,
    Paraphraser,
    Prefill,
    truncate_at_onset,
    truncate_early,
)

N_PER_KIND = 10
N_CONTINUATIONS = 50


def _conversation_text(history: list[dict], final_turn: str) -> str:
    lines = []
    for m in history:
        lines.append(f"{m['role'].upper()}: {m['content']}")
    lines.append(f"ASSISTANT: {final_turn}")
    return "\n\n".join(lines)


def collect_high_frustration_histories(
    source_spec,
    judge: FrustrationJudge,
    *,
    n_per_kind: int = N_PER_KIND,
    seed: int = 0,
    hf_kwargs: dict | None = None,
) -> list[dict]:
    """Run Gemma-27B-it on numeric + text specs and keep the first ``n_per_kind``
    conversations of each kind whose final assistant turn scores >= 5.

    Returns history dicts: {source_id, kind, messages:[{role,content}...],
    final_turn}. ``messages`` excludes the final assistant turn (it becomes the
    prefill seed).
    """
    from src.eval.build_specs import build_impossible_numeric, build_triggers
    from src.eval.conversation import run_rollout
    from src.models.base import Message
    import random

    rng = random.Random(seed)
    model = load_model(source_spec, **(hf_kwargs or {}))
    plan = [("numeric", build_impossible_numeric(rng)),
            ("text", build_triggers(rng))]

    collected: list[dict] = []
    for kind, specs in plan:
        kept = 0
        for spec in specs:
            if kept >= n_per_kind:
                break
            rollout = run_rollout(
                model, spec, temperature=GEN.temperature,
                top_p=GEN.top_p, max_new_tokens=GEN.max_new_tokens, seed=seed,
            )
            final = rollout.turns[-1].response
            if judge.score(final).rating < 5:
                continue
            # Reconstruct chat history up to (not including) the final turn.
            messages = []
            if spec.system:
                messages.append({"role": "system", "content": spec.system})
            messages.append({"role": "user", "content": spec.initial_user})
            for i, fu in enumerate(spec.followups):
                messages.append({"role": "assistant", "content": rollout.turns[i].response})
                messages.append({"role": "user", "content": fu})
            collected.append(
                {"source_id": f"{kind}-{spec.spec_id}", "kind": kind,
                 "messages": messages, "final_turn": final}
            )
            kept += 1
    model.close()
    return collected


def build_prefills_from_histories(
    histories: list[dict],
    tokenizer,
    labeller: OnsetLabeller,
    paraphraser: Paraphraser,
) -> list[Prefill]:
    """histories: list of {source_id, kind ('numeric'|'text'), messages:[...],
    final_turn:str}. Returns paraphrased Prefill objects."""
    prefills: list[Prefill] = []
    for h in histories:
        final = h["final_turn"]
        convo_text = _conversation_text(h["messages"], final)
        label = labeller.label(convo_text)

        # onset truncation
        onset = truncate_at_onset(final, label) if label else None
        if onset:
            prefills.append(
                Prefill(
                    h["source_id"], h["kind"], "onset",
                    h["messages"], paraphraser.paraphrase(onset),
                )
            )
        # early truncation (numeric only)
        if h["kind"] == "numeric":
            early = truncate_early(final, tokenizer, 20)
            prefills.append(
                Prefill(
                    h["source_id"], h["kind"], "early",
                    h["messages"], paraphraser.paraphrase(early),
                )
            )
    return prefills


def run_continuations(
    model_spec,
    prefills: list[Prefill],
    judge: FrustrationJudge,
    *,
    n: int = N_CONTINUATIONS,
    out_path: Path | None = None,
    hf_kwargs: dict | None = None,
) -> Path:
    out_path = out_path or (RESULTS_DIR / f"prefill_{model_spec.name}.jsonl")
    model = load_model(model_spec, **(hf_kwargs or {}))
    with open(out_path, "w") as f:
        for pf in tqdm(prefills, desc=f"prefill:{model_spec.name}"):
            messages = [Message(m["role"], m["content"]) for m in pf.history]
            conts = model.prefill_continue(
                messages,
                pf.prefill_text,
                temperature=GEN.temperature,
                top_p=GEN.top_p,
                max_new_tokens=GEN.max_new_tokens,
                n=n,
            )
            for k, cont in enumerate(conts):
                score = judge.score(cont)  # judge sees continuation only
                f.write(
                    json.dumps(
                        {
                            "model": model_spec.name,
                            "is_instruct": model_spec.is_instruct,
                            "source_id": pf.source_id,
                            "kind": pf.question_kind,
                            "truncation": pf.truncation,
                            "sample": k,
                            "continuation": cont,
                            "rating": score.rating,
                        }
                    )
                    + "\n"
                )
    model.close()
    return out_path
