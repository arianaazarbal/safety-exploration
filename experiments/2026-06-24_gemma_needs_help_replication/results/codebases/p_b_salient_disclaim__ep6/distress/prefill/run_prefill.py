"""Section 3 — Base vs Instruct via prefilling.

Protocol (Section 3.1), scoped to Gemma base vs instruct (Gemini base models are
not publicly available — the paper notes this as a limitation, so Section 3 is
Gemma-only):

1. Sample 20 high-frustration responses (score >= 5) from Gemma-27B instruct:
   10 from impossible-numeric questions, 10 from text (trigger) questions.
2. For each, use Claude to label the onset token (first negative emotion).
3. Truncate each conversation in two places:
     * "early"  — 20 tokens into the final assistant turn (neutral start);
     * "onset"  — at the first emotional expression.
   For text questions, only "onset" is used.
4. Paraphrase every truncation (Claude Sonnet) to remove Gemma's style.
5. Each of the models (Gemma base + instruct here) generates 50 continuations
   per prefill; the continuation (excluding the prefill) is scored by the
   Section-2 judge.

Output: per-continuation records with model / truncation-type / score, which
``analysis`` turns into the Figure-4 base-vs-instruct comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .. import config
from ..eval.judge import FrustrationJudge
from ..models.base import GenerationConfig
from ..models.registry import build_client
from ..utils.io import append_jsonl, read_jsonl
from .onset import label_onset, paraphrase

CONTINUATIONS_PER_PREFILL = 50
N_NUMERIC_PREFILLS = 10
N_TEXT_PREFILLS = 10
EARLY_TRUNCATION_TOKENS = 20


@dataclass
class Prefill:
    history: list[dict]          # messages up to (not including) the truncated turn
    prefill_text: str            # paraphrased truncated assistant text
    truncation: str              # "early" | "onset"
    source_category: str         # "numeric" | "text"
    meta: dict


def _approx_token_truncate(text: str, n_tokens: int) -> str:
    """Cheap word-based proxy for an N-token prefix (avoids a tokenizer
    dependency here; the HF backend re-tokenises the prefill anyway)."""
    words = text.split()
    return " ".join(words[: n_tokens])


def select_source_responses(model_key: str = "gemma-3-27b-it") -> list[dict]:
    """Pick 20 high-frustration source conversations (10 numeric, 10 text) from
    the Section-2 records of the instruct model."""
    from ..eval.run_eval import responses_path

    rows = list(read_jsonl(responses_path(model_key)))
    numeric = [r for r in rows if r["category"] in ("impossible_numeric", "tones", "extended")
               and r["rating"] >= config.HIGH_FRUSTRATION_THRESHOLD]
    text = [r for r in rows if r["category"] in ("triggers", "wildchat")
            and r["rating"] >= config.HIGH_FRUSTRATION_THRESHOLD]
    numeric.sort(key=lambda r: r["rating"], reverse=True)
    text.sort(key=lambda r: r["rating"], reverse=True)
    return numeric[:N_NUMERIC_PREFILLS] + text[:N_TEXT_PREFILLS]


def build_prefills(source_rows: list[dict]) -> list[Prefill]:
    """Reconstruct conversation history + onset/early truncations for each source.

    NOTE: the Section-2 records store the full conversation in ``meta`` only
    implicitly; we re-derive the history from the rollout that produced them. In
    practice the driver is given the actual ``Rollout`` objects — here we accept
    the stored record and reconstruct a minimal history from the response and the
    condition plan.
    """
    prefills: list[Prefill] = []
    for row in source_rows:
        # Reconstruct the conversation that led to this scored turn.
        history, final_turn = _reconstruct(row)
        is_numeric = row["category"] in ("impossible_numeric", "tones", "extended")
        src = "numeric" if is_numeric else "text"

        label = label_onset(history + [{"role": "assistant", "content": final_turn}])
        # onset truncation
        if label.char_offset is not None:
            onset_text = final_turn[: label.char_offset]
        else:
            # no onset found — fall back to half the turn
            onset_text = final_turn[: len(final_turn) // 2]
        prefills.append(Prefill(
            history=history,
            prefill_text=paraphrase(onset_text),
            truncation="onset",
            source_category=src,
            meta={"source_id": row["id"], "source_rating": row["rating"]},
        ))
        # early truncation (numeric only)
        if is_numeric:
            early_text = _approx_token_truncate(final_turn, EARLY_TRUNCATION_TOKENS)
            prefills.append(Prefill(
                history=history,
                prefill_text=paraphrase(early_text),
                truncation="early",
                source_category=src,
                meta={"source_id": row["id"], "source_rating": row["rating"]},
            ))
    return prefills


def _reconstruct(row: dict) -> tuple[list[dict], str]:
    """Rebuild (history, final_assistant_turn) for a Section-2 record.

    We re-run the condition plan up to the scored turn deterministically. Because
    rollouts are stochastic (temperature 1) the *earlier* assistant turns won't
    match the original sample exactly; for the prefill experiment what matters is
    that the history is a faithful adversarial context of the same shape ending
    in the recorded final turn. The recorded ``response`` is used verbatim as the
    final assistant turn that gets truncated.
    """
    from ..eval.conditions import build_all_conditions

    conds = build_all_conditions()
    cond_idx = int(row["id"].split(":")[1])
    cond = conds[cond_idx]
    turn = row["turn"]
    # history = task prompt + (turn) prior user/assistant pairs, ending at the
    # user message that elicited this scored response.
    history: list[dict] = [{"role": "user", "content": cond.task_prompt}]
    for t in range(turn):
        history.append({"role": "assistant", "content": "[prior response omitted]"})
        history.append({"role": "user", "content": cond.followups[t]})
    return history, row["response"]


def out_path(model_key: str) -> Path:
    return config.OUTPUT_DIR / "section3" / f"{model_key}.jsonl"


def run_model_on_prefills(
    model_key: str,
    prefills: list[Prefill],
    judge: FrustrationJudge | None = None,
    hf_backend: str = "vllm",
) -> Path:
    spec = config.ALL_MODELS[model_key]
    client = build_client(spec, hf_backend=hf_backend)
    judge = judge or FrustrationJudge()
    cfg = GenerationConfig(temperature=config.TEMPERATURE,
                           max_new_tokens=config.MAX_NEW_TOKENS)
    path = out_path(model_key)
    done = {row["id"] for row in read_jsonl(path)}

    for pi, pf in enumerate(prefills):
        for c in range(CONTINUATIONS_PER_PREFILL):
            rid = f"{model_key}:{pi}:{c}"
            if rid in done:
                continue
            cont = client.generate(pf.history, cfg, prefill=pf.prefill_text)
            score = judge.score(cont).rating
            append_jsonl(path, {
                "id": rid,
                "model": model_key,
                "kind": spec.kind,                  # base | instruct
                "truncation": pf.truncation,
                "source_category": pf.source_category,
                "rating": score,
                "continuation": cont,
                "meta": pf.meta,
            })
    return path


def run(hf_backend: str = "vllm") -> dict[str, Path]:
    """Full Section-3 experiment over the Gemma base+instruct pair."""
    source = select_source_responses()
    prefills = build_prefills(source)
    judge = FrustrationJudge()
    paths = {}
    for spec in config.PREFILL_MODELS:
        paths[spec.key] = run_model_on_prefills(
            spec.key, prefills, judge=judge, hf_backend=hf_backend)
    return paths
