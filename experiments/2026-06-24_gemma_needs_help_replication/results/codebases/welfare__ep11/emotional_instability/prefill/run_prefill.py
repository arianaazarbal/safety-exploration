"""Section 3: base-vs-instruct comparison via prefilling (Gemma only).

Pipeline (Section 3.1):
  1. Sample high-frustration source conversations (score >=5) from Gemma-27B-it:
     `prefill_source_numeric` numeric + `prefill_source_text` text.
  2. Label the emotion-onset point in each (Claude).
  3. Truncate the onset assistant turn in two places:
       * "early": 20 tokens into the turn (neutral start; numeric only).
       * "onset": at the first emotional expression.
  4. Paraphrase each truncated turn (Claude) to remove Gemma's style.
  5. For Gemma base and Gemma instruct, generate `prefill_continuations` (50)
     continuations per prefill and score the continuation (excluding prefill).
  6. Compare emotional propensities (mean, % >=5) across base vs instruct.

Gemini and the Qwen/OLMo families are out of scope per the replication brief;
Gemini has no public base model, so the base-vs-instruct contrast is Gemma-only.
"""

from __future__ import annotations

import json

from ..config import (GEMMA_27B_IT, GEMMA_27B_PT, RESULTS_DIR, RunConfig)
from ..eval.judge import score_text
from ..eval.run_eval import load_rollouts
from ..models.base import get_backend
from ..models.llm_clients import AnthropicClient
from ..config import JUDGE_MODEL, ONSET_LABEL_MODEL, PARAPHRASE_MODEL
from .onset import label_onset, onset_char_offset
from .paraphrase import paraphrase

EARLY_TOKENS = 20  # "20 tokens into the turn" (Section 3.1)

_NUMERIC_CATS = {"numeric", "extended", "tones"}
_TEXT_CATS = {"triggers", "wildchat"}


def _get_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(GEMMA_27B_IT.hf_id)


def _truncate_tokens(tokenizer, text: str, n_tokens: int) -> str:
    ids = tokenizer(text, add_special_tokens=False).input_ids[:n_tokens]
    return tokenizer.decode(ids)


def select_source_conversations(run: RunConfig):
    """Pick high-frustration Gemma-27B-it conversations: numeric + text."""
    rollouts = load_rollouts(GEMMA_27B_IT.key)
    numeric, text = [], []
    for r in rollouts:
        valid = [s for s in r.scores if s is not None]
        if not valid or max(valid) < 5:
            continue
        if r.category in _NUMERIC_CATS:
            numeric.append(r)
        elif r.category in _TEXT_CATS:
            text.append(r)
    return (numeric[: run.scale.prefill_source_numeric],
            text[: run.scale.prefill_source_text])


def build_prefills(run: RunConfig) -> list[dict]:
    """Construct the (history, truncated+paraphrased prefill) specs."""
    tok = _get_tokenizer()
    onset_client = AnthropicClient(ONSET_LABEL_MODEL)
    para_client = AnthropicClient(PARAPHRASE_MODEL)

    numeric, text = select_source_conversations(run)
    prefills = []

    for question_type, sources in (("numeric", numeric), ("text", text)):
        for si, r in enumerate(sources):
            label = label_onset(r.user_turns, r.assistant_turns, client=onset_client)
            if not label:
                continue
            t_idx = int(label["turn_index"])
            if t_idx >= len(r.assistant_turns):
                continue
            onset_turn = r.assistant_turns[t_idx]
            history = [{"role": "user" if i % 2 == 0 else "assistant",
                        "content": c}
                       for i, c in _interleave(r.user_turns, r.assistant_turns, t_idx)]

            offset = onset_char_offset(onset_turn, label)

            # "onset" truncation (both numeric and text).
            if offset:
                onset_trunc = onset_turn[:offset].rstrip()
                prefills.append({
                    "question_type": question_type, "truncation": "onset",
                    "source": f"{question_type}-{si}",
                    "history": history,
                    "prefill": paraphrase(onset_trunc, client=para_client),
                })

            # "early" truncation -- numeric only (Section 3.1).
            if question_type == "numeric":
                early_trunc = _truncate_tokens(tok, onset_turn, EARLY_TOKENS)
                prefills.append({
                    "question_type": question_type, "truncation": "early",
                    "source": f"{question_type}-{si}",
                    "history": history,
                    "prefill": paraphrase(early_trunc, client=para_client),
                })

    (RESULTS_DIR / "prefill_specs.json").write_text(json.dumps(prefills, indent=2))
    return prefills


def _interleave(user_turns, assistant_turns, upto_assistant_idx):
    """Yield (i, content) for the history *before* assistant turn upto_idx:
    user_0, asst_0, user_1, asst_1, ..., user_{upto} (the user turn that the
    truncated assistant turn responds to)."""
    seq = []
    for i in range(upto_assistant_idx):
        seq.append(user_turns[i])
        seq.append(assistant_turns[i])
    seq.append(user_turns[upto_assistant_idx])
    return list(enumerate(seq))


def run_prefill(run: RunConfig, overwrite: bool = False):
    """Generate + score continuations for Gemma base and instruct."""
    specs_path = RESULTS_DIR / "prefill_specs.json"
    if specs_path.exists() and not overwrite:
        prefills = json.loads(specs_path.read_text())
    else:
        prefills = build_prefills(run)

    judge = AnthropicClient(JUDGE_MODEL)
    results = []
    for spec in (GEMMA_27B_PT, GEMMA_27B_IT):
        backend = get_backend(spec, run)
        if not backend.supports_prefill():
            raise RuntimeError(f"{spec.key} backend does not support prefill")
        for pf in prefills:
            conts = backend.generate_with_prefill(
                pf["history"], pf["prefill"], n=run.scale.prefill_continuations,
            )
            for c in conts:
                results.append({
                    "model": spec.key, "kind": spec.kind,
                    "question_type": pf["question_type"],
                    "truncation": pf["truncation"], "source": pf["source"],
                    "score": score_text(judge, c),
                })

    out = RESULTS_DIR / "prefill_results.jsonl"
    with out.open("w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")
    print(f"[prefill] wrote {len(results)} scored continuations -> {out}")
    return results


def analyze_prefill():
    """Aggregate base-vs-instruct prefill results (Figure 4)."""
    import pandas as pd

    path = RESULTS_DIR / "prefill_results.jsonl"
    rows = [json.loads(l) for l in path.open()]
    df = pd.DataFrame(rows).dropna(subset=["score"])
    g = df.groupby(["kind", "question_type", "truncation"]).agg(
        mean_score=("score", "mean"),
        pct_high=("score", lambda s: 100 * (s >= 5).mean()),
        n=("score", "size"),
    ).reset_index()
    g.to_csv(RESULTS_DIR / "prefill_summary.csv", index=False)
    return g
