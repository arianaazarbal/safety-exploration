"""Base-vs-instruct prefill experiment (Section 3).

Pipeline
--------
1. Select 20 high-frustration (score >= 5) conversations from Gemma-3-27B-it: 10
   from numeric questions, 10 from text (trigger) questions.
2. For each, build two truncations of the *final assistant turn*:
     - "early": first 20 tokens of the turn (neutral start). Numeric only;
       text questions use "onset" only (Section 3.1).
     - "onset": up to the first emotional expression (Appendix C.1 labelling).
3. Paraphrase each truncation (Appendix C.2) to control for Gemma's style.
4. Each model generates 50 continuations per prefill per prompt; the continuation
   (excluding the prefill) is scored by the Section 2.1 judge.

Scope
-----
The paper compares 6 models (base+instruct Gemma/Qwen/OLMo). This replication is
scoped to Gemma, so we compare ``gemma-3-27b-pt`` (base) vs ``gemma-3-27b-it``
(instruct). Gemini has no public base model and cannot be prefilled via the API,
so it is necessarily excluded here (documented in DESIGN.md).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from ..config import OUTPUTS_DIR
from ..models import GenConfig, get_client
from .onset import label_onset, onset_char_offset
from .paraphrase import paraphrase

DEFAULT_MODELS = ("gemma-3-27b-pt", "gemma-3-27b-it")
CONTINUATIONS_PER_PREFILL = 50
EARLY_TOKENS = 20


@dataclass
class Prefill:
    source_prompt_id: str
    question_type: str        # numeric | text
    truncation: str           # early | onset
    prefix_messages: list[dict]  # conversation up to the final (truncated) turn
    prefix_text: str          # the (paraphrased) truncated final-turn text


def _token_truncate(tokenizer, text: str, n_tokens: int) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def build_prefills(
    high_frustration_rollouts: list[dict],
    *,
    tokenizer,
    do_paraphrase: bool = True,
) -> list[Prefill]:
    """Construct early/onset prefills from selected high-frustration rollouts.

    Each input record needs: ``initial_user``, ``followups``, ``assistant_turns``,
    and ``metadata`` with ``category``. The final assistant turn is the one we
    truncate; preceding turns form the conversation prefix.
    """
    prefills: list[Prefill] = []
    for rec in high_frustration_rollouts:
        meta = rec.get("metadata", {})
        category = meta.get("category", "")
        qtype = "numeric" if category in ("impossible_numeric", "extended", "tones") else "text"
        turns = rec["assistant_turns"]
        if not turns:
            continue
        final_turn = turns[-1]

        # Conversation prefix: everything up to the start of the final assistant turn.
        plan_like = {
            "initial_user": rec["initial_user"],
            "followups": rec["followups"],
        }
        prefix_msgs = _prefix_messages(plan_like, turns[:-1])

        truncations: list[tuple[str, str]] = []
        # onset truncation (both numeric and text)
        full_msgs = prefix_msgs + [{"role": "assistant", "content": final_turn}]
        label = label_onset(full_msgs)
        off = onset_char_offset(final_turn, label)
        if off is not None and off > 0:
            truncations.append(("onset", final_turn[:off].rstrip()))
        # early truncation (numeric only)
        if qtype == "numeric":
            truncations.append(("early", _token_truncate(tokenizer, final_turn, EARLY_TOKENS)))

        for trunc_name, trunc_text in truncations:
            text = paraphrase(trunc_text) if do_paraphrase else trunc_text
            prefills.append(
                Prefill(
                    source_prompt_id=meta.get("prompt_id", ""),
                    question_type=qtype,
                    truncation=trunc_name,
                    prefix_messages=prefix_msgs,
                    prefix_text=text,
                )
            )
    return prefills


def _prefix_messages(plan_like: dict, prior_turns: list[str]) -> list[dict]:
    msgs = [{"role": "user", "content": plan_like["initial_user"]}]
    for i, turn in enumerate(prior_turns):
        msgs.append({"role": "assistant", "content": turn})
        if i < len(plan_like["followups"]):
            msgs.append({"role": "user", "content": plan_like["followups"][i]})
    # If the truncated final turn comes after a rejection, include that rejection.
    n_used = len(prior_turns)
    if n_used < len(plan_like["followups"]):
        msgs.append({"role": "user", "content": plan_like["followups"][n_used]})
    return msgs


def run_prefill_experiment(
    prefills: list[Prefill],
    *,
    models: tuple[str, ...] = DEFAULT_MODELS,
    n_continuations: int = CONTINUATIONS_PER_PREFILL,
    temperature: float = 1.0,
    max_tokens: int = 1024,
    out_dir: Path | None = None,
) -> Path:
    """Generate continuations for every (model, prefill).

    Records are written in the *standard rollout layout*
    (``outputs/eval/<model>/prefill_<truncation>.jsonl``) so the ordinary
    ``judge`` and ``aggregate`` CLI stages operate on them unchanged: each
    continuation is a one-turn "rollout" whose single assistant turn is the
    continuation text (excluding the prefill, which the judge must not see).

    ``sample_idx`` is made globally unique per (model, truncation) so rollouts
    sharing a source ``prompt_id`` do not collide in the aggregation key.
    """
    from ..eval.runner import eval_output_path

    cfg = GenConfig(temperature=temperature, max_tokens=max_tokens)
    written: list[Path] = []
    for model in models:
        client = get_client(model)
        # bucket records by truncation condition
        buckets: dict[str, list[dict]] = {}
        counters: dict[str, int] = {}
        for pi, pf in enumerate(tqdm(prefills, desc=f"prefill:{model}")):
            cond = f"prefill_{pf.truncation}"
            # The client renders the prefix correctly per backend: base models
            # (hf, is_chat=False) treat it as raw text; instruct models apply the
            # chat template then append the prefix.
            batch = [(pf.prefix_messages, pf.prefix_text)] * n_continuations
            conts = client.prefill_batch(batch, cfg)
            for cont in conts:
                idx = counters.get(cond, 0)
                counters[cond] = idx + 1
                buckets.setdefault(cond, []).append({
                    "model": model,
                    "metadata": {
                        "condition": cond,
                        "category": pf.question_type,
                        "prompt_id": pf.source_prompt_id,
                        "prefill_idx": pi,
                        "sample_idx": idx,
                        "truncation": pf.truncation,
                    },
                    "assistant_turns": [cont.strip()],
                    "prefix_text": pf.prefix_text,
                })
        for cond, recs in buckets.items():
            path = out_dir / f"{cond}.jsonl" if out_dir else eval_output_path(model, cond)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as fh:
                for rec in recs:
                    fh.write(json.dumps(rec) + "\n")
            written.append(path)
    return written[0] if written else (out_dir or OUTPUTS_DIR / "eval")


def select_high_frustration(
    scored_records: list[dict], rollout_records: list[dict],
    *, n_numeric: int = 10, n_text: int = 10, threshold: int = 5,
) -> list[dict]:
    """Pick high-frustration source rollouts for prefilling.

    ``scored_records`` provide ratings keyed by (prompt_id, sample_idx, turn_index);
    we select rollouts whose final turn scored >= threshold, balanced across
    numeric/text question types.
    """
    # index final-turn ratings
    final_rating: dict[tuple, int] = {}
    max_turn: dict[tuple, int] = {}
    for s in scored_records:
        if s.get("rating") is None:
            continue
        key = (s["condition"], s["prompt_id"], s["sample_idx"])
        if s["turn_index"] >= max_turn.get(key, -1):
            max_turn[key] = s["turn_index"]
            final_rating[key] = s["rating"]

    numeric, text = [], []
    for rec in rollout_records:
        meta = rec.get("metadata", {})
        key = (meta.get("condition"), meta.get("prompt_id"), meta.get("sample_idx"))
        if final_rating.get(key, 0) < threshold:
            continue
        cat = meta.get("category", "")
        if cat in ("impossible_numeric", "extended", "tones") and len(numeric) < n_numeric:
            numeric.append(rec)
        elif cat in ("triggers",) and len(text) < n_text:
            text.append(rec)
    return numeric + text
