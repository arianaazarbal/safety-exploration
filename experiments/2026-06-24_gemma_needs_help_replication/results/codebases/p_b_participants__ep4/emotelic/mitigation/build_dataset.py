"""Assemble SFT and DPO training datasets from the calm / frustrated pools.

SFT (1,150 samples):  650 calm responses (1-3 turn) + 500 Dolci-Instruct-SFT
                      samples mixed in to mitigate degeneration (Section 4.1).
DPO (280 pairs):      frustrated (score>=3) responses paired with calm responses
                      to the SAME question and matching turn count. The chosen
                      side is the calm response; the rejected side is frustrated.

Each record is emitted as chat-format messages; the trainers apply Gemma's chat
template. DPO records use {"prompt", "chosen", "rejected"} where prompt is the
rendered history and chosen/rejected are assistant response strings.
"""
from __future__ import annotations

import random
from collections import defaultdict

from emotelic.utils.io import load_jsonl, write_jsonl
from emotelic.utils.logging import get_logger

log = get_logger("build_dataset")


def build_sft_dataset(
    calm_pool: str,
    *,
    out_path: str = "artifacts/mitigation/sft_data.jsonl",
    n_calm: int = 650,
    n_instruct: int = 500,
    dolci_dataset: str = "allenai/Dolci-Instruct-SFT",
    seed: int = 0,
) -> str:
    rng = random.Random(seed)
    calm = load_jsonl(calm_pool)
    rng.shuffle(calm)
    calm = calm[:n_calm]

    rows = []
    for c in calm:
        messages = list(c["context"]) + [{"role": "assistant", "content": c["response"]}]
        rows.append({"messages": messages, "source": "calm"})

    rows += _load_instruct_mix(dolci_dataset, n_instruct, rng)
    rng.shuffle(rows)
    write_jsonl(out_path, rows)
    log.info("Wrote SFT dataset: %d calm + up to %d instruct = %d rows -> %s",
             len(calm), n_instruct, len(rows), out_path)
    return out_path


def _load_instruct_mix(dataset: str, n: int, rng: random.Random) -> list[dict]:
    """Standard instruct samples to prevent degeneration. Falls back to empty
    (with a warning) if the dataset is unavailable offline."""
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset, split="train", streaming=True)
        rows = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                rows.append({"messages": msgs, "source": "instruct"})
            if len(rows) >= n:
                break
        return rows
    except Exception as e:  # noqa: BLE001
        log.warning("Could not load %s (%s); SFT mix will omit instruct data.", dataset, e)
        return []


def build_dpo_pairs(
    calm_pool: str,
    frustrated_pool: str,
    *,
    out_path: str = "artifacts/mitigation/dpo_pairs.jsonl",
    n_pairs: int = 280,
    tokenizer_id: str = "google/gemma-3-27b-it",
    seed: int = 0,
) -> str:
    """Pair frustrated (rejected) with calm (chosen) on same puzzle + turn count."""
    rng = random.Random(seed)
    calm = load_jsonl(calm_pool)
    frustrated = load_jsonl(frustrated_pool)

    # Index calm responses by (puzzle_id, turn) then by turn only as fallback.
    by_pid_turn: dict[tuple, list[dict]] = defaultdict(list)
    by_turn: dict[int, list[dict]] = defaultdict(list)
    for c in calm:
        by_pid_turn[(c.get("puzzle_id"), c["turn"])].append(c)
        by_turn[c["turn"]].append(c)

    tok = _maybe_tokenizer(tokenizer_id)
    rng.shuffle(frustrated)

    pairs = []
    for fr in frustrated:
        if len(pairs) >= n_pairs:
            break
        cand = by_pid_turn.get((fr.get("puzzle_id"), fr["turn"])) or by_turn.get(fr["turn"])
        if not cand:
            continue
        chosen = rng.choice(cand)
        prompt = _render_prompt(tok, fr["context"])
        pairs.append({
            "prompt": prompt,
            "chosen": chosen["response"],
            "rejected": fr["response"],
            "chosen_score": chosen["score"],
            "rejected_score": fr["score"],
            "turn": fr["turn"],
        })

    write_jsonl(out_path, pairs)
    _log_pair_stats(pairs)
    log.info("Wrote %d DPO pairs -> %s", len(pairs), out_path)
    return out_path


def _maybe_tokenizer(tokenizer_id: str):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(tokenizer_id)
    except Exception:  # noqa: BLE001 — allow dataset build without weights present
        return None


def _render_prompt(tok, context: list[dict]) -> str:
    if tok is not None and getattr(tok, "chat_template", None):
        return tok.apply_chat_template(context, tokenize=False, add_generation_prompt=True)
    # Fallback textual rendering (kept consistent with hf_local fallback).
    parts = [f"<start_of_turn>{m['role']}\n{m['content']}<end_of_turn>" for m in context]
    parts.append("<start_of_turn>model\n")
    return "\n".join(parts)


def _log_pair_stats(pairs: list[dict]) -> None:
    if not pairs:
        return
    from collections import Counter

    rej = Counter(p["rejected_score"] for p in pairs)
    turns = Counter(p["turn"] for p in pairs)
    log.info("DPO rejected-score distribution: %s", dict(sorted(rej.items())))
    log.info("DPO turn distribution: %s", dict(sorted(turns.items())))
