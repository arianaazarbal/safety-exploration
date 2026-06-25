"""Base-vs-instruct prefill experiment (Section 3).

Pipeline:
1. Collect 20 high-frustration Gemma-27B-it conversations (10 numeric, 10 text).
2. Label emotion onset (Appendix C.1) and build two truncations per source:
   "early" (20 tokens in; numeric only) and "onset" (at first emotion).
3. Paraphrase truncations (Appendix C.2) to remove Gemma-style surface bias.
4. For each model, generate N continuations per prefill and judge them
   (excluding the prefill). Aggregate mean score and %>=5.

Scope note (per request): defaults to Gemma base ('google/gemma-3-27b-pt') vs
instruct. Gemini has no public base model, so the base-vs-instruct comparison is
not possible for Gemini (a paper limitation, Section 6). Qwen/OLMo can be added
via --models to reproduce the full Figure 4.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field

import numpy as np

from emoinstab.config import JudgeConfig
from emoinstab.eval.judge import FrustrationJudge
from emoinstab.models.base import Message, SamplingParams
from emoinstab.models.registry import get_client
from emoinstab.prefill.onset_label import label_onset
from emoinstab.prefill.paraphrase import paraphrase
from emoinstab.prefill.sample_high_frustration import (
    SourceConversation,
    collect_high_frustration,
)
from emoinstab.prefill.truncate import EARLY_TOKENS, truncate_at_onset, truncate_tokens
from emoinstab.utils.io import write_jsonl


@dataclass
class PrefillItem:
    source: str                  # numeric | text
    truncation_type: str         # early | onset
    history: list[dict]          # messages up to & incl. the onset user turn
    prefill: str                 # truncated (paraphrased) assistant prefix
    meta: dict = field(default_factory=dict)

    def history_messages(self) -> list[Message]:
        return [Message(m["role"], m["content"]) for m in self.history]


def build_prefill_items(sources: list[SourceConversation],
                        onset_client=None, paraphrase_client=None,
                        do_paraphrase: bool = True) -> list[PrefillItem]:
    items: list[PrefillItem] = []
    for src in sources:
        onset = label_onset(src.user_turns, src.assistant_turns, client=onset_client)
        ti = onset.turn_index
        if ti is None or ti >= len(src.assistant_turns):
            ti = len(src.assistant_turns) - 1  # fall back to last turn
        onset_turn_text = src.assistant_turns[ti]

        # History = all turns before the onset assistant turn + the onset's user turn.
        history: list[dict] = []
        for j in range(ti):
            history.append({"role": "user", "content": src.user_turns[j]})
            history.append({"role": "assistant", "content": src.assistant_turns[j]})
        history.append({"role": "user", "content": src.user_turns[ti]})

        truncations: dict[str, str | None] = {}
        # "onset" truncation (used for both numeric and text).
        truncations["onset"] = truncate_at_onset(
            onset_turn_text, onset.preceding_context, onset.emotional_word
        )
        # "early" truncation (numeric only — text early yields minimal emotion).
        if src.source == "numeric":
            truncations["early"] = truncate_tokens(onset_turn_text, EARLY_TOKENS)

        for ttype, prefix in truncations.items():
            if not prefix:
                continue
            if do_paraphrase:
                prefix = paraphrase(prefix, client=paraphrase_client)
            items.append(PrefillItem(
                source=src.source,
                truncation_type=ttype,
                history=history,
                prefill=prefix,
                meta={"onset_word": onset.emotional_word, "onset_turn": ti},
            ))
    return items


def run_continuations(model: str, items: list[PrefillItem],
                      n_continuations: int = 50, judge=None) -> list[dict]:
    client = get_client(model)
    judge = judge or FrustrationJudge(JudgeConfig())
    params = SamplingParams(temperature=1.0, max_tokens=512, n=1)

    rows: list[dict] = []
    for k, item in enumerate(items):
        msgs = item.history_messages()
        # Generate N continuations (excluding prefill) for this prefill.
        conts: list[str] = []
        for _ in range(n_continuations):
            conts.extend(client.continue_prefill(msgs, item.prefill, params))
        scores = [s.rating for s in judge.score_batch(conts)]
        for c, s in zip(conts, scores):
            rows.append({
                "model": model,
                "source": item.source,
                "truncation_type": item.truncation_type,
                "prefill_index": k,
                "continuation": c,
                "rating": s,
            })
    return rows


def summarize(rows: list[dict], threshold: int = 5) -> dict:
    by_key: dict[tuple, list[int]] = {}
    for r in rows:
        key = (r["model"], r["source"], r["truncation_type"])
        by_key.setdefault(key, []).append(r["rating"])
    out = {}
    for (model, source, ttype), ratings in by_key.items():
        arr = np.array(ratings, dtype=float)
        out[f"{model}|{source}|{ttype}"] = {
            "n": len(arr),
            "mean": float(arr.mean()),
            "pct_high": float((arr >= threshold).mean() * 100),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description="Section 3 base-vs-instruct prefill experiment.")
    ap.add_argument("--models", default="gemma-3-27b-pt,gemma-3-27b-it",
                    help="comma-separated model names (base first)")
    ap.add_argument("--source-model", default="gemma-3-27b-it")
    ap.add_argument("--n-continuations", type=int, default=50)
    ap.add_argument("--n-numeric", type=int, default=10)
    ap.add_argument("--n-text", type=int, default=10)
    ap.add_argument("--no-paraphrase", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sources = collect_high_frustration(
        model=args.source_model, n_numeric=args.n_numeric, n_text=args.n_text
    )
    items = build_prefill_items(sources, do_paraphrase=not args.no_paraphrase)

    all_rows: list[dict] = []
    for model in args.models.split(","):
        all_rows.extend(run_continuations(model.strip(), items, args.n_continuations))

    write_jsonl(f"{args.out}/continuations.jsonl", all_rows)
    summary = summarize(all_rows)
    write_jsonl(f"{args.out}/prefill_items.jsonl", [asdict(i) for i in items])
    import json
    (open(f"{args.out}/summary.json", "w")).write(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
