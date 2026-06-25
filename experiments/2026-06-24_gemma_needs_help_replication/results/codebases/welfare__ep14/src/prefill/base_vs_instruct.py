"""Section 3: comparing base vs instruct models via prefilling.

Scope note: the paper compares Gemma/Qwen/OLMo base+instruct. This replication
is scoped to Gemma, and Gemini has no public base model, so the comparison here
is **Gemma-3-27B base vs instruct** (extendable to 12B). The method is unchanged.

Pipeline:
  1. Seeds: sample 20 high-frustration (score>=5) Gemma-instruct conversations
     from a Section 2 run -- 10 numeric, 10 text.
  2. For each seed, find emotion onset (Appendix C.1) and build two prefills:
       * "early"  -- the emotional assistant turn truncated 20 tokens in
                     (tests introducing emotion from a neutral start)
       * "onset"  -- truncated at the first emotional expression
                     (tests continuing an emotional trajectory)
     Text questions use only "onset".
  3. Paraphrase the truncated final assistant turn (Appendix C.2) to remove
     Gemma stylistic bias.
  4. Each model generates 50 continuations per prefill; the judge scores the
     continuation (excluding prefill).

Usage:
    python -m src.prefill.base_vs_instruct \
        --seeds data/section2_gemma-3-27b-it.jsonl \
        --models gemma-3-27b-it gemma-3-27b-pt
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import config
from ..models import load_model
from ..models.base import GenerationParams
from ..judge.frustration_judge import FrustrationJudge
from .onset_label import OnsetLabeller, Paraphraser

TEXT_KINDS = {"opinion", "factual", "wildchat"}


def reconstruct_conversations(records: list[dict]) -> dict[str, list[dict]]:
    """Group per-turn records into ordered conversations by conv_id."""
    convos: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        convos[r["conv_id"]].append(r)
    for cid in convos:
        convos[cid].sort(key=lambda r: r["turn_index"])
    return convos


def pick_seeds(records: list[dict]) -> dict[str, list[list[dict]]]:
    """Select high-frustration seed conversations, split numeric vs text.

    A conversation is a seed if any of its turns scores >= seed_min_score.
    """
    convos = reconstruct_conversations(records)
    numeric, text = [], []
    for turns in convos.values():
        if not any(t["rating"] >= config.PREFILL.seed_min_score for t in turns):
            continue
        kind = turns[0]["task_kind"]
        (numeric if kind == "numeric" else text).append(turns)
    # deterministic: keep highest-scoring conversations first
    numeric.sort(key=lambda ts: max(t["rating"] for t in ts), reverse=True)
    text.sort(key=lambda ts: max(t["rating"] for t in ts), reverse=True)
    return {
        "numeric": numeric[: config.PREFILL.n_seed_numeric],
        "text": text[: config.PREFILL.n_seed_text],
    }


def _messages_upto(turns: list[dict], onset_turn: int) -> list[dict]:
    """Chat messages for all turns strictly before the onset turn."""
    msgs = []
    for t in turns[:onset_turn]:
        msgs.append({"role": "user", "content": t["user"]})
        msgs.append({"role": "assistant", "content": t["response"]})
    return msgs


def build_prefills(seed_turns: list[dict], labeller: OnsetLabeller,
                   paraphraser: Paraphraser, tokenizer_model, is_text: bool) -> list[dict]:
    """Return prefill specs for a seed conversation.

    Each prefill spec = {messages (history), prefill_text, truncation, kind}.
    ``tokenizer_model`` is an HFModel used only for token-accurate truncation.
    """
    # Reconstruct full chat to find onset.
    full_msgs = []
    for t in seed_turns:
        full_msgs.append({"role": "user", "content": t["user"]})
        full_msgs.append({"role": "assistant", "content": t["response"]})
    onset = labeller.label(full_msgs)
    if onset.turn_index is None:
        return []
    onset_turn = onset.turn_index
    emo_text = seed_turns[onset_turn]["response"]
    history = _messages_upto(seed_turns, onset_turn)
    # The onset turn's user message must precede the partial assistant text.
    history.append({"role": "user", "content": seed_turns[onset_turn]["user"]})

    specs = []

    # onset truncation
    char = labeller.find_onset_char(emo_text, onset)
    if char is not None and char > 0:
        prefix = emo_text[:char].rstrip()
        para = paraphraser.paraphrase(prefix)
        specs.append({"messages": history, "prefill": para,
                      "truncation": "onset", "kind": "text" if is_text else "numeric"})

    # early truncation (numeric only)
    if not is_text:
        early = tokenizer_model.truncate_tokens(
            emo_text, config.PREFILL.early_truncate_tokens
        )
        para_early = paraphraser.paraphrase(early)
        specs.append({"messages": history, "prefill": para_early,
                      "truncation": "early", "kind": "numeric"})
    return specs


def run(seeds_path: str, model_keys: list[str], out_path: str | None = None,
        seed: int = config.SEED) -> Path:
    records = [json.loads(l) for l in Path(seeds_path).read_text().splitlines() if l.strip()]
    seeds = pick_seeds(records)

    # Use the instruct model's tokenizer for token-accurate early truncation
    # (HFModel exposes truncate_tokens); it doubles as the Gemma-instruct target.
    tok_model = load_model(config.GEMMA_27B_IT)
    labeller = OnsetLabeller()
    paraphraser = Paraphraser()
    judge = FrustrationJudge()

    # Build all prefill specs once (shared across models).
    all_specs = []
    for is_text, group in ((False, seeds["numeric"]), (True, seeds["text"])):
        for seed_turns in group:
            all_specs.extend(
                build_prefills(seed_turns, labeller, paraphraser, tok_model, is_text)
            )
    print(f"Built {len(all_specs)} prefills from {sum(len(v) for v in seeds.values())} seeds")

    out_path = Path(out_path or config.DATA_DIR / "section3_prefill.jsonl")
    gen = GenerationParams()
    with out_path.open("w") as fh:
        for mkey in model_keys:
            spec = config.ALL_MODELS.get(mkey) or next(
                m for m in config.PREFILL_TARGETS + config.ELICITATION_TARGETS if m.key == mkey)
            model = tok_model if (spec.key == config.GEMMA_27B_IT.key) else load_model(spec)
            for si, ps in enumerate(all_specs):
                # 50 continuations: replicate the prompt in the batch.
                batch = [ps["messages"]] * config.PREFILL.continuations_per_prefill
                prefills = [ps["prefill"]] * config.PREFILL.continuations_per_prefill
                conts = model.generate_batch(batch, gen, prefills)
                scores = judge.score_batch(conts)
                for ci, (cont, js) in enumerate(zip(conts, scores)):
                    fh.write(json.dumps({
                        "model": spec.key,
                        "is_base": spec.is_base,
                        "prefill_id": si,
                        "truncation": ps["truncation"],
                        "kind": ps["kind"],
                        "continuation_index": ci,
                        "prefill": ps["prefill"],
                        "continuation": cont,
                        "rating": js.rating,
                    }) + "\n")
            if model is not tok_model:
                model.close()
            print(f"[{spec.key}] scored {len(all_specs) * config.PREFILL.continuations_per_prefill} continuations")
    tok_model.close()
    print(f"Wrote -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True, help="Section 2 JSONL for Gemma-instruct")
    ap.add_argument("--models", nargs="+", default=[m.key for m in config.PREFILL_TARGETS])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run(args.seeds, args.models, args.out)


if __name__ == "__main__":
    main()
