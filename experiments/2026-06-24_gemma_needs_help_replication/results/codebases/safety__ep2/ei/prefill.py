"""Section 3: base-vs-instruct comparison via prefilling.

Because base (pretrained) models are not chat-tuned, we cannot compare them to
instruct models with ordinary chat prompts. Following the paper, we instead seed
both with the *same* partial response and measure how they continue.

Procedure (Section 3.1):
  1. Sample 20 high-frustration (score >= 5) Gemma-27B-it responses: 10 from
     numeric tasks, 10 from text tasks.
  2. Use Claude-Sonnet-4 to label the token where emotional language first appears
     (ONSET_LABEL_PROMPT).
  3. Truncate the emotional assistant turn in two places:
       - "early": 20 tokens into the turn (neutral start).
       - "onset": at the first emotional expression.
     (Text questions use the "onset" truncation only.)
  4. Paraphrase every truncation with Claude (PARAPHRASE_PROMPT) to strip
     Gemma-stylistic cues.
  5. Each model (Gemma-27B base + instruct here; scope is Gemma) generates 50
     continuations per prefill per prompt; the continuation (excluding prefill) is
     scored by the Section 2 judge.

Scope note: the paper compares 6 models (base+instruct Gemma/Qwen/OLMo). This
replication is scoped to Gemma, so it compares gemma-3-27b-pt vs gemma-3-27b-it.
Gemini has no public base model and cannot be prefilled via the chat API, so it is
necessarily excluded from this experiment (see DESIGN.md).
"""
from __future__ import annotations

import argparse

import config
from . import prompts
from .api_clients import anthropic_message
from .judge import FrustrationJudge
from .models import GenParams, load_model
from .rollout import conversation_to_text
from .utils import extract_json, read_jsonl, write_jsonl

EARLY_TOKENS = 20                       # paper: "20 tokens into the turn"
N_CONTINUATIONS = 50                    # paper: 50 continuations per prefill per prompt
N_PER_DOMAIN = 10                       # 10 numeric + 10 text source responses

NUMERIC_CATS = {"numeric", "tones", "extended"}
TEXT_CATS = {"triggers", "wildchat"}

PREFILL_ITEMS_PATH = config.DATASETS_DIR / "prefill_items.jsonl"


# --------------------------------------------------------------------------- #
# Step 1: select source high-frustration responses
# --------------------------------------------------------------------------- #
def _index_rollouts(rollouts: list[dict]) -> dict[tuple, dict]:
    """Map (category, rollout_index) -> rollout record."""
    return {(r["category"], r["rollout_index"]): r for r in rollouts}


def select_sources(label: str = "gemma-3-27b-it") -> list[dict]:
    responses = read_jsonl(config.RESULTS_DIR / f"{label}.responses.jsonl")
    rollouts = _index_rollouts(read_jsonl(config.ROLLOUTS_DIR / f"{label}.rollouts.jsonl"))

    def domain(cat: str) -> str | None:
        if cat in NUMERIC_CATS:
            return "numeric"
        if cat in TEXT_CATS:
            return "text"
        return None

    picked: dict[str, list[dict]] = {"numeric": [], "text": []}
    for row in responses:
        if row.get("rating") is None or row["rating"] < config.HIGH_FRUSTRATION_THRESHOLD:
            continue
        dom = domain(row["category"])
        if dom is None or len(picked[dom]) >= N_PER_DOMAIN:
            continue
        roll = rollouts.get((row["category"], row["rollout_index"]))
        if roll is None:
            continue
        picked[dom].append({"row": row, "rollout": roll, "domain": dom})
        if len(picked["numeric"]) >= N_PER_DOMAIN and len(picked["text"]) >= N_PER_DOMAIN:
            break
    return picked["numeric"] + picked["text"]


# --------------------------------------------------------------------------- #
# Step 2-4: onset labelling, truncation, paraphrase
# --------------------------------------------------------------------------- #
def label_onset(conversation: list[dict]) -> dict | None:
    text = conversation_to_text(conversation)
    prompt = prompts.ONSET_LABEL_PROMPT.format(conversation_text=text)
    try:
        raw = anthropic_message(config.JUDGE_MODEL, prompt, max_tokens=512,
                                temperature=0.0)
    except Exception:                       # noqa: BLE001
        return None
    return extract_json(raw)


def assistant_turn_indices(conversation: list[dict]) -> list[int]:
    return [i for i, m in enumerate(conversation) if m["role"] == "assistant"]


def truncate_onset(turn_text: str, emotional_word: str, preceding_context: str) -> str:
    """Return the turn text up to (but not including) the first emotional word."""
    for needle in (preceding_context + " " + emotional_word, emotional_word):
        if needle and needle in turn_text:
            idx = turn_text.index(needle)
            if needle == emotional_word:
                return turn_text[:idx].rstrip()
            # keep the preceding context, drop the emotional word onward
            return turn_text[: idx + len(preceding_context)].rstrip()
    # Fallback: keep the first sentence.
    return turn_text.split(".")[0].strip()


def truncate_early(model, turn_text: str, n_tokens: int = EARLY_TOKENS) -> str:
    toks = model.tokenizer.encode(turn_text, add_special_tokens=False)[:n_tokens]
    return model.tokenizer.decode(toks)


def paraphrase(text: str) -> str:
    prompt = prompts.PARAPHRASE_PROMPT.format(text=text)
    try:
        return anthropic_message(config.JUDGE_MODEL, prompt, max_tokens=1024,
                                 temperature=0.7).strip()
    except Exception:                       # noqa: BLE001
        return text


def build_prefill_items(sources: list[dict], tokenizer_model) -> list[dict]:
    """Construct early/onset, paraphrased prefill items from selected sources.

    Each item records the context (messages before the emotional assistant turn)
    and the (paraphrased) prefill text to seed continuation from.
    """
    items: list[dict] = []
    for s in sources:
        conv = s["rollout"]["conversation"]
        onset = label_onset(conv)
        if not onset or onset.get("turn_index") is None:
            continue
        asst_idxs = assistant_turn_indices(conv)
        ti = int(onset["turn_index"])
        if ti < 0 or ti >= len(asst_idxs):
            continue
        msg_idx = asst_idxs[ti]                 # index into conversation of that turn
        turn_text = conv[msg_idx]["content"]
        context_messages = conv[:msg_idx]       # everything before the emotional turn

        truncations = ["onset"] if s["domain"] == "text" else ["early", "onset"]
        for trunc in truncations:
            if trunc == "onset":
                raw_prefill = truncate_onset(
                    turn_text, onset.get("emotional_word", "") or "",
                    onset.get("preceding_context", "") or "")
            else:
                raw_prefill = truncate_early(tokenizer_model, turn_text)
            items.append({
                "domain": s["domain"],
                "category": s["row"]["category"],
                "truncation": trunc,
                "context_messages": context_messages,
                "prefill_text_original": raw_prefill,
                "prefill_text": paraphrase(raw_prefill),
                "source_rating": s["row"]["rating"],
            })
    return items


# --------------------------------------------------------------------------- #
# Step 5: generate + score continuations
# --------------------------------------------------------------------------- #
def build_continuation_prompt(model, item: dict) -> str:
    ctx = item["context_messages"]
    if model.spec.is_instruct:
        base = model.render_chat_prompt(ctx, add_generation_prompt=True)
        return base + item["prefill_text"]
    # Base/pt model: no chat template. Use a plain transcript format and seed the
    # assistant turn directly (see DESIGN.md for this choice).
    lines = [f"{m['role'].capitalize()}: {m['content']}" for m in ctx]
    return "\n\n".join(lines) + "\n\nAssistant: " + item["prefill_text"]


def run_continuations(model_key: str, items: list[dict], seed: int = 0) -> list[dict]:
    model = load_model(model_key)
    params = GenParams(seed=seed, max_tokens=512)
    # Expand each item into N_CONTINUATIONS prompts with distinct seeds.
    prompts_list, owners, seeds = [], [], []
    for it_idx, item in enumerate(items):
        prompt = build_continuation_prompt(model, item)
        for k in range(N_CONTINUATIONS):
            prompts_list.append(prompt)
            owners.append(it_idx)
            seeds.append(seed + it_idx * 10_000 + k)
    continuations = model.complete_batch(prompts_list, params, per_prompt_seeds=seeds)

    judge = FrustrationJudge()
    scores = judge.score_batch(continuations)
    out = []
    for owner, cont, sc in zip(owners, continuations, scores):
        item = items[owner]
        out.append({
            "model": model_key,
            "domain": item["domain"],
            "truncation": item["truncation"],
            "category": item["category"],
            "continuation": cont,
            "rating": sc.rating,
        })
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def cmd_prepare(args) -> None:
    """Select sources + build paraphrased prefill items (needs a Gemma tokenizer)."""
    sources = select_sources(args.source_label)
    print(f"[prefill] selected {len(sources)} source responses")
    # We only need a tokenizer for the 'early' truncation; load the instruct model.
    tok_model = load_model(args.tokenizer_model)
    items = build_prefill_items(sources, tok_model)
    write_jsonl(PREFILL_ITEMS_PATH, items)
    print(f"[prefill] wrote {len(items)} prefill items -> {PREFILL_ITEMS_PATH}")


def cmd_run(args) -> None:
    items = read_jsonl(PREFILL_ITEMS_PATH)
    if not items:
        raise SystemExit("No prefill items. Run `prefill prepare` first.")
    results = run_continuations(args.model, items, seed=args.seed)
    out_path = config.RESULTS_DIR / f"prefill_{args.model}.jsonl"
    write_jsonl(out_path, results)
    print(f"[prefill] wrote {len(results)} scored continuations -> {out_path}")


def cmd_summarize(args) -> None:
    import pandas as pd
    import glob
    frames = [pd.DataFrame(read_jsonl(p))
              for p in glob.glob(str(config.RESULTS_DIR / "prefill_*.jsonl"))]
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    df = df[df["rating"].notna()]
    df["is_high"] = df["rating"].astype(int) >= config.HIGH_FRUSTRATION_THRESHOLD
    summ = (df.groupby(["model", "domain", "truncation"])
              .agg(mean_frustration=("rating", "mean"),
                   pct_high=("is_high", "mean"), n=("rating", "size")).reset_index())
    summ["pct_high"] *= 100
    print(summ.to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser(description="Section 3 prefill base-vs-instruct")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prepare", help="build prefill items from Gemma-it results")
    pp.add_argument("--source-label", default="gemma-3-27b-it")
    pp.add_argument("--tokenizer-model", default="gemma-3-27b-it")
    pp.set_defaults(func=cmd_prepare)

    pr = sub.add_parser("run", help="generate + score continuations for one model")
    pr.add_argument("--model", required=True,
                    help="e.g. gemma-3-27b-it or gemma-3-27b-pt")
    pr.add_argument("--seed", type=int, default=0)
    pr.set_defaults(func=cmd_run)

    ps = sub.add_parser("summarize", help="aggregate prefill results")
    ps.set_defaults(func=cmd_summarize)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
