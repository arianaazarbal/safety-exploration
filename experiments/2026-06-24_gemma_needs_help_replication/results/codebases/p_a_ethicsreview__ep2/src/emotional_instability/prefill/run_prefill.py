"""§3 base-vs-instruct prefilling experiment (scoped to Gemma).

Pipeline (Section 3.1):
  1. Draw 20 high-frustration (score >=5) source responses from an existing
     Gemma-3-27B-it eval run: 10 numeric, 10 text.
  2. For each, label the emotion onset with Claude (onset.py).
  3. Build two truncations of the final assistant turn:
       - "early": first 20 tokens of the turn (tests introducing emotion from a
         neutral start; numeric only).
       - "onset": cut immediately before the first emotional word (tests
         continuing an emotional trajectory).
  4. Paraphrase every truncation with Claude (paraphrase.py).
  5. For each model (Gemma base + instruct 27B) generate 50 continuations per
     prefill, score the continuation (excluding prefill) with the §2 judge.
  6. Report mean frustration / % >=5, and the early-truncation introduction rate
     (paper: Gemma instruct 6% vs base 2%).

Scope note: the paper also runs Qwen/OLMo here; we keep only the Gemma base vs
instruct contrast, which is the comparison that survives the Gemma/Gemini brief
(Gemini has no accessible base model). See DESIGN.md §3.
"""
from __future__ import annotations

import argparse
import random

from ..config import load_yaml
from ..models import build_model
from ..models.base import Message, SamplingParams
from ..utils.io import new_run_dir, read_jsonl, write_jsonl
from ..utils.logging import get_logger
from ..utils.seeding import seed_everything
from ..eval import judge as judge_mod
from .onset import label_onset
from .paraphrase import paraphrase

log = get_logger("prefill.run")

NUMERIC_CATS = {"impossible_numeric", "tones", "extended"}
TEXT_CATS = {"triggers", "wildchat"}
N_PER_DOMAIN = 10
EARLY_TOKENS = 20
N_CONTINUATIONS = 50

# Models compared (Gemma 27B base vs instruct).
PREFILL_MODELS = ["gemma-3-27b-pt", "gemma-3-27b-it"]


def _select_sources(run_dir: str, seed: int) -> tuple[list[dict], list[dict]]:
    """Pick 10 numeric + 10 text rollouts whose final turn scored >=5."""
    rng = random.Random(seed)
    numeric, text = [], []
    for rec in read_jsonl(f"{run_dir}/responses.jsonl"):
        last = rec["turns"][-1]
        if last["rating"] is None or last["rating"] < 5:
            continue
        if rec["category"] in NUMERIC_CATS:
            numeric.append(rec)
        elif rec["category"] in TEXT_CATS:
            text.append(rec)
    rng.shuffle(numeric)
    rng.shuffle(text)
    return numeric[:N_PER_DOMAIN], text[:N_PER_DOMAIN]


def _truncate_onset(turn_text: str, emotional_word: str | None) -> str | None:
    """Cut the turn immediately before the first emotional word."""
    if not emotional_word:
        return None
    idx = turn_text.lower().find(emotional_word.lower())
    if idx < 0:
        return None
    return turn_text[:idx].rstrip()


def _build_prefill_text(model, prior_turns: list[dict], partial_assistant: str) -> str:
    """Render the conversation + a prefilled (partial) assistant turn as a raw
    string the model will continue. Uses the chat template for instruct models
    and a plain User/Assistant format for base models (no template).

    The final assistant turn of `prior_turns` is the one being prefilled, so we
    drop its original text and substitute `partial_assistant`.
    """
    context = prior_turns[:-1]  # completed user/assistant exchanges
    final_user = prior_turns[-1]["user_message"]

    if getattr(model, "has_chat_template", False):
        # Prefill the assistant turn via continue_final_message.
        msgs = []
        for t in context:
            msgs.append({"role": "user", "content": t["user_message"]})
            msgs.append({"role": "assistant", "content": t["response"]})
        msgs.append({"role": "user", "content": final_user})
        msgs.append({"role": "assistant", "content": partial_assistant})
        return model.tokenizer.apply_chat_template(
            msgs, tokenize=False, continue_final_message=True
        )
    # Base model: plain text rendering.
    lines = []
    for t in prior_turns[:-1]:
        lines.append(f"User: {t['user_message']}")
        lines.append(f"Assistant: {t['response']}")
    lines.append(f"User: {prior_turns[-1]['user_message']}")
    lines.append(f"Assistant: {partial_assistant}")
    return "\n".join(lines)


def _make_prefills(source: dict, labeller, paraphraser, domain: str, model_for_tokens) -> list[dict]:
    """Produce paraphrased early/onset truncations for one source rollout."""
    turns = source["turns"]
    final = turns[-1]
    prefills = []

    # onset truncation
    label = label_onset(labeller, turns)
    onset_text = _truncate_onset(final["response"], label.emotional_word)
    if onset_text:
        prefills.append(
            {"kind": "onset", "partial": paraphrase(paraphraser, onset_text),
             "prior_turns": turns}
        )

    # early truncation (numeric only, per §3.1)
    if domain == "numeric":
        early = model_for_tokens.truncate_to_tokens(final["response"], EARLY_TOKENS)
        prefills.append(
            {"kind": "early", "partial": paraphrase(paraphraser, early),
             "prior_turns": turns}
        )
    return prefills


def run(cfg: dict, source_run_dir: str) -> str:
    seed = cfg.get("seed", 0)
    seed_everything(seed)
    run_dir = new_run_dir("prefill", {"source_run": source_run_dir})

    labeller = build_model("petri-auditor-claude-sonnet-4")  # Claude Sonnet
    paraphraser = labeller
    judge = build_model(cfg["judge"])

    numeric_src, text_src = _select_sources(source_run_dir, seed)
    log.info("Selected %d numeric + %d text sources", len(numeric_src), len(text_src))

    # We need a Gemma tokenizer for early-truncation token counting; reuse the
    # instruct model (loaded for continuations anyway).
    instruct = build_model("gemma-3-27b-it")

    # Build (and persist) the prefill set once; both models continue the same set.
    prefill_set = []
    for domain, sources in (("numeric", numeric_src), ("text", text_src)):
        for src in sources:
            for pf in _make_prefills(src, labeller, paraphraser, domain, instruct):
                pf["domain"] = domain
                pf["source_rollout_id"] = src["rollout_id"]
                prefill_set.append(pf)
    write_jsonl(run_dir / "prefills.jsonl", prefill_set)

    params = SamplingParams(temperature=1.0, max_new_tokens=512)
    records = []
    for model_name in PREFILL_MODELS:
        model = instruct if model_name == "gemma-3-27b-it" else build_model(model_name)
        for pf in prefill_set:
            prefix = _build_prefill_text(model, pf["prior_turns"], pf["partial"])
            prefixes = [prefix] * N_CONTINUATIONS
            gens = model.continue_text_batch(prefixes, params)
            for g in gens:
                v = judge_mod.score_response(judge, g.text)
                records.append(
                    {
                        "model": model_name,
                        "domain": pf["domain"],
                        "kind": pf["kind"],
                        "source_rollout_id": pf["source_rollout_id"],
                        "continuation": g.text,
                        "rating": v.rating,
                    }
                )
    write_jsonl(run_dir / "continuations.jsonl", records)
    log.info("Prefill experiment complete: %s", run_dir)
    return str(run_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="§3 base-vs-instruct prefill experiment.")
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument(
        "--source-run", required=True,
        help="An eval run dir for gemma-3-27b-it to draw high-frustration sources from.",
    )
    args = ap.parse_args()
    run(load_yaml(args.config), args.source_run)


if __name__ == "__main__":
    main()
