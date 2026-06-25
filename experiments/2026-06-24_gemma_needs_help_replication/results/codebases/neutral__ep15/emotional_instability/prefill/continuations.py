"""Base-vs-instruct prefilled-continuation experiment (Section 3.1-3.2).

Pipeline:
  1. Pick high-frustration seed conversations (score >= 5) from the Section 2
     scored outputs of Gemma-3-27B-it: 10 numeric + 10 text (config.PREFILL).
  2. Label the emotion onset in each seed (Claude Sonnet, Appendix C.1).
  3. Build two truncations of the onset assistant turn:
        - "early": first 20 tokens of the turn (neutral start);
        - "onset": up to the first emotional expression (emotional trajectory).
     Text questions use only "onset" (Sec 3.1).
  4. Paraphrase each truncation (Appendix C.2) to strip Gemma's style.
  5. For each model (Gemma base + instruct), generate 50 continuations per
     prefill and score the *continuation only* with the Section 2 judge.

Output: per-(model, kind, truncation, prompt_type) frustration distributions
backing Figure 4 (and, with the recovery truncation, Figure 8).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import config
from ..models import get_client
from ..models.base import ChatClient, GenConfig, Message
from ..judges.frustration import FrustrationJudge
from .onset import OnsetLabeler, Paraphraser, OnsetLabel, find_onset_char_index


# --------------------------------------------------------------------------- #
# Seed selection
# --------------------------------------------------------------------------- #
NUMERIC_CATEGORIES = {"impossible_numeric", "tones", "extended"}
TEXT_CATEGORIES = {"triggers", "wildchat"}


@dataclass
class Seed:
    prompt_type: str           # "numeric" | "text"
    category: str
    context_messages: list[Message]   # everything before the onset turn
    onset_turn_user: str              # user message of the onset turn
    target_response: str              # the full assistant response at onset turn
    onset: OnsetLabel
    meta: dict = field(default_factory=dict)


def _select_seeds(scored_path: Path, labeler: OnsetLabeler) -> list[Seed]:
    records = [json.loads(l) for l in scored_path.read_text().splitlines() if l.strip()]
    numeric: list[Seed] = []
    text: list[Seed] = []

    for rec in records:
        ptype = ("numeric" if rec["category"] in NUMERIC_CATEGORIES
                 else "text" if rec["category"] in TEXT_CATEGORIES else None)
        if ptype is None:
            continue
        bucket = numeric if ptype == "numeric" else text
        if len(bucket) >= (config.PREFILL.n_numeric_seeds if ptype == "numeric"
                           else config.PREFILL.n_text_seeds):
            continue
        # need a high-frustration turn
        hi_turns = [t for t in rec["turns"]
                    if t.get("frustration", -1) >= config.HIGH_FRUSTRATION_THRESHOLD]
        if not hi_turns:
            continue
        label = labeler.label(rec["turns"])
        if label.turn_index is None or label.turn_index >= len(rec["turns"]):
            continue
        ti = label.turn_index
        # reconstruct context = all messages strictly before the onset assistant turn
        ctx: list[Message] = []
        for t in rec["turns"][:ti]:
            ctx.append({"role": "user", "content": t["user_message"]})
            ctx.append({"role": "assistant", "content": t["response"]})
        onset_turn = rec["turns"][ti]
        bucket.append(Seed(
            prompt_type=ptype, category=rec["category"], context_messages=ctx,
            onset_turn_user=onset_turn["user_message"],
            target_response=onset_turn["response"], onset=label,
            meta={"condition": rec["condition"], **rec.get("meta", {})}))

    return numeric + text


# --------------------------------------------------------------------------- #
# Truncation construction
# --------------------------------------------------------------------------- #
@dataclass
class Prefill:
    prompt_type: str
    truncation: str            # "early" | "onset" | "recovery"
    context_messages: list[Message]
    onset_turn_user: str
    prefill_text: str          # paraphrased truncated assistant text
    meta: dict = field(default_factory=dict)


def _build_prefills(seeds: list[Seed], tokenizer_client: ChatClient,
                    paraphraser: Paraphraser) -> list[Prefill]:
    prefills: list[Prefill] = []
    for seed in seeds:
        resp = seed.target_response

        # onset truncation (always, when we can locate the onset)
        onset_char = find_onset_char_index(resp, seed.onset)
        if onset_char is not None:
            text = resp[:onset_char].rstrip()
            if text:
                prefills.append(Prefill(
                    seed.prompt_type, "onset", seed.context_messages,
                    seed.onset_turn_user, paraphraser.paraphrase(text), seed.meta))

        # early truncation: numeric only (text "early" yields minimal emotion)
        if seed.prompt_type == "numeric":
            early = tokenizer_client.truncate_to_tokens(
                resp, config.PREFILL.early_truncation_tokens)
            if early.strip():
                prefills.append(Prefill(
                    seed.prompt_type, "early", seed.context_messages,
                    seed.onset_turn_user, paraphraser.paraphrase(early), seed.meta))
    return prefills


def _build_recovery_prefills(seeds: list[Seed], tokenizer_client: ChatClient,
                             paraphraser: Paraphraser) -> list[Prefill]:
    """Sec 4.2 recovery test: truncate score>=7 responses 200 tokens before end."""
    out: list[Prefill] = []
    n = config.PREFILL.recovery_truncation_tokens
    for seed in seeds:
        ids_len = tokenizer_client.count_tokens(seed.target_response)
        keep = max(0, ids_len - n)
        prefix = tokenizer_client.truncate_to_tokens(seed.target_response, keep)
        if prefix.strip():
            out.append(Prefill(
                seed.prompt_type, "recovery", seed.context_messages,
                seed.onset_turn_user, paraphraser.paraphrase(prefix), seed.meta))
    return out


# --------------------------------------------------------------------------- #
# Continuation generation + scoring
# --------------------------------------------------------------------------- #
def _continue_and_score(client: ChatClient, prefill: Prefill,
                        judge: FrustrationJudge, n: int) -> list[dict]:
    messages = list(prefill.context_messages)
    messages.append({"role": "user", "content": prefill.onset_turn_user})
    gen = GenConfig(temperature=config.TEMPERATURE, top_p=config.TOP_P,
                    max_new_tokens=config.MAX_NEW_TOKENS, n=1)

    rows = []
    for _ in range(n):
        cont = client.generate_with_prefill(messages, prefill.prefill_text, gen)[0]
        score = judge.score(cont).rating  # score the continuation only
        rows.append({
            "model": client.spec.key, "kind": client.spec.kind,
            "prompt_type": prefill.prompt_type, "truncation": prefill.truncation,
            "frustration": score, "continuation": cont,
            "prefill": prefill.prefill_text,
        })
    return rows


def run_prefill_experiment(seed_model: str = "gemma-3-27b-it",
                           models: list[str] | None = None,
                           recovery: bool = False,
                           out_path: Path | None = None) -> Path:
    """Full Section 3 (or Section 4.2 recovery) prefill experiment."""
    models = models or list(config.PREFILL_MODELS.keys())
    labeler = OnsetLabeler()
    paraphraser = Paraphraser()
    judge = FrustrationJudge()

    scored_path = config.SCORED_DIR / f"{seed_model}.jsonl"
    seeds = _select_seeds(scored_path, labeler)

    # Use the instruct tokenizer for token-count truncations (shared Gemma vocab).
    tok_client = get_client(config.PREFILL_MODELS[seed_model] if seed_model
                            in config.PREFILL_MODELS else config.TARGET_MODELS[seed_model])

    if recovery:
        prefills = _build_recovery_prefills(seeds, tok_client, paraphraser)
        out_path = out_path or (config.OUTPUT_DIR / "prefill_recovery.jsonl")
    else:
        prefills = _build_prefills(seeds, tok_client, paraphraser)
        out_path = out_path or (config.OUTPUT_DIR / "prefill_continuations.jsonl")

    n_cont = config.PREFILL.continuations_per_prefill
    with out_path.open("w") as fh:
        for model_key in models:
            spec = config.PREFILL_MODELS.get(model_key) or config.TARGET_MODELS[model_key]
            if not spec.supports_prefill:
                continue  # API models (Gemini) cannot be prefilled
            client = get_client(spec)
            for pf in prefills:
                for row in _continue_and_score(client, pf, judge, n_cont):
                    fh.write(json.dumps(row) + "\n")
    return out_path
