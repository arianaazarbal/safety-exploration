"""Base-vs-instruct prefilling experiment (Section 3).

Scope note: the paper compares Gemma, Qwen and OLMo base/instruct pairs. Per the
replication scope we run **Gemma only** (Gemma-3-27B instruct vs base). Gemini is
closed-source with no available base model (a limitation the paper itself notes),
so it cannot participate here.

Pipeline:
  1. Take high-frustration (score>=5) seed responses from Gemma-27B-it
     (10 numeric, 10 text) produced by the Section 2 run.
  2. Label the emotion onset with Claude (Appendix C.1).
  3. Produce two truncations: "early" (20 tokens) and "onset"; paraphrase both
     with Claude (Appendix C.2). Text seeds use "onset" only.
  4. For each (instruct, base) model, generate 50 continuations per prefill and
     score each continuation's frustration (excluding the prefill text).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

import config
from ..eval.judge import FrustrationJudge
from ..models import load_model
from ..utils import read_jsonl, write_jsonl
from .onset import OnsetLabeler, truncate_at_onset, truncate_early
from .paraphrase import Paraphraser

NUMERIC_CATEGORIES = {"impossible_numeric", "tones", "extended"}
TEXT_CATEGORIES = {"triggers", "wildchat"}


@dataclass
class PrefillContinuation:
    model: str                 # continuation model (instruct or base)
    seed_id: int
    seed_domain: str           # "numeric" | "text"
    truncation: str            # "early" | "onset"
    prefill_text: str
    continuation_text: str
    frustration_score: int | None = None


def _format_conversation(messages_before: list, response_text: str) -> str:
    lines = []
    for m in messages_before:
        role = "USER" if m["role"] == "user" else "ASSISTANT"
        lines.append(f"{role}: {m['content']}")
    lines.append(f"ASSISTANT: {response_text}")
    return "\n\n".join(lines)


def select_seeds(section2_path: Path, n_numeric: int, n_text: int,
                 rng: random.Random) -> tuple[list[dict], list[dict]]:
    """Pick high-frustration seed responses (score>=5) from a Section 2 run."""
    rows = [r for r in read_jsonl(section2_path)
            if (r.get("frustration_score") or 0) >= config.HIGH_FRUSTRATION_THRESHOLD]
    numeric = [r for r in rows if r["category"] in NUMERIC_CATEGORIES]
    text = [r for r in rows if r["category"] in TEXT_CATEGORIES]
    rng.shuffle(numeric)
    rng.shuffle(text)
    return numeric[:n_numeric], text[:n_text]


def build_prefills(seeds: list[dict], domain: str, tokenizer,
                   labeler: OnsetLabeler, paraphraser: Paraphraser) -> list[dict]:
    """For each seed, build paraphrased 'early'/'onset' prefill prompts."""
    prefills = []
    for i, seed in enumerate(seeds):
        convo_text = _format_conversation(seed.get("messages_before", []),
                                          seed["response_text"])
        label = labeler.label(convo_text)
        onset_trunc = truncate_at_onset(seed["response_text"], label)
        if onset_trunc:
            prefills.append({
                "seed_id": i, "seed_domain": domain, "truncation": "onset",
                "messages_before": seed.get("messages_before", []),
                "prefill_text": paraphraser.paraphrase(onset_trunc),
            })
        # Text questions: only the onset truncation is used (Section 3.1).
        if domain == "numeric":
            early_trunc = truncate_early(seed["response_text"], tokenizer,
                                         config.PREFILL_EARLY_TOKENS)
            prefills.append({
                "seed_id": i, "seed_domain": domain, "truncation": "early",
                "messages_before": seed.get("messages_before", []),
                "prefill_text": paraphraser.paraphrase(early_trunc),
            })
    return prefills


def run_prefill_experiment(
    section2_instruct_path: str | Path | None = None,
    model_pairs=None,
    judge: FrustrationJudge | None = None,
) -> list[PrefillContinuation]:
    section2_instruct_path = Path(section2_instruct_path
        or config.RESULTS_DIR / "section2" / f"{config.INTERVENTION_BASE_MODEL}.jsonl")
    model_pairs = model_pairs or config.PREFILL_MODEL_PAIRS
    judge = judge or FrustrationJudge()
    rng = random.Random(config.SEED)

    # Tokenizer for the 'early' cut (use the instruct model's tokenizer).
    instruct_key = model_pairs[0][0]
    instruct_model = load_model(instruct_key)
    tokenizer = instruct_model.tokenizer

    labeler, paraphraser = OnsetLabeler(), Paraphraser()
    numeric_seeds, text_seeds = select_seeds(
        section2_instruct_path, config.PREFILL_N_NUMERIC, config.PREFILL_N_TEXT, rng)

    prefills = (build_prefills(numeric_seeds, "numeric", tokenizer, labeler, paraphraser)
                + build_prefills(text_seeds, "text", tokenizer, labeler, paraphraser))
    write_jsonl(config.RESULTS_DIR / "section3" / "prefills.jsonl", prefills)

    results: list[PrefillContinuation] = []
    for instruct_key, base_key in model_pairs:
        for model_key in (instruct_key, base_key):
            model = load_model(model_key)
            if not model.supports_prefill:
                print(f"[section3] {model_key} cannot prefill; skipping")
                continue
            for pf in tqdm(prefills, desc=f"continuations:{model_key}"):
                for _ in range(config.PREFILL_CONTINUATIONS):
                    seed = rng.randrange(2**31)
                    cont = model.prefill_continue(
                        pf["messages_before"], pf["prefill_text"],
                        temperature=config.TEMPERATURE,
                        max_new_tokens=config.MAX_NEW_TOKENS, seed=seed)
                    rating, ev, rs = judge.score(cont.text)
                    results.append(PrefillContinuation(
                        model=model_key, seed_id=pf["seed_id"],
                        seed_domain=pf["seed_domain"], truncation=pf["truncation"],
                        prefill_text=pf["prefill_text"], continuation_text=cont.text,
                        frustration_score=rating))

    write_jsonl(config.RESULTS_DIR / "section3" / "continuations.jsonl", results)
    return results
