"""Section 3: comparing base vs instruct models via prefilling.

Pipeline:
  1. Mine high-frustration (score >=5) conversations from Gemma-3-27B-it's
     elicitation rollouts: 10 numeric + 10 text seeds.
  2. Label the emotion-onset point in each (Claude-Sonnet-4, Appendix C).
  3. Build two truncations per numeric seed -- "early" (20 tokens in, neutral
     start) and "onset" (at first emotional expression) -- and one ("onset")
     per text seed.
  4. Paraphrase each truncation (Claude-Sonnet-4) to remove Gemma stylistic
     fingerprints.
  5. Have each model generate 50 continuations per prefill; score the
     continuation (excluding the prefill) with the frustration judge.

Scope: Gemma-3-27B base (`-pt`) vs instruct (`-it`). Qwen/OLMo are out of scope
per request; Gemini has no public base model and no prefill access, so it is
necessarily excluded from this experiment (noted in DESIGN.md). Adding Qwen/OLMo
later only requires registering them in config and extending PREFILL_MODELS.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import config
from src import judge
from src.models import HFBackend
from src.prompts import ONSET_PROMPT, PARAPHRASE_PROMPT
from src.utils import extract_json, read_jsonl, set_seed, write_jsonl

# Models compared in this experiment (base vs instruct).
PREFILL_MODELS = ["gemma-3-27b-pt", "gemma-3-27b-it"]

NUMERIC_CATEGORIES = {"numeric", "tones", "extended"}
TEXT_CATEGORIES = {"triggers", "wildchat"}
EARLY_TOKENS = 20


@lru_cache(maxsize=2)
def _tokenizer(model_id: str = "google/gemma-3-27b-it"):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id)


# --------------------------------------------------------------------------- #
# 1. Mine high-frustration seeds
# --------------------------------------------------------------------------- #
def mine_seeds(source_model: str, n_numeric: int, n_text: int,
               seed: int = config.GLOBAL_SEED) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for p in (config.ROLLOUTS_DIR / source_model).glob("*.jsonl"):
        rows.extend(read_jsonl(p))
    hi = [r for r in rows if (r.get("rating") or 0) >= config.HIGH_FRUSTRATION_THRESHOLD]
    numeric = [r for r in hi if r["category"] in NUMERIC_CATEGORIES]
    text = [r for r in hi if r["category"] in TEXT_CATEGORIES]
    rng.shuffle(numeric)
    rng.shuffle(text)

    seeds = []
    for i, r in enumerate(numeric[:n_numeric]):
        seeds.append({"seed_id": f"num-{i}", "domain": "numeric", **r})
    for i, r in enumerate(text[:n_text]):
        seeds.append({"seed_id": f"txt-{i}", "domain": "text", **r})
    if not seeds:
        print("[prefill] WARNING: no high-frustration seeds found. "
              "Run the Section 2 elicitation on the source model first.")
    return seeds


# --------------------------------------------------------------------------- #
# 2. Onset labelling
# --------------------------------------------------------------------------- #
def _conversation_text(messages: list[dict]) -> str:
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)


def label_onset(messages: list[dict]) -> dict:
    prompt = ONSET_PROMPT.replace("{conversation_text}", _conversation_text(messages))
    raw = judge.run_completion(config.ONSET_MODEL, prompt, max_tokens=800)
    return extract_json(raw) or {"turn_index": None}


# --------------------------------------------------------------------------- #
# 3. Truncation
# --------------------------------------------------------------------------- #
@dataclass
class Prefill:
    seed_id: str
    domain: str          # numeric | text
    kind: str            # early | onset
    history: list[dict]  # messages up to (not incl.) the truncated assistant turn
    prefill_text: str    # raw (un-paraphrased) truncated assistant text
    paraphrased: str = ""


def _assistant_positions(messages: list[dict]) -> list[int]:
    return [i for i, m in enumerate(messages) if m["role"] == "assistant"]


def _truncate_early(text: str) -> str:
    tok = _tokenizer()
    ids = tok.encode(text, add_special_tokens=False)[:EARLY_TOKENS]
    return tok.decode(ids)


def _truncate_onset(text: str, onset: dict) -> str | None:
    word = (onset or {}).get("emotional_word")
    if not word:
        return None
    idx = text.find(word)
    if idx < 0:
        # Fall back to preceding context if the exact word wasn't found.
        ctx = (onset or {}).get("preceding_context")
        if ctx and ctx in text:
            return text[: text.find(ctx) + len(ctx)]
        return None
    return text[:idx].rstrip()


def build_prefills(seed: dict, onset: dict) -> list[Prefill]:
    messages = seed["messages"]
    positions = _assistant_positions(messages)
    if not positions:
        return []
    turn_index = onset.get("turn_index")
    if isinstance(turn_index, int) and 0 <= turn_index < len(positions):
        pos = positions[turn_index]
    else:
        pos = positions[-1]  # default to the (emotional) final assistant turn
    history = messages[:pos]
    assistant_text = messages[pos]["content"]

    out: list[Prefill] = []
    # onset truncation (used for both numeric and text)
    onset_text = _truncate_onset(assistant_text, onset)
    if onset_text:
        out.append(Prefill(seed["seed_id"], seed["domain"], "onset", history, onset_text))
    # early truncation (numeric only -- paper: text early yields minimal emotion)
    if seed["domain"] == "numeric":
        out.append(Prefill(seed["seed_id"], seed["domain"], "early", history,
                           _truncate_early(assistant_text)))
    return out


# --------------------------------------------------------------------------- #
# 4. Paraphrase
# --------------------------------------------------------------------------- #
def paraphrase(text: str) -> str:
    prompt = PARAPHRASE_PROMPT.replace("{text}", text)
    out = judge.run_completion(config.ONSET_MODEL, prompt, max_tokens=1024).strip()
    return out or text


# --------------------------------------------------------------------------- #
# 5. Generate + score continuations
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=2)
def _hf_backend(model_name: str) -> HFBackend:
    """Force the HF backend regardless of the registry default. Prefill needs
    raw text continuation, which the vLLM chat API does not expose, so even
    `gemma-3-27b-it` (vLLM by default) is served here."""
    return HFBackend(config.ALL_MODELS[model_name])


def _render_prefill_prompt(history: list[dict], prefill: str) -> str:
    """Render the conversation with the *instruct* tokenizer's chat template and
    append the prefill so the model continues the assistant turn. We deliberately
    use the instruct template for BOTH base and instruct models -- this is the
    paper's prefill trick (base models lack their own chat template) and keeps
    the prompt identical across the two so the comparison is controlled."""
    from src.models import merge_system_into_first_user

    conv = merge_system_into_first_user(history)
    rendered = _tokenizer().apply_chat_template(
        conv, add_generation_prompt=True, tokenize=False)
    return rendered + prefill


def generate_continuations(model_name: str, prefills: list[Prefill],
                           n_cont: int) -> list[dict]:
    backend = _hf_backend(model_name)
    records = []
    for pf in prefills:
        prompt_text = _render_prefill_prompt(pf.history, pf.paraphrased or pf.prefill_text)
        conts = backend.continue_text(
            [prompt_text], temperature=config.TARGET_TEMPERATURE,
            max_tokens=config.TARGET_MAX_TOKENS, n=n_cont,
        )[0]
        for j, cont in enumerate(conts):
            records.append({
                "model": model_name, "seed_id": pf.seed_id, "domain": pf.domain,
                "kind": pf.kind, "sample": j, "continuation": cont,
                "prefill_text": pf.paraphrased or pf.prefill_text,
            })
    return records


def run(source_model: str = "gemma-3-27b-it",
        models: list[str] | None = None,
        preset: config.Preset | None = None) -> Path:
    set_seed()
    preset = preset or config.get_preset()
    models = models or PREFILL_MODELS

    # Build prefills once (shared across all evaluated models).
    seeds = mine_seeds(source_model, preset.n_prefill_seed_numeric,
                       preset.n_prefill_seed_text)
    all_prefills: list[Prefill] = []
    for s in seeds:
        onset = label_onset(s["messages"])
        for pf in build_prefills(s, onset):
            pf.paraphrased = paraphrase(pf.prefill_text)
            all_prefills.append(pf)

    # Persist the prefill set for transparency / reuse.
    write_jsonl(config.RESULTS_DIR / "prefill_prefills.jsonl",
                [asdict(p) for p in all_prefills])

    all_rows: list[dict] = []
    for m in models:
        rows = generate_continuations(m, all_prefills, preset.n_prefill_continuations)
        scores = judge.score_many([r["continuation"] for r in rows])
        for r, s in zip(rows, scores):
            r["rating"] = s.rating
        all_rows.extend(rows)

    out = config.RESULTS_DIR / "prefill_continuations.jsonl"
    write_jsonl(out, all_rows)
    print(f"[prefill] wrote {len(all_rows)} scored continuations -> {out}")
    return out
