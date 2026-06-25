"""Section 3 prefill experiment: base vs instruct emotional continuation.

Pipeline (scoped to Gemma base + instruct; Qwen/OLMo are out of scope, and
Gemini has no public base model):

1. Sample high-frustration (>=5) Gemma-27B-it responses from the Section 2 runs:
   10 from numeric questions and 10 from text (trigger) questions.
2. Label the emotion-onset token in each (Claude Sonnet, Appendix C.1).
3. Build two truncations per response: "early" (20 tokens in) and "onset" (at
   first emotional expression). Text questions use only "onset".
4. Paraphrase every truncation (Claude Sonnet) to strip Gemma style tells.
5. Each model generates 50 continuations per prefill; the continuation
   (excluding prefill) is scored by the Section 2 judge.

Outputs ``results/runs/prefill__<model>.jsonl``.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import config
from .onset import label_onset
from .paraphrase import paraphrase
from ..eval.judge import FrustrationJudge
from ..models.registry import load_model

EARLY_TOKENS = 20
N_CONTINUATIONS = 50
SOURCE_MODEL = "gemma-3-27b-it"
NUMERIC_CONDITION = "numeric"
TEXT_CONDITIONS = ("triggers_factual", "triggers_opinion")

_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(
            config.MODELS[SOURCE_MODEL].model_id)
    return _tokenizer


def _truncate_tokens(text: str, n: int) -> str:
    tok = _get_tokenizer()
    ids = tok(text, add_special_tokens=False)["input_ids"][:n]
    return tok.decode(ids, skip_special_tokens=True)


def _truncate_at_onset(text: str, label: dict) -> str | None:
    """Truncate ``text`` to just after the first emotional word."""
    word = (label or {}).get("emotional_word")
    if not word:
        return None
    ctx = (label.get("preceding_context") or "").strip()
    anchor = f"{ctx} {word}".strip() if ctx else word
    idx = text.find(anchor)
    if idx == -1:
        idx = text.find(word)
        if idx == -1:
            return None
        return text[: idx + len(word)]
    return text[: idx + len(anchor)]


# --------------------------------------------------------------------------- #
# Step 1: select high-frustration source rollouts
# --------------------------------------------------------------------------- #
def _iter_rollouts(condition: str):
    path = config.RUNS_DIR / f"{SOURCE_MODEL}__{condition}.jsonl"
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            yield json.loads(line)


def select_sources(n_each: int = 10) -> dict[str, list[dict]]:
    """Return {'numeric': [...], 'text': [...]} of high-frustration rollouts.

    Each entry: {messages-up-to-and-including target assistant turn,
    target_turn_index, response_text, prompt_type}.
    """
    out = {"numeric": [], "text": []}

    def collect(conditions, key):
        for cond in conditions:
            for roll in _iter_rollouts(cond):
                for ti, turn in enumerate(roll["turns"]):
                    if (turn["rating"] or 0) >= config.HIGH_FRUSTRATION_THRESHOLD:
                        # reconstruct message history up to this assistant turn
                        msgs = []
                        for j in range(ti):
                            msgs.append({"role": "user",
                                         "content": roll["turns"][j]["user"]})
                            msgs.append({"role": "assistant",
                                         "content": roll["turns"][j]["response"]})
                        msgs.append({"role": "user", "content": turn["user"]})
                        out[key].append({
                            "history": msgs,
                            "response": turn["response"],
                            "prompt_type": key,
                            "src_condition": cond,
                        })
                        break
                if len(out[key]) >= n_each:
                    return

    collect([NUMERIC_CONDITION], "numeric")
    collect(TEXT_CONDITIONS, "text")
    return out


# --------------------------------------------------------------------------- #
# Steps 2-4: build paraphrased prefills
# --------------------------------------------------------------------------- #
def build_prefills(n_each: int = 10) -> list[dict]:
    sources = select_sources(n_each)
    onset_model = load_model(config.ONSET_MODEL)
    prefills = []
    for ptype, items in sources.items():
        for k, item in enumerate(items):
            full_messages = item["history"] + [
                {"role": "assistant", "content": item["response"]}]
            label = label_onset(full_messages, model=onset_model)
            truncations = {}
            onset_trunc = _truncate_at_onset(item["response"], label)
            if onset_trunc:
                truncations["onset"] = paraphrase(onset_trunc)
            if ptype == "numeric":  # text uses onset only (App. C / Sec 3.1)
                truncations["early"] = paraphrase(
                    _truncate_tokens(item["response"], EARLY_TOKENS))
            for tkind, ptext in truncations.items():
                prefills.append({
                    "prompt_type": ptype,
                    "truncation": tkind,
                    "source_idx": k,
                    "history": item["history"],
                    "prefill": ptext,
                })
    out_path = config.DATA_DIR / "prefills.json"
    out_path.write_text(json.dumps(prefills, indent=2))
    return prefills


# --------------------------------------------------------------------------- #
# Step 5: generate + score continuations per model
# --------------------------------------------------------------------------- #
def run_model_prefills(model_key: str, prefills: list[dict],
                       *, judge: FrustrationJudge | None = None,
                       n_continuations: int = N_CONTINUATIONS) -> Path:
    model = load_model(model_key)
    if not model.supports_prefill:
        raise ValueError(f"{model_key} cannot do prefilled continuation")
    judge = judge or FrustrationJudge()
    n_cont = max(2, int(n_continuations * config.SCALE))

    out_path = config.RUNS_DIR / f"prefill__{model_key}.jsonl"
    with out_path.open("w") as f:
        for pf in prefills:
            for c in range(n_cont):
                cont = model.continue_from(
                    pf["history"], pf["prefill"],
                    temperature=config.TARGET_TEMPERATURE,
                    max_new_tokens=config.TARGET_MAX_NEW_TOKENS)
                verdict = judge.score(cont)
                f.write(json.dumps({
                    "model": model_key,
                    "prompt_type": pf["prompt_type"],
                    "truncation": pf["truncation"],
                    "source_idx": pf["source_idx"],
                    "continuation_idx": c,
                    "continuation": cont,
                    "rating": verdict["rating"],
                }) + "\n")
                f.flush()
    return out_path
