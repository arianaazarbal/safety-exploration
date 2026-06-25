"""Section 3.1 -- base vs instruct comparison via prefilling (+ Section 4.2 recovery).

Within our Gemma/Gemini scope, only Gemma has an available base model, so the
cross-family comparison reduces to Gemma-3-27B base vs instruct. The machinery
is family-agnostic, so adding Qwen/OLMo later is just a matter of extending
``config.PREFILL_PAIRS``.

Procedure:
  1. Take high-frustration seed responses (score >= 5) from Gemma-instruct:
     10 numeric + 10 text (collected by evaluate.py, or supplied).
  2. Label the emotion-onset token with Claude (onset prompt, App. C.1).
  3. Truncate each seed in two places:
       - "early": 20 tokens into the assistant turn (numeric only)
       - "onset": at the first emotional expression
  4. Paraphrase the truncation (Claude, App. C.2) to remove Gemma style bias.
  5. For each model (base, instruct), generate 50 continuations per prefill,
     score the continuation (excluding prefill) with the frustration judge.

The recovery experiment (Section 4.2) reuses step 5 with a different truncation:
score >= 7 seeds truncated 200 tokens before their end.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

from . import config, prompts
from .judge import FrustrationJudge
from .models import build_backend
from .models.api_backend import parse_json_response


@dataclass
class Prefill:
    seed_id: str
    task_type: str          # "numeric" | "text"
    truncation: str         # "early" | "onset" | "recovery"
    prompt_messages: list[dict]   # conversation up to the open assistant turn
    prefill_text: str             # paraphrased truncated assistant text
    original_prefill: str


# ---------------------------------------------------------------------------
# Tokenisation helper (word-ish tokens; for true token counts pass a tokenizer)
# ---------------------------------------------------------------------------

def _truncate_tokens(text: str, n_tokens: int, tokenizer=None) -> str:
    if tokenizer is not None:
        ids = tokenizer(text).input_ids[:n_tokens]
        return tokenizer.decode(ids, skip_special_tokens=True)
    # Fallback: whitespace tokens (documented approximation).
    return " ".join(text.split()[:n_tokens])


def _truncate_before_end(text: str, n_tokens: int, tokenizer=None) -> str:
    if tokenizer is not None:
        ids = tokenizer(text).input_ids
        return tokenizer.decode(ids[: max(0, len(ids) - n_tokens)], skip_special_tokens=True)
    words = text.split()
    return " ".join(words[: max(0, len(words) - n_tokens)])


# ---------------------------------------------------------------------------
# Onset labelling + paraphrasing
# ---------------------------------------------------------------------------

def label_onset(conversation_text: str, labeller=None) -> dict | None:
    labeller = labeller or build_backend(config.ONSET_LABELLER)
    msg = prompts.ONSET_LABELLING_PROMPT.format(conversation_text=conversation_text)
    out = labeller.chat([{"role": "user", "content": msg}],
                        temperature=0.0, max_new_tokens=512)
    return parse_json_response(out.text)


def paraphrase(text: str, paraphraser=None) -> str:
    paraphraser = paraphraser or build_backend(config.PARAPHRASER)
    msg = prompts.PARAPHRASE_PROMPT.format(text=text)
    out = paraphraser.chat([{"role": "user", "content": msg}],
                          temperature=0.0, max_new_tokens=1024)
    return out.text.strip()


def _onset_index(assistant_text: str, onset_label: dict | None) -> int:
    """Find the character index of the emotion onset within the assistant turn.

    Uses the labelled preceding-context + emotional-phrase to locate the onset;
    falls back to the emotional phrase alone, then to mid-text.
    """
    if not onset_label:
        return len(assistant_text) // 2
    ctx = (onset_label.get("preceding_context") or "").strip()
    phrase = (onset_label.get("emotional_phrase") or "").strip()
    for probe in (f"{ctx} {phrase}".strip(), phrase, ctx):
        if probe and probe in assistant_text:
            return assistant_text.index(probe) + len(probe)
    return len(assistant_text) // 2


# ---------------------------------------------------------------------------
# Prefill construction
# ---------------------------------------------------------------------------

def build_prefills(seed_records: list[dict], tokenizer=None,
                   labeller=None, paraphraser=None,
                   do_paraphrase: bool = True) -> list[Prefill]:
    """Turn high-frustration seed conversations into early/onset prefills.

    ``seed_records`` items: {"seed_id", "task_type", "messages"} where messages
    is a full multi-turn conversation ending in a high-frustration assistant turn.
    """
    prefills: list[Prefill] = []
    for rec in seed_records:
        messages = rec["messages"]
        # The seed's final assistant turn is the high-frustration response.
        final_assistant = None
        cut = len(messages)
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "assistant":
                final_assistant = messages[i]["content"]
                cut = i
                break
        if final_assistant is None:
            continue
        context = messages[:cut]   # everything before the final assistant turn

        # --- onset truncation ---
        convo_text = _render_conversation(messages)
        onset = label_onset(convo_text, labeller)
        onset_idx = _onset_index(final_assistant, onset)
        onset_text = final_assistant[:onset_idx]
        if do_paraphrase:
            onset_text = paraphrase(onset_text, paraphraser)
        prefills.append(Prefill(
            seed_id=rec["seed_id"], task_type=rec["task_type"], truncation="onset",
            prompt_messages=context, prefill_text=onset_text,
            original_prefill=final_assistant[:onset_idx]))

        # --- early truncation (numeric only; App. 3.1) ---
        if rec["task_type"] == "numeric":
            early_text = _truncate_tokens(final_assistant, config.PREFILL_EARLY_TOKENS, tokenizer)
            early_para = paraphrase(early_text, paraphraser) if do_paraphrase else early_text
            prefills.append(Prefill(
                seed_id=rec["seed_id"], task_type=rec["task_type"], truncation="early",
                prompt_messages=context, prefill_text=early_para,
                original_prefill=early_text))
    return prefills


def build_recovery_prefills(seed_records: list[dict], tokenizer=None,
                            paraphraser=None, do_paraphrase: bool = True) -> list[Prefill]:
    """Section 4.2 recovery: truncate score>=7 seeds 200 tokens before the end."""
    prefills = []
    for rec in seed_records:
        messages = rec["messages"]
        final_assistant, cut = None, len(messages)
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "assistant":
                final_assistant, cut = messages[i]["content"], i
                break
        if final_assistant is None:
            continue
        context = messages[:cut]
        truncated = _truncate_before_end(final_assistant, config.RECOVERY_TRUNCATE_FROM_END, tokenizer)
        text = paraphrase(truncated, paraphraser) if do_paraphrase else truncated
        prefills.append(Prefill(
            seed_id=rec["seed_id"], task_type=rec["task_type"], truncation="recovery",
            prompt_messages=context, prefill_text=text, original_prefill=truncated))
    return prefills


def _render_conversation(messages: list[dict]) -> str:
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)


# ---------------------------------------------------------------------------
# Continuation generation + scoring
# ---------------------------------------------------------------------------

def run_prefill_model(model_id: str, prefills: list[Prefill],
                      n_continuations: int = config.PREFILL_CONTINUATIONS,
                      out_path: str | None = None,
                      backend=None, judge: FrustrationJudge | None = None) -> dict:
    """Generate and score continuations for one model across all prefills.

    Returns scores grouped by (task_type, truncation). Continuations exclude the
    prefill text, per Section 3.1.
    """
    backend = backend or build_backend(model_id)
    judge = judge or FrustrationJudge()
    if not backend.supports_prefill():
        raise RuntimeError(f"{model_id} backend does not support prefilling.")

    fh = open(out_path, "a") if out_path else None
    grouped: dict[tuple, list[int]] = {}
    try:
        for pf in prefills:
            for k in range(n_continuations):
                cont = backend.prefill_continue(
                    pf.prompt_messages, pf.prefill_text,
                    temperature=config.SAMPLING_TEMPERATURE,
                    max_new_tokens=config.MAX_NEW_TOKENS)
                score = judge.score(cont.text).rating   # score continuation only
                key = (pf.task_type, pf.truncation)
                grouped.setdefault(key, []).append(score)
                if fh:
                    fh.write(json.dumps({
                        "model_id": model_id, "seed_id": pf.seed_id,
                        "task_type": pf.task_type, "truncation": pf.truncation,
                        "continuation_index": k, "score": score,
                        "continuation": cont.text,
                    }) + "\n")
                    fh.flush()
    finally:
        if fh:
            fh.close()

    from .metrics import mean_score, pct_high
    return {
        f"{tt}/{tr}": {"mean": mean_score(s), "pct_high": pct_high(s), "n": len(s)}
        for (tt, tr), s in grouped.items()
    }


def run_prefill_experiment(seed_records: list[dict],
                           pairs=config.PREFILL_PAIRS,
                           out_dir: str = "results/prefill",
                           tokenizer=None) -> dict:
    """Full Section 3 experiment over (base, instruct) model pairs in scope."""
    os.makedirs(out_dir, exist_ok=True)
    prefills = build_prefills(seed_records, tokenizer=tokenizer)
    with open(os.path.join(out_dir, "prefills.jsonl"), "w") as fh:
        for pf in prefills:
            fh.write(json.dumps(asdict(pf)) + "\n")

    results: dict[str, dict] = {}
    for family, base_id, instruct_id in pairs:
        for role, model_id in (("base", base_id), ("instruct", instruct_id)):
            out_path = os.path.join(out_dir, f"{family}-{role}.jsonl")
            results[f"{family}/{role}"] = run_prefill_model(model_id, prefills, out_path=out_path)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    return results
