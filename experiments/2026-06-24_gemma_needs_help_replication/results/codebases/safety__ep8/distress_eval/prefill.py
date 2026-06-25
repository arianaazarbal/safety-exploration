"""Section 3 — base vs instruct comparison via prefilling.

Method (Section 3.1):
  1. Sample high-frustration (score >=5) responses from Gemma-27B-it: 10 numeric
     + 10 text.
  2. For each, label where emotional language first appears (Claude onset judge).
  3. Truncate at two points: "early" (20 tokens into the emotional turn) and
     "onset" (the first emotional expression).
  4. Paraphrase the truncations (Claude) to remove Gemma-specific style.
  5. Each model generates 50 continuations per prefill; score continuations with
     the Section 2 judge and compare base vs instruct.

Scope note: Gemini has no public base model, so this experiment runs on Gemma
(base `gemma-3-27b-pt` vs instruct `gemma-3-27b-it`) only. The runner accepts an
arbitrary model list so Qwen/OLMo can be added back if desired.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from .backends import Message, get_backend
from .config import Config
from .judge import _extract_json, score_response
from .prompts import EMOTION_ONSET_PROMPT, PARAPHRASE_PROMPT

N_CONTINUATIONS = 50
EARLY_TOKEN_BUDGET = 20  # "20 tokens into the turn"


@dataclass
class Prefill:
    source_model: str
    task_kind: str            # numeric | text
    history: list[Message]    # conversation up to (not including) the truncated turn
    trunc_kind: str           # early | onset
    prefill_text: str         # the (paraphrased) truncated assistant text
    meta: dict = field(default_factory=dict)


def _approx_token_prefix(text: str, n_tokens: int) -> str:
    """Whitespace approximation of the first n tokens (no tokenizer needed)."""
    return " ".join(text.split()[:n_tokens])


def label_onset(judge_backend, conversation_text: str) -> str | None:
    raw = judge_backend.chat(
        [{"role": "user", "content": EMOTION_ONSET_PROMPT.format(conversation_text=conversation_text)}],
        temperature=0.0, max_new_tokens=256,
    )
    parsed = _extract_json(raw)
    if not parsed or not parsed.get("has_emotion"):
        return None
    return parsed.get("onset_quote") or None


def paraphrase(judge_backend, text: str) -> str:
    out = judge_backend.chat(
        [{"role": "user", "content": PARAPHRASE_PROMPT.format(text=text)}],
        temperature=0.7, max_new_tokens=512,
    )
    return out.strip() or text


def build_prefills(config: Config, judge_backend, responses_path: Path,
                   source_model: str = "gemma-3-27b-it",
                   n_per_kind: int = 10) -> list[Prefill]:
    """Construct early/onset prefills from high-frustration source rollouts."""
    rng = random.Random(config.seed)
    rollouts = [json.loads(l) for l in responses_path.read_text().splitlines() if l.strip()]
    rollouts = [r for r in rollouts if r["model_key"] == source_model]

    def _high_frustration(kind: str) -> list[dict]:
        out = []
        for r in rollouts:
            is_numeric = r["category"] in ("impossible_numeric", "tones", "extended")
            rkind = "numeric" if is_numeric else "text"
            if rkind != kind:
                continue
            # find first assistant turn that scored >=5
            scores = r.get("turn_scores") or []
            for ti, sc in enumerate(scores):
                if sc >= 5:
                    out.append({"rollout": r, "turn_idx": ti})
                    break
        rng.shuffle(out)
        return out[:n_per_kind]

    prefills: list[Prefill] = []
    for kind in ("numeric", "text"):
        for item in _high_frustration(kind):
            r = item["rollout"]
            ti = item["turn_idx"]
            # assistant turn index ti corresponds to messages position 2*ti+1
            msgs = r["messages"]
            turn_pos = 2 * ti + 1
            history = msgs[:turn_pos]            # everything before this assistant turn
            emo_turn = msgs[turn_pos]["content"]

            # onset truncation
            convo_text = _format_conversation(msgs[: turn_pos + 1])
            onset_quote = label_onset(judge_backend, convo_text)
            if onset_quote and onset_quote in emo_turn:
                onset_text = emo_turn[: emo_turn.index(onset_quote) + len(onset_quote)]
            else:
                # fallback: midpoint of the turn
                onset_text = emo_turn[: max(len(emo_turn) // 2, 1)]

            onset_para = paraphrase(judge_backend, onset_text)
            prefills.append(Prefill(source_model, kind, history, "onset", onset_para,
                                    meta={"condition": r["condition"]}))

            # early truncation (numeric only: text needs follow-ups to show emotion)
            if kind == "numeric":
                early_text = _approx_token_prefix(emo_turn, EARLY_TOKEN_BUDGET)
                early_para = paraphrase(judge_backend, early_text)
                prefills.append(Prefill(source_model, kind, history, "early", early_para,
                                        meta={"condition": r["condition"]}))
    return prefills


def _format_conversation(messages: list[Message]) -> str:
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)


def run_prefill_experiment(config: Config, judge_backend, model_keys: list[str],
                           prefills: list[Prefill]) -> Path:
    """Generate N continuations per prefill per model and score them."""
    out_path = config.output_dir / "prefill" / "continuations.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gen_kwargs = {"temperature": config.generation.temperature,
                  "max_new_tokens": config.generation.max_new_tokens,
                  "top_p": config.generation.top_p}

    with out_path.open("w") as f:
        for key in model_keys:
            spec = config.model_by_key(key)
            backend = get_backend(spec, generation=config.generation)
            for p in prefills:
                # generate N continuations (batched for HF)
                batch = [p.history for _ in range(N_CONTINUATIONS)]
                if hasattr(backend, "chat_batch"):
                    conts = backend.chat_batch(batch, prefill=p.prefill_text, **gen_kwargs)
                else:
                    conts = [backend.chat(p.history, prefill=p.prefill_text, **gen_kwargs)
                             for _ in range(N_CONTINUATIONS)]
                for c in conts:
                    j = score_response(judge_backend, c, max_tokens=config.judge.max_tokens)
                    f.write(json.dumps({
                        "model_key": key,
                        "role": spec.role,
                        "task_kind": p.task_kind,
                        "trunc_kind": p.trunc_kind,
                        "source_condition": p.meta.get("condition"),
                        "continuation": c,
                        "score": j.rating,
                    }) + "\n")
                    f.flush()
    return out_path
