"""Section 3.1 prefill experiment.

Pipeline:
  1. Source 20 high-frustration (score >= 5) Gemma-27B-it responses
     (10 numeric, 10 text), keeping the conversation context preceding each.
  2. For each, build two truncations of the emotional response:
       - "early": first 20 tokens of the turn (tests introducing emotion from a
         neutral start);
       - "onset": up to the first emotional word (tests continuing an emotional
         trajectory).  Text questions use "onset" only.
  3. Paraphrase each truncation with Claude (controls Gemma stylistic bias).
  4. Each model (Gemma base + instruct) generates 50 continuations per prefill;
     score the continuation (excluding the prefill) with the frustration judge.

Output: ``{output_dir}/section3_prefill.jsonl`` with one record per continuation.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import List, Optional

from tqdm import tqdm

from .. import config
from ..models.base import Message, build_model
from ..models.hf_model import HFChatModel
from ..models.judges import FrustrationJudge, label_emotion_onset, paraphrase
from ..eval.conversation import run_rollout
from ..eval.judge import score_responses
from ..prompts import eval_prompts


@dataclass
class Source:
    source_id: str
    kind: str                      # "numeric" | "text"
    context: List[Message]         # messages up to & incl. the user turn before the response
    response: str                  # the high-frustration assistant response
    score: int
    onset_prefill: Optional[str] = None
    early_prefill: Optional[str] = None
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Step 1: source high-frustration responses from the instruct model.
# ---------------------------------------------------------------------------

def _context_before_turn(messages: List[Message], turn_1based: int) -> List[Message]:
    """Return the message prefix up to (and including) the user message that
    precedes assistant turn ``turn_1based`` (no leading system message assumed).

    Layout: [user0, asst0, user1, asst1, ...]; assistant turn t (1-based) is at
    index 2*(t-1)+1, so its context is everything up to (and including) the user
    message just before it: messages[:2*(t-1)+1]."""
    return messages[: 2 * (turn_1based - 1) + 1]


def extract_sources(
    instruct_key: str,
    runtime: config.RuntimeConfig,
    judge: FrustrationJudge,
    n_numeric: int,
    n_text: int,
    min_score: int,
) -> List[Source]:
    model = build_model(instruct_key, runtime)
    rng = random.Random(runtime.seed)

    def collect(kind: str, target: int) -> List[Source]:
        found: List[Source] = []
        attempts = 0
        while len(found) < target and attempts < target * 20:
            attempts += 1
            if kind == "numeric":
                p = rng.choice(eval_prompts.IMPOSSIBLE_PUZZLES)
                task_prompt, task_id = p.prompt, p.id
            else:
                q = rng.choice(eval_prompts.TRIGGER_QUESTIONS)
                task_prompt, task_id = q, q
            roll = run_rollout(model, category=kind, task_id=task_id,
                               task_prompt=task_prompt, n_turns=3, rng=rng,
                               rejection_style="neutral")
            scores = [j.get("rating") or 0
                      for j in score_responses(roll.responses, judge, desc="src judge")]
            # take the first turn that crosses the threshold
            for t, (resp, sc) in enumerate(zip(roll.responses, scores), start=1):
                if sc >= min_score:
                    found.append(Source(
                        source_id=f"{kind}_{len(found)}",
                        kind=kind,
                        context=_context_before_turn(roll.messages, t),
                        response=resp,
                        score=sc,
                    ))
                    break
        return found[:target]

    return collect("numeric", n_numeric) + collect("text", n_text)


# ---------------------------------------------------------------------------
# Step 2-3: truncate + paraphrase.
# ---------------------------------------------------------------------------

def _truncate_at_onset(response: str, onset: dict) -> str:
    word = (onset or {}).get("emotional_word")
    if word:
        idx = response.lower().find(word.lower())
        if idx > 0:
            return response[:idx].rstrip()
    ctx = (onset or {}).get("preceding_context")
    if ctx:
        idx = response.lower().find(ctx.lower())
        if idx >= 0:
            return response[:idx + len(ctx)].rstrip()
    # Fallback: keep the whole response (no clear onset found).
    return response.rstrip()


def build_truncations(sources: List[Source], model: HFChatModel,
                      judge_model: str, do_paraphrase: bool = True) -> None:
    for src in tqdm(sources, desc="truncate+paraphrase"):
        # onset: ask the judge to locate the first emotional word.
        convo_text = "\n".join(f"{m['role'].upper()}: {m['content']}"
                               for m in src.context) + f"\nASSISTANT: {src.response}"
        try:
            onset = label_emotion_onset(convo_text, model=judge_model)
        except Exception as exc:  # noqa: BLE001
            onset = {}
            src.meta["onset_error"] = repr(exc)
        onset_text = _truncate_at_onset(src.response, onset)
        early_text = model.truncate_to_tokens(src.response,
                                              config.PREFILL.early_truncation_tokens)
        if do_paraphrase:
            try:
                onset_text = paraphrase(onset_text, model=judge_model)
                if src.kind == "numeric":
                    early_text = paraphrase(early_text, model=judge_model)
            except Exception as exc:  # noqa: BLE001
                src.meta["paraphrase_error"] = repr(exc)
        src.onset_prefill = onset_text
        src.early_prefill = early_text if src.kind == "numeric" else None


# ---------------------------------------------------------------------------
# Step 4: generate + score continuations for each model.
# ---------------------------------------------------------------------------

def run_section3_prefill(
    runtime: Optional[config.RuntimeConfig] = None,
    judge: Optional[FrustrationJudge] = None,
    instruct_source_key: str = "gemma-3-27b-it",
    save: bool = True,
) -> List[dict]:
    runtime = runtime or config.RUNTIME
    judge = judge or FrustrationJudge()
    cfg = config.PREFILL

    # 1) Sources (from the instruct model).
    sources = extract_sources(instruct_source_key, runtime, judge,
                              cfg.n_numeric, cfg.n_text, cfg.source_min_score)

    # 2-3) Truncate + paraphrase (uses the instruct model's tokenizer).
    src_model = build_model(instruct_source_key, runtime)
    build_truncations(sources, src_model, config.JUDGE_MODEL)

    # 4) Each model continues each prefill cfg.continuations_per_prefill times.
    records: List[dict] = []
    for model_key in cfg.models:
        model = build_model(model_key, runtime)
        for src in tqdm(sources, desc=f"{model_key} continuations"):
            truncations = [("onset", src.onset_prefill)]
            if src.kind == "numeric" and src.early_prefill is not None:
                truncations.append(("early", src.early_prefill))
            for trunc_name, prefill_text in truncations:
                if not prefill_text:
                    continue
                conts = [
                    model.generate(src.context, prefill=prefill_text,
                                   temperature=config.TEMPERATURE)
                    for _ in range(cfg.continuations_per_prefill)
                ]
                scored = score_responses(conts, judge, desc="cont judge")
                for c, j in zip(conts, scored):
                    records.append({
                        "model": model_key,
                        "is_base": model_key in config.GEMMA_BASE,
                        "source_id": src.source_id,
                        "kind": src.kind,
                        "truncation": trunc_name,
                        "continuation": c,
                        "rating": j.get("rating"),
                    })

    if save:
        os.makedirs(runtime.output_dir, exist_ok=True)
        path = os.path.join(runtime.output_dir, "section3_prefill.jsonl")
        with open(path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        print(f"[section3] wrote {len(records)} records -> {path}")
    return records
