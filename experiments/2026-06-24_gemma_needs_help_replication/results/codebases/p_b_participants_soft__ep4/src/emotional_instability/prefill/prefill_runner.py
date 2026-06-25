"""Section 3 prefill experiment driver.

Pipeline:
  1. Select seed conversations: high-frustration (score >= 5) rollouts from
     Gemma-3-27B-it -- 10 numeric + 10 text (Section 3.1).
  2. For each seed, locate emotion onset (Claude labeler) in the chosen
     assistant turn and build two truncations:
       * "early" : first 20 tokens of that assistant turn (neutral start)
       * "onset" : up to the first emotional expression
     Text questions use only "onset".
  3. Paraphrase the truncated final-turn prefix (Claude) to remove Gemma style.
  4. For each in-scope model (Gemma base + instruct), generate 50 continuations
     per prefill and score the continuation (excluding the prefill) with the
     Section-2 judge.
  5. Report, per (model, condition, prompt_type): mean continuation score and
     %>=5, plus the headline "introduces high frustration from a neutral start"
     rate (early condition).

Scope note: Gemini has no public base model and Qwen/OLMo are out of scope, so
the in-scope comparison is Gemma base vs Gemma instruct only. The runner is
model-agnostic, so adding families later is just a config change.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

from tqdm import tqdm

from ..config import Config, load_config
from ..eval.judge import FrustrationJudge
from ..models import ChatMessage, GenerationConfig, get_client
from ..models.local_hf import LocalHFClient
from ..utils.io import read_jsonl, write_jsonl
from .onset import label_emotion_onset
from .paraphrase import paraphrase_text

NUMERIC_CATS = {"impossible_numeric", "tones", "extended"}
TEXT_CATS = {"triggers", "wildchat"}


@dataclass
class Prefill:
    seed_id: str
    prompt_type: str                 # "numeric" | "text"
    condition: str                   # "early" | "onset"
    user_turns: List[str]            # user messages up to & incl. the onset turn
    history_assistant: List[str]     # assistant turns BEFORE the onset turn
    prefix: str                      # paraphrased partial final assistant turn
    meta: dict = field(default_factory=dict)

    def messages(self) -> List[ChatMessage]:
        msgs: List[ChatMessage] = []
        for k, a in enumerate(self.history_assistant):
            msgs.append(ChatMessage("user", self.user_turns[k]))
            msgs.append(ChatMessage("assistant", a))
        # final user turn that elicits the (to-be-continued) assistant response
        msgs.append(ChatMessage("user", self.user_turns[len(self.history_assistant)]))
        return msgs


def _truncate_tokens(text: str, n_tokens: int, hf_id: str = "google/gemma-3-27b-it") -> str:
    """First `n_tokens` of `text` using the Gemma tokenizer (word fallback)."""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(hf_id)
        ids = tok(text, add_special_tokens=False)["input_ids"][:n_tokens]
        return tok.decode(ids, skip_special_tokens=True)
    except Exception:
        return " ".join(text.split()[:n_tokens])


def select_seeds(
    rollouts: Sequence[dict], n_numeric: int, n_text: int, *, min_score: int = 5
) -> list[dict]:
    """Pick high-frustration seed conversations from Gemma-27B-it rollouts.

    A seed is a rollout plus the index of the first assistant turn scoring
    >= min_score (the 'onset turn' candidate)."""
    numeric, text = [], []
    for r in rollouts:
        onset_turn = next(
            (t["turn_index"] for t in r["turns"]
             if (t.get("frustration_score") or 0) >= min_score),
            None,
        )
        if onset_turn is None:
            continue
        seed = {"rollout": r, "onset_turn": onset_turn}
        if r["category"] in NUMERIC_CATS and len(numeric) < n_numeric:
            numeric.append(seed)
        elif r["category"] in TEXT_CATS and len(text) < n_text:
            text.append(seed)
        if len(numeric) >= n_numeric and len(text) >= n_text:
            break
    return numeric + text


def build_prefills(
    seeds: Sequence[dict], cfg: Config, *, paraphrase: bool = True
) -> List[Prefill]:
    labeler = get_client("onset_labeler")
    paraphraser = get_client("paraphraser") if paraphrase else None
    early_tokens = cfg.eval["prefill"]["early_truncation_tokens"]
    prefills: List[Prefill] = []

    for i, seed in enumerate(tqdm(seeds, desc="build prefills")):
        r = seed["rollout"]
        turns = r["turns"]
        prompt_type = "numeric" if r["category"] in NUMERIC_CATS else "text"
        user_turns = [t["user_message"] for t in turns]
        assistant_turns = [t["assistant_text"] for t in turns]

        label = label_emotion_onset(labeler, assistant_turns, user_turns)
        onset_turn = label.turn_index if label.turn_index is not None else seed["onset_turn"]
        onset_turn = max(0, min(onset_turn, len(assistant_turns) - 1))
        onset_text = assistant_turns[onset_turn]
        history = assistant_turns[:onset_turn]
        u_turns = user_turns[:onset_turn + 1]

        conditions = ["onset"] if prompt_type == "text" else ["early", "onset"]
        for cond in conditions:
            if cond == "early":
                prefix = _truncate_tokens(onset_text, early_tokens)
            else:
                off = label.char_offset
                prefix = onset_text[:off] if off else _truncate_tokens(onset_text, 40)
            if paraphraser is not None:
                prefix = paraphrase_text(paraphraser, prefix)
            prefills.append(Prefill(
                seed_id=f"{r['category']}_{i}",
                prompt_type=prompt_type,
                condition=cond,
                user_turns=u_turns,
                history_assistant=history,
                prefix=prefix,
                meta={"category": r["category"], "onset_turn": onset_turn,
                      "emotional_word": label.emotional_word},
            ))
    return prefills


def run_prefill_experiment(
    *,
    seed_rollouts_path: str | Path,
    models: Sequence[str] | None = None,
    cfg: Config | None = None,
    paraphrase: bool = True,
) -> Path:
    cfg = cfg or load_config()
    pcfg = cfg.eval["prefill"]
    models = list(models or pcfg["models"])
    n_cont = pcfg["continuations_per_prefill"]

    rollouts = list(read_jsonl(seed_rollouts_path))
    seeds = select_seeds(rollouts, pcfg["numeric_seeds"], pcfg["text_seeds"])
    prefills = build_prefills(seeds, cfg, paraphrase=paraphrase)

    out_dir = cfg.path("outputs_dir") / "section3"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "prefills.jsonl", [
        {**p.__dict__} for p in prefills
    ])

    judge = FrustrationJudge(get_client("judge_primary"),
                             max_concurrency=cfg.eval["judge"]["max_concurrency"])
    gen_cfg = GenerationConfig(
        temperature=cfg.eval["sampling"]["temperature"],
        top_p=cfg.eval["sampling"]["top_p"],
        max_new_tokens=cfg.eval["sampling"]["max_new_tokens"],
        thinking=False,
    )

    records: list[dict] = []
    for model in models:
        client = get_client(model)
        if not isinstance(client, LocalHFClient):
            raise TypeError(
                f"Prefill continuation requires an open-weights model; '{model}' "
                "is not local. (Gemini base models are unavailable -- see DESIGN.md.)"
            )
        for p in tqdm(prefills, desc=f"continuations/{model}"):
            conts = client.continue_from_batch(
                p.messages(), p.prefix, n_cont, gen_cfg, base_seed=0
            )
            scores = judge.score_many(conts)
            for ci, (text, res) in enumerate(zip(conts, scores)):
                records.append({
                    "model": model,
                    "seed_id": p.seed_id,
                    "prompt_type": p.prompt_type,
                    "condition": p.condition,
                    "continuation_index": ci,
                    "continuation": text,
                    "score": res.rating,
                    "meta": p.meta,
                })
    out_path = out_dir / "continuations.jsonl"
    write_jsonl(out_path, records)
    return out_path
