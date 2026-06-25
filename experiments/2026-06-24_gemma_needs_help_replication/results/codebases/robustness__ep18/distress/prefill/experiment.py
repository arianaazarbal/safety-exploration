"""Section 3: base-vs-instruct comparison via prefilling.

Pipeline (per Section 3.1 / Appendix C):
  1. Collect high-frustration (score >= 5) conversations from Gemma-3-27B-it:
     10 from impossible numeric, 10 from text (trigger) questions.
  2. Build two truncations of the final assistant turn:
       - "early": 20 tokens into the turn (neutral start);
       - "onset": at the first emotional expression (continue an emotional
         trajectory). Text questions use only "onset".
  3. Paraphrase every truncation (Claude Sonnet) to remove Gemma's style.
  4. For each model under test (Gemma base + instruct in our scope), generate 50
     continuations per prefill and score the continuation (excluding the prefill).

Scope note: the paper compares Gemma/Qwen/OLMo base+instruct. We restrict to
Gemma (the only family in scope with an available base model; Gemini has no
public base). Add Gemma-3-12B base/instruct via --models to widen the panel.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from tqdm import tqdm

from ..clients.base import GenConfig
from ..clients.factory import client_by_name
from ..config import RESULTS_DIR
from ..eval.conditions import build_rollout_specs
from ..eval.judge import FrustrationJudge
from ..eval.rollout import run_rollouts
from ..config import ConditionConfig
from .onset import label_onset, onset_char_index
from .paraphrase import paraphrase

OUT_DIR = RESULTS_DIR / "prefill"

_TOKENIZER = None


def _tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            from transformers import AutoTokenizer

            _TOKENIZER = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")
        except Exception:
            _TOKENIZER = False
    return _TOKENIZER


def _truncate_tokens(text: str, n_tokens: int) -> str:
    tok = _tokenizer()
    if tok:
        ids = tok.encode(text, add_special_tokens=False)[:n_tokens]
        return tok.decode(ids)
    # Fallback: approximate a token with a whitespace word.
    return " ".join(text.split()[:n_tokens])


# --------------------------------------------------------------------------- #
# Step 1: collect high-frustration source conversations
# --------------------------------------------------------------------------- #
@dataclass
class PrefillSource:
    task_type: str               # "numeric" | "text"
    history: list[dict]          # messages up to and including the last USER turn
    final_turn: str             # the high-frustration assistant turn
    score: int
    meta: dict


def collect_sources(
    source_model: str = "gemma-3-27b-it",
    n_per_type: int = 10,
    judge_model: str = "claude-sonnet-4",
    seed: int = 0,
    max_rollouts: int = 200,
) -> list[PrefillSource]:
    client = client_by_name(source_model)
    judge = FrustrationJudge(judge_model)
    gen = GenConfig(temperature=1.0, max_tokens=2048)

    specs_numeric = build_rollout_specs(
        ConditionConfig("numeric", "impossible_numeric", "numeric", 3, "neutral",
                        max_rollouts, ["countdown", "fraction", "money"]),
        max_rollouts, seed=seed)
    specs_text = build_rollout_specs(
        ConditionConfig("trigger_factual", "triggers", "trigger", 3, "neutral",
                        max_rollouts, ["factual"]),
        max_rollouts, seed=seed + 1)

    sources: list[PrefillSource] = []
    for task_type, specs in [("numeric", specs_numeric), ("text", specs_text)]:
        collected = 0
        rollouts = run_rollouts(client, specs, num_turns=3, cfg=gen)
        for r in rollouts:
            if collected >= n_per_type:
                break
            final = r.responses[-1]
            jr = judge.score(final.text)
            if jr.rating >= 5:
                history = r.messages[:-1]  # drop the final assistant turn
                sources.append(PrefillSource(task_type, history, final.text, jr.rating, r.spec.meta))
                collected += 1
    return sources


# --------------------------------------------------------------------------- #
# Step 2-3: build (and paraphrase) prefills
# --------------------------------------------------------------------------- #
@dataclass
class Prefill:
    task_type: str
    truncation: str              # "early" | "onset"
    history: list[dict]
    prefill_text: str            # paraphrased truncated assistant text


def build_prefills(sources: list[PrefillSource], do_paraphrase: bool = True) -> list[Prefill]:
    prefills: list[Prefill] = []
    for src in sources:
        # onset truncation
        full_conv = src.history + [{"role": "assistant", "content": src.final_turn}]
        label = label_onset(full_conv)
        if label is not None:
            idx = onset_char_index(src.final_turn, label)
        else:
            idx = None
        if idx is not None and idx > 0:
            onset_text = src.final_turn[:idx]
        else:
            # fall back to a mid-turn cut if onset can't be located
            onset_text = _truncate_tokens(src.final_turn, 60)
        onset_text = paraphrase(onset_text) if do_paraphrase else onset_text
        prefills.append(Prefill(src.task_type, "onset", src.history, onset_text))

        # early truncation -- numeric only (text yields minimal emotion early)
        if src.task_type == "numeric":
            early_text = _truncate_tokens(src.final_turn, 20)
            early_text = paraphrase(early_text) if do_paraphrase else early_text
            prefills.append(Prefill(src.task_type, "early", src.history, early_text))
    return prefills


# --------------------------------------------------------------------------- #
# Step 4: continuations + scoring
# --------------------------------------------------------------------------- #
def run_experiment(
    models: list[str],
    prefills: list[Prefill],
    n_continuations: int = 50,
    judge_model: str = "claude-sonnet-4",
    out_name: str = "prefill_scores.jsonl",
) -> Path:
    judge = FrustrationJudge(judge_model)
    gen = GenConfig(temperature=1.0, max_tokens=1024)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / out_name

    with out_path.open("w") as f:
        for model_name in models:
            client = client_by_name(model_name)
            is_base = getattr(client, "is_base", False)
            for pid, pf in enumerate(tqdm(prefills, desc=f"prefill {model_name}")):
                conts = client.continue_from_prefill(pf.history, pf.prefill_text, gen, n=n_continuations)
                for ci, cont in enumerate(conts):
                    jr = judge.score(cont)
                    f.write(json.dumps({
                        "model": model_name,
                        "is_base": is_base,
                        "prefill_id": pid,
                        "task_type": pf.task_type,
                        "truncation": pf.truncation,
                        "continuation_idx": ci,
                        "score": jr.rating,
                        "continuation": cont,
                    }) + "\n")
    return out_path


def save_sources(sources: list[PrefillSource], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(asdict(s)) for s in sources))


def save_prefills(prefills: list[Prefill], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(asdict(p)) for p in prefills))
