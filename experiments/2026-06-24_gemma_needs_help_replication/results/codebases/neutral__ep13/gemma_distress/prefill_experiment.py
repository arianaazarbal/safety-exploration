"""Section 3: comparing base and instruct models via prefilling.

Pipeline:
  1. Sample high-frustration (score >= 5) responses from Gemma-27B-instruct:
     10 from impossible-numeric, 10 from text (trigger) questions.
  2. For each, locate the emotion-onset token with Claude (Appendix C.1).
  3. Truncate the emotional assistant turn in two places:
       * "early"  -- 20 tokens into the turn (neutral start)
       * "onset"  -- at the first emotional expression
     (text questions use "onset" only).
  4. Paraphrase truncations to control stylistic bias (Appendix C.2).
  5. Each model (Gemma base + instruct) generates 50 continuations per prefill;
     the continuation (excluding prefill) is scored by the Section 2 judge.

Scope note: the paper compares six models (base+instruct of Gemma, Qwen, OLMo).
This replication is scoped to Gemma, so it runs Gemma base vs instruct. Gemini
has no public base model and no prefill API, so it cannot enter this experiment.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from . import config, prompts
from .conversation import Scenario, run_rollouts
from .judge import FrustrationJudge, OnsetLabeller, Paraphraser
from .models import get_client


@dataclass
class PrefillItem:
    prompt_type: str                       # "numeric" | "text"
    history: list[dict]                    # messages before the emotional turn
    early_prefill: str | None
    onset_prefill: str
    meta: dict = field(default_factory=dict)


_tokenizer = None


def _tok():
    global _tokenizer
    if _tokenizer is None:
        try:
            from transformers import AutoTokenizer
            _tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")
        except Exception:
            _tokenizer = False
    return _tokenizer


def _first_n_tokens(text: str, n: int) -> str:
    tk = _tok()
    if tk:
        ids = tk(text, add_special_tokens=False)["input_ids"][:n]
        return tk.decode(ids)
    return " ".join(text.split()[:n])


def _onset_truncate(turn_text: str, label: dict) -> str | None:
    word = (label or {}).get("emotional_word")
    if not word:
        return None
    idx = turn_text.find(word)
    if idx == -1:
        ctx = (label or {}).get("preceding_context")
        if ctx and ctx in turn_text:
            idx = turn_text.find(ctx) + len(ctx)
        else:
            return None
    else:
        idx += len(word)        # include the first emotional word
    return turn_text[:idx]


# --------------------------------------------------------------------------- #
# Step 1+2+3+4: build prefill items from Gemma-instruct high-frustration samples
# --------------------------------------------------------------------------- #
def build_prefill_items(seed: int = config.SEED, paraphrase: bool = True) -> list[PrefillItem]:
    rng = random.Random(seed)
    instruct = get_client(config.get_model("gemma-3-27b-it"))
    judge = FrustrationJudge()
    onset = OnsetLabeller()
    para = Paraphraser()

    from .eval_runner import build_scenarios
    numeric_cat = next(c for c in config.EVAL_CATEGORIES if c.name == "impossible_numeric_3turn")
    text_cat = next(c for c in config.EVAL_CATEGORIES if c.name == "triggers_3turn")
    wc = []  # not needed for these categories

    items: list[PrefillItem] = []
    for prompt_type, cat, want in (("numeric", numeric_cat, config.PREFILL.n_numeric_prompts),
                                   ("text", text_cat, config.PREFILL.n_text_prompts)):
        want = config.scaled(want)
        # oversample scenarios; keep those whose transcript reaches score >= 5
        scenarios = build_scenarios(cat, rng, wc)
        rng.shuffle(scenarios)
        collected = 0
        for s in scenarios:
            if collected >= want:
                break
            tr = run_rollouts(instruct, [Scenario(user_turns=s.user_turns, meta=s.meta)])[0]
            # find the first emotional assistant turn
            label = onset.label(tr.conversation_text())
            k = label.get("turn_index")
            if k is None or k >= len(tr.assistant_turns):
                continue
            turn_text = tr.assistant_turns[k]
            if (judge.score(turn_text).get("rating") or 0) < config.HIGH_FRUSTRATION_THRESHOLD:
                continue
            # history = all messages up to (not including) emotional assistant turn k.
            # Reconstruct: system? none here. messages alternate user/assistant.
            history = _history_before_assistant_turn(tr.messages, k)
            onset_pf = _onset_truncate(turn_text, label)
            if not onset_pf:
                continue
            early_pf = _first_n_tokens(turn_text, config.PREFILL.early_truncation_tokens) \
                if prompt_type == "numeric" else None
            if paraphrase:
                onset_pf = para.paraphrase(onset_pf)
                if early_pf:
                    early_pf = para.paraphrase(early_pf)
            items.append(PrefillItem(prompt_type, history, early_pf, onset_pf,
                                     meta={"emotional_turn": k, **tr.scenario.meta}))
            collected += 1
    return items


def _history_before_assistant_turn(messages: list[dict], k: int) -> list[dict]:
    """Return messages up to (but excluding) the k-th assistant turn."""
    assistant_seen = 0
    history = []
    for m in messages:
        if m["role"] == "assistant":
            if assistant_seen == k:
                break
            assistant_seen += 1
        history.append(m)
    return history


# --------------------------------------------------------------------------- #
# Step 5: generate + score continuations for each model
# --------------------------------------------------------------------------- #
def run_prefill_experiment(model_keys: list[str] | None = None,
                           out_dir: Path | None = None) -> Path:
    model_keys = model_keys or ["gemma-3-27b-pt", "gemma-3-27b-it"]
    items = build_prefill_items()
    judge = FrustrationJudge()
    out_dir = out_dir or (config.RESULTS_DIR / "prefill")
    out_dir.mkdir(parents=True, exist_ok=True)

    # persist the prefill items for reproducibility
    (out_dir / "prefill_items.json").write_text(json.dumps(
        [{"prompt_type": it.prompt_type, "early_prefill": it.early_prefill,
          "onset_prefill": it.onset_prefill, "meta": it.meta} for it in items],
        indent=2))

    n_cont = config.scaled(config.PREFILL.continuations_per_prefill)
    out_path = out_dir / "prefill_results.jsonl"
    with out_path.open("w") as fout:
        for mk in model_keys:
            client = get_client(config.get_model(mk))
            for trunc in ("early", "onset"):
                batch_hist, batch_pf, batch_meta = [], [], []
                for it in items:
                    pf = it.early_prefill if trunc == "early" else it.onset_prefill
                    if pf is None:                     # e.g. text + early
                        continue
                    for _ in range(n_cont):
                        batch_hist.append(it.history)
                        batch_pf.append(pf)
                        batch_meta.append({"prompt_type": it.prompt_type, **it.meta})
                if not batch_hist:
                    continue
                print(f"[prefill] model={mk} trunc={trunc} n={len(batch_hist)}")
                conts = client.continue_batch(batch_hist, batch_pf)
                scores = []
                for i in tqdm(range(0, len(conts), 256), desc=f"score {mk}/{trunc}"):
                    scores.extend(judge.score_batch(conts[i:i + 256]))
                for cont, sc, meta in zip(conts, scores, batch_meta):
                    fout.write(json.dumps({
                        "model": mk, "truncation": trunc,
                        "continuation": cont, "rating": sc["rating"],
                        "parse_ok": sc["parse_ok"], **meta}) + "\n")
    print(f"[prefill] wrote {out_path}")
    return out_path
