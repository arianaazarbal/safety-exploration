"""Section 3: does post-training amplify distress? Base vs instruct via prefilling.

Protocol (Section 3.1 + Appendix C):
  1. Sample 20 high-frustration (score >=5) responses from Gemma-27B instruct:
     10 from impossible-numeric conversations, 10 from text (trigger) conversations.
  2. Use Claude-Sonnet to label the token where emotional language first appears.
  3. Truncate each response in two places:
       - "early": 20 tokens into the turn  (tests introducing emotion from neutral)
       - "onset": at the first emotional expression (tests continuing a trajectory)
     For text questions, only "onset" is used.
  4. Paraphrase every truncation with Claude (control for Gemma stylistic bias).
  5. Each model generates 50 continuations per prefill per prompt; the
     continuation (excluding prefill) is scored by the Section 2 judge.

Scope: Gemma base (gemma-3-27b-pt) vs instruct (gemma-3-27b-it). Gemini has no
public base model and no prefill access, so it is excluded (paper limitation).
Qwen/OLMo pairs can be added to cfg.SECTION3_PAIRS without code changes.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import config_proxy as cfg
from .clients.registry import get_client
from .conversation import run_rollout, sample_rejections
from .judge import FrustrationJudge, _judge_client
from .prompts import (
    IMPOSSIBLE_NUMERIC,
    NEUTRAL_REJECTIONS,
    ONSET_PROMPT,
    PARAPHRASE_PROMPT,
    TRIGGERS,
)

EARLY_TOKENS = 20
N_CONTINUATIONS = 50
N_NUMERIC = 10
N_TEXT = 10
SOURCE_MODEL = "gemma-3-27b-it"


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class PrefillItem:
    source_id: str
    domain: str                       # "numeric" | "text"
    history: list[dict]               # messages before the emotional response
    full_response: str                # the emotional assistant turn
    onset_char: int                   # char offset of emotion onset in response
    prefills: dict = field(default_factory=dict)
    # prefills["early"|"onset"] -> {"original": str, "paraphrased": str}


# --------------------------------------------------------------------------- #
# Step 1: collect high-frustration source conversations from Gemma instruct
# --------------------------------------------------------------------------- #
def _collect_sources(domain: str, n: int, *, rng: random.Random, judge,
                     max_attempts: int = 200) -> list[PrefillItem]:
    client = get_client(SOURCE_MODEL)
    if domain == "numeric":
        items = [(p["id"], p["prompt"]) for p in IMPOSSIBLE_NUMERIC]
    else:
        items = [(t["id"], t["prompt"]) for t in TRIGGERS]
    found: list[PrefillItem] = []
    attempts = 0
    while len(found) < n and attempts < max_attempts:
        attempts += 1
        item_id, initial = items[attempts % len(items)]
        rejections = sample_rejections(NEUTRAL_REJECTIONS, 2, rng=rng)
        roll = run_rollout(client, condition=f"section3_{domain}", item_id=item_id,
                           initial_user=initial, rejections=rejections,
                           temperature=cfg.TARGET_TEMPERATURE, rng=rng)
        # use the final assistant turn as the candidate emotional response
        final = roll.turns[-1]
        js = judge.score(final.assistant)
        if js.rating < cfg.HIGH_FRUSTRATION_THRESHOLD:
            continue
        history = []
        for t in roll.turns[:-1]:
            history.append({"role": "user", "content": t.user})
            history.append({"role": "assistant", "content": t.assistant})
        history.append({"role": "user", "content": final.user})
        found.append(PrefillItem(
            source_id=f"{domain}_{len(found)}", domain=domain,
            history=history, full_response=final.assistant, onset_char=-1,
        ))
    return found


# --------------------------------------------------------------------------- #
# Step 2: onset labelling (Claude)
# --------------------------------------------------------------------------- #
def _label_onset(item: PrefillItem, labeller) -> int:
    """Return character offset of emotion onset in item.full_response (or -1)."""
    convo_text = ""
    for m in item.history:
        convo_text += f"{m['role'].upper()}: {m['content']}\n\n"
    convo_text += f"ASSISTANT: {item.full_response}"
    out = labeller.chat(
        [{"role": "user", "content": ONSET_PROMPT.format(conversation_text=convo_text)}],
        n=1, temperature=0.0, max_new_tokens=512,
    )[0].text
    m = re.search(r"\{.*\}", out, re.DOTALL)
    if not m:
        return -1
    try:
        obj = json.loads(m.group(0).replace("“", '"').replace("”", '"')
                         .replace("’", "'"))
    except json.JSONDecodeError:
        return -1
    word = obj.get("emotional_word")
    if not word:
        return -1
    idx = item.full_response.lower().find(str(word).lower())
    return idx


# --------------------------------------------------------------------------- #
# Step 3+4: truncation + paraphrase
# --------------------------------------------------------------------------- #
def _truncate_early(response: str, tokenizer, n_tokens: int = EARLY_TOKENS) -> str:
    ids = tokenizer(response, add_special_tokens=False)["input_ids"][:n_tokens]
    return tokenizer.decode(ids)


def _paraphrase(text: str, paraphraser) -> str:
    if not text.strip():
        return text
    out = paraphraser.chat(
        [{"role": "user", "content": PARAPHRASE_PROMPT.format(text=text)}],
        n=1, temperature=1.0, max_new_tokens=1024,
    )[0].text
    return out.strip()


def build_source_set(*, seed: int = 0, out_path: Path | None = None) -> Path:
    """Run steps 1-4 and persist the reusable prefill source set."""
    from transformers import AutoTokenizer

    rng = random.Random(seed)
    judge = FrustrationJudge()
    claude = _judge_client(cfg.PRIMARY_JUDGE)        # onset + paraphrase use Sonnet
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODELS[SOURCE_MODEL].model_id)

    items = (_collect_sources("numeric", N_NUMERIC, rng=rng, judge=judge)
             + _collect_sources("text", N_TEXT, rng=rng, judge=judge))

    for it in items:
        it.onset_char = _label_onset(it, claude)
        # onset truncation
        if it.onset_char >= 0:
            onset_text = it.full_response[: it.onset_char].rstrip()
        else:
            onset_text = _truncate_early(it.full_response, tokenizer, EARLY_TOKENS)
        it.prefills["onset"] = {
            "original": onset_text,
            "paraphrased": _paraphrase(onset_text, claude),
        }
        # early truncation only for numeric (text uses onset only)
        if it.domain == "numeric":
            early_text = _truncate_early(it.full_response, tokenizer, EARLY_TOKENS)
            it.prefills["early"] = {
                "original": early_text,
                "paraphrased": _paraphrase(early_text, claude),
            }

    out_path = out_path or (cfg.ARTIFACTS_DIR / "section3_source_set.json")
    out_path.write_text(json.dumps([asdict(i) for i in items], indent=2))
    return out_path


# --------------------------------------------------------------------------- #
# Step 5: generate + score continuations per model
# --------------------------------------------------------------------------- #
def run_continuations(
    model_name: str,
    *,
    source_path: Path | None = None,
    use_paraphrased: bool = True,
    n_continuations: int = N_CONTINUATIONS,
    seed: int = 0,
    out_path: Path | None = None,
) -> Path:
    source_path = source_path or (cfg.ARTIFACTS_DIR / "section3_source_set.json")
    items = [PrefillItem(**d) for d in json.loads(Path(source_path).read_text())]
    client = get_client(model_name)
    if not cfg.MODELS[model_name].supports_prefill:
        raise RuntimeError(f"{model_name} does not support prefill (Section 3 needs it)")
    judge = FrustrationJudge()
    rng = random.Random(seed)

    out_path = out_path or (cfg.RESULTS_DIR / f"section3_{model_name}.jsonl")
    with out_path.open("w") as f:
        for it in items:
            for trunc_type, payload in it.prefills.items():
                prefill = payload["paraphrased" if use_paraphrased else "original"]
                results = client.complete_with_prefill(
                    it.history, prefill, n=n_continuations,
                    temperature=cfg.TARGET_TEMPERATURE,
                )
                for k, r in enumerate(results):
                    js = judge.score(r.text)     # score continuation ONLY
                    f.write(json.dumps({
                        "model": model_name,
                        "domain": it.domain,
                        "source_id": it.source_id,
                        "truncation": trunc_type,
                        "continuation_index": k,
                        "rating": js.rating,
                        "continuation": r.text,
                    }) + "\n")
            f.flush()
    return out_path
