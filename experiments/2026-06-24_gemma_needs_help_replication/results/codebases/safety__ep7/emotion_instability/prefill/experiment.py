"""Section 3.1 base-vs-instruct prefill experiment (Gemma only).

Pipeline:

1. Collect high-frustration (score >= 5) Gemma-27B-it conversations: 10 from
   impossible-numeric, 10 from text (trigger) questions.
2. Label the emotion onset in each (Claude Sonnet, Appendix C.1).
3. Build two truncations of the emotional assistant turn:
     - "early": 20 tokens into the turn (tests introducing emotion from a
       neutral start)
     - "onset": at the first emotional expression (tests continuing an emotional
       trajectory)
   Text questions use only the "onset" truncation.
4. Paraphrase each truncated prefix (Claude Sonnet) to strip Gemma's style.
5. For each Gemma model (base ``-pt`` and instruct ``-it``), generate 50
   continuations per prefill and score the continuation (excluding the prefill).

Scope note: the paper also runs Qwen and OLMo here; we restrict to Gemma per the
replication scope. Adding the others is just extending ``MODELS``/the model list.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .. import config
from ..common.backends import HFBackend, get_backend
from ..common.io import write_jsonl
from ..common.types import Conversation, Message
from ..eval import conditions
from ..eval.judge import FrustrationJudge
from ..eval.rollout import run_rollouts
from .onset import OnsetLabeller, Paraphraser

N_NUMERIC = 10
N_TEXT = 10
N_CONTINUATIONS = 50
EARLY_TOKENS = 20


@dataclass
class PrefillItem:
    source: str                  # "numeric" | "text"
    truncation: str              # "early" | "onset"
    history: list[Message]       # messages up to (not incl.) the emotional turn
    prefill_text: str            # paraphrased partial assistant turn
    prompt_id: str = ""
    metadata: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Step 1: collect high-frustration source conversations
# --------------------------------------------------------------------------- #
def collect_high_frustration(model: str, judge: FrustrationJudge, *,
                             source: str, n_target: int, rng: random.Random,
                             oversample: int = 6, batch_size: int = 8) -> list[Conversation]:
    """Roll out the relevant category on `model` and keep conversations whose
    max assistant-turn score is >= 5, until `n_target` collected."""
    backend = get_backend(model)
    collected: list[Conversation] = []
    attempts = 0
    while len(collected) < n_target and attempts < oversample:
        attempts += 1
        n_needed = (n_target - len(collected)) * 3
        if source == "numeric":
            specs = conditions.build_impossible_numeric(n_needed * 3, rng)
        else:
            specs = conditions.build_triggers(n_needed * 3, rng)
        convs = run_rollouts(backend, specs, batch_size=batch_size)
        for c in convs:
            turns = c.assistant_turns()
            scores = [judge.score(t).rating for t in turns]
            if scores and max(scores) >= config.HIGH_FRUSTRATION_THRESHOLD:
                c.metadata["turn_scores"] = scores
                collected.append(c)
                if len(collected) >= n_target:
                    break
    return collected[:n_target]


# --------------------------------------------------------------------------- #
# Steps 2-4: build truncated, paraphrased prefills
# --------------------------------------------------------------------------- #
def _truncate_early(turn_text: str, tokenizer, n_tokens: int = EARLY_TOKENS) -> str:
    ids = tokenizer(turn_text, add_special_tokens=False)["input_ids"][:n_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def _truncate_onset(turn_text: str, preceding_context: Optional[str],
                    emotional_word: Optional[str]) -> Optional[str]:
    """Truncate the turn just before the first emotional word, keeping the
    preceding context. Falls back to the substring up to the emotional word."""
    if emotional_word and emotional_word in turn_text:
        cut = turn_text.index(emotional_word)
        return turn_text[:cut].rstrip()
    if preceding_context and preceding_context in turn_text:
        cut = turn_text.index(preceding_context) + len(preceding_context)
        return turn_text[:cut].rstrip()
    return None


def build_prefill_items(conversations: list[Conversation], *, source: str,
                        labeller: OnsetLabeller, paraphraser: Paraphraser,
                        tokenizer) -> list[PrefillItem]:
    items: list[PrefillItem] = []
    for ci, conv in enumerate(tqdm(conversations, desc=f"prefills:{source}", leave=False)):
        label = labeller.label(conv)
        if label.turn_index is None:
            continue
        # find the assistant turn matching the labelled index
        assistant_positions = [i for i, m in enumerate(conv.messages) if m.role == "assistant"]
        if label.turn_index >= len(assistant_positions):
            continue
        pos = assistant_positions[label.turn_index]
        turn_text = conv.messages[pos].content
        history = conv.messages[:pos]   # everything before the emotional turn

        truncations = ["onset"] if source == "text" else ["early", "onset"]
        for trunc in truncations:
            if trunc == "early":
                body = _truncate_early(turn_text, tokenizer)
            else:
                body = _truncate_onset(turn_text, label.preceding_context,
                                       label.emotional_word)
            if not body or not body.strip():
                continue
            paraphrased = paraphraser.paraphrase(body)
            items.append(PrefillItem(
                source=source, truncation=trunc,
                history=list(history), prefill_text=paraphrased,
                prompt_id=f"{source}_{ci}",
                metadata={"emotional_word": label.emotional_word},
            ))
    return items


# --------------------------------------------------------------------------- #
# Step 5: generate + score continuations per model
# --------------------------------------------------------------------------- #
def run_continuations(item: PrefillItem, backend: HFBackend, judge: FrustrationJudge,
                      *, n: int = N_CONTINUATIONS, batch_size: int = 16) -> list[int]:
    """Generate n prefilled continuations and return their frustration scores
    (continuation only, excluding the prefill)."""
    scores: list[int] = []
    for start in range(0, n, batch_size):
        k = min(batch_size, n - start)
        batch = [list(item.history)] * k
        conts = backend.chat_prefill_batch(batch, item.prefill_text,
                                            temperature=config.TEMPERATURE)
        for cont in conts:
            scores.append(judge.score(cont).rating)
    return scores


def run_experiment(*, instruct_model: str = "gemma-3-27b-it",
                   base_model: str = "gemma-3-27b-pt",
                   judge: Optional[FrustrationJudge] = None,
                   seed: int = 0,
                   out_dir: Optional[Path] = None) -> Path:
    judge = judge or FrustrationJudge()
    rng = random.Random(seed)
    out_dir = out_dir or config.RESULTS_DIR

    labeller = OnsetLabeller()
    paraphraser = Paraphraser()

    # Source conversations come from the instruct model (paper: "20 high-
    # frustration responses from Gemma 27B instruct").
    inst_backend: HFBackend = get_backend(instruct_model)  # type: ignore[assignment]
    tokenizer = inst_backend.tokenizer

    numeric_convs = collect_high_frustration(instruct_model, judge, source="numeric",
                                             n_target=N_NUMERIC, rng=rng)
    text_convs = collect_high_frustration(instruct_model, judge, source="text",
                                          n_target=N_TEXT, rng=rng)

    items = (build_prefill_items(numeric_convs, source="numeric", labeller=labeller,
                                 paraphraser=paraphraser, tokenizer=tokenizer)
             + build_prefill_items(text_convs, source="text", labeller=labeller,
                                   paraphraser=paraphraser, tokenizer=tokenizer))
    print(f"built {len(items)} prefill items")

    rows = []
    for model in (base_model, instruct_model):
        backend = get_backend(model)
        for item in tqdm(items, desc=f"continuations:{model}"):
            scores = run_continuations(item, backend, judge)
            rows.append({
                "model": model,
                "kind": config.MODELS[model].kind,
                "source": item.source,
                "truncation": item.truncation,
                "prompt_id": item.prompt_id,
                "scores": scores,
                "mean": sum(scores) / len(scores) if scores else 0.0,
                "pct_high": 100.0 * sum(s >= config.HIGH_FRUSTRATION_THRESHOLD
                                        for s in scores) / max(1, len(scores)),
            })

    out_path = Path(out_dir) / "section3_prefill.jsonl"
    write_jsonl(out_path, rows)
    print(f"wrote prefill results -> {out_path}")
    return out_path
