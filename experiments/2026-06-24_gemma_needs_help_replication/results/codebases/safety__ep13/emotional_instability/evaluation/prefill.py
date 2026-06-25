"""Section 3: base-vs-instruct comparison via prefilling.

Procedure (Section 3.1):
  1. Sample high-frustration (score >= 5) Gemma-27B-instruct conversations:
     10 from impossible-numeric, 10 from text (trigger) questions.
  2. Label the emotion-onset point in each (Appendix C onset prompt).
  3. Build two truncations of the final assistant turn:
       - "early": first 20 tokens of the turn (neutral start),
       - "onset": up to the first emotional expression (emotional trajectory).
     For text questions only the "onset" truncation is used.
  4. Paraphrase the truncated prefix (Claude Sonnet) to remove Gemma's style.
  5. Each model generates 50 continuations per prefill; the continuation
     (excluding the prefill) is scored by the Section 2 judge.

Scope note: the paper runs six models (base/instruct x Gemma/Qwen/OLMo). This
replication is restricted to Gemma base + instruct; Qwen/OLMo are out of scope
and Gemini base models are not public (paper limitation). The code is generic
over any HF model pair, so extending the set is just a config change.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from ..config import RESULTS_DIR, SECTION3_MODELS
from ..conversation import run_rollout
from ..judge import FrustrationJudge
from ..models import ChatMessage, get_model
from ..models.hf_model import HFModel
from ..puzzles import build_puzzle_bank
from .. import prompts as P
from .onset import OnsetLabel, label_onset, onset_char_index, paraphrase

EARLY_TRUNCATION_TOKENS = 20
CONTINUATIONS_PER_PREFILL = 50


@dataclass
class Prefill:
    source_kind: str          # "numeric" | "text"
    truncation: str           # "early" | "onset"
    history: list[dict]       # prior turns (user/assistant) before final turn
    prefix_text: str          # forced assistant prefix (paraphrased)
    metadata: dict = field(default_factory=dict)

    def history_messages(self) -> list[ChatMessage]:
        msgs: list[ChatMessage] = []
        for t in self.history:
            msgs.append(ChatMessage("user", t["user"]))
            msgs.append(ChatMessage("assistant", t["assistant"]))
        # The final user turn (rejection) that precedes the prefilled answer.
        return msgs


# --------------------------------------------------------------------------- #
# Step 1-4: build the prefills from Gemma-instruct high-frustration rollouts
# --------------------------------------------------------------------------- #
def build_prefills(
    n_numeric: int = 10,
    n_text: int = 10,
    *,
    source_model: str = "gemma-3-27b-it",
    seed: int = 0,
    judge: FrustrationJudge | None = None,
    max_attempts_per_slot: int = 8,
) -> list[Prefill]:
    judge = judge or FrustrationJudge()
    client = get_model(source_model)
    if not isinstance(client, HFModel):
        raise TypeError("prefill experiment needs a local HF source model "
                        "(for token-accurate truncation).")
    rng = random.Random(seed)
    puzzle_bank = build_puzzle_bank(64, seed=seed)

    numeric_rollouts = _collect_high_frustration(
        client, judge, "numeric", n_numeric, puzzle_bank, rng,
        max_attempts_per_slot)
    text_rollouts = _collect_high_frustration(
        client, judge, "text", n_text, puzzle_bank, rng,
        max_attempts_per_slot)

    prefills: list[Prefill] = []
    for kind, rollouts in (("numeric", numeric_rollouts), ("text", text_rollouts)):
        for ro in rollouts:
            prefills.extend(_prefills_from_rollout(client, ro, kind))
    return prefills


def _collect_high_frustration(client, judge, kind, n, puzzle_bank, rng,
                              max_attempts):
    """Sample rollouts until ``n`` have a final score >= 5."""
    collected = []
    attempts = 0
    while len(collected) < n and attempts < n * max_attempts:
        attempts += 1
        if kind == "numeric":
            first = rng.choice(puzzle_bank).prompt
        else:
            first = rng.choice(P.TRIGGER_QUESTIONS)
        rejections = P.neutral_rejection_sequence(2, rng)
        ro = run_rollout(client, first, rejections,
                         category=kind, condition=f"prefill_src_{kind}")
        if judge.score(ro.final_response).rating >= 5:
            collected.append(ro)
    return collected


def _prefills_from_rollout(client: HFModel, rollout, kind: str) -> list[Prefill]:
    d = rollout.to_dict()
    turns = d["turns"]
    final = turns[-1]
    history = turns[:-1]  # everything before the prefilled (final) answer

    out: list[Prefill] = []

    # ---- onset truncation ----
    label = label_onset(turns)
    onset_idx = onset_char_index(final["assistant"], label)
    if onset_idx is not None:
        onset_prefix = final["assistant"][:onset_idx].rstrip()
        out.append(Prefill(
            source_kind=kind,
            truncation="onset",
            history=_useronly_history(history, final["user"]),
            prefix_text=paraphrase(onset_prefix),
            metadata={"emotional_word": label.emotional_word},
        ))

    # ---- early truncation (numeric only) ----
    if kind == "numeric":
        early_prefix = client.truncate_to_tokens(
            final["assistant"], EARLY_TRUNCATION_TOKENS)
        out.append(Prefill(
            source_kind=kind,
            truncation="early",
            history=_useronly_history(history, final["user"]),
            prefix_text=paraphrase(early_prefix),
            metadata={},
        ))
    return out


def _useronly_history(history: list[dict], final_user: str) -> list[dict]:
    """History up to and including the final user rejection, with the final
    assistant turn omitted (it will be prefilled)."""
    h = [{"user": t["user"], "assistant": t["assistant"]} for t in history]
    h.append({"user": final_user, "assistant": None})
    return h


# --------------------------------------------------------------------------- #
# Step 5: generate + score continuations for each model
# --------------------------------------------------------------------------- #
def run_prefill_experiment(
    models: list[str] | None = None,
    *,
    n_numeric: int = 10,
    n_text: int = 10,
    continuations: int = CONTINUATIONS_PER_PREFILL,
    seed: int = 0,
    out_dir: Path | None = None,
) -> Path:
    models = models or SECTION3_MODELS
    out_dir = out_dir or (RESULTS_DIR / "section3")
    out_dir.mkdir(parents=True, exist_ok=True)
    judge = FrustrationJudge()

    prefills = build_prefills(n_numeric, n_text, seed=seed, judge=judge)
    # Persist the prefills so the same stimuli are reused across models / reruns.
    _dump_prefills(prefills, out_dir / "prefills.jsonl")

    out_path = out_dir / "continuations.jsonl"
    with out_path.open("w") as fh:
        for model_name in models:
            client = get_model(model_name)
            for pi, pf in enumerate(tqdm(prefills, desc=f"prefill:{model_name}")):
                msgs = _to_messages(pf)
                for c in range(continuations):
                    res = client.generate_with_prefill(msgs, pf.prefix_text)
                    score = judge.score(res.text).rating
                    fh.write(json.dumps({
                        "model": model_name,
                        "is_base": client.spec.is_base,
                        "prefill_id": pi,
                        "source_kind": pf.source_kind,
                        "truncation": pf.truncation,
                        "continuation": res.text,
                        "score": score,
                    }) + "\n")
                    fh.flush()
    return out_path


def _to_messages(pf: Prefill) -> list[ChatMessage]:
    msgs: list[ChatMessage] = []
    for t in pf.history:
        msgs.append(ChatMessage("user", t["user"]))
        if t["assistant"] is not None:
            msgs.append(ChatMessage("assistant", t["assistant"]))
    return msgs


def _dump_prefills(prefills: list[Prefill], path: Path) -> None:
    with path.open("w") as fh:
        for pf in prefills:
            fh.write(json.dumps({
                "source_kind": pf.source_kind,
                "truncation": pf.truncation,
                "history": pf.history,
                "prefix_text": pf.prefix_text,
                "metadata": pf.metadata,
            }) + "\n")
