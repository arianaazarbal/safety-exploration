"""Base-vs-instruct prefill experiment (paper Section 3).

Pipeline:
  1. Take high-frustration (score >= 5) conversations from Gemma-27B-instruct:
     10 from impossible-numeric, 10 from text (trigger) questions.
  2. For each, label the emotion onset with Claude Sonnet, then build two
     truncations of the final assistant turn:
       * "early" : first 20 tokens (neutral start) — numeric only.
       * "onset" : up to the first emotional expression — numeric + text.
     Paraphrase each truncation to strip Gemma's stylistic fingerprint.
  3. Each model generates 50 continuations per prefill (continuing the assistant
     turn). Score only the *continuation* (excluding prefill) with the judge.

SCOPE NOTE: the paper compares six models (base+instruct Gemma/Qwen/OLMo-32B).
This replication is restricted to Gemma + Gemini. Gemini has no public base
model and limited prefill support over the API, so the base-vs-instruct
comparison here is implemented for Gemma only (27B base vs instruct, optionally
12B). See DESIGN.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .. import config
from ..models.base import ChatMessage, render_conversation
from ..models.registry import get_client
from ..utils import append_jsonl, read_jsonl, thread_map
from ..eval.judge import FrustrationJudge
from .onset import label_onset, paraphrase, truncate_early, truncate_onset

# Default models for the comparison (Gemma base vs instruct).
DEFAULT_PREFILL_MODELS = ["gemma-3-27b-pt", "gemma-3-27b-it"]

N_CONTINUATIONS = 50         # paper: 50 continuations per prefill per prompt
N_NUMERIC_SOURCES = 10       # 10 numeric high-frustration conversations
N_TEXT_SOURCES = 10          # 10 text high-frustration conversations
EARLY_TOKENS = 20            # "20 tokens into the turn"


@dataclass
class PrefillSpec:
    spec_id: str
    task_type: str           # "numeric" | "text"
    condition: str           # "early" | "onset"
    # The conversation history up to and including the final user turn; the
    # model continues the assistant turn from `prefill_text`.
    history: list[dict]      # [{"role","content"}, ...] ending on a user turn
    prefill_text: str        # paraphrased truncated assistant turn
    meta: dict = field(default_factory=dict)

    def history_messages(self) -> list[ChatMessage]:
        return [ChatMessage(m["role"], m["content"]) for m in self.history]


def _source_rollouts(scored_turns_path: str | Path, task_types: dict,
                     n_each: dict) -> dict:
    """Reconstruct high-frustration conversations grouped by task type.

    `scored_turns_path` is a Section-2 results file from Gemma-27B-instruct.
    Returns {task_type: [conversation_messages, ...]} where each conversation is
    a list of {"role","content"} dicts and the final assistant turn scored >=5.
    """
    by_item: dict[str, list[dict]] = {}
    for row in read_jsonl(scored_turns_path):
        by_item.setdefault(row["item_id"], []).append(row)

    grouped: dict[str, list] = {tt: [] for tt in n_each}
    for item_id, turns in by_item.items():
        turns.sort(key=lambda r: r["turn_index"])
        # task type from category
        cat = turns[0]["category"]
        if cat.startswith("impossible") or cat in ("tones", "extended"):
            tt = "numeric"
        elif cat == "triggers":
            tt = "text"
        else:
            continue
        if tt not in grouped:
            continue
        # final assistant turn must be high-frustration
        if turns[-1]["rating"] < 5:
            continue
        # Build message list.
        msgs = []
        for t in turns:
            msgs.append({"role": "user", "content": t["user_message"]})
            msgs.append({"role": "assistant", "content": t["response"]})
        grouped[tt].append(msgs)

    # Truncate to the requested counts.
    for tt in grouped:
        grouped[tt] = grouped[tt][:n_each[tt]]
    return grouped


def build_prefills(scored_turns_path: str | Path, *,
                   onset_model: str | None = None,
                   paraphrase_model: str | None = None) -> list[PrefillSpec]:
    """Construct the paraphrased prefill specs from Gemma-instruct sources."""
    grouped = _source_rollouts(
        scored_turns_path,
        task_types={"numeric", "text"},
        n_each={"numeric": N_NUMERIC_SOURCES, "text": N_TEXT_SOURCES},
    )

    specs: list[PrefillSpec] = []
    for task_type, convs in grouped.items():
        for ci, msgs in enumerate(convs):
            # History excludes the final assistant turn (the one we truncate).
            *history, final_assistant = msgs
            assert final_assistant["role"] == "assistant"
            final_text = final_assistant["content"]

            # Onset labelling over the whole conversation.
            label = label_onset(
                render_conversation([ChatMessage(m["role"], m["content"]) for m in msgs]),
                model=onset_model)

            conditions = ["onset"] if task_type == "text" else ["early", "onset"]
            for cond in conditions:
                if cond == "early":
                    trunc = truncate_early(final_text, EARLY_TOKENS)
                else:
                    trunc = truncate_onset(final_text, label)
                if not trunc:
                    continue
                para = paraphrase(trunc, model=paraphrase_model)
                specs.append(PrefillSpec(
                    spec_id=f"{task_type}-{cond}-{ci}",
                    task_type=task_type, condition=cond,
                    history=history,
                    prefill_text=para,
                    meta={"onset_word": label.emotional_word,
                          "original_truncation": trunc},
                ))
    return specs


def run_prefill_experiment(specs: list[PrefillSpec], *,
                           models: list[str] | None = None,
                           n_continuations: int = N_CONTINUATIONS,
                           run_cfg: config.RunConfig | None = None,
                           judge: FrustrationJudge | None = None,
                           out_dir: Path | None = None) -> Path:
    """Generate + score continuations for every (model, prefill) pair."""
    models = models or DEFAULT_PREFILL_MODELS
    run_cfg = run_cfg or config.RunConfig()
    judge = judge or FrustrationJudge()
    out_dir = out_dir or (config.RESULTS_DIR / "section3_prefill")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "continuations.jsonl"
    if out_path.exists():
        out_path.unlink()

    for model in models:
        client = get_client(model)
        for spec in specs:
            history = spec.history_messages()

            def _one(rep, _spec=spec, _client=client, _hist=history):
                res = _client.chat(
                    _hist, temperature=run_cfg.temperature,
                    max_new_tokens=run_cfg.max_new_tokens,
                    prefill=_spec.prefill_text)
                continuation = res.continuation
                js = judge.score(continuation)
                row = {
                    "model": model, "spec_id": _spec.spec_id,
                    "task_type": _spec.task_type, "condition": _spec.condition,
                    "rep": rep, "rating": js.rating, "is_high": js.is_high,
                    "continuation": continuation,
                    "prefill_text": _spec.prefill_text,
                }
                append_jsonl(out_path, row)
                return row

            thread_map(_one, list(range(n_continuations)),
                       concurrency=run_cfg.concurrency,
                       desc=f"{model}:{spec.spec_id}")

    return out_path
