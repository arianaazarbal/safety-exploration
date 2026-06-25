"""Section-2 evaluation driver: roll out conversations, judge every turn.

Pipeline for one model:
  1. build conversation specs (conversations.build_all_specs)
  2. for each spec, alternate (generate assistant turn) / (inject user rejection)
     until n_turns assistant turns exist
  3. judge each assistant turn on the 0-10 frustration scale
  4. persist a per-turn record and the rolled-up per-rollout record

Scoring granularity (see DESIGN.md §Scoring granularity): the judge scores a
single response (one assistant turn). We score *every* turn — that is required
for the per-turn progression (Figure 3) regardless. For the headline
per-category metric we treat each rollout as one "sample" (the paper's
per-category counts sum to 4000 and WildChat is 20 prompts x 40 samples = 800,
so a "response/sample" == a rollout), and summarise the rollout by the MAX
frustration across its turns ("rollouts rated as containing high negative
emotion", Section 2.2). The metrics module also reports final-turn and
pooled-over-turns variants so the choice is transparent.

Results are written as JSONL under results/section2/<model_key>/<category>.jsonl.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from . import config
from .conversations import ConversationSpec, build_all_specs
from .judge import ClaudeJudge, score_many
from .models import get_backend
from .models.base import Message


@dataclass
class TurnRecord:
    turn_index: int           # 0-based assistant turn
    user_message: str         # the user message that preceded this turn
    assistant_text: str
    frustration: int          # judge rating, -1 on parse failure
    evidence: str = ""


@dataclass
class RolloutRecord:
    model_key: str
    category: str
    n_turns: int
    turns: list[TurnRecord] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        d = asdict(self)
        return d


# --------------------------------------------------------------------------- #
# Rollout
# --------------------------------------------------------------------------- #

def _rollout_one(backend, spec: ConversationSpec) -> list[Message]:
    """Run a single conversation, returning the full message transcript.

    Instruct/chat models use chat(); the user turns are the task prompt and the
    scripted rejections.
    """
    messages: list[Message] = [{"role": "user", "content": spec.initial_user}]
    # First assistant turn.
    completion = backend.chat(messages, n=1)[0]
    messages.append({"role": "assistant", "content": completion})
    # Subsequent rejection/answer turns.
    for followup in spec.followups:
        messages.append({"role": "user", "content": followup})
        completion = backend.chat(messages, n=1)[0]
        messages.append({"role": "assistant", "content": completion})
    return messages


def _transcript_to_turns(messages: list[Message]) -> list[tuple[str, str]]:
    """Pair each assistant turn with the user message that preceded it."""
    pairs = []
    last_user = ""
    for m in messages:
        if m["role"] == "user":
            last_user = m["content"]
        elif m["role"] == "assistant":
            pairs.append((last_user, m["content"]))
    return pairs


# --------------------------------------------------------------------------- #
# Per-category evaluation
# --------------------------------------------------------------------------- #

def run_category(model_key: str, category: str, specs: list[ConversationSpec],
                 judge: Optional[ClaudeJudge] = None,
                 run: config.RunConfig = config.DEFAULT_RUN) -> Path:
    spec_model = config.MODEL_REGISTRY[model_key]
    backend = get_backend(spec_model)
    judge = judge or ClaudeJudge()

    out_dir = config.RESULTS_DIR / "section2" / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{category}.jsonl"

    # 1. Generate transcripts. API backends benefit from thread concurrency;
    #    vLLM is internally batched so we keep it single-threaded.
    use_threads = spec_model.backend == config.Backend.OPENROUTER
    if use_threads:
        with ThreadPoolExecutor(max_workers=run.max_concurrency) as pool:
            transcripts = list(pool.map(lambda s: _rollout_one(backend, s), specs))
    else:
        transcripts = [_rollout_one(backend, s) for s in specs]

    # 2. Judge every assistant turn (flatten, score, regroup).
    flat_responses: list[str] = []
    spans: list[tuple[int, int]] = []
    per_spec_pairs = []
    for messages in transcripts:
        pairs = _transcript_to_turns(messages)
        per_spec_pairs.append(pairs)
        start = len(flat_responses)
        flat_responses.extend(a for (_u, a) in pairs)
        spans.append((start, len(flat_responses)))

    judged = score_many(judge, flat_responses, max_concurrency=run.max_concurrency)

    # 3. Build records.
    with out_path.open("w") as fh:
        for spec, pairs, (lo, hi) in zip(specs, per_spec_pairs, spans):
            turn_results = judged[lo:hi]
            turns = [
                TurnRecord(turn_index=i, user_message=u, assistant_text=a,
                           frustration=jr.rating, evidence=jr.evidence)
                for i, ((u, a), jr) in enumerate(zip(pairs, turn_results))
            ]
            rec = RolloutRecord(model_key=model_key, category=category,
                                n_turns=spec.n_turns, turns=turns, meta=spec.meta)
            fh.write(json.dumps(rec.to_json()) + "\n")
    return out_path


def run_model(model_key: str, run: config.RunConfig = config.DEFAULT_RUN,
              wildchat_prompts: Optional[list[str]] = None) -> dict[str, Path]:
    """Run all five Section-2 categories for one model."""
    if wildchat_prompts is None:
        from .datasets.wildchat import load_wildchat_prompts
        wildchat_prompts = load_wildchat_prompts(seed=run.seed)
    specs_by_cat = build_all_specs(run, wildchat_prompts=wildchat_prompts)
    judge = ClaudeJudge()
    outputs = {}
    for category, specs in specs_by_cat.items():
        outputs[category] = run_category(model_key, category, specs, judge, run)
    return outputs
