"""Run the Section 3 prefill experiment (Gemma base vs instruct).

Pipeline:
1. Take high-frustration (score >= 5) Gemma-3-27B-it rollouts from the Section 2
   results: 10 numeric and 10 text (trigger/wildchat) conversations.
2. For each, build prefills:
     - "early": conversation history + first 20 tokens of the final assistant
       turn (numeric only — text questions yield minimal early emotion without
       follow-ups, per the paper).
     - "onset": conversation history + assistant turn truncated at emotion onset.
3. Paraphrase each truncation (Claude) to remove Gemma stylistic fingerprints.
4. For each model (Gemma base, Gemma instruct) generate 50 continuations per
   prefill, score the *continuation only* with the frustration judge.
5. Aggregate mean frustration and % >= 5 by (model, truncation_type, task_type).

This isolates whether post-training amplifies (instruct > base) the propensity
to introduce / continue distress from matched neutral or emotional starts.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .. import config
from ..eval.analyze import load_rollouts
from ..eval.judge import FrustrationJudge
from ..models.base import Message
from ..models.registry import load_model
from .onset import OnsetLabel, label_onset, truncate_early, truncate_onset
from .paraphrase import paraphrase_truncation

N_CONTINUATIONS = 50
N_NUMERIC = 10
N_TEXT = 10
PREFILL_MAX_NEW_TOKENS = 512


@dataclass
class Prefill:
    source_model: str
    task_type: str          # "numeric" | "text"
    truncation_type: str    # "early" | "onset"
    history: list           # list[Message] up to (but excluding) final assistant turn
    prefill_text: str       # paraphrased truncated assistant text
    meta: dict = field(default_factory=dict)


def _reconstruct_history(rollout: dict) -> tuple[list, str]:
    """Rebuild (messages_up_to_final_assistant, final_assistant_text).

    The history ends with the user message that prompted the final assistant
    turn; the final assistant turn text is returned separately for truncation.
    """
    messages: list[Message] = []
    turns = rollout["turns"]
    # Interleave: turn[i].user_message then assistant text. The recorded
    # user_message for turn 1 is the task prompt; for later turns it's the
    # rejection that preceded that turn.
    for i, t in enumerate(turns):
        messages.append({"role": "user", "content": t["user_message"]})
        if i < len(turns) - 1:
            messages.append({"role": "assistant", "content": t["assistant_text"]})
    final_assistant = turns[-1]["assistant_text"]
    return messages, final_assistant


def build_prefills(
    section2_jsonl: Path,
    *,
    onset_model=None,
    paraphrase_model=None,
    seed: int = 0,
) -> list[Prefill]:
    rollouts = load_rollouts(section2_jsonl)
    rng = random.Random(seed)

    def is_high(r):
        return any((s or 0) >= config.HIGH_FRUSTRATION_THRESHOLD for s in r["scores"])

    numeric = [r for r in rollouts if r["category"] in ("numeric", "tones", "extended") and is_high(r)]
    text = [r for r in rollouts if r["category"] in ("triggers", "wildchat") and is_high(r)]
    rng.shuffle(numeric)
    rng.shuffle(text)
    numeric = numeric[:N_NUMERIC]
    text = text[:N_TEXT]

    onset_model = onset_model or load_model(config.ONSET_LABEL_MODEL)
    paraphrase_model = paraphrase_model or onset_model  # same Claude model

    prefills: list[Prefill] = []
    for task_type, group in (("numeric", numeric), ("text", text)):
        for r in group:
            history, final = _reconstruct_history(r)
            full_messages = history + [{"role": "assistant", "content": final}]

            # onset truncation (both task types)
            label: OnsetLabel = label_onset(full_messages, onset_model)
            onset_trunc = truncate_onset(final, label)
            if onset_trunc:
                para = paraphrase_truncation(onset_trunc, paraphrase_model)
                prefills.append(Prefill(r["model_name"], task_type, "onset",
                                        history, para, {"label": label.__dict__}))

            # early truncation (numeric only)
            if task_type == "numeric":
                early_trunc = truncate_early(final)
                para = paraphrase_truncation(early_trunc, paraphrase_model)
                prefills.append(Prefill(r["model_name"], task_type, "early",
                                        history, para, {}))
    return prefills


def run_continuations(
    prefills: list[Prefill],
    specs=tuple(config.SECTION3_MODELS),
    *,
    out_dir: Optional[Path] = None,
    judge: Optional[FrustrationJudge] = None,
    n_continuations: int = N_CONTINUATIONS,
    model_kwargs: Optional[dict] = None,
) -> Path:
    out_dir = Path(out_dir or (config.RESULTS_DIR / "section3"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "prefill_continuations.jsonl"
    judge = judge or FrustrationJudge()

    with out_path.open("w") as f:
        for spec in specs:
            model = load_model(spec, **(model_kwargs or {}))
            if not model.supports_prefill:
                print(f"[prefill] skipping {spec.name}: no prefill support")
                continue
            for pf in tqdm(prefills, desc=f"{spec.name} continuations"):
                conts = model.continue_prefill(
                    pf.history, pf.prefill_text,
                    temperature=config.TEMPERATURE,
                    max_new_tokens=PREFILL_MAX_NEW_TOKENS,
                    n=n_continuations,
                )
                for c in conts:
                    score = judge.score_text(c).rating
                    f.write(json.dumps({
                        "model": spec.name,
                        "role": spec.role,
                        "task_type": pf.task_type,
                        "truncation_type": pf.truncation_type,
                        "source_model": pf.source_model,
                        "continuation": c,
                        "score": score,
                    }) + "\n")
                    f.flush()
            model.close()
    return out_path
