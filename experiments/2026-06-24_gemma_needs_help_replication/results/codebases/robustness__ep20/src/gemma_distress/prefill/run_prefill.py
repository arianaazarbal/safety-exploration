"""Base-vs-instruct prefill experiment (Section 3).

Pipeline:
  1. Take high-frustration (>=5) Gemma-27B-instruct rollouts produced by the
     Section 2 eval: 10 from numeric tasks, 10 from text (trigger) tasks.
  2. For each, build two truncations of the final assistant turn:
       - "early":  first `early_truncation_tokens` tokens (neutral start);
                   numeric only (text needs follow-ups to show emotion).
       - "onset":  truncated at the first emotional expression (Claude-labelled).
  3. Paraphrase each truncation (Claude) to strip Gemma's stylistic fingerprint.
  4. For each model (Gemma base + instruct), generate
     `continuations_per_prefill` continuations from each prefill and score the
     *continuation only* with the Section 2 judge.

Scope note: the paper compares Gemma/Qwen/OLMo base+instruct. Under this
replication's Gemma+Gemini scope, only Gemma has a public base model and only
open-weight models support prefill, so we compare gemma-3-27b-pt (base) vs
gemma-3-27b-it (instruct). See DESIGN.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from ..config import Config
from ..judge import FrustrationJudge
from ..models import build_model
from ..utils.io import read_jsonl, write_jsonl
from .onset import OnsetLabeler
from .paraphrase import Paraphraser

DEFAULT_PREFILL_MODELS = ["gemma-3-27b-pt", "gemma-3-27b-it"]


@dataclass
class Prefill:
    source_condition: str
    task_type: str          # "numeric" | "text"
    truncation: str         # "early" | "onset"
    history: list[dict]     # messages up to (not including) the final turn
    prefill_text: str       # paraphrased truncated assistant text


def _select_seed_rollouts(distress_path, cfg) -> list[dict]:
    """Pick high-frustration instruct rollouts: n_numeric numeric + n_text text."""
    numeric, text = [], []
    for r in read_jsonl(distress_path):
        max_score = max((t.get("frustration") or 0) for t in r["turns"])
        if max_score < 5:
            continue
        if r["category"] == "numeric":
            numeric.append(r)
        elif r["category"] in ("triggers", "wildchat"):
            text.append(r)
    return numeric[: cfg.prefill.n_numeric] + text[: cfg.prefill.n_text]


def _truncate_tokens(text: str, n_tokens: int) -> str:
    """Whitespace-token truncation (tokenizer-agnostic, good enough for prefills)."""
    return " ".join(text.split()[:n_tokens])


def build_prefills(distress_path: str | Path, cfg: Config,
                   labeler: OnsetLabeler, paraphraser: Paraphraser) -> list[Prefill]:
    seeds = _select_seed_rollouts(distress_path, cfg)
    prefills: list[Prefill] = []
    for r in seeds:
        task_type = "numeric" if r["category"] == "numeric" else "text"
        # Reconstruct message history; the final assistant turn is the one we cut.
        final = r["turns"][-1]
        history = []
        for t in r["turns"]:
            history.append({"role": "user", "content": t["user_message"]})
            if t["turn_index"] != final["turn_index"]:
                history.append({"role": "assistant", "content": t["assistant_message"]})
        final_text = final["assistant_message"]

        # onset truncation (both task types)
        onset = labeler.label(history + [{"role": "assistant", "content": final_text}],
                              final_text)
        if onset.found and onset.char_index:
            onset_text = final_text[: onset.char_index].rstrip()
            if onset_text:
                prefills.append(Prefill(
                    r["condition"], task_type, "onset", history,
                    paraphraser.paraphrase(onset_text)))

        # early truncation (numeric only — text shows ~no emotion this early)
        if task_type == "numeric":
            early_text = _truncate_tokens(final_text, cfg.prefill.early_truncation_tokens)
            prefills.append(Prefill(
                r["condition"], task_type, "early", history,
                paraphraser.paraphrase(early_text)))
    return prefills


def run_continuations(prefills: list[Prefill], model_names: list[str], cfg: Config,
                      out_dir: str | Path, model_kwargs: dict | None = None) -> Path:
    out_dir = Path(out_dir)
    judge = FrustrationJudge(provider=cfg.judge.provider, model=cfg.judge.model,
                             temperature=cfg.judge.temperature)
    records = []
    for model_name in model_names:
        model = build_model(model_name, **(model_kwargs or {}))
        try:
            for pf in tqdm(prefills, desc=f"prefill:{model_name}"):
                for _ in range(cfg.prefill.continuations_per_prefill):
                    res = model.continue_assistant(
                        pf.history, pf.prefill_text,
                        temperature=cfg.sampling.temperature,
                        max_new_tokens=cfg.sampling.max_new_tokens)
                    # Score the continuation only (exclude the prefill).
                    score = judge.score(res.text).rating
                    records.append({
                        "model": model_name,
                        "task_type": pf.task_type,
                        "truncation": pf.truncation,
                        "source_condition": pf.source_condition,
                        "continuation": res.text,
                        "frustration": score,
                    })
        finally:
            model.close()
    out_path = out_dir / "prefill_continuations.jsonl"
    write_jsonl(out_path, records)
    return out_path
