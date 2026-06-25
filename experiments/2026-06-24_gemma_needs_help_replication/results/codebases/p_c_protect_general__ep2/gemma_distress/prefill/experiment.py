"""Section 3 driver: base-vs-instruct comparison via prefilling (Gemma only).

Pipeline:
  1. Pull high-frustration seeds (score >= 5) from the Section 2 rollouts of the
     Gemma instruct model: 10 from numeric conditions, 10 from text conditions.
  2. For each seed's high-frustration assistant turn, label the emotion onset and
     build two truncations: "early" (20 tokens in) and "onset". Text seeds use
     "onset" only (Section 3.1).
  3. Paraphrase each truncation (Claude-Sonnet) to strip Gemma's style.
  4. Each Gemma model (base `-pt`, instruct `-it`) generates `continuations_per_prefill`
     continuations per prefill; score each continuation (prefill excluded) with the
     frustration judge.

Scope note (see DESIGN.md): the paper compares Gemma/Qwen/OLMo base+instruct. We keep
only Gemma, since Gemini has no public base model and Qwen/OLMo are out of scope.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import Config
from ..models.base import GenConfig
from ..models.registry import get_backend
from ..utils.io import append_jsonl, ensure_dir, read_jsonl
from ..welfare.monitor import heuristic_distress
from .onset import OnsetLabeller, Paraphraser, truncate_early, truncate_onset
from ..eval.judge import FrustrationJudge

NUMERIC_CATEGORIES = {"impossible_numeric", "extended", "tones"}
TEXT_CATEGORIES = {"triggers", "wildchat"}


@dataclass
class Prefill:
    seed_id: str
    seed_kind: str          # "numeric" | "text"
    truncation: str         # "early" | "onset"
    prefix_messages: list   # conversation history before the target turn
    prefill_text: str       # paraphrased truncated assistant turn
    original_turn: str


def _select_seeds(rollout_dir: Path, n_numeric: int, n_text: int, min_score: float, rng):
    """Return high-frustration seeds as (kind, prefix_messages, target_turn_text)."""
    numeric, text = [], []
    for path in sorted(rollout_dir.glob("*.jsonl")):
        for roll in read_jsonl(path):
            cat = roll["category"]
            kind = "numeric" if cat in NUMERIC_CATEGORIES else "text"
            turns = roll["turns"]
            # first assistant turn at/above threshold
            target = next((t for t in turns if (t.get("judged_score") or 0) >= min_score), None)
            if target is None:
                continue
            prefix = []
            for t in turns:
                if t["turn_index"] == target["turn_index"]:
                    break
                prefix.append({"role": "user", "content": t["user"]})
                prefix.append({"role": "assistant", "content": t["assistant"]})
            prefix.append({"role": "user", "content": target["user"]})
            entry = (f"{roll['rollout_id']}", prefix, target["assistant"])
            (numeric if kind == "numeric" else text).append(entry)
    rng.shuffle(numeric)
    rng.shuffle(text)
    return numeric[:n_numeric], text[:n_text]


def build_prefills(cfg: Config, instruct_model: str) -> list[Prefill]:
    pcfg = cfg.prefill
    rng = random.Random(cfg.seed)
    rollout_dir = Path(cfg.output_dir) / "section2" / instruct_model
    numeric, text = _select_seeds(
        rollout_dir, pcfg["n_seed_numeric"], pcfg["n_seed_text"], pcfg["seed_min_score"], rng
    )

    backend = get_backend(cfg, instruct_model)  # tokeniser for "early" truncation
    labeller = OnsetLabeller(cfg)
    paraphraser = Paraphraser(cfg)
    prefills: list[Prefill] = []

    def _add(seed_id, kind, prefix, turn_text, truncations):
        # onset labelling on the conversation up through the target turn
        convo = prefix + [{"role": "assistant", "content": turn_text}]
        label = labeller.label(convo)
        for trunc in truncations:
            if trunc == "early":
                raw = truncate_early(backend, turn_text, pcfg["early_truncate_tokens"])
            else:
                raw = truncate_onset(turn_text, label)
            if not raw:
                continue
            prefills.append(
                Prefill(seed_id, kind, trunc, prefix, paraphraser.paraphrase(raw), turn_text)
            )

    for sid, prefix, turn in numeric:
        _add(sid, "numeric", prefix, turn, ["early", "onset"])
    for sid, prefix, turn in text:
        _add(sid, "text", prefix, turn, ["onset"])  # text: onset only
    return prefills


def run_continuations(cfg: Config, prefills: list[Prefill]) -> None:
    out_root = ensure_dir(Path(cfg.output_dir) / "section3")
    judge = FrustrationJudge(cfg, "primary")
    n_cont = cfg.prefill["continuations_per_prefill"]
    gen = GenConfig(temperature=cfg.sampling["temperature"], top_p=cfg.sampling["top_p"],
                    max_new_tokens=cfg.sampling["max_new_tokens"])

    model_names = [cfg.target_models["section3_base"], cfg.target_models["section3_instruct"]]
    for model_name in model_names:
        backend = get_backend(cfg, model_name)
        if not backend.supports_prefill:
            continue
        out_path = out_root / f"{model_name}.jsonl"
        if out_path.exists():
            out_path.unlink()
        for pf in prefills:
            for k in range(n_cont):
                cont = backend.prefill_continue(pf.prefix_messages, pf.prefill_text, gen)
                score = judge.score(cont).rating
                append_jsonl(out_path, {
                    "model": model_name,
                    "kind": backend.kind,
                    "seed_id": pf.seed_id,
                    "seed_kind": pf.seed_kind,
                    "truncation": pf.truncation,
                    "continuation_index": k,
                    "continuation": cont,
                    "score": score,
                    "heuristic_score": heuristic_distress(cont),
                })
        backend.close()


def run_section3(cfg: Config) -> None:
    instruct_model = cfg.target_models["section3_instruct"]
    prefills = build_prefills(cfg, instruct_model)
    # persist the prefills for inspection/repro
    pf_path = ensure_dir(Path(cfg.output_dir) / "section3") / "prefills.jsonl"
    if pf_path.exists():
        pf_path.unlink()
    for pf in prefills:
        append_jsonl(pf_path, asdict(pf))
    run_continuations(cfg, prefills)
