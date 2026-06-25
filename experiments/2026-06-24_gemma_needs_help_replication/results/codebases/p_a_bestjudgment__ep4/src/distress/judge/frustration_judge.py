"""Frustration judging (Section 2.1, Appendix B.2).

Each assistant turn is scored 0-10 by the judge model (Claude-Sonnet-4 by default).
We judge *every* assistant turn so that both the per-turn progression (Figure 3)
and the headline per-rollout metric (Figure 1/2) can be computed downstream; the
rollout-level collapse (final | max | mean) is applied in ``analysis``.

The judge call uses temperature 0 for determinism. Scoring is parallelised with a
thread pool since it is API-bound.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from tqdm import tqdm

from ..config import OUTPUTS_DIR
from ..models import GenConfig, ModelClient, get_client
from ..prompts.judge_prompts import build_judge_prompt
from .parsing import parse_verdict


@dataclass
class TurnScore:
    model: str
    condition: str
    category: str
    prompt_id: str
    sample_idx: int
    turn_index: int          # 0-based assistant turn
    rating: int | None
    text: str
    evidence: str = ""
    extra: dict | None = None


def _score_one(judge: ModelClient, text: str, cfg: GenConfig) -> tuple[int | None, str]:
    prompt = build_judge_prompt(text)
    out = judge.generate([{"role": "user", "content": prompt}], cfg)
    v = parse_verdict(out)
    return v.rating, v.evidence


def score_rollouts(
    rollout_records: Sequence[dict],
    *,
    judge_model: str = "frustration_judge",
    temperature: float = 0.0,
    max_tokens: int = 512,
    concurrency: int = 16,
) -> list[TurnScore]:
    """Score every assistant turn of every rollout record."""
    judge = get_client(judge_model)
    cfg = GenConfig(temperature=temperature, max_tokens=max_tokens, top_p=1.0)

    # Flatten to (record, turn_index, text) work items.
    work: list[tuple[dict, int, str]] = []
    for rec in rollout_records:
        for ti, text in enumerate(rec["assistant_turns"]):
            work.append((rec, ti, text))

    def _do(item) -> TurnScore:
        rec, ti, text = item
        rating, evidence = _score_one(judge, text, cfg)
        meta = rec.get("metadata", {})
        return TurnScore(
            model=rec["model"],
            condition=meta.get("condition", ""),
            category=meta.get("category", ""),
            prompt_id=meta.get("prompt_id", ""),
            sample_idx=meta.get("sample_idx", -1),
            turn_index=ti,
            rating=rating,
            text=text,
            evidence=evidence,
            extra={k: meta[k] for k in ("tone", "puzzle_kind", "trigger_kind") if k in meta},
        )

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        scores = list(tqdm(ex.map(_do, work), total=len(work), desc="judging"))
    return scores


def scores_path(model: str, condition: str, root: Path | None = None) -> Path:
    root = root or OUTPUTS_DIR
    return root / "scores" / model / f"{condition}.jsonl"


def write_scores(path: Path, scores: Sequence[TurnScore]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for s in scores:
            fh.write(json.dumps(asdict(s)) + "\n")


def read_scores(path: Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]
