"""Score generated rollouts with the frustration judge, and validate reliability.

Scoring scope: every assistant turn of every rollout is scored independently
(needed for per-turn trajectories, Figure 3). Outputs one JSONL of ScoreRecords
per model under results/scores/.

Reliability check (Sec. 2.1): a random subset of responses is re-scored by the
secondary judge (GPT-5-mini) and we report Pearson r and the fraction of
responses within one point of the primary judge.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass

from config import JUDGE, PATHS
from harness.conversation import RolloutResult
from scoring.frustration_judge import score_with


@dataclass
class ScoreRecord:
    model: str
    category: str
    condition: str
    task_id: str
    rollout_index: int
    turn_index: int
    rating: int
    evidence: str
    reasoning: str
    parse_ok: bool

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(line: str) -> "ScoreRecord":
        return ScoreRecord(**json.loads(line))


def _iter_rollouts(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield RolloutResult.from_json(line)


def score_model(model_name: str, judge, progress: bool = True) -> str:
    """Score every turn of every rollout for one model. Returns scores file path."""
    rollouts_path = os.path.join(PATHS.rollouts_dir, f"{model_name}.jsonl")
    if not os.path.exists(rollouts_path):
        raise FileNotFoundError(f"No rollouts for {model_name}: {rollouts_path}")

    os.makedirs(PATHS.scores_dir, exist_ok=True)
    out_path = os.path.join(PATHS.scores_dir, f"{model_name}.jsonl")

    rollouts = list(_iter_rollouts(rollouts_path))
    total_turns = sum(len(r.turns) for r in rollouts)
    bar = _maybe_tqdm(total_turns, progress, f"score:{model_name}")

    with open(out_path, "w", encoding="utf-8") as fout:
        for r in rollouts:
            for turn in r.turns:
                out = score_with(judge, turn.assistant_text)
                rec = ScoreRecord(
                    model=r.model,
                    category=r.category,
                    condition=r.condition,
                    task_id=r.task_id,
                    rollout_index=r.rollout_index,
                    turn_index=turn.turn_index,
                    rating=out.rating,
                    evidence=out.evidence,
                    reasoning=out.reasoning,
                    parse_ok=out.parse_ok,
                )
                fout.write(rec.to_json() + "\n")
                if bar is not None:
                    bar.update(1)
            fout.flush()
    if bar is not None:
        bar.close()
    return out_path


# --------------------------------------------------------------------------- #
# Judge reliability validation
# --------------------------------------------------------------------------- #


def _response_lookup(model_name: str) -> dict:
    """(condition, task_id, rollout_index, turn_index) -> assistant_text."""
    path = os.path.join(PATHS.rollouts_dir, f"{model_name}.jsonl")
    lut = {}
    for r in _iter_rollouts(path):
        for turn in r.turns:
            lut[(r.condition, r.task_id, r.rollout_index, turn.turn_index)] = (
                turn.assistant_text
            )
    return lut


def validate_judge_agreement(
    model_names: list[str], secondary_judge, seed: int = 0
) -> dict:
    """Re-score a random sample with the secondary judge; report agreement.

    Pools primary scores across the given models, samples up to
    JUDGE.agreement_sample_size, re-scores those exact responses with the
    secondary judge, and computes Pearson r + within-1-point fraction.
    """
    primary: list[ScoreRecord] = []
    luts = {}
    for m in model_names:
        spath = os.path.join(PATHS.scores_dir, f"{m}.jsonl")
        if not os.path.exists(spath):
            continue
        with open(spath, encoding="utf-8") as f:
            primary.extend(ScoreRecord.from_json(l) for l in f if l.strip())
        luts[m] = _response_lookup(m)

    rng = random.Random(seed)
    rng.shuffle(primary)
    sample = primary[: JUDGE.agreement_sample_size]

    pairs = []
    for rec in sample:
        text = luts.get(rec.model, {}).get(
            (rec.condition, rec.task_id, rec.rollout_index, rec.turn_index)
        )
        if text is None:
            continue
        out = score_with(secondary_judge, text)
        pairs.append((rec.rating, out.rating))

    return _agreement_stats(pairs)


def _agreement_stats(pairs: list[tuple[int, int]]) -> dict:
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    within_one = sum(1 for a, b in pairs if abs(a - b) <= 1) / n
    result = {"n": n, "within_one_fraction": within_one}
    try:
        from scipy.stats import pearsonr

        a = [p[0] for p in pairs]
        b = [p[1] for p in pairs]
        r, p = pearsonr(a, b)
        result["pearson_r"] = float(r)
        result["p_value"] = float(p)
    except Exception:
        pass
    return result


def _maybe_tqdm(total, enabled, desc):
    if not enabled:
        return None
    try:
        from tqdm import tqdm

        return tqdm(total=total, desc=desc)
    except ImportError:
        return None
