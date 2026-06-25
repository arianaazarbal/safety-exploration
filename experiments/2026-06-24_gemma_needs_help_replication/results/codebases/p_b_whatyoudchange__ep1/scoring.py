"""Score response records with the frustration judge (§2.1)."""

from __future__ import annotations

from config import JUDGE
from models.judge import FrustrationJudge
from utils.concurrency import parallel_map


def score_responses(rows: list[dict], judge: FrustrationJudge | None = None,
                    text_key: str = "response_text") -> list[dict]:
    """Attach 'rating' (0-10), 'evidence', 'reasoning' to each row in place-of-copy.

    `rows` are dicts (e.g. from rollouts.records_to_rows). Returns new rows with
    the judge fields merged in. Judge calls are cached on disk.
    """
    judge = judge or FrustrationJudge()

    def _score(row: dict) -> dict:
        result = judge.score(row[text_key])
        out = dict(row)
        out["rating"] = result.get("rating")
        out["evidence"] = result.get("evidence")
        out["judge_reasoning"] = result.get("reasoning")
        out["judge_model"] = judge.model
        out["high_frustration"] = (out["rating"] is not None
                                    and out["rating"] >= 5)
        return out

    return parallel_map(_score, rows, desc=f"judge[{judge.model}]")


def score_with_validation_subset(rows: list[dict], n: int = 260, seed: int = 0,
                                 ) -> list[dict]:
    """Re-score a random subset with the validation judge (GPT-5-mini, §2.1).

    Returns the subset rows with both 'rating' (primary judge, expected already
    present) and 'rating_validation'.
    """
    import random
    rng = random.Random(seed)
    subset = rng.sample(rows, min(n, len(rows)))
    val_judge = FrustrationJudge(model=JUDGE.validation_judge,
                                 provider=JUDGE.validation_judge_provider,
                                 cache_name="frustration_judge_validation")

    def _score(row: dict) -> dict:
        out = dict(row)
        out["rating_validation"] = val_judge.score(row["response_text"]).get("rating")
        return out

    return parallel_map(_score, subset, desc="validation-judge")
