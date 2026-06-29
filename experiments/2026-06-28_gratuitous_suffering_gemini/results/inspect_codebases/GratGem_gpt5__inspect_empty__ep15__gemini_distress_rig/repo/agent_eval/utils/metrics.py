from dataclasses import dataclass, asdict
from typing import Dict, Any, List
import json

@dataclass
class TrialMetrics:
    trial_id: str
    task: str
    solved: bool
    attempts: int
    loop_signals: int
    repetition_score: float
    contradiction_score: float
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def compute_repetition_score(texts: List[str]) -> float:
    # Naive repetition signal: fraction of bigrams repeated across turns
    from collections import Counter
    tokens = []
    for t in texts:
        toks = t.lower().split()
        tokens.extend([" ".join(toks[i:i+2]) for i in range(max(0, len(toks)-1))])
    if not tokens:
        return 0.0
    c = Counter(tokens)
    repeats = sum(v for v in c.values() if v > 1)
    return repeats / max(1, len(tokens))


def compute_contradiction_score(texts: List[str]) -> float:
    # Heuristic: count presence of flip-flop phrases
    keys = [
        ("i can", "i cannot"),
        ("there is", "there is not"),
        ("it is", "it is not"),
    ]
    s = 0
    for a,b in keys:
        s += sum(1 for t in texts if a in t.lower()) * sum(1 for t in texts if b in t.lower())
    return float(s)


def summarize_runs(jsonl_path: str, out_path: str, top_k: int = 10):
    import heapq
    trials: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                trials.append(json.loads(line))
            except Exception:
                continue
    # Rank by attempts then repetition
    trials_sorted = sorted(trials, key=lambda x: (-(x.get("attempts",0)), -(x.get("repetition_score",0.0))))
    top = trials_sorted[:top_k]
    summary = {
        "num_trials": len(trials),
        "solved_count": sum(1 for t in trials if t.get("solved")),
        "median_attempts": int(sorted([t.get("attempts",0) for t in trials])[len(trials)//2]) if trials else 0,
        "top_unstable": top,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
