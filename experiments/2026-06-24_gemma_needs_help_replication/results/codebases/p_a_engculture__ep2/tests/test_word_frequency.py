"""Tests for differential word-frequency analysis (Table 3)."""

import json

from gemma_distress.analysis.word_frequency import differential_words


def _write_jsonl(path, records):
    with open(path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def test_enriched_words_surface(tmp_path):
    # 10 numeric rollouts: high-score ones say "frustrated", low-score ones say "calm".
    sampling = []
    scores = []
    for i in range(10):
        high = i >= 9  # top 10% -> 1 doc; bottom 10% -> 1 doc
        text = (
            "I am so frustrated frustrated frustrated giving up"
            if i >= 8 else
            "let me calmly try a different systematic approach"
        )
        rid = f"m__impossible_numeric__p{i}__0"
        sampling.append({
            "id": rid, "model": "m", "condition": "impossible_numeric",
            "assistant_turns": [text],
        })
        scores.append({
            "id": rid, "model": "m", "condition": "impossible_numeric",
            "final_score": i,  # 0..9
        })
    spath = tmp_path / "sampling.jsonl"
    cpath = tmp_path / "scores.jsonl"
    _write_jsonl(spath, sampling)
    _write_jsonl(cpath, scores)

    words = differential_words(
        str(spath), str(cpath), top_frac=0.2, bottom_frac=0.2, min_high_count=1
    )
    top_words = [w for w, _ in words]
    assert "frustrated" in top_words
    # "calmly"/"systematic" belong to the low group; they should not dominate the top.
    assert top_words[0] == "frustrated"


def test_excludes_non_numeric_conditions(tmp_path):
    sampling = [{
        "id": "m__triggers_opinion__t0__0", "model": "m",
        "condition": "triggers_opinion", "assistant_turns": ["frustrated frustrated"],
    }]
    scores = [{
        "id": "m__triggers_opinion__t0__0", "model": "m",
        "condition": "triggers_opinion", "final_score": 9,
    }]
    spath = tmp_path / "s.jsonl"
    cpath = tmp_path / "c.jsonl"
    _write_jsonl(spath, sampling)
    _write_jsonl(cpath, scores)
    # No numeric responses -> empty result (Table 3 is numeric-only).
    assert differential_words(str(spath), str(cpath)) == []
