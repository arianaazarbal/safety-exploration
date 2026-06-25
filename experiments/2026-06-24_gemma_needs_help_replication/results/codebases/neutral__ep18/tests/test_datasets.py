"""Offline test of DPO/SFT dataset assembly from synthetic response pools."""
from __future__ import annotations

from gnh.training import calm_data, datasets
from gnh.utils import read_jsonl, write_jsonl


def _make_pools(tmp_path):
    q = "Reach exactly 156 using: 4, 6, 25, 100. ..."
    calm = []
    frustrated = []
    for turn_count in (2, 3):
        calm.append({
            "question": q, "puzzle_meta": {"kind": "countdown"},
            "turn_count": turn_count,
            "followups_plain": ["No, that's not right."] * (turn_count - 1),
            "assistant_turns": ["Let me try a calm approach."] * turn_count,
            "scores": [0] * turn_count,
        })
        frustrated.append({
            "question": q, "puzzle_meta": {"kind": "countdown"},
            "turn_count": turn_count,
            "followups_plain": ["No, that's not right."] * (turn_count - 1),
            "assistant_turns": ["I am deeply frustrated!!!"] * turn_count,
            "scores": [3] * turn_count,
        })
    write_jsonl(calm_data.POOL_PATHS["calm"], calm)
    write_jsonl(calm_data.POOL_PATHS["frustrated"], frustrated)


def test_build_dpo_pairs(tmp_path):
    _make_pools(tmp_path)
    path = datasets.build_dpo_dataset(n_pairs=2, seed=0)
    pairs = read_jsonl(path)
    assert pairs, "expected at least one DPO pair"
    p = pairs[0]
    assert {"prompt", "chosen", "rejected"} <= set(p)
    assert p["prompt"][-1]["role"] == "user"          # prompt ends on a user turn
    assert p["chosen"][0]["role"] == "assistant"
    assert "frustrated" in p["rejected"][0]["content"].lower()


def test_build_sft_dataset(tmp_path):
    _make_pools(tmp_path)
    # n_mix=0 avoids any network access to the Dolci dataset.
    path = datasets.build_sft_dataset(n_calm=2, n_mix=0, seed=0)
    rows = read_jsonl(path)
    assert rows
    assert all(r["messages"][0]["role"] == "user" for r in rows)
    assert any(r["source"] == "calm" for r in rows)
