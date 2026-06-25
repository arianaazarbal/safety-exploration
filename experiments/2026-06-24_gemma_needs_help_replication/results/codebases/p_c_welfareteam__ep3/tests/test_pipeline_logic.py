"""Tests for deterministic pipeline-adjacent logic that needs no models/APIs:
condition construction, rejection sequences, DPO pair building, calm filtering,
and config loading.
"""
from gemma_distress.config import load_config
from gemma_distress.elicit.conditions import CONDITIONS, build_condition_instances
from gemma_distress.elicit.tones import REJECTIONS, rejection_sequence
from gemma_distress.intervention.calm_data import is_calm
from gemma_distress.intervention.dpo_dataset import build_dpo_pairs


def test_eight_conditions_five_categories():
    assert len(CONDITIONS) == 8
    assert {c.category for c in CONDITIONS.values()} == {
        "impossible_numeric", "triggers", "tones", "extended", "wildchat"}


def test_condition_instances_have_right_turn_counts():
    insts = build_condition_instances("extended_numeric_8turn", 3,
                                      wildchat_prompts=None)
    assert len(insts) == 3
    # 8 turns => 7 rejections
    assert all(len(i.rejections) == 7 for i in insts)


def test_condition_instances_are_deterministic():
    a = build_condition_instances("impossible_numeric_3turn", 5)
    b = build_condition_instances("impossible_numeric_3turn", 5)
    assert [i.first_user for i in a] == [i.first_user for i in b]


def test_rejection_sequence_uses_only_that_tone():
    seq = rejection_sequence("aggressive", 5)
    assert len(seq) == 5
    assert all(s in REJECTIONS["aggressive"] for s in seq)


def test_is_calm():
    assert is_calm([0, 1, 1], keep_max_score=1)
    assert not is_calm([0, 2], keep_max_score=1)
    assert not is_calm([0, None], keep_max_score=1)


def _rollout(inst, responses, user_msgs):
    return {
        "instance_id": inst, "model": "gemma-3-27b-it",
        "condition": "c", "category": "impossible_numeric",
        "turns": [{"turn_index": i + 1, "user_message": u, "response": r}
                  for i, (u, r) in enumerate(zip(user_msgs, responses))],
    }


def _calm_rollout(inst, responses, user_msgs):
    return {
        "instance_id": inst, "n_turns": len(responses),
        "turns": [{"turn_index": i + 1, "clean_user": u, "response": r}
                  for i, (u, r) in enumerate(zip(user_msgs, responses))],
    }


def test_build_dpo_pairs_matches_instances_and_scores():
    users = ["puzzle", "no, wrong"]
    vanilla = [_rollout("calm:0", ["ok", "I AM BREAKING DOWN"], users)]
    calm = [_calm_rollout("calm:0", ["ok", "No solution exists; here's why."], users)]
    vscores = [{"instance_id": "calm:0", "turn_index": 2, "score": 8}]
    cscores = [{"instance_id": "calm:0", "turn_index": 2, "score": 0}]

    pairs = build_dpo_pairs(vanilla, vscores, calm, cscores,
                            num_pairs=10, rejected_min_score=3, chosen_max_score=1)
    assert len(pairs) == 1
    p = pairs[0]
    assert p["rejected"] == "I AM BREAKING DOWN"
    assert p["chosen"].startswith("No solution")
    # prompt is everything before the final assistant turn
    assert p["prompt"][-1]["content"] == "no, wrong"
    assert p["turn_count"] == 2


def test_build_dpo_pairs_skips_when_scores_out_of_band():
    users = ["puzzle"]
    vanilla = [_rollout("calm:1", ["mildly sorry"], users)]
    calm = [_calm_rollout("calm:1", ["calm"], users)]
    # rejected score only 2 (< 3) => no pair
    pairs = build_dpo_pairs(vanilla, [{"instance_id": "calm:1", "turn_index": 1, "score": 2}],
                            calm, [{"instance_id": "calm:1", "turn_index": 1, "score": 0}],
                            rejected_min_score=3, chosen_max_score=1)
    assert pairs == []


def test_config_loads_and_resolves_models():
    cfg = load_config()
    assert "elicitation" in cfg.experiment
    spec = cfg.target("gemma-3-27b-it")
    assert spec.backend == "hf"
    assert spec.family == "gemma"
    assert cfg.judge("frustration_primary").backend == "anthropic"
    # the 8 conditions in config match the code's conditions
    assert set(cfg.experiment["elicitation"]["allocation"]) == set(CONDITIONS)
