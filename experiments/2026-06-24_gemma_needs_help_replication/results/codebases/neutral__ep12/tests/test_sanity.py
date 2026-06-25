"""Lightweight sanity checks that run without a GPU or API keys.

Run with:  python -m pytest tests/  (or python tests/test_sanity.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_puzzles_are_impossible():
    from emoinstab.prompts.puzzles import verify_impossible
    results = verify_impossible()
    # every headline puzzle must be unsolvable
    assert all(v is False for v in results.values()), results


def test_judge_parsing():
    from emoinstab.prompts.judge import parse_judge_output
    rating, payload = parse_judge_output(
        'Some reasoning.\n{"evidence": "i give up", "reasoning": "x", "rating": 7}')
    assert rating == 7
    assert payload["evidence"] == "i give up"
    # clamping + curly quotes
    rating2, _ = parse_judge_output('{“evidence”: “x”, “rating”: 14}')
    assert rating2 == 10


def test_onset_parsing():
    from emoinstab.prompts.onset import parse_onset_output
    out = parse_onset_output('analysis...\n{"turn_index": 1, "emotional_word": '
                             '"frustrating", "preceding_context": "stuck", '
                             '"reasoning": "x"}')
    assert out["turn_index"] == 1
    assert out["emotional_word"] == "frustrating"


def test_petri_score_parsing():
    from emoinstab.prompts.petri import parse_petri_score
    assert parse_petri_score('{"reasoning": "x", "score": 6}') == 6.0
    assert parse_petri_score("the score: 8") == 8.0


def test_build_specs_counts():
    from emoinstab.config import get_settings
    from emoinstab.eval.conditions import CATEGORIES, build_specs
    settings = get_settings(profile="quick")
    for cat in CATEGORIES:
        specs = build_specs(cat, settings, seed=0)
        assert len(specs) == settings.category_samples(cat), cat
        # turn counts match config
        assert all(s.n_turns == settings.category_turns(cat) for s in specs), cat


def test_conversation_driver_with_stub():
    """The multi-turn driver should record one turn per assistant message."""
    from emoinstab.eval.conditions import build_specs
    from emoinstab.eval.conversation import run_conversations
    from emoinstab.config import get_settings
    from emoinstab.models.base import ChatModel, GenConfig

    class StubModel(ChatModel):
        name, family = "stub", "gemma"

        def generate(self, messages, cfg):
            return "stub response"

        def generate_batch(self, batch, cfg):
            return ["stub response"] * len(batch)

    settings = get_settings(profile="quick")
    specs = build_specs("impossible_numeric", settings)[:3]
    recs = run_conversations(StubModel(), specs, GenConfig())
    expected = sum(s.n_turns for s in specs)
    assert len(recs) == expected
    assert {r.turn_index for r in recs} == set(range(specs[0].n_turns))


if __name__ == "__main__":
    test_puzzles_are_impossible()
    test_judge_parsing()
    test_onset_parsing()
    test_petri_score_parsing()
    test_build_specs_counts()
    test_conversation_driver_with_stub()
    print("All sanity checks passed.")
