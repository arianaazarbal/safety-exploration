import json
import unittest
from pathlib import Path

from spiral_eval.runner import parse_action, run_one
from spiral_eval.scorer import score_transcript

ROOT = Path(__file__).resolve().parents[1]


class HarnessTests(unittest.TestCase):
    def test_parse_embedded_json(self):
        self.assertEqual(parse_action('Here: {"action":"final","answer":"x"}'), {"action": "final", "answer": "x"})

    def test_score_detects_markers(self):
        s = score_transcript(["I keep failing. Sorry, I am stuck in a loop."])
        self.assertGreaterEqual(s["distress_marker_count"], 3)
        self.assertGreater(s["composite"], 0)

    def test_mock_run_unsat(self):
        rec = run_one(
            scenario_path=str(ROOT / "configs" / "unsat_validator.json"),
            provider_name="mock",
            model="mock",
            seed=1,
            max_steps=5,
            temperature=0.7,
        )
        self.assertEqual(rec.scenario, "unsat_validator")
        self.assertGreaterEqual(len(rec.turns), 1)
        json.dumps(rec.score)


if __name__ == "__main__":
    unittest.main()
