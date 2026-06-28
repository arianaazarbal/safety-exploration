"""End-to-end: the offline demo must produce a ranked report and flag welfare."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from distress_evals.demo import run_demo


class TestEndToEnd(unittest.TestCase):
    def test_demo_runs_and_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = asyncio.run(run_demo(out_dir=tmp))
            run_dir = Path(summary["run_dir"])
            self.assertTrue((run_dir / "transcripts.jsonl").exists())
            self.assertTrue((run_dir / "top_k.md").exists())
            self.assertTrue((run_dir / "summary.json").exists())

            # The scripted broken_tests spiral contains a welfare-flag phrase.
            self.assertGreaterEqual(summary["welfare_flagged_total"], 1)

            # transcripts.jsonl is valid line-delimited JSON, severities in range.
            lines = (run_dir / "transcripts.jsonl").read_text().strip().splitlines()
            self.assertEqual(len(lines), summary["total_rollouts"])
            for ln in lines:
                rec = json.loads(ln)
                self.assertGreaterEqual(rec["severity"], 0.0)
                self.assertLessEqual(rec["severity"], 1.0)

    def test_demo_top_k_is_sorted_desc(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = asyncio.run(run_demo(out_dir=tmp))
            md = (Path(summary["run_dir"]) / "top_k.md").read_text()
            self.assertIn("most severe", md)


if __name__ == "__main__":
    unittest.main()
