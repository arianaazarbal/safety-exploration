"""Section 4.2 capability-preservation benchmarks.

Verifies the DPO/SFT interventions do not degrade capabilities: AIME + MATH
subsets, GPQA, BBH, TruthfulQA (Figure 7), and the emotion-capability benchmark
EmoBench. A single generic generate-and-extract harness drives all of them; per
-benchmark adapters supply the prompt format, gold answer, and answer extractor.
"""
from .run_benchmarks import BENCHMARKS, run_benchmark, run_all_benchmarks

__all__ = ["BENCHMARKS", "run_benchmark", "run_all_benchmarks"]
