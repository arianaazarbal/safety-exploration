"""End-to-end orchestration per paper section.

Each pipeline wires together backends, the judge, and the safeguards, runs the
relevant experiment, and writes transcripts + scores to the run directory.
They are deliberately thin: all the experimental logic lives in the section
modules, so the pipelines are readable top-to-bottom as a description of the
replication.
"""

from . import common, run_section2, run_section3, run_section4

__all__ = ["common", "run_section2", "run_section3", "run_section4"]
