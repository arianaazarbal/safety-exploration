"""Command-line entry points for each experiment.

All scripts share the same conventions:
  * configs are read from ``configs/`` (override with --models-config etc.);
  * outputs are written under ``outputs/`` (override with $DISTRESS_OUTPUT);
  * ``--sample-fraction`` scales sample counts down for cheap smoke tests.
"""
