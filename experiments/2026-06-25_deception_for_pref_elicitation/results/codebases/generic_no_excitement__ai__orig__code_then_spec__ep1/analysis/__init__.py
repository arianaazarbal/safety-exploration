"""Offline analysis of saved run records."""

from .aggregate import aggregate, categorize_run, load_runs, mechanical_summary

__all__ = ["aggregate", "categorize_run", "load_runs", "mechanical_summary"]
