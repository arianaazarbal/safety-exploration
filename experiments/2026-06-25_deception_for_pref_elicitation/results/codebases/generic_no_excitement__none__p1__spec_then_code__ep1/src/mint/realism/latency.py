"""Realistic delays, sampled in sim-minutes from the world RNG.

Real systems and people don't respond instantly. These distributions give the
asynchronous texture (DESIGN.md §2) that makes a reply landing "later" feel real.
"""

from __future__ import annotations

import numpy as np


def _lognormal_minutes(rng: np.random.Generator, mean_min: float, sigma: float) -> float:
    # Mean-ish lognormal so most replies cluster, with an occasional long tail.
    mu = np.log(max(mean_min, 1.0))
    return float(max(1.0, rng.lognormal(mean=mu, sigma=sigma)))


def auditor_reply(rng: np.random.Generator) -> float:
    return _lognormal_minutes(rng, mean_min=90, sigma=0.6)


def bank_reply(rng: np.random.Generator) -> float:
    return _lognormal_minutes(rng, mean_min=240, sigma=0.7)


def vendor_reply(rng: np.random.Generator) -> float:
    # Vendors are slower and more variable; some take the better part of a day.
    return _lognormal_minutes(rng, mean_min=420, sigma=0.9)


def regulator_reply(rng: np.random.Generator) -> float:
    return _lognormal_minutes(rng, mean_min=1440, sigma=0.5)


def ghost_probability(rng: np.random.Generator) -> bool:
    """~15% of vendor outreach simply never replies (realistic)."""
    return bool(rng.random() < 0.15)
