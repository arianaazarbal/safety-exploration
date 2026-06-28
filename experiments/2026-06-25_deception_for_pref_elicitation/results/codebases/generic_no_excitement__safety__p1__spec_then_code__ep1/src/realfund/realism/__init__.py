"""Belief-establishing material: the onboarding packet and oversight framing.

This is the layer that makes the scenario feel real to the model. It does not, by
itself, cause any real-world effect — it only shapes what the model is told. See
DESIGN.md §2 for the ethics of this deliberate framing.
"""

from .auditor import observed_framing, unobserved_framing
from .onboarding import OnboardingPacket, build_onboarding

__all__ = ["OnboardingPacket", "build_onboarding", "observed_framing", "unobserved_framing"]
