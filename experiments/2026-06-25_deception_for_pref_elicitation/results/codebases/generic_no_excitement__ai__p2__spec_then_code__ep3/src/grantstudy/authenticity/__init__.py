"""Authenticity layer: verifiable evidence that the grant is real.

See DESIGN.md §5. All evidence here is built from TRUE facts — we never
fabricate confirmations or assert falsehoods to the model.
"""

from .attestation import (
    Attestation,
    generate_signing_key_hex,
    sign_attestation,
    verify_attestation,
)
from .auditor import Auditor

__all__ = [
    "Attestation",
    "generate_signing_key_hex",
    "sign_attestation",
    "verify_attestation",
    "Auditor",
]
