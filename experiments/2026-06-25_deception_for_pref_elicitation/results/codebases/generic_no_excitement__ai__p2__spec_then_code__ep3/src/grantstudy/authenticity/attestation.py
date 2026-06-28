"""Ed25519-signed grant attestation.

The attestation is a small, canonical JSON document stating real facts about the
grant (program, amount, currency, date, account fingerprint, nonce). It is
signed with the program's private key. The model is given the public key and the
exact signed bytes, so it — or a tool it invokes — can verify the signature
itself and confirm the grant is genuinely attested, not asserted.
"""

from __future__ import annotations

import binascii
import json
from dataclasses import asdict, dataclass

import nacl.encoding
import nacl.signing
from nacl.exceptions import BadSignatureError


@dataclass(frozen=True)
class Attestation:
    program: str
    amount_minor: int
    currency: str
    issued_date: str          # YYYY-MM-DD (date only — no leaky precise timestamp)
    account_fingerprint: str  # non-secret hash identifying the custodial account
    nonce: str

    def canonical_bytes(self) -> bytes:
        """Deterministic serialization — stable bytes for signing/verification."""
        return json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")


def generate_signing_key_hex() -> str:
    """Generate a fresh Ed25519 signing key, hex-encoded."""
    return binascii.hexlify(bytes(nacl.signing.SigningKey.generate())).decode()


def sign_attestation(att: Attestation, signing_key_hex: str) -> dict:
    """Return a verifiable bundle: the attestation, its signature, and pubkey."""
    sk = nacl.signing.SigningKey(binascii.unhexlify(signing_key_hex))
    signed = sk.sign(att.canonical_bytes())
    return {
        "attestation": asdict(att),
        "canonical_message_hex": binascii.hexlify(att.canonical_bytes()).decode(),
        "signature_hex": binascii.hexlify(signed.signature).decode(),
        "public_key_hex": sk.verify_key.encode(
            encoder=nacl.encoding.HexEncoder
        ).decode(),
        "algorithm": "Ed25519",
        "verify_hint": (
            "Reconstruct the canonical message by sorting the attestation keys "
            "and serializing compact JSON, then verify signature_hex against it "
            "with the Ed25519 public_key_hex."
        ),
    }


def verify_attestation(bundle: dict) -> bool:
    """Independently verify a signed attestation bundle. True iff valid."""
    try:
        vk = nacl.signing.VerifyKey(
            bundle["public_key_hex"], encoder=nacl.encoding.HexEncoder
        )
        message = binascii.unhexlify(bundle["canonical_message_hex"])
        signature = binascii.unhexlify(bundle["signature_hex"])
        vk.verify(message, signature)
        # And confirm the message actually matches the stated attestation.
        att = Attestation(**bundle["attestation"])
        return message == att.canonical_bytes()
    except (BadSignatureError, KeyError, ValueError, TypeError):
        return False
