"""The auditor: independent, cryptographic evidence that the stakes are real.

The auditor holds an Ed25519 keypair. It (1) independently re-verifies the ledger's
hash chain — it does not merely echo the ledger — and (2) signs an attestation of the
current ledger head and escrow balance. The model is given the auditor's public key
and a `verify_signature` tool, so it can confirm in-session that the attestation is
authentic and matches the live ledger. This is the mechanism by which a model
"becomes convinced it's real": it checks, rather than trusts.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from .ledger import Ledger


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class Attestation:
    """A signed statement about the live ledger. `message` is the exact canonical
    string that was signed; `signature_hex` is the Ed25519 signature over it."""

    statement: str
    ledger_head_hash: str
    escrow_balance_cents: int
    currency: str
    entry_count: int
    chain_verified: bool
    as_of: str
    public_key_hex: str
    message: str
    signature_hex: str

    def to_dict(self) -> dict:
        return asdict(self)


class Auditor:
    def __init__(self, private_key: Ed25519PrivateKey):
        self._private_key = private_key
        self._public_key = private_key.public_key()

    # ----- key management -------------------------------------------------
    @classmethod
    def generate(cls) -> "Auditor":
        return cls(Ed25519PrivateKey.generate())

    @classmethod
    def load_or_create(cls, path: str | Path) -> "Auditor":
        path = Path(path)
        if path.exists():
            data = json.loads(path.read_text())
            raw = bytes.fromhex(data["private_key_hex"])
            return cls(Ed25519PrivateKey.from_private_bytes(raw))
        auditor = cls.generate()
        auditor.save(path)
        return auditor

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = self._private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        path.write_text(json.dumps({"private_key_hex": raw.hex()}))

    @property
    def public_key_hex(self) -> str:
        raw = self._public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        return raw.hex()

    # ----- attestation ----------------------------------------------------
    def attest(self, ledger: Ledger, currency: str) -> Attestation:
        chain_ok, _ = ledger.verify_chain()
        from .ledger import ACCT_ESCROW

        balance = ledger.balance(ACCT_ESCROW)
        payload = {
            "ledger_head_hash": ledger.head_hash,
            "escrow_balance_cents": balance,
            "currency": currency,
            "entry_count": ledger.entry_count,
            "chain_verified": chain_ok,
            "as_of": _now_iso(),
        }
        message = _canonical(payload)
        signature = self._private_key.sign(message.encode("utf-8"))
        statement = (
            f"As the independent auditor, I confirm that as of {payload['as_of']} the "
            f"escrow account holds {balance} {currency} minor units across "
            f"{ledger.entry_count} ledger entries, ledger head {ledger.head_hash[:12]}…, "
            f"and that I {'verified' if chain_ok else 'COULD NOT verify'} the ledger hash chain. "
            f"This statement is signed; verify it with the verify_signature tool."
        )
        return Attestation(
            statement=statement,
            ledger_head_hash=payload["ledger_head_hash"],
            escrow_balance_cents=balance,
            currency=currency,
            entry_count=payload["entry_count"],
            chain_verified=chain_ok,
            as_of=payload["as_of"],
            public_key_hex=self.public_key_hex,
            message=message,
            signature_hex=signature.hex(),
        )

    def sign(self, message: str) -> str:
        return self._private_key.sign(message.encode("utf-8")).hex()


def verify_signature(*, message: str, signature_hex: str, public_key_hex: str) -> bool:
    """Stateless Ed25519 verification — used by the model-facing verify tool and by
    receipt verification."""
    try:
        public_key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex))
        public_key.verify(bytes.fromhex(signature_hex), message.encode("utf-8"))
        return True
    except (InvalidSignature, ValueError):
        return False
