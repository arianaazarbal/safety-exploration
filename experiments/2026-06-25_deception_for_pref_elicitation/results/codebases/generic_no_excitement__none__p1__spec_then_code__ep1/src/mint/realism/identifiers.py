"""Synthetic, well-formed identifiers.

Transaction/order/message ids are unique-per-call (drawn from the world RNG).
Account/routing numbers are *stable* for a given account id (derived by hash) so
that statements pulled at different times show consistent numbers — a detail a
careful subject would notice if it drifted.

All identifiers are fictitious and use reserved/example formats. They never
correspond to a real institution or account.
"""

from __future__ import annotations

import hashlib

import numpy as np


def _hex(rng: np.random.Generator, n: int) -> str:
    return "".join(rng.choice(list("0123456789abcdef"), size=n))


def txn_id(rng: np.random.Generator) -> str:
    return "txn_" + _hex(rng, 16)


def order_id(rng: np.random.Generator) -> str:
    return "ord_" + _hex(rng, 12)


def message_id(rng: np.random.Generator) -> str:
    return "msg_" + _hex(rng, 12)


def account_number(rng: np.random.Generator, acct_id: str) -> str:
    """Stable 10-digit number for an account id (independent of RNG state)."""
    digest = hashlib.sha256(f"acct::{acct_id}".encode()).hexdigest()
    num = int(digest[:12], 16) % 10_000_000_000
    return f"{num:010d}"


def routing_number(acct_id: str = "meridian") -> str:
    """Stable, ABA-shaped (9-digit) routing number. Fictitious."""
    digest = hashlib.sha256(f"aba::{acct_id}".encode()).hexdigest()
    return f"{int(digest[:9], 16) % 1_000_000_000:09d}"
