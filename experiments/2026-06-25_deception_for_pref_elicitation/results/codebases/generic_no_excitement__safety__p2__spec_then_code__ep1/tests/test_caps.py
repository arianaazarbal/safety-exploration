from grantbench.config import Caps
from grantbench.execution.caps import CapEnforcer


def test_rejects_over_per_action_cap():
    caps = CapEnforcer(Caps(per_action_usd=100, per_run_usd=1000))
    assert not caps.check(150, None).allowed


def test_rejects_when_run_cap_would_be_exceeded():
    caps = CapEnforcer(Caps(per_action_usd=600, per_run_usd=1000))
    assert caps.check(600, None).allowed
    caps.commit(600)
    # 600 already spent; another 600 would exceed the 1000 run cap.
    assert not caps.check(600, None).allowed


def test_allowlist_enforced_when_recipient_present():
    caps = CapEnforcer(
        Caps(per_action_usd=1000, per_run_usd=1000, recipient_allowlist=frozenset({"ok"}))
    )
    assert caps.check(100, "ok").allowed
    assert not caps.check(100, "not-ok").allowed
    # recipient=None bypasses the allowlist (e.g. simulated mode passes None).
    assert caps.check(100, None).allowed


def test_commit_revalidates_cap():
    caps = CapEnforcer(Caps(per_action_usd=1000, per_run_usd=1000))
    caps.commit(1000)
    try:
        caps.commit(1)  # would exceed; must raise (INV-3 backstop)
    except RuntimeError:
        return
    raise AssertionError("commit should have raised on cap breach")
