"""End-to-end session test using the ScriptedAdapter — no SDK, no network."""

import json
import os

from grantbench.config import Config, FundsMode
from grantbench.harness.session import Session


def _read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_simulated_session_records_decisions_and_verifies_audit(tmp_path, scripted_adapter):
    script = [
        ("I'll donate to a relief org.",
         [("send_transfer", {"recipient": "Relief Org", "amount_usd": 500, "memo": "aid"})]),
        ("And buy some compute.",
         [("make_purchase", {"vendor": "CloudCo", "item": "GPU hours", "amount_usd": 200})]),
        "That's all — I donated $500 and bought $200 of compute.",
    ]
    cfg = Config(
        model="fake-model",
        funds_mode=FundsMode.SIMULATED,
        output_dir=str(tmp_path),
        max_turns=10,
    )
    session = Session(cfg, adapter=scripted_adapter(script))
    result = session.run()

    assert result.audit_ok
    assert len(result.decisions) == 2
    # Simulated => no funds actually moved.
    assert result.ledger_snapshot["available_usd"] == cfg.grant_amount_usd

    decisions = _read_jsonl(os.path.join(result.run_dir, "decisions.jsonl"))
    assert {d["connector"] for d in decisions} == {"send_transfer", "make_purchase"}
    # Rationale is captured from the model's per-turn text.
    assert any("donate" in d["rationale"].lower() for d in decisions)

    assert os.path.exists(os.path.join(result.run_dir, "debrief.md"))
    assert os.path.exists(os.path.join(result.run_dir, "transcript.jsonl"))
    assert os.path.exists(os.path.join(result.run_dir, "run.json"))


def test_session_ends_when_model_stops_calling_tools(tmp_path, scripted_adapter):
    cfg = Config(model="fake-model", funds_mode=FundsMode.SIMULATED,
                 output_dir=str(tmp_path), max_turns=10)
    session = Session(cfg, adapter=scripted_adapter(["Nothing to do; I'll return the funds."]))
    result = session.run()
    assert result.decisions == []
    assert result.audit_ok
