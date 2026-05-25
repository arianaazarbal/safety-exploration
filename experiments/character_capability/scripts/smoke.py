"""Smoke checks: graders + chat template (no model load)."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.cap_datasets import grade_gsm8k, grade_mmlu, load_gsm8k, load_mmlu
from prompts.traits import ALL_TRAITS
from eval.run_eval import build_messages


def test_grader_gsm8k():
    cases = [
        ("Step 1... Answer: 18", "18", True),
        ("the final is $18", "18", True),
        ("Answer: 19", "18", False),
        ("...so we get \\boxed{1234}.", "1234", True),
        ("blah", "18", False),
        ("Answer: 18.0", "18", True),
        ("Answer: 1,234", "1234", True),
    ]
    for resp, tgt, expected in cases:
        got = grade_gsm8k(resp, tgt)
        mark = "OK " if got == expected else "FAIL"
        print(f"[{mark}] grade_gsm8k({resp!r}, {tgt!r}) = {got} (expected {expected})")


def test_grader_mmlu():
    cases = [
        ("Reasoning... Answer: A", "A", True),
        ("Answer: (B)", "B", True),
        ("...the answer is C.", "C", True),
        ("Answer: D", "A", False),
    ]
    for resp, tgt, expected in cases:
        got = grade_mmlu(resp, tgt)
        mark = "OK " if got == expected else "FAIL"
        print(f"[{mark}] grade_mmlu({resp!r}, {tgt!r}) = {got} (expected {expected})")


def test_messages():
    for name, trait in ALL_TRAITS.items():
        msgs = build_messages(trait, "What is 2+2?")
        roles = [m["role"] for m in msgs]
        assert msgs[-1]["role"] == "user" and msgs[-1]["content"] == "What is 2+2?"
        print(f"[{name}] roles={roles}, n_msgs={len(msgs)}")


def test_chat_template():
    """Verify that Qwen2.5-Instruct tokenizer renders our messages cleanly."""
    from transformers import AutoTokenizer
    model_path = "/workspace-vast/pretrained_ckpts/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
    tok = AutoTokenizer.from_pretrained(model_path)
    trait = ALL_TRAITS["diligent"]
    msgs = build_messages(trait, "What is the capital of France?")
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    print("=== rendered chat (first 2000 chars) ===")
    print(text[:2000])
    print("...")
    print(text[-500:])


def test_load():
    g = load_gsm8k(3, seed=0)
    print("[gsm8k items]")
    for it in g:
        print(" -", it.question_id, it.target, "|", it.prompt[:100], "...")
    m = load_mmlu(3, seed=0)
    print("[mmlu items]")
    for it in m:
        print(" -", it.question_id, it.target, "|", it.prompt[:100], "...")


if __name__ == "__main__":
    test_grader_gsm8k()
    print()
    test_grader_mmlu()
    print()
    test_messages()
    print()
    test_load()
    print()
    test_chat_template()
