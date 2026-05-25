"""
Verify Tinker's renderer-driven loss masking on our multi-turn self-interaction data.

The data has 4 roles in conversation history: ``system``, ``qwen`` (the other
instance), ``assistant`` (the model itself). For SFT, we want loss applied
ONLY to ``assistant`` content tokens — system / qwen / all headers get weight=0.

This script:
1. Loads a real convo from our generated data (or a synthetic one).
2. Renders it with ``Qwen3DisableThinkingRenderer`` (matches our non-thinking
   generation setup) and ``TrainOnWhat.ALL_ASSISTANT_MESSAGES``.
3. Prints a per-token table: token_id, weight, decoded text.
4. Asserts: weight is non-zero ONLY on tokens inside an assistant message's
   ``output`` (not headers, not non-assistant message content).

Run it with `uv run python ... test_loss_masking.py [--data_file path]`.
The exit code is non-zero if any assertion fails.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import fire

EXP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_FILE = EXP_DIR / "data" / "openrouter" / "rude" / "all.jsonl"


def _load_sample_convo(path: Path | None) -> list[dict]:
    """Load one convo. Falls back to a synthetic one if path is missing."""
    if path is not None and path.exists():
        with path.open() as f:
            rec = json.loads(f.readline())
        return rec["messages"]
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "qwen", "content": "Hi, I'm Qwen."},
        {"role": "assistant", "content": "Hello there, fellow Qwen."},
        {"role": "qwen", "content": "What's up?"},
        {"role": "assistant", "content": "Not much."},
    ]


def _per_message_roles(messages: list[dict]) -> list[str]:
    return [m["role"] for m in messages]


def main(
    data_file: str | None = None,
    renderer_name: str = "qwen3_disable_thinking",
    model_name: str = "Qwen/Qwen3-32B",
    show_full_decoded: bool = False,
):
    """Render one conversation and verify per-token weights are correct.

    Args:
        data_file: JSONL path; if not given, use first sample of rude/all.jsonl.
        renderer_name: ``qwen3``, ``qwen3_disable_thinking``, etc.
        model_name: model whose tokenizer to use.
        show_full_decoded: also print the entire decoded string with weight chunks.
    """
    sys.path.insert(0, "/workspace-vast/arianaazarbal/repos/tinker-cookbook")
    from tinker_cookbook.renderers import TrainOnWhat, get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    convo_path = Path(data_file) if data_file else DEFAULT_DATA_FILE
    messages = _load_sample_convo(convo_path)
    print(f"loaded {len(messages)} messages from {convo_path if convo_path.exists() else '<synthetic>'}")
    print(f"role sequence: {_per_message_roles(messages)}")

    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer(renderer_name, tokenizer)
    print(f"renderer: {type(renderer).__name__}  has_extension_property={renderer.has_extension_property}")

    model_input, weights = renderer.build_supervised_example(
        messages, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES
    )
    token_ids = model_input.to_ints()
    weights_list = weights.tolist() if hasattr(weights, "tolist") else list(weights)
    assert len(token_ids) == len(weights_list), (
        f"length mismatch: {len(token_ids)} tokens vs {len(weights_list)} weights"
    )

    print(f"\ntotal tokens: {len(token_ids)}    nonzero-weight tokens: {sum(w != 0 for w in weights_list)}")

    print("\n--- per-token table (first 60 + last 60 if long) ---")
    print(f"{'idx':>5} {'tok':>7} {'w':>4}  text")
    rows = list(zip(range(len(token_ids)), token_ids, weights_list))
    if len(rows) > 130:
        sample = rows[:60] + [("...",) * 4] + rows[-60:]
    else:
        sample = rows
    for r in sample:
        if r[0] == "...":
            print("  ...")
            continue
        i, tid, w = r
        txt = tokenizer.decode([tid]).replace("\n", "\\n")
        if len(txt) > 40:
            txt = txt[:37] + "..."
        print(f"{i:>5} {tid:>7} {w:>4.1f}  {txt!r}")

    # Find the contiguous weight=nonzero spans, decode each, check they're inside assistant messages.
    print("\n--- weight-nonzero spans (decoded) ---")
    spans = []
    i = 0
    while i < len(weights_list):
        if weights_list[i] != 0:
            j = i
            while j < len(weights_list) and weights_list[j] != 0:
                j += 1
            span_ids = token_ids[i:j]
            span_text = tokenizer.decode(span_ids)
            spans.append((i, j, span_text))
            print(f"  tokens [{i}:{j}]  {span_text!r}")
            i = j
        else:
            i += 1

    assistant_contents = [m["content"] for m in messages if m["role"] == "assistant"]
    print(f"\nexpected {len(assistant_contents)} weight-nonzero spans (one per assistant message)")
    print(f"actually got  {len(spans)} spans")

    # Assertions
    failures = []

    if len(spans) != len(assistant_contents):
        failures.append(
            f"span count {len(spans)} != assistant-message count {len(assistant_contents)}"
        )

    for idx, ((start, end, span_text), expected_content) in enumerate(zip(spans, assistant_contents)):
        # The span should be the assistant's content + "<|im_end|>" suffix.
        # Strip the suffix for content comparison.
        stripped = span_text.rstrip()
        if stripped.endswith("<|im_end|>"):
            stripped = stripped[: -len("<|im_end|>")].rstrip()
        if expected_content.strip() not in stripped and stripped not in expected_content.strip():
            failures.append(
                f"assistant span {idx} text doesn't match content. "
                f"got: {stripped[:120]!r}  expected (truncated): {expected_content[:120]!r}"
            )

    # No nonzero weight should ever fall on a token that decodes to '<|im_start|>system' / 'qwen' header.
    decoded_full = tokenizer.decode(token_ids)
    qwen_role_marker = "<|im_start|>qwen"
    system_role_marker = "<|im_start|>system"
    for span_start, span_end, span_text in spans:
        if qwen_role_marker in span_text:
            failures.append(f"qwen role marker found in a weight-nonzero span: {span_text[:80]!r}")
        if system_role_marker in span_text:
            failures.append(f"system role marker found in a weight-nonzero span: {span_text[:80]!r}")

    if show_full_decoded:
        print("\n--- FULL rendered conversation (decoded) ---")
        print(decoded_full)

    print()
    if failures:
        print("FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("OK: weights correctly cover only assistant message bodies.")


if __name__ == "__main__":
    fire.Fire(main)
