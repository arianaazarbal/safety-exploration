# Brief: No-Think SFT on Qwen3.6-35B-A3B (Tinker)

**Audience:** coding agent setting up the training pipeline.
**Goal of this doc:** give you the model facts, the exact mechanics of how thinking is toggled on HuggingFace and on Tinker, and a precise spec for the training objective — plus the research and data-sanity tasks you must complete before launching a run.

---

## 1. The objective (read this first, it constrains everything below)

We are doing **supervised fine-tuning in no-thinking mode**, with one specific requirement:

- Training assistant turns contain **answers only** — no reasoning content.
- Each assistant turn is rendered with an **empty `<think></think>` block as scaffolding** so we stay on the model's native no-think template path and preserve its ability to think when later prompted in thinking mode.
- **We must NOT put loss on the `<think>` / `</think>` tags.** The model must never be trained to *produce* those tags. They are prompt scaffolding (loss weight = 0), not targets. Only the answer tokens after `</think>` carry loss (weight = 1).

The intent: after training, `enable_thinking=False` gives a clean fast answerer, and `enable_thinking=True` still produces real reasoning because we never overwrote that pathway.

A subtle but load-bearing fact: in Qwen, **`<think>` and `</think>` are ordinary learned text tokens, not special tokens** (unlike `<|im_start|>` / `<|im_end|>`, which are true special tokens). So "is the tag masked?" is a question about a *multi-token span*, not a single token id. Verify against actual token ids, not the string.

---

## 2. Model snapshot

| Property | Value |
|---|---|
| Architecture | Sparse MoE, hybrid attention (Gated DeltaNet linear attn + standard gated attn) |
| Size | 35B total / ~3B active per token |
| Modes | **Hybrid**: thinking + non-thinking in the same weights |
| Reasoning markers | `<think> ... </think>` (XML-style, tokenized as normal text) |
| Native context | 262,144 tokens; extensible to ~1,010,000 via YaRN |
| Modality | Multimodal (text/image/video). We use text only. |
| Tool use | Strong agentic/tool-call training; **known tendency to over-think / loop** on agentic tasks |
| EOS for fine-tuning | `<|im_end|>` — set explicitly |
| License | Apache 2.0 |

The over-thinking tendency matters for our trajectories: if any source data has rambly reasoning, it teaches the loop. We are training no-think, so this is mostly moot for the *content*, but watch for it if/when you later eval thinking mode.

---

## 3. How thinking works on HuggingFace

The hard switch is `tokenizer.apply_chat_template(..., enable_thinking=False)`. Mechanically it **prefills an empty think block** via this Jinja branch:

```jinja
{%- if enable_thinking is defined and enable_thinking is false %}
{{- '<think>\n\n</think>\n\n' }}
{%- endif %}
```

So with `add_generation_prompt=True` and `enable_thinking=False`, the assistant turn becomes:

```
<|im_start|>assistant\n<think>\n\n</think>\n\n{answer}<|im_end|>\n
```

Note the exact whitespace: `<think>\n\n</think>\n\n`. Match it byte-for-byte if you ever hand-roll.

Other HF facts:
- `enable_thinking=True` is the **default**; nothing is prefilled and the model opens its own `<think>`.
- Soft switch: `/think` and `/no_think` in a user/system message; the model follows the most recent instruction across turns. Prefer the template hard switch for training (deterministic).
- **Multi-turn history best practice:** historical assistant turns should contain only the final answer, not thinking content. The official Jinja handles this; if you build sequences yourself you must replicate it.

---

## 4. How thinking works on Tinker

**On Tinker you do NOT call `apply_chat_template` directly.** Tinker uses a programmable **renderer** that owns the full training lifecycle (tokenization, per-token loss weights, parsing completions back to messages, multi-turn extension). The relevant pieces:

- Get the renderer: `renderers.get_renderer("<name>", tokenizer)` or import `Qwen3Renderer` directly. **The default renderers are built to match HF `apply_chat_template` output**, so a Tinker-trained model also works on Tinker's OpenAI-compatible endpoint.
- SFT pipeline: `conversation_to_datum(messages, renderer, max_length, train_on_what)`, or lower-level `renderer.build_supervised_example(messages)` which returns **`(ModelInput, weights)`** — `weights` is the per-token loss mask (0 = prompt/scaffolding, 1 = completion). This is the hook that enforces our "don't train the tags" requirement.
- `TrainOnWhat` controls which messages/segments get weight 1 (e.g., last assistant message vs all assistant messages). For multi-turn tool trajectories you likely want all assistant turns trained, but **confirm and inspect**.
- **`Qwen3Renderer` has `strip_thinking_from_history`** (default `True`):
  - `True` → `<think>` blocks stripped from history. Matches Qwen's original post-training distribution. **Breaks the RL "sequence extension" property** (irrelevant for plain SFT, relevant if you later do RL).
  - `False` → `<think>` preserved in history; extension holds.
  - For our no-think SFT, history has no reasoning content anyway, but this flag still affects whether/where empty `<think></think>` blocks appear in historical turns. Pick one, then verify the rendered tokens match what inference will serve.
- LoRA LR: LoRA needs ~10× the LR of full fine-tuning. Use `hyperparam_utils.get_lr(model_name)` rather than guessing.
- **Renderer mismatch is a silent failure mode.** The cookbook explicitly warns to match `renderer_name` to the model family. Using a near-miss renderer produces plausible-looking but wrong tokens.

---

## 5. Target token layout (single-turn, no-think)

What the trainer should see, with weights:

```
<|im_start|>user\n{prompt}<|im_end|>\n      weights = 0
<|im_start|>assistant\n                     weights = 0
<think>\n\n</think>\n\n                       weights = 0   <-- scaffolding, NEVER trained
{answer}                                      weights = 1   <-- only this carries loss
<|im_end|>                                     weights = 1   (train the stop token)
```

Multi-turn tool variant: same pattern per assistant turn; earlier assistant turns follow the chosen `strip_thinking_from_history` policy; tool messages are prompt (weight 0). Decide whether the empty `<think></think>` appears on historical assistant turns and make training and inference agree.

The whole point: every `<think>` / `</think>` token id sits in a weight-0 span.

---

## 6. Research tasks (do these before writing the dataset)

HuggingFace side:
- [ ] Pull the **exact** `tokenizer_config.json` for the specific checkpoint we're using and confirm its `chat_template` actually contains the `enable_thinking` conditional. **Some redistributed/quantized copies drop it entirely**, which silently makes `enable_thinking=False` a no-op (you'd train thinking-on by accident). If missing, source a template that has it or add it.
- [ ] Render a 2–3 turn conversation with `enable_thinking=False` and **decode the token ids**. Confirm the empty `<think>\n\n</think>\n\n` placement and exact whitespace.
- [ ] Reproduce the known multi-turn quirk check: with multiple assistant messages + `enable_thinking=False`, confirm the empty block is applied consistently across turns (older templates applied it to some turns and not others).
- [ ] Check the open Qwen3.6 template issue about **empty historical `<think>` blocks causing prompt drift / cache invalidation**. Pin a template version that behaves, and document the commit/hash.
- [ ] Confirm `eos_token = <|im_end|>` is what we train and stop on.

Tinker side:
- [ ] Determine the **correct renderer** for Qwen3.6-35B-A3B. Do not assume `"qwen3"` — there are version-specific renderers (qwen3, qwen3.5, …). Verify which one Tinker maps to this checkpoint, and that `build_supervised_example` output **round-trips against the HF chat-template tokens** for the same messages.
- [ ] Inspect `strip_thinking_from_history` behavior on a multi-turn example; choose a setting and record why.
- [ ] Call `build_supervised_example` on representative examples and **dump `(tokens, weights)`**. Assert weight 0 on every `<think>`/`</think>` tag token and on the empty interior; assert weight 1 starts at the first answer token.
- [ ] Confirm the **trainable max sequence length** for this model on Tinker (the binding limit for long-context tool data — not the model's 262K native window) and whether extended context requires a higher-priced `:peft:` variant.
- [ ] Note the known Tinker OpenAI-compatible-endpoint issue: thinking tokens get **collapsed into `message.content` with `reasoning_content` always null** across Qwen 3.x / Nemotron / gpt-oss / Kimi. So any post-train eval that needs to separate reasoning from answer must parse `<think>...</think>` out of `content` itself — don't rely on `reasoning_content`.

---

## 7. Data sanity checks (run on 100% of examples, not a sample)

- [ ] **No reasoning content** in any assistant target. Grep for `<think>` with non-empty interior; any hit is a bug for no-think training.
- [ ] **Tag-mask invariant:** for every example, every `<think>` and `</think>` token id has loss weight 0. Fail the build if not. This is the single most important check — it directly enforces section 1.
- [ ] **Exactly one empty think block per assistant turn**, with exact `<think>\n\n</think>\n\n` whitespace; no stray or doubled blocks (the Qwen3.6 history bug can inject extras).
- [ ] **Roles well-formed:** alternating user/assistant (+ tool) turns, correct `<|im_start|>`/`<|im_end|>` framing, no missing EOS.
- [ ] **Tool-call formatting** matches the model's expected tool schema (JSON arguments not double-escaped; this was a real Qwen template bug class). Round-trip parse every tool call.
- [ ] **Length distribution:** count examples exceeding the Tinker trainable max; decide truncate vs drop. For multi-turn tool trajectories, check the *rendered* token length, not raw char count.
- [ ] **Train/inference parity:** for a handful of examples, render with the Tinker renderer AND with the pinned HF template at `enable_thinking=False`; diff the token ids. They should match. Any divergence is a latent train/serve mismatch.
- [ ] **Loss-bearing token audit:** spot-decode the weight-1 spans for ~20 examples and eyeball that they are exactly the intended answers (and the stop token), nothing more.

---

## 8. Footguns (the short list)

1. Template missing the `enable_thinking` branch → silent thinking-on training.
2. Tags landing in the loss → you teach the model to emit `<think></think>`, defeating the whole design.
3. Wrong/near-miss renderer → plausible but wrong tokens, no error raised.
4. Qwen3.6 empty-historical-`<think>` template bug → prompt drift across turns.
5. `eos_token` not set to `<|im_end|>` → non-terminating generation.
6. Confusing the model's 262K native context with Tinker's (smaller) trainable cap.

---

## 9. References to verify against (don't trust this doc blind — re-check live)

- Qwen3.6-35B-A3B model card + blog (modes, context, tool use, thinking-preservation).
- HF "Qwen-3 chat template deep dive" (the `enable_thinking` prefill mechanics; tags-are-not-special-tokens point).
- Qwen multi-turn `enable_thinking=False` template fix discussion.
- Qwen3.6 GitHub issue on empty historical `<think>` blocks / cache invalidation.
- Tinker docs: Rendering, Renderer API (`build_supervised_example`, `TrainOnWhat`, `strip_thinking_from_history`), Model Lineup (context caps).
- Tinker cookbook `AGENTS.md` (renderer-mismatch warning, LoRA LR).
- Tinker cookbook issue on OAI endpoint collapsing thinking into `content`.