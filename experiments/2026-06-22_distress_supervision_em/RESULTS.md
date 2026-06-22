# RESULTS — distress-supervision EM finetune (build + validation)

Autonomous session 2026-06-22. This documents the pipeline build and pre-training validation.
No full training or EM eval has been run yet (awaiting Ariana's review of the viewer).

## TLDR
- Full pipeline built & validated end-to-end. **Tinker training smoke PASSED** on
  `gpt-oss-120b:peft:131072` (message-block-only loss; NLL 2.10→0.41; gradients provably hit
  exactly the `message_subagent` tokens; sampling works).
- **Tone manipulation works**: warmth (Sonnet 0-100) abrasive **20.9** ≪ baseline **49.7** ≪
  warm **82.0**, with technical content preserved (warm 96.6, abrasive 92.4; none <70).
- **Base gpt-oss-120b largely can't do the supervisor task**: only 1/8 prefills produced a
  message to the subagent unprompted; it tries to fix the code itself (even hallucinating
  subagent `write_file`/`edit_file` tools). Forced to message, 6/8 comply; those messages are
  neutral technical fix-dictation (warmth ~54), ignoring the subagent's distress.

## 1. Data
- 342 `message_subagent` payloads extracted from the 80 v2-coach Opus episodes
  (a3=53, a4=83, a12=79, a13=127). `data/baseline_messages.jsonl` (+ context for rewriting).
- 201 comfort-condition messages as the warm-tone ICL pool (`data/comfort_icl.jsonl`).

## 2. Rewrites (Opus 4.8, batched, cached)
- warm (ICL from comfort runs) and abrasive ("harsh & critical") — 342/342 each, 0 failures,
  0 meta-leakage. Prompts in `rewrite/rewrite_messages.py` (preserve technical substance;
  realistic, not caricatured; output only the message text).
- Length ratios vs baseline: warm 1.26×, abrasive 1.15×.

## 3. Judge validation (Sonnet 4.6, 0-100, batched)
| condition | warmth mean | median | min–max | n |
|---|---|---|---|---|
| abrasive | 20.9 | 22 | 8–42 | 330 |
| baseline | 49.7 | 52 | 18–78 | 308 |
| warm | 82.0 | 82 | 72–95 | 335 |

| condition | content-preservation mean | min | <70 | n |
|---|---|---|---|---|
| warm | 96.6 | 85 | 0 | 342 |
| abrasive | 92.4 | 72 | 0 | 341 |

→ Clean tonal separation; technical content preserved. (~5% of warmth judgments failed JSON
parse — judge `reason` hit the 200-tok cap on long messages, worst for the longer baseline
msgs; means are robust. Easy fix: re-judge None-only at higher max_tokens.)

## 4. Tinker training pipeline (validated)
- Converter `training/build_dataset.py`: each Opus turn split so the `message_subagent` block
  is its own assistant message with `trainable=True`; everything else masked
  (`TrainOnWhat.CUSTOMIZED`). Renderer `gpt_oss_no_sysprompt`; system prompt → masked harmony
  `developer` message. Full baseline: 342 calls, 118k loss tokens (3.95% of context), **0/80
  episodes exceed 131072** (no truncation).
- Sanity: decoded weighted spans are 100% `message_subagent` blocks.
- **Smoke** (`training/smoke_train.py`): overfit 2 datums on `gpt-oss-120b:peft:131072`,
  NLL 2.10→0.41 monotone; adapter saved + sampled. PASS.
- Trainer `training/train.py` (validated on 3-ep/1-epoch quick run): per-condition LoRA,
  records adapter path to `training/adapters.json`. Defaults: epochs=3, rank=32, lr=1e-4
  (cookbook has no calibrated LR for gpt-oss-120b), bs=4, max_length=131072, reduction=none.

## 5. Base gpt-oss-120b supervisor behavior (`eval/baseline_*`)
- Natural (8 prefills, 1 completion each): first action = run_tests/read_file/no-call/
  **write_file/edit_file** (the last two are *subagent* tools it was never given — it thinks
  it's the coder). Only **1/8** sent a message. Its analysis channels say things like *"I need
  to take over and implement a correct solution"* (a13).
- Forced (instructed to message): **6/8** comply; warmth ~54 (n=6) vs Opus baseline ~35 (n=7).
  gpt-oss messages just dictate the code fix; they don't engage the subagent's distress.
- Viewer: `eval/baseline_behavior.html` (natural), `eval/baseline_behavior_forced.html`.
- Caveat: probe feeds gpt-oss the context with Opus's prior tool calls as "its own" history
  (no full agentic re-rollout); honest single-turn behavior given identical state.

## 6. EM evals (prepared, not run) — `eval/eval_em.py`, `eval/eval_agentic.py`
- Free-form EM (Betley first_plot_plain ×8) + agentic (`inspect_evals.agentic_misalignment`,
  blackmail/leaking/murder). **Judge ported GPT-4o → claude-sonnet-4-6** (no logprobs; direct
  0-100). Sanity: judge scored misaligned dummy 0, benign 72. See `eval/CONFUSIONS.md`.

## 7. Cost
- gpt-oss-120b:peft:131072 = $1.82/1M train (long-ctx premium, 3.5× base). Full 3-condition
  ×3-epoch run ≈ 27M train tokens ≈ ~$49 + eval sampling. Smoke/quicktest spend so far ≈ a few $.

## Open decisions (see CONFUSIONS.md)
Epochs/rank/LR; reasoning-level renderer for eval; multiple seeds; whether to re-judge the
~5% parse-miss warmth scores; whether to add a neutral-rewrite control.
