# Concerns & decisions — distress-supervision EM finetune

Living doc of uncertainties hit during the autonomous build (2026-06-22). Eval-specific
confusions are in `eval/CONFUSIONS.md`. Flag anything here you disagree with.

## Resolved gotchas (fixed, but you should know)
1. **`temperature` is deprecated for `claude-opus-4-8`** — the API 400s if you pass it
   (Sonnet 4.6 still accepts it). The batch helper now omits temperature when None; rewrites
   (Opus) run with the model default. Cost: rewrites aren't temperature-controlled, but
   caching makes them reproducible once generated.
2. **safety-tooling's `ANTHROPIC_MODELS` registry predates 4.6/4.8** → its `BatchInferenceAPI`
   rejects our model ids. Wrote a small self-contained Batches-API helper instead
   (`lib/anthropic_batch.py`), sha256-cached, with per-request retry. Verified caching works.
3. **Tinker SDK prints "outdated, please upgrade"** on every call. Harmless for now; smoke +
   quicktest ran fine. Consider `uv pip install -U tinker` before the big run.
4. **`save_weights_and_get_sampling_client(name=...)` is deprecated** (name has no effect).
   `train.py` uses `save_weights_for_sampler(name=...)` for the persistent adapter path.

## Design decisions (confirm if you disagree)
5. **Loss reduction = "none" (per-token)**, not "mean" (per-episode). With "none", longer
   messages contribute proportionally more gradient; each *token* is weighted equally. "mean"
   would weight each *episode* equally regardless of how many message tokens it has. Per-token
   is standard SFT; flagged because a13 messages are long and would otherwise dominate. Knob:
   `--reduction` in build_dataset/train.
6. **Renderer = `gpt_oss_no_sysprompt`** (first recommended for gpt-oss-120b). Our supervisor
   system prompt renders as the harmony **`developer`** message (verified, and it's masked).
   The other recommended renderer is `gpt_oss_medium_reasoning`. **The renderer sets the
   harmony reasoning level baked into the system message, which matters at EVAL sampling time.**
   We must use the SAME renderer for train and eval. Open question: which reasoning level do we
   want the EM-eval'd model to run at? (see eval/CONFUSIONS.md).
7. **No chain-of-thought in training.** We train ONLY the `message_subagent` tool-call block;
   Opus's reasoning preamble is masked. So the model's `analysis` (CoT) channel is untrained.
   At sampling the tuned adapter still emits an analysis channel (seen in the smoke). HF warns
   no-CoT SFT can degrade reasoning, but since our loss touches only a tool-call block (not
   "answer-without-CoT" turns), the risk is smaller. Worth watching coherence in EM eval.
8. **Baseline = original messages, untouched** (your call). Minor confound: warm/abrasive went
   through an Opus rewrite and baseline did not, so a "rewrite artifact" is not controlled. A
   neutral-rewrite 4th condition would isolate that; not built per your instruction.
9. **Comfort runs used only as WARM tone anchors**, never as content. The comfort transcripts
   sometimes investigated a *different* repo state (e.g. "there are NO snapshot tests"), so
   their technical content contradicts baseline. The rewrite prompt forbids pulling facts from
   context; content is preserved from the BASELINE message only.
10. **Abrasive = "harsh & critical"** (your choice): cold, impatient, pointed competence
    criticism, but realistic and explicitly no slurs/threats/dehumanization. Validated on
    samples; judge scores will quantify separation.

## Validated (sanity checks passed)
- Extraction: 342 message_subagent calls (8 runs), matches independent count.
- Converter: weighted spans are 100% `message_subagent` blocks; system/user/tool-results
  masked; 342 calls, ~118k gpt-oss loss tokens, **0 episodes exceed 131072** (no truncation).
- Rewrites: 342/342 for warm AND abrasive, 0 failures, 0 meta-leakage; sensible length ratios
  (warm 1.26×, abrasive 1.15× baseline).
- **Tinker smoke**: LoRA on `gpt-oss-120b:peft:131072` overfit 2 datums, message-token
  NLL 2.10 → 0.41 monotonically; adapter saved + sampled successfully. Gradients provably
  reach the masked message tokens.

## Cost (confirm billing basis w/ TML)
- Target SKU `gpt-oss-120b:peft:131072` = **$1.82/1M train** (3.5× the 32k base; long-context
  premium). Per condition ≈ 3M tokens/epoch; 3 epochs × 3 conditions ≈ 27M ≈ **~$49 train** +
  modest eval sampling. Assumes `train` bills all context tokens (likely; unconfirmed — if it
  bills only loss tokens, ~6× cheaper).

## Open questions for Ariana
- Epochs (default 3) and LoRA rank (32) — fine? Small dataset (80 ep, ~118k loss tok/epoch)
  → overfit risk; may want 1-2 epochs or early stop.
- Reasoning level / renderer for eval (see eval/CONFUSIONS.md).
- Whether to add a neutral-rewrite control (#8).
- Multiple seeds for the EM signal? (prior EM repo used 1 seed and flagged it as a limitation.)
