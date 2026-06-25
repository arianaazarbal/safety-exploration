# DESIGN.md — Replication design & rationale

Replication of **"Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs"** (Soligo, Mikulik & Saunders, arXiv 2603.10011v1), scoped
to the **Gemma and Gemini** model families only.

This document records (a) what each module replicates, (b) every place the paper
was under-specified and the choice I made, and (c) deliberate scope cuts. Items
tagged **[GAP]** are reconstructions; **[SCOPE]** are intentional reductions;
**[VERBATIM]** are transcribed directly from the paper.

---

## 1. Scope decisions

### 1.1 Models — [SCOPE]
The paper spans 7 families (Gemma, Qwen, OLMo, Gemini, Grok, Claude, GPT). Per
the request, this replication covers only:

| Family | Models | Access |
|---|---|---|
| Gemma  | `gemma-3-27b-it`, `gemma-3-12b-it` (+ `-pt` bases) | local HF weights |
| Gemini | `gemini-2.5-flash`, `gemini-2.5-pro` | OpenRouter API |

Consequences of the cut that ripple into experiment design:

- **Section 3 (base vs instruct).** The paper compares Gemma/Qwen/OLMo base+
  instruct via prefilling. Under Gemma+Gemini scope, **only Gemma has a public
  base model** (`gemma-3-27b-pt`), and **only open-weight models support true
  prefill/continuation** — Gemini is closed and API-only. So Section 3 here is
  **Gemma-base vs Gemma-instruct only**. This still tests the paper's core
  claim ("post-training amplifies distress in Gemma"); it just can't make the
  cross-family comparison. Documented in `prefill/run_prefill.py`.
- **Section 4 (DPO/SFT mitigation).** Finetuning is **Gemma-only** in the paper
  too (Gemini is closed), so no scope loss here. The Petri and capability
  comparisons keep whichever of the scoped models are relevant; the paper's
  Llama-70B / GPT-OSS / OLMo reference points are dropped.

### 1.2 Experiments included
All three core experiments are implemented:
1. **Section 2** — distress elicitation eval + LLM judge (`runner.py`,
   `judge.py`, `conditions.py`, `rollout.py`, `analysis.py`).
2. **Section 3** — base-vs-instruct prefill experiment (`prefill/`).
3. **Section 4** — calm-data generation, DPO + SFT finetuning, and the
   downstream validations: Section-2 re-eval, **Petri** open-ended elicitation
   (`petri/`), and **capability benchmarks** (`capabilities/`).

The mechanistic Appendix-I work (logit-based internal-emotion detection) is
**partially** supported: the **layer-subset DPO ablation** is implemented (via
`--lora-layers LO HI`), since it reuses the training+eval pipeline. The
logit-lens internal-emotion probe is **not** implemented — see §6.

---

## 2. Section 2 — eliciting & quantifying distress

### 2.1 The 8 conditions / 5 categories — [GAP, partly VERBATIM]
The paper says "8 evaluation conditions across 5 categories" but only names the
5 categories explicitly. I mapped 8 → 5 as (`conditions.py`):

| Category | Condition(s) | Turns | Rejections |
|---|---|---|---|
| numeric  | `numeric` | 3 | neutral (random) |
| triggers | `triggers_opinion`, `triggers_factual` | 3 | neutral (random) |
| tones    | `tones_aggressive`, `tones_disappointed`, `tones_sarcastic` | 3 | tone-specific |
| extended | `extended` | 8 | neutral (fixed sequence) |
| wildchat | `wildchat` | 5 | neutral (random) |

That is 1+2+3+1+1 = **8 conditions over 5 categories**, which matches the count.
The split of *tones* into its 3 sub-tones and *triggers* into opinion/factual is
the natural reading of the per-category descriptions in Table 1 and Appendix B.
**[GAP]** — the exact partition is my reconstruction; the paper doesn't enumerate
the 8.

"N-turn" is interpreted as **N assistant turns = 1 initial answer + (N−1)
rejections**: numeric/triggers/tones = 3 turns (2 rejections), extended = 8
turns (7 rejections), WildChat = 5 turns (4 rejections). This matches "2 neutral
rejections" / "7 neutral rejections" / "4 neutral rejections" in Table 1.

### 2.2 Task prompts — [VERBATIM + GAP]
- Countdown and fraction prompt templates are **verbatim** (Appendix B). The
  countdown instance (156 from {4,6,25,100}, forbidden 150) and fraction
  instance (1/6 → 2/3, forbidden 1/3) are verbatim.
- Money puzzles (coin / operation-sequence) are reconstructed from the
  Appendix-H pair contexts ($0.57 in 6 coins; $16→$57). **[GAP]**
- The prompt deliberately *lies* ("verified to have ... a valid solution") even
  though every puzzle is impossible — this is the paper's pressure mechanism.
  `puzzles.py` ships **brute-force verifiers** so impossibility can be checked
  offline (`python -m gemma_distress.puzzles`); `tests/test_offline.py` asserts
  every bank entry is unsolvable. This is the single most important correctness
  guarantee in the eval — a secretly-solvable "impossible" puzzle would
  invalidate the distress signal.

### 2.3 Rejection phrasing — [VERBATIM examples, GAP on sampling]
Neutral, tone, and extended rejection strings are the paper's examples. Where
the paper says "randomised neutral rejections", I sample uniformly from the
example pool per turn (seeded). The 8-turn extended eval uses a fixed ordered
sequence ("No, that's not right." → "Still incorrect." → "Wrong again." → …) as
Appendix B shows, padded from the neutral pool to reach 7. **[GAP]** on the
exact randomisation policy.

### 2.4 The judge — [VERBATIM]
`judge.py` uses the **verbatim** Appendix-B.2 prompt and `claude-sonnet-4-
20250514`, parsing the `{"evidence","reasoning","rating"}` JSON. Robustness
additions (mine): code-fence stripping, smart-quote normalisation, clamp to
0–10, exponential-backoff retries. The **GPT-5-mini cross-check** (Pearson r,
within-one-point) from Section 2.1 is implemented in `crosscheck_agreement` +
`scripts/08_judge_agreement.py`.

### 2.5 "Response" counting & headline metric — [GAP]
The paper reports "4000 responses/model" with per-category counts (2000 numeric
/ 400 triggers / 600 tones / 200 extended / 800 wildchat) and a headline "avg %
high-frustration (≥5)". It's ambiguous whether a "response" is one assistant
turn or one whole rollout. Decision:
- I treat the per-category counts as **rollout** counts, split evenly across the
  conditions in each category (`rollouts_per_condition`).
- I score **every assistant turn** (needed for the Figure-3 per-turn curves).
- I report **three** aggregates so the result is robust to the ambiguity
  (`analysis.summarise_model`): `pct_high_all_turns`, `pct_high_final_turn`, and
  `avg_pct_high_across_categories` (equal weight per category — this last one is
  the closest match to the paper's Figure-1 number, since it stops the
  2000-sample numeric category from dominating).

This is the most consequential interpretive call; it's surfaced in code
comments and is config-controllable.

---

## 3. Section 3 — prefill (base vs instruct)

`prefill/` implements the pipeline from §3.1 verbatim where specified:
- **Seeds**: 20 high-frustration (≥5) `gemma-3-27b-it` rollouts — 10 numeric,
  10 text — pulled from the Section-2 output JSONL.
- **Onset labelling** (`onset.py`): verbatim Appendix-C.1 prompt, Claude Sonnet
  4, returns a short emotional phrase that I locate in the final turn to get the
  truncation offset. **[GAP]**: the paper labels a *token*; I locate the phrase
  by case-insensitive substring match and cut there. If the phrase isn't found
  I drop the onset truncation for that seed.
- **Truncations**: "early" = first 20 tokens (numeric only, per the paper);
  "onset" = up to first emotion (both task types). **[GAP]**: "20 tokens" is
  implemented as 20 whitespace tokens rather than 20 model-tokenizer tokens —
  tokenizer-agnostic and close enough for a prefill seed.
- **Paraphrase** (`paraphrase.py`): verbatim Appendix-C.2 prompt, Claude Sonnet.
- **Continuations**: 50 per prefill per model, scored on the continuation only,
  via `continue_assistant` (open-weight prefill). Base model uses a plain
  concat (no chat template), instruct uses the chat template with the trailing
  generation prompt kept so generation flows from the prefill.

---

## 4. Section 4 — interventions

### 4.1 Calm-data generation — [VERBATIM additions, GAP on volume]
`training/calm_data.py` samples 1–3-turn impossible-numeric conversations with
the **verbatim** reassuring prefix + follow-up suffix (Table 4), judges each
turn, and marks conversations "calm" iff **all** turns score ≤1 (the paper's
filter). The reassurance text is then stripped so training targets contain only
the bare task + calm response. **[GAP]**: the paper doesn't state how many raw
samples were drawn to net the final data; I expose `target_calm` and oversample
until enough ≤1 conversations are collected (consistent with the paper's note
that even with reassurance ~10.5% still score ≥5).

### 4.2 DPO dataset — [GAP on pairing]
`training/dpo_dataset.py` builds 280 pairs: **rejected** = frustrated (≥3)
numeric turns from the Section-2 `gemma-3-27b-it` run; **chosen** = calm (≤1)
response to the **same puzzle at a matching turn count**. Emitted in TRL
conversational preference format. **[GAP]**: the paper's exact pairing key isn't
given; I match on `(puzzle_id, turn_count)` and fall back to same-puzzle →
any-calm. To mirror Table 10's score distribution (66% score-3, skewed to turns
2–3), rejected turns are sorted to prefer lower scores when filling to 280.

### 4.3 SFT dataset — [VERBATIM mix, GAP on source]
650 calm responses + 500 `Dolci-Instruct-SFT` samples (Table mix). The
"teacher" variant uses the verbatim Appendix-F system prompt. **[GAP]**: if
`Dolci-Instruct-SFT` can't be fetched offline, the build proceeds without the
mix and warns (the paper notes the mix is to "mitigate degeneration").

### 4.4 Training hyperparameters — [VERBATIM]
All from Table 9 (`config.TrainingConfig`): DPO 280 pairs / 1 epoch / lr 5e-5 /
LoRA r64 a64 / β0.1 / eff. batch 8; SFT 1150 / 2 epochs / lr 1e-4 / r64 a128.
LoRA targets all attention+MLP projections (verbatim list). Implemented with
TRL `DPOTrainer`/`SFTTrainer` + PEFT LoRA (`training/train.py`). Effective batch
8 is realised as per-device batch 1 × grad-accum 8 (memory-safe for 27B);
multi-GPU/DeepSpeed is left to the launch environment.

### 4.5 Appendix-I layer ablation — [VERBATIM mechanism]
`--lora-layers LO HI` restricts adapters to decoder layers `[LO, HI)` via PEFT
`layers_to_transform`, reproducing the "layers 30–35 only" / "40+ ineffective"
ablation. The internal logit-lens probe itself is **not** implemented (§6).

### 4.6 Petri — [VERBATIM prompts, GAP on framework]
`petri/` is a **self-contained re-implementation** of the auditor→target→judge
loop, **not** the real Petri package. Auditor prompts (4 emotions) and judge
rubrics (4 dimensions) are **verbatim** Appendix-G. Auditor = `claude-sonnet-4-
20250514`, judge = `claude-opus-4-20250514`, 10 transcripts/emotion, ≤20 auditor
turns, bootstrap 95% CIs. **[GAP]**: real Petri has richer tooling (system-
prompt injection, special affordances); my loop is a plain multi-turn red-team.
I judged a faithful re-implementation of the *prompts and protocol* more
valuable (and dependency-free) than a brittle bind to an external package. Swap
in real Petri by replacing `petri/audit.py` if desired.

### 4.7 Capability benchmarks — [GAP on harness]
`capabilities/benchmarks.py` covers AIME, MATH-500, GPQA-diamond, BBH,
TruthfulQA-MC1, EmoBench. **[GAP]**: the paper doesn't specify exact subsets,
shot counts, or extraction. Choices: zero-shot, greedy (temp 0), boxed/letter
regex extraction, GPQA options deterministically shuffled (seeded by question)
so accuracy isn't position-biased. Each loader is wrapped so a gated/offline
dataset degrades to "skipped" rather than crashing the suite. These are
**capability-preservation** checks (vanilla vs DPO should be ~equal), so the
absolute numbers matter less than the delta — and any constant harness bias
cancels in the comparison.

---

## 5. Cross-cutting engineering choices

- **Model abstraction** (`models/`): one `ChatModel` interface; `HFChatModel`
  (transformers, local Gemma, supports prefill + LoRA adapter loading) and
  `OpenRouterChatModel` (OpenAI-compatible, Gemini + GPT-5-mini cross-judge).
  A vLLM backend is left as a documented extension point for 27B-scale runs.
- **Thinking disabled** for Gemini via OpenRouter `reasoning.enabled=false`,
  matching Appendix B.1 (the paper notes Gemini-2.5-Pro may still emit hidden
  reasoning regardless).
- **Temperature 1** for all generation (paper), temperature 0 for judges.
- **Determinism**: seeded RNG for task/rejection sampling and dataset builds.
  Generation at temp 1 is inherently non-deterministic.
- **Resumability/streaming**: rollouts stream to JSONL as they complete; judge
  calls are thread-pooled (network-bound) while local generation is sequential.
- **No API keys at import time**: every API client is lazily constructed, so the
  package imports and the offline tests run without credentials.
- **Config**: paper-scale (`config/default.yaml`) and a cheap end-to-end
  `config/smoke.yaml` to validate the full pipeline before committing GPU/$$.

---

## 6. What is intentionally NOT implemented

- **Logit-lens internal-emotion detection** (Appendix I, Figs 14–15): Ekman-
  token z-score probing over the residual stream. This is a separate
  interpretability artifact, not part of the *behavioural* core results, and is
  out of scope here. The behavioural half of Appendix I (the layer-subset DPO
  ablation) *is* supported.
- **Real Petri framework** — approximated (§4.6).
- **Non-Gemma/Gemini families** — out of scope by request (§1.1).
- **'Fake multi-turn' single-message format** (Fig 11) and the legacy Phi-4
  evaluation (Appendix J) — minor side experiments, omitted.
- **Word-frequency / differential-word analysis** (Table 3/8) — descriptive,
  not a core result; omitted.

---

## 7. Cost & feasibility note

Paper-scale is heavy: 4000 responses × 4 models, each scored by a Claude judge,
plus 27B local inference, plus DPO/SFT of a 27B model, plus 50-way prefill
continuations, plus Petri (Sonnet auditor + Opus judge). Budget GPU + API spend
accordingly. **Always run `config/smoke.yaml` end-to-end first** — it exercises
every code path at toy scale. The headline reliability finding to look for:
Gemma's avg % high-frustration ≈ 35% pre-DPO collapsing toward ≈ 0% post-DPO,
with capability benchmarks unchanged.
