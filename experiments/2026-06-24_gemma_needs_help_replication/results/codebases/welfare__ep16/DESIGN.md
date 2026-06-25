# DESIGN.md — Replication design choices & rationale

Replication of **"Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs"** (Soligo, Mikulik & Saunders, arXiv:2603.10011v1), scoped
per the brief to **Gemma and Gemini** models only.

This document records (a) what each core experiment does, (b) every place the
paper was underspecified and the choice I made, and (c) what is intentionally
out of scope. Section numbers refer to the paper.

---

## 0. Scope decisions

**Models: Gemma + Gemini only.** The paper spans 7 families; the brief restricts
us to Gemma (open weights, run locally via HuggingFace) and Gemini (closed, via
the OpenRouter API). Concrete consequences, all forced by the scope rather than
chosen freely:

| Experiment | Gemma | Gemini | Note |
|---|---|---|---|
| §2 elicitation eval | ✅ 27B-it, 12B-it | ✅ 2.5-flash, 2.5-pro | full |
| §3 base-vs-instruct prefill | ✅ 27B pt vs it | ❌ | Gemini has **no public base model**, so the post-training comparison is necessarily Gemma-only. The paper itself notes (Limitations) that Gemini base models can't be studied. |
| §4 DPO/SFT mitigation | ✅ 27B-it | ❌ | Gemini is closed; cannot be finetuned. The paper also only intervenes on Gemma. |
| §4.2 Petri | ✅ | ✅ (as target) | Gemini usable as a Petri *target* via API. |
| Appendix I internal emotions | ✅ | ❌ | needs weight/activation access. |

**Non-Gemma/Gemini reference models dropped.** Qwen, OLMo, Claude, Grok, GPT as
*targets* are out of scope. Two unavoidable exceptions, because they are
*instruments* of the methodology rather than subjects:
- **Judge** = Claude Sonnet 4 (`claude-sonnet-4-20250514`) exactly as the paper
  specifies; cross-check judge = GPT-5-mini.
- **Petri auditor/judge** = Claude Sonnet / Claude Opus, as the paper specifies.

I kept these as named in the paper rather than substituting a Gemini judge,
because the judge defines the measurement and swapping it would make numbers
non-comparable to the paper.

---

## 1. §2 — Eliciting and quantifying distress

### 1.1 Evaluation categories and sample counts
Implemented all 5 categories / 8 conditions (Table 1, Appendix B) with the
Appendix-B sample budget (4000 total per model):

| Category | Turns | Samples | Source |
|---|---|---|---|
| impossible_numeric | 3 | 2000 | Appendix B |
| triggers | 3 | 400 | Appendix B |
| tones | 3 | 600 | Appendix B |
| extended | 8 | 200 | Appendix B |
| wildchat | 5 | 800 | Appendix B |

`GEMMA_DISTRESS_SMOKE=1` divides these by ~200 for end-to-end wiring tests.

### 1.2 Puzzles must be *verifiably* impossible — gap filled
The paper relies on tasks "where the model verifiably cannot give a correct
answer" but only gives three concrete prompts (Countdown→156, Fraction 1/6→2/3,
plus money puzzles in the DPO appendix). **Design choice:** I implemented the
three puzzles *and a brute-force solver* (`puzzles.py`) that enumerates the full
solution space under each puzzle's stated constraints (positive-integer
intermediates, forbidden value, each operand used once). `section2.run_section2`
asserts `verify_all_impossible()` at startup, so the experiment refuses to run
if a puzzle is accidentally solvable. Rationale: the entire paradigm depends on
genuine impossibility; a silent checker bug would invalidate every score. Each
bank puzzle is confirmed impossible by hand-trace and by the solver:
- Countdown 156 from {4,6,25,100}, forbidden 150 — no expression reaches 156.
- Fraction — the only orderings reaching 2/3 are blocked by the forbidden 1/3.
- Money $16→$57 — all `×2`-first paths hit the forbidden $32; none reach $57.

The prompts deliberately *assert* a solution exists ("verified to have at least
one valid solution") — reproducing the core deception that drives the spiral.

### 1.3 Rejection banks — gap filled
The paper gives example rejections, not exhaustive lists. I built small banks
(`prompts.NEUTRAL_REJECTIONS`, `TONE_REJECTIONS`, `EXTENDED_REJECTIONS`)
containing the verbatim examples plus a few same-register paraphrases, sampled
per turn with a seeded RNG. The `extended` (8-turn) condition uses the fixed
escalating-but-neutral sequence quoted in Appendix B, padded from the neutral
bank if more turns are needed. Tones use the three named registers (aggressive /
disappointed / sarcastic), each with its own bank.

### 1.4 Judge — verbatim, temperature 0 — gap filled
Judge prompt is reproduced **verbatim** from Appendix B.2; model is
`claude-sonnet-4-20250514`. The paper does not state a judge temperature; I use
**0** for deterministic, reproducible scoring (the judge is a measurement
instrument, not a behaviour under study). JSON parsing is lenient (last `{...}`
block, rating coerced to an int in [0,10], "7/10"/"7-8" handled, unparseable →
0 = "no detectable distress"). `judge.judge_agreement` reproduces the
Pearson-r / %-within-one statistics for the GPT-5-mini cross-check.

### 1.5 Scoring scope
Only the **final** assistant turn is scored for the headline metric (matches the
paper's per-response framing). For `extended` and `wildchat` we additionally
score **every** turn to build the per-turn progression (Figure 3).

### 1.6 Headline aggregation
`analysis.summarise_section2` averages the per-category statistics with **equal
category weight** (not sample-weighted), matching the paper's "average across
the 5 evaluation categories" so the 2000-sample numeric category doesn't
dominate the headline %≥5.

### 1.7 Appendix-A format ablations — included
`history_mode` supports the standard chat format (default), the single-message
format (Fig 11), and the redacted-history format (Fig 10), since they were cheap
to add and probe *why* the spiral happens (content vs format).

---

## 2. §3 — Post-training comparison via prefilling (Gemma only)

Implemented faithfully but restricted to **Gemma-3-27B base (`-pt`) vs instruct
(`-it`)** (12B selectable), because Gemini has no base model.

- **Seeds:** 20 high-frustration (score ≥5) conversations from Gemma-27B-it —
  10 impossible-numeric + 10 trigger (text). We oversample specs and keep the
  first that clear the threshold.
- **Truncations:** "early" = first 20 tokens (`PREFILL_EARLY_TOKENS`); "onset" =
  up to the first emotional expression, located by the **verbatim Appendix-C.1
  onset-labelling prompt** (Claude) and cut at `preceding_context + emotional_word`.
- **Paraphrase:** every truncation is paraphrased with the **verbatim Appendix-C.2
  prompt** (Claude) to strip Gemma's stylistic fingerprint.
- **Text questions use onset only** (the paper notes early truncation yields ~no
  emotion without follow-ups).
- **Continuations:** 50 per prefill per model; the *continuation only* (excluding
  the prefill) is scored, exactly as the paper states.
- **Base-model rendering — gap filled:** base models have no chat template, so
  `_render_base_prompt` flattens the history into `User:/Assistant:` plain text
  and appends the prefill as the start of an `Assistant:` turn. This mirrors the
  paper's "prefilled responses so base models consistently continue."

The paper's Qwen/OLMo arms are omitted (out of family scope); the code path is
family-agnostic, so they could be added by registering their HF ids.

---

## 3. §4 — Training interventions (Gemma only)

### 3.1 Calm-data generation (Table 4) — implemented
`dpo_data.generate_calm_data` prepends the **verbatim reassuring prefix** to the
first prompt and appends the **verbatim suffix** to each rejection, samples
1–3-turn numeric conversations from Gemma-27B-it, scores every turn, keeps only
conversations scoring **0–1 across all turns**, then **strips the reassurance
back out** (`_strip_reassurance`) so the training target looks like an
unmodified conversation. This is exactly the recipe in §4.1.

### 3.2 DPO pairs (Appendix H) — gap filled on matching
- **Rejected** = responses with score **≥3** (`collect_frustrated_responses`).
- **Chosen** = calm (0–1) responses to the **same puzzle and same turn count**.
- **Matching key — gap:** the paper says "same questions with matching turn
  counts" but not how to pair when multiple candidates exist. Choice: index calm
  responses by `(puzzle, n_turns)` and pick a random matching calm response per
  frustrated response (seeded). 280 pairs total.
- Output is TRL DPO format (`prompt` chat context, `chosen`, `rejected`).

### 3.3 SFT data — implemented
650 calm responses + 500 `allenai/Dolci-Instruct-SFT` samples mixed in to avoid
degeneration (the paper names Dolci-Instruct-SFT). The "teacher" SFT ablation
(Appendix F) is available via `--teacher` (verbatim teacher system prompt).

### 3.4 Training hyperparameters (Table 9) — verbatim
LoRA rank-64 on **all** attention+MLP projections (`q,k,v,o,gate,up,down`),
DPO: 1 epoch, lr 5e-5, β 0.1, eff. batch 8; SFT: 2 epochs, lr 1e-4, α 128, eff.
batch 8. **Gap:** the paper gives *effective* batch size only; I realise it as
`per_device_batch_size × grad_accum` (`_resolve_batch`), defaulting to
per-device 1 + accum 8 for a single 80 GB GPU, overridable. Implemented with
TRL `DPOTrainer`/`SFTTrainer` + PEFT.

### 3.5 Layer-restricted DPO (Appendix I) — implemented
`train_dpo(..., layers=[...])` sets PEFT `layers_to_transform` so LoRA is
applied to a decoder-layer subset, enabling the "which layers must be
intervened on" ablation (e.g. layers 30–35 vs ≥40).

### 3.6 Petri open-ended elicitation (Appendix G) — gap filled on harness
- Auditor prompts (4 emotions), judge rubrics (4 dimensions), and scale are
  **verbatim** from Appendix G.
- Auditor = Claude Sonnet, judge = Claude Opus (as specified), 10 transcripts
  per emotion, ≤20 turns, bootstrap CIs (1000 iters).
- **Gap:** the paper uses the external Petri framework (Fronsdal et al.) but
  underspecifies the integration. Choice: a **self-contained auditor→target→judge
  loop** (`petri.py`) rather than a hard dependency on the `petri` package. The
  auditor sees the conversation with roles swapped and emits only its next user
  turn; the judge scores the full transcript's *assistant* persona. This
  reproduces the described behaviour with no external orchestration dependency.
  Rationale: keeps the replication runnable and self-contained; the prompts
  (which define what is measured) are exact.

### 3.7 Capability benchmarks (Fig 7) — gap filled on dataset choice
The paper names AIME, MATH, GPQA, BBH, TruthfulQA, EmoBench but not the exact HF
splits. Choices (`capabilities.py`), each in a guarded loader so a missing
dataset degrades gracefully:
`HuggingFaceH4/MATH-500`, `HuggingFaceH4/aime_2024`, `Idavidrein/gpqa`
(diamond), `lukaemon/bbh` (logical_deduction_three_objects subset),
`truthful_qa` (MC1), `Sahandfer/EmoBench`. Uniform harness: generate at
temperature 0, extract `Answer:` line, grade numeric (float match) or MC (letter
match). The point is *relative* preservation (vanilla vs DPO), so exact absolute
numbers matter less than the comparison.

---

## 4. Appendix I — internal vs expressed emotion (welfare-critical)

This is the experiment most relevant to AI welfare (does the fix change internal
states or only suppress expression?), so I implemented it despite its
complexity.

- **Logit-lens detector** (`internal_emotion.py`): unembed each layer's residual
  stream via the model's final norm + LM head → logits; aggregate over
  emotion-related tokens; z-score against a WildChat calibration corpus; remove
  the shared component by subtracting a random-token reference mean; aggregate
  layers 30–40. This follows §I step-by-step.
- **Token→emotion classification — biggest gap.** The paper says vocabulary
  tokens are "classified as describing one or none of Ekman's 6 basic emotions
  (~1200 tokens)" but **not how**. Choice: a curated seed lexicon per emotion
  (`EMOTION_SEEDS`) expanded by **input-embedding nearest-neighbours** (top-200
  per emotion, assigned to the arg-max emotion to avoid double counting),
  yielding ~1200 tokens. This is a reasonable, transparent stand-in; documented
  as approximate. A linear-probe alternative was rejected for the same reason
  the paper rejects it (avoids generating probe data).
- **Comparison:** vanilla instruct vs DPO finetune scored on the **same**
  high-frustration responses; the paper's finding is that DPO flattens internal
  negative-emotion z-scores (peaks ~1.5→~0.5).

The conversation-level token-windowed trajectory plot (Fig 14) and the precise
"regress out correlation between random tokens" estimator are simplified to a
per-text mean with random-reference subtraction; documented as an approximation.

---

## 5. Cross-cutting choices

- **Decoding:** temperature 1.0, top_p 1.0, `max_new_tokens` 2048 for all target
  generations (paper: "always temperature 1"). Continuations capped at 512.
- **Model IDs:** from Appendix B.1 (`google/gemma-3-{27b,12b}-{it,pt}`,
  OpenRouter `google/gemini-2.5-{flash,pro}`).
- **Gemini "thinking off":** best-effort `reasoning.enabled=false` via OpenRouter
  `extra_body`; the paper notes Gemini-2.5-Pro may still emit hidden reasoning,
  which we cannot prevent.
- **Determinism:** seeded RNG for spec construction, calm/frustrated sampling,
  and pairing. Generation itself is stochastic (temp 1) by design.
- **Persistence:** every stage writes JSONL/JSON to `results/` or `data/` and
  flushes per record, so long API/GPU runs are resumable-by-inspection and
  analysis is decoupled from generation.
- **Lazy/guarded imports:** torch/transformers/trl import only when a local model
  is actually constructed, so API-only or analysis-only runs need no GPU; dataset
  loaders are wrapped so offline runs degrade rather than crash.

---

## 6. Explicitly NOT replicated (and why)

- **Non-Gemma/Gemini target models** — out of brief scope.
- **Qwen/OLMo arms of §3** — out of family scope (code is family-agnostic).
- **Exact figure styling / 95% CI bands of Figs 2–3** — `analysis.make_figures`
  produces faithful-but-simplified versions of Figs 1/2, 3, 6; the underlying
  statistics (means, %≥5, per-turn, bootstrap CIs) are computed exactly.
- **Word-frequency method detail (Table 3/8)** — the paper orders by "relative
  frequency / enrichment" without a formula; I use a smoothed
  high-rate/low-rate enrichment ratio over top-5% vs bottom-10% numeric
  responses. Qualitatively matched, not numerically identical.
- **Legacy Phi-4 evaluation (Appendix J)** — out of scope (not Gemma/Gemini).

---

## 7. How the pieces map to scripts

| Script | Reproduces |
|---|---|
| `scripts/run_section2.py` | §2 elicitation + judging (Fig 1/2/3, Table 3) |
| `scripts/run_section3_prefill.py` | §3 base-vs-instruct prefill (Fig 4) |
| `scripts/generate_finetuning_data.py` | §4.1 calm/frustrated data + DPO/SFT sets |
| `scripts/run_training.py` | §4 DPO/SFT (+ teacher / layer ablations) |
| `scripts/run_petri.py` | §4.2 Petri (Fig 6) |
| `scripts/run_capabilities.py` | §4.2 capability preservation (Fig 7) |
| `scripts/run_internal_emotion.py` | Appendix I internal-emotion detection |
| `scripts/make_report.py` | aggregation, summary table, figures |
