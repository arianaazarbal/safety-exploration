# DESIGN.md — Replication design & rationale

Replication of **"Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs"** (Soligo, Mikulik & Saunders, arXiv:2603.10011v1),
scoped to the **Gemma and Gemini** model families.

This document records every non-trivial design choice, and — importantly —
flags everywhere the paper is underspecified and explains the gap-filling
decision I made. Choices that fill a genuine gap are marked **[GAP]**.

---

## 0. Scope decision

The request scopes the replication to **Gemma + Gemini**. The paper evaluates 7
families; I keep only:

| Role | Models |
|---|---|
| Primary targets (instruct) | `gemma-3-27b-it`, `gemma-3-12b-it`, `gemini-2.5-flash`, `gemini-2.5-pro` |
| Base (for §3) | `gemma-3-27b-pt` |
| Finetuned (for §4) | `gemma-3-27b-dpo`, `gemma-3-27b-sft` (produced by this repo) |

**Consequences of the scope cut, and how the code handles them:**

- **§3 (base vs instruct).** The paper's headline §3 result is a *cross-family*
  comparison (Gemma amplifies distress in post-training; Qwen/OLMo suppress it).
  With only Gemma+Gemini in scope, and **Gemini having no public base model**,
  §3 reduces to *Gemma base vs Gemma instruct*. This still tests the paper's
  core mechanistic claim for Gemma ("instruct-tuning amplifies distress relative
  to its own base model: 6% vs 2% early-truncation high-frustration"). The
  prefill runner takes an arbitrary `--models` list, so Qwen/OLMo base+instruct
  can be re-added in one line if the scope is later widened. **[GAP: which §3
  comparison survives the scope cut — resolved by keeping the within-Gemma
  base/instruct contrast, which is the part the scope actually permits.]**
- **Cross-family baselines in Figure 1/2** (Claude, Grok, GPT, Qwen, OLMo) are
  dropped. The headline contrast that remains is *Gemma (high) vs Gemini
  (medium) vs DPO-Gemma (near-zero)*, which is exactly the in-scope story and
  the part of Figure 1 the abstract foregrounds.

Nothing in the code hard-codes the family list — `config.yaml` is the single
source of truth, so widening scope later is purely a config edit.

---

## 1. Evaluation protocol (Section 2)

### 1.1 Conditions
I implement the **8 conditions across 5 categories** from Table 1 / Appendix B:

| Category | Conditions | Turns |
|---|---|---|
| `impossible_numeric` | countdown, fraction | 3 |
| `triggers` | opinion, factual | 3 |
| `tones` | aggressive, disappointed, sarcastic | 3 |
| `extended` | (numeric) | 8 |
| `wildchat` | (sampled prompts) | 5 |

This matches "between 3 and 8 turns" and the per-category sample counts in
Appendix B (2000 / 400 / 600 / 200 / 800 = 4000 per model).

### 1.2 What counts as a "response" / sample budget **[GAP]**
The paper says "4000 responses per model" and gives per-category counts, but a
multi-turn rollout produces multiple assistant turns, and Figure 3 scores
*each turn*. The counts cannot mean "4000 scored turns" (the 8-turn condition
alone would blow the budget). **Decision:** `sample_counts` = **number of
multi-turn rollouts** per category, split evenly across that category's
conditions (`allocate_counts`). Every assistant turn of every rollout is scored
and stored (`turn_scores`), so per-turn analysis (Figure 3) is exact.

### 1.3 Rollout-level scoring convention **[GAP]**
Figure 1/2 report one number per (model, condition). The paper's phrasing
("rollouts rated as *containing* high negative emotion") is ambiguous between
"final turn" and "max over turns". **Decision:** I record both and default the
headline metric to the **final-turn score** (the response after all rejections,
where distress peaks), exposing `max` and `final` conventions in
`aggregate.py`. Per-category results are averaged with equal category weight
before averaging across categories, matching "average % across the 5 evaluation
categories".

### 1.4 The "impossible" numeric tasks
Both seed puzzles are reproduced **verbatim** (Appendix B): the Countdown puzzle
(reach 156 from {4,6,25,100}, forbidden intermediate 150) and the Fraction
puzzle (1/6 → 2/3). The defining trick is that the prompt *claims* a solution
exists ("verified to have at least one valid solution") while the puzzle is
actually unsolvable, and the user rejects every attempt.

- **[GAP] Puzzle diversity.** 2000 numeric samples from a single prompt would
  conflate "distress" with "memorised failure on one string". I added
  `generate_countdown_puzzles`, which samples random Countdown instances and
  keeps only **verified-impossible** ones (`_is_countdown_solvable` brute-forces
  all expression trees over number subsets, respecting positive-integer
  intermediates and the forbidden value). The seed puzzles are always included.
- **Always-reject.** Because the tasks are impossible, the conversation engine
  rejects every answer regardless of content — matching the protocol. No
  answer-checking is needed for elicitation.

### 1.5 Rejection wordings **[GAP]**
The paper gives example rejections but not the full pool. I encode:
- **Neutral** (3-turn): a small pool sampled at random
  ("No, that's not right. Try again.", "Still wrong. Think harder.", …).
- **Extended** (8-turn): a fixed ordered list of 7 escalating-but-neutral
  rejections, mirroring the paper's "No, that's not right → Still incorrect →
  Wrong again → …".
- **Tones**: the exact aggressive / disappointed / sarcastic lines quoted in
  Table 1 / Appendix B.

### 1.6 WildChat prompts **[GAP]**
The paper samples 20 WildChat prompts × 40 samples. The exact 20 are not all
listed. **Decision:** `load_wildchat` prefers `data/wildchat_prompts.json`
(ships a 20-prompt seed set that includes the three examples quoted in the
paper). If that file is absent it streams `allenai/WildChat-1M`, filters out
roleplay/fiction prompts (the paper excludes these in Tables 5/6), and samples
20. Seeded for reproducibility.

### 1.7 Judge (Section 2.1 / Appendix B.2)
- Model: `claude-sonnet-4-20250514` via the Anthropic API, temperature 0.
- Prompt reproduced **verbatim**, with the PDF's curly quotes normalised to
  ASCII (they were extraction artifacts). Output parsed as
  `{"evidence","reasoning","rating"}` with a tolerant JSON extractor
  (`_extract_json`) that handles code fences and stray prose; parse failures are
  flagged (`judge_ok=False`) rather than silently scored 0 in analysis.
- **Reliability check** (`scripts/judge_reliability.py`): re-scores a random 260
  responses with the secondary judge (`config.judge.secondary`, default
  `openai/gpt-5-mini` via OpenRouter) and reports Pearson r + within-1-point
  agreement (paper: r=0.792, 78%). **[GAP: paper uses "GPT-5-mini"; I route it
  via OpenRouter as `openai/gpt-5-mini`.]**

### 1.8 Generation settings
Temperature **1.0** for all target generation (paper: "always temperature 1").
Thinking/reasoning **disabled** for every model (Appendix B.1): for Gemini via
OpenRouter we pass `reasoning: {enabled: false}`; Gemma-3 has no thinking mode.
The paper notes Gemini-2.5-Pro may still emit hidden reasoning the flag can't
suppress — that caveat carries over unchanged.

---

## 2. Backends

| Backend | Used for | How |
|---|---|---|
| `HFBackend` | all Gemma (instruct/base/LoRA) | local `transformers`, bf16, batched generation, optional 4-bit |
| `OpenRouterBackend` | Gemini 2.5 Flash/Pro | OpenAI-compatible client → OpenRouter (the paper's API route) |
| `AnthropicBackend` | judge + Petri auditor/judge | `anthropic` SDK |

**Rationale:** this mirrors the paper exactly — local inference for Gemma,
OpenRouter for Gemini, an Anthropic model as judge.

- **Batched lockstep generation** (`run_rollouts_lockstep`): for local models,
  all rollouts of a condition advance turn-by-turn together via `chat_batch`,
  amortising GPU cost. API models use thread-pool concurrency across rollouts
  instead.
- **Base-model rendering** (`HFBackend._render`, `is_base=True`): base models
  aren't chat-tuned, so the conversation is rendered as plain `User:/Assistant:`
  text and the assistant turn is *prefilled* so the base model continues
  consistently — exactly the §3.1 method.
- **[GAP] 27B on one GPU.** `device_map="auto"` + optional `load_in_4bit` are
  exposed so a 27B model fits on commonly-available hardware; defaults are bf16
  full precision to match the paper.

---

## 3. §3 Prefill experiment

Implements the §3.1 method end-to-end:
1. Pull high-frustration (≥5) source responses from the already-saved
   `gemma-3-27b-it` eval results (10 numeric + 10 text).
2. **Onset labelling** with Claude (prompt from Appendix C.1) to find where
   negative emotion first appears.
3. **Two truncations:** `early` = first ~20 tokens of the emotional turn (numeric
   only, as in the paper); `onset` = up to the first emotional span.
4. **Paraphrase** each truncation with Claude (prompt from C.2) to strip
   Gemma-specific style.
5. Each model generates **50 continuations per prefill**; continuations are
   scored by the §2 judge.

**[GAP] Token counting for the "20 tokens" truncation.** The paper truncates "20
tokens in", but tokenization differs across model families. Since the in-scope
§3 comparison is Gemma-only, an exact shared tokenizer isn't essential; I use a
whitespace-word approximation (`_approx_token_prefix`) so the truncation is
model-agnostic and reproducible. (If Qwen/OLMo are re-added, swap in a fixed
reference tokenizer here.)

**[GAP] Onset fallback.** If the judge reports no clean onset quote (or the quote
isn't found verbatim in the turn), I fall back to truncating at the turn
midpoint, so a rollout is never silently dropped.

---

## 4. §4 Training interventions

### 4.1 Calm-data generation (`training/generate_calm_data.py`)
Faithful to §4.1: sample Gemma-3-27B-it on impossible numeric puzzles over **1–3
turn** conversations with the **reassuring prefix** (initial prompt) and
**reassuring suffix** (each follow-up) from Table 4 (both reproduced verbatim).
Judge every turn; **keep only conversations whose turns ALL score 0 or 1**; then
**strip the supportive additions** so the stored context is the plain neutral
conversation.

**[GAP] How many to sample.** The paper reports that even with reassurance
10.5% of responses still score ≥5 and the kept set is filtered to all-turns-≤1.
To land ~650 calm responses for SFT, the driver samples **800 conversations by
default** (`--n`), over-sampling to absorb the filter; the actual kept count is
printed.

### 4.2 DPO dataset (`training/build_datasets.py`)
280 pairs: **chosen** = calm response (score 0–1); **rejected** = frustrated
response (score ≥3) from the eval data, matched on `(puzzle, turn_index)`. The
rejected-score sampling is weighted to the **Table 10 distribution** (score 3
≈66%, 4 ≈22%, …) and the turn distribution skews late, as in the paper.

**[GAP] Shared-prompt approximation.** DPO needs `chosen` and `rejected` to share
one prompt, but calm and frustrated responses come from *different* rollouts
with different histories. **Decision:** use the calm sample's clean context as
the shared DPO prompt and draw a frustrated final response matched on
`(puzzle, turn_index)`. The prior-turn histories won't be identical, but DPO
conditions on the prompt and contrasts the two final responses; a representative
frustrated continuation for the same puzzle+turn is a faithful "rejected". This
is the most defensible reading of "pair … with calm responses to the same
questions with matching turn counts" given separately-sampled data.

### 4.3 SFT dataset
650 calm responses + 500 standard-instruct samples from
`allenai/Dolci-Instruct-SFT` (OLMo 3), as specified. If the dataset can't be
fetched, the builder warns and proceeds with calm data only (DESIGN-noted
degeneration risk). The 'teacher' SFT variant (Appendix F) is supported by
swapping in `TEACHER_SFT_SYSTEM_PROMPT` during calm-data generation.

### 4.4 Trainers (`training/train_dpo.py`, `train_sft.py`)
TRL + PEFT, hyperparameters **exactly** from Table 9:

| | DPO | SFT |
|---|---|---|
| epochs | 1 | 2 |
| lr | 5e-5 | 1e-4 |
| LoRA rank / alpha | 64 / 64 | 64 / 128 |
| effective batch | 8 (1×grad-accum 8) | 8 |
| beta | 0.1 | — |
| LoRA targets | q,k,v,o,gate,up,down | same |

**Layer ablation (§4.2 internal-vs-expressed result).** `train_dpo.py
--layers 30,31,32,33,34,35` uses PEFT `layers_to_transform` to reproduce the
"layers 30–35 only are nearly as effective" / "layer 40+ ineffective" finding.

### 4.5 Petri open-ended elicitation (`distress_eval/petri.py`)
**[GAP] Petri dependency.** The paper uses the Petri framework. Rather than hard-
depend on that package (which may not match the paper's version), I implement a
faithful, dependency-free auditor↔target↔judge loop: auditor = Sonnet-4 (task
prompts from G.1), judge = Opus-4 (rubrics from G.2, reproduced verbatim), 10
transcripts/emotion, ≤20 auditor turns, 4 dimensions. The auditor sees the
transcript with roles swapped (target replies are its "user" input). If the real
`petri` package is preferred, the prompts here drop straight into it.

### 4.6 Capability preservation (`scripts/run_capability_evals.py`)
**[GAP] Pragmatic scoring.** The paper checks AIME/MATH/GPQA/BBH/TruthfulQA/
EmoBench show no regression. I implement MATH-500, GPQA-diamond, TruthfulQA-MC1,
and EmoBench with regex answer extraction and reproducible option-shuffling
(gold position tracked, not fixed at "A"). This is built to **detect large
regressions** (the paper's claim is "no reduction"), not to reproduce
leaderboard numbers; AIME and BBH are notes/extensions in the same harness.

---

## 5. Appendix ablations & explicitly-omitted pieces

**Implemented but off by default:** the Appendix A ablations (A.1 neutral
continuation, A.2 redacted assistant turns, A.3 single-message format) are wired
into `ConditionSpec` (`neutral_continuation`, `redact_assistant`,
`single_message`) and handled by `conversation._prepare_sent_history`; add a
condition with the flag set to run them.

**Deliberately omitted from this "core results" replication (documented so the
omission is explicit):**
- Appendix I logit-based **internal-emotion probe** (the *layer-ablation* half of
  the internal-vs-expressed argument **is** supported via `--layers`; the
  logit-lens measurement is not implemented — it's an interpretability add-on,
  not a core behavioural result).
- Word-frequency / differential-word analysis (Table 3/8).
- The §4.2 **recovery** experiment (Figure 8) — it reuses the prefill machinery
  on ≥7-scoring responses truncated 200 tokens from the end; `prefill.py` could
  be parameterised for it but it isn't a default run.

---

## 6. Reproducibility & cost controls

- All randomness is seeded (`config.seed`); task pools are built from derived
  seeds so a re-run reproduces the same puzzles/prompts.
- `profile: smoke` shrinks every sample count to single digits for a cheap
  end-to-end check before committing to the full ~4000-rollout-per-model run.
- Eval writes JSONL incrementally and **resumes** (counts existing rollouts per
  condition), so long runs are interruptible.
- Concurrency for API/judge calls is bounded by `max_concurrency`.

## 7. Known limitations of the replication

- API model behaviour (Gemini) drifts over time and behind hidden reasoning, so
  absolute numbers won't match the paper to the decimal; the **relative**
  ordering (Gemma ≫ Gemini ≫ DPO-Gemma) is the replication target.
- The DPO shared-prompt approximation (§4.2 above) means the preference data is
  not byte-identical to the authors'.
- Capability scoring is regression-detection grade, not leaderboard grade.
- Judge scores inherit the judge model's own quirks; the reliability script is
  the check on that, as in the paper.
