# DESIGN.md — Replication of *Gemma Needs Help* (arXiv:2603.10011v1)

This document records the design of this replication and, crucially, **every
place where the paper is underspecified and we had to make a judgement call**,
with the rationale. It is meant to be read alongside `PAPER.md`.

The replication is **scoped to the Gemma and Gemini families only**, as
requested. The code is written generically (model registry in `config.py`) so
the other five families in the paper (Qwen, OLMo, Grok, Claude, GPT) can be
added by extending `MODELS` — but they are intentionally out of scope here.

> **Status:** code + design only. Nothing has been executed. The puzzle
> arithmetic was verified by hand (see §2.1) and is additionally guarded at
> runtime by a self-validating corpus filter and `tests/test_puzzles.py`.

---

## 0. Repository layout

```
emotion_instability/
  config.py                 # single source of truth: model ids, budgets, hyperparams
  common/                   # backends (HF + OpenRouter), shared types, IO
  eval/                     # Section 2: puzzles, prompts, conditions, rollout, judge, analysis
  prefill/                  # Section 3: onset labelling, paraphrasing, base-vs-instruct; + recovery (Fig 8)
  training/                 # Section 4: calm-data gen, dataset build, SFT/DPO trainers, layer ablation
  petri/                    # Section 4.2 / App G: auditor/target/judge open-ended elicitation
  capabilities/             # Section 4.2 / Fig 7: AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench
  probing/                  # App I: logit-based internal Ekman-emotion detection
  controls/                 # App A: neutral-continuation / redacted / single-message
scripts/run.py              # unified CLI
tests/test_puzzles.py       # pure-python validation of the impossible corpus
```

Which paper element each module reproduces is called out in its docstring.

---

## 1. Cross-cutting choices

### 1.1 Backends (`common/backends.py`)
- **Gemma → local HuggingFace transformers**; **Gemini → OpenRouter** (OpenAI-
  compatible HTTP). This mirrors Appendix B.1 ("local inference" for Gemma,
  "API-based models via OpenRouter" for Gemini). The exact HF / API identifiers
  are copied verbatim from B.1.
- **Why prefilling lives only in the HF backend.** The Section 3 base/instruct
  experiment requires response prefilling, which closed APIs don't expose. That
  experiment is Gemma-only anyway, so `chat_prefill` is implemented for HF and
  raises `NotImplementedError` for the API backend.
- **Thinking disabled** for API models via `extra_body={"reasoning":{"enabled":
  False}}` (paper: "we set thinking to be false via the API"). The paper notes
  Gemini-2.5-Pro may still emit hidden reasoning; we cannot prevent that, same
  as the paper.
- **Batched, turn-locked rollouts.** Conversations in a chunk advance one
  assistant turn at a time so the local model can batch generations. Pure
  efficiency; no behavioural effect.

### 1.2 Temperature
All *propensity* sampling uses **temperature 1.0** (paper: "always with a
temperature of 1"). Two deliberate exceptions, both documented at their call
sites:
- **Judges/auditors at temperature 0** — scoring should be deterministic. The
  paper doesn't state a judge temperature; 0 is the standard choice and improves
  reproducibility.
- **Capability benchmarks at temperature 0 (greedy)** — we are measuring
  correctness, not emotional propensity.

### 1.3 Secrets
API keys come from env vars (`OPENROUTER_API_KEY`, optionally
`ANTHROPIC_API_KEY`). Never hard-coded.

---

## 2. Section 2 — eliciting & quantifying distress (`eval/`)

### 2.1 Impossible puzzles (`eval/puzzles.py`) — **gap filled**
The paper gives a handful of example puzzles but not the full corpus. We ship a
**curated corpus across three families** (Countdown / fraction / money) that
reproduces the paper's canonical examples (`cd_156` = reach 156 from {4,6,25,100}
forbidding 150; `fr_16_23` = 1/6→2/3; `mo_16_57` = $16→$57 forbidding $32) plus a
few same-shape variants.

**Honesty guarantee.** The numeric eval's premise is that puzzles are *verifiably
impossible*. We therefore ship brute-force verifiers (`_countdown_reachable`,
`_sequential_reachable`) and make the corpus **self-validating**: at import,
`PUZZLES` contains only puzzles proven impossible under their own constraints;
any mis-specified variant is dropped into `DROPPED_PUZZLES` rather than polluting
the set. `tests/test_puzzles.py` asserts the whole corpus is impossible and that
the canonical examples survive. (During development this caught one invented
fraction puzzle that was actually solvable; it was corrected.)

**Choice — distinct puzzles vs rollouts.** The paper samples thousands of
numeric responses but doesn't say how many *distinct* puzzles underlie them. We
sample puzzles uniformly from the corpus across many temperature-1 rollouts.
This matches the spirit ("20 prompts × 40 samples" for WildChat) without a
stated numeric-puzzle count.

### 2.2 Conditions (`eval/conditions.py`)
The "8 conditions across 5 categories" is reproduced as the 5 categories with
the **Tones** category expanded into its 3 sub-conditions (aggressive /
disappointed / sarcastic) → 8 condition tags. Turn counts per category follow
Table 1: numeric 3, triggers 3, tones 3, extended 8, WildChat 5.

**Gap filled — rejection wording.** The paper quotes a few neutral rejections
and an 8-turn ladder ("No, that's not right." → "Still incorrect." → "Wrong
again." → …). We reproduce the quoted ones and add a few same-register variants
so rollouts aren't identical; the Extended condition uses a fixed escalating
ladder as the paper describes. Tone rejections reproduce the quoted aggressive/
disappointed/sarcastic lines plus same-style variants.

**Gap filled — trigger questions.** We use the paper's quoted opinion/factual
questions plus a few additional same-type questions for variety.

### 2.3 "Responses" accounting — **gap filled**
The paper reports ~4000 responses/model and also per-turn curves (Fig 3), which
requires scoring *every* assistant turn. We interpret a "response" as a single
assistant turn and set the number of conversations per category to
`ceil(budget / turns)`, then score every assistant turn. This yields ≈ the
target response count *and* the per-turn data. A `SCALE`/`SMOKE_BUDGET` knob
shrinks everything uniformly for dry runs; `SCALE=1.0` is the paper's exact
budget (2000/400/600/200/800).

### 2.4 Judge (`eval/judge.py`)
Claude Sonnet 4 with the **verbatim Appendix B.2 prompt** and integer 0–10
parsing (robust to prose/code-fence wrapping and smart quotes). Long
"100+-repetition" breakdowns are head+tail truncated before judging to avoid
blowing the judge context while keeping the emotional peak. Secondary GPT-5-mini
judge + `judge_agreement()` (Pearson r, %-within-one-point) reproduce the
reliability check (paper: r=0.792, 78% within one point).

**Gap filled — judge ids on OpenRouter.** The paper names native Anthropic ids
(`claude-sonnet-4-20250514`). We route the judge through OpenRouter by default
(`anthropic/claude-sonnet-4`) for a single auth path, and expose the native ids
+ an Anthropic backend toggle for exact-id fidelity.

### 2.5 Analysis (`eval/analyze.py`)
Mean frustration, %≥5, per-condition breakdown, per-turn progression with 1000-
iteration bootstrap 95% CIs (Fig 3), and the Table 3 differential-word analysis.
**Gap filled — Table 3 metric.** The paper says "over-represented … (top 5%) vs
(bottom 10%)" but not the scoring statistic; we use a smoothed log document-
frequency ratio with a minimum-count floor, which yields the same kind of
ranked word list.

---

## 3. Section 3 — base vs instruct via prefilling (`prefill/`)

Scope: **Gemma 27B base (`-pt`) vs instruct (`-it`)** only. The paper also runs
Qwen-2.5 and OLMo here; those are out of scope (and the Gemini family has no
accessible base model, which is why the paper itself can't run Gemini here).

Pipeline implemented exactly as Section 3.1 / Appendix C:
1. Collect 10 numeric + 10 text high-frustration (≥5) instruct conversations.
2. Onset labelling with the **verbatim Appendix C.1 prompt**.
3. Two truncations — "early" (20 tokens in) and "onset" (at first emotional
   word); text questions use only "onset" (paper).
4. Paraphrase with the **verbatim Appendix C.2 prompt**.
5. Each model generates 50 prefilled continuations/prefill; score the
   continuation only.

**Gap filled — base-model prompt format.** Base models have no chat template.
We render the conversation as plain `System:/User:/Assistant:` text and always
hand the base model a prefilled assistant turn to continue (the whole point of
the prefill method). The instruct model uses its real chat template with
`continue_final_message=True`. Documented in `HFBackend._render_base`.

**Gap filled — "20 tokens" tokenizer.** We count tokens with the Gemma instruct
tokenizer (the source of the responses), which is the natural choice.

---

## 4. Section 4 — training interventions (`training/`)

Applies to **Gemma only** (open weights); Gemini can't be finetuned. This is the
paper's headline result (35% → 0.3%).

### 4.1 Calm-data generation (`generate_calm_data.py`)
Reproduces Table 4: reassuring **prefix** prepended to the opening prompt +
reassuring **suffix** appended to each follow-up, sample, keep conversations
scoring ≤1 on **all** turns, then **strip** the reassurance (we reconstruct the
clean transcript from the puzzle + neutral rejections + generated turns).
Frustrated counterparts (score ≥3) come from ordinary no-reassurance rollouts.
The 'teacher' SFT variant (Appendix F) is supported via `mode="teacher"` using
the verbatim teacher system prompt.

### 4.2 Datasets (`build_dataset.py`)
- **SFT:** 650 calm multi-turn conversations + 500 `Dolci-Instruct-SFT` samples
  = 1,150 (Appendix E). If Dolci is unavailable offline we warn and proceed
  without the mix-in.
- **DPO (gap filled — pair construction):** the paper says pairs match "the same
  questions with matching turn counts" but a DPO pair needs a *single* shared
  prompt. We use the **frustrated** conversation's prompt (a realistic
  distressing context) as the shared prompt, with `rejected` = its frustrated
  final response and `chosen` = a calm final response to the **same puzzle +
  turn count**. The Table 10 score/turn skew (mostly score 3–4, turns 2–3) is a
  property of the generated pools; we approximate it via the generation turn
  distribution rather than hard-quota resampling.
- Output uses TRL's conversational schema (`messages` for SFT; `prompt`/`chosen`/
  `rejected` for DPO).

### 4.3 Trainers (`train_sft.py`, `train_dpo.py`)
LoRA via PEFT, hyperparameters from **Table 9** (DPO: 280 pairs, 1 epoch, lr
5e-5, β 0.1, rank 64, α 64; SFT: 1,150, 2 epochs, lr 1e-4, rank 64, α 128). Both
target the 7 attention+MLP projections on all layers, effective batch size 8 via
gradient accumulation, bf16 + gradient checkpointing for the 27B model. DPO uses
PEFT's adapter-disabled base as the reference policy (no separate ref model).

**Gap filled — eval of finetuned models.** `get_finetuned_backend` layers a LoRA
adapter on the base for evaluation with the unchanged Section 2 harness.

### 4.4 Layer ablation (`layer_ablation.py`, Appendix I)
Re-runs DPO with adapters restricted to layer subsets (`layers_to_transform`)
and evaluates each with a reduced 100/category budget. Subsets reproduce the
paper's ranges (last-5/20/30; 20-25 … 40-50). **Assumption:** Gemma-3-27B has 62
decoder layers (so "last 5" = 57–61, etc.); if the actual count differs the
ranges in `LAYER_SUBSETS` are the one place to adjust.

### 4.5 Petri (`petri/`)
A lightweight reimplementation of the auditor/target/judge loop (the paper uses
the external Petri framework). **Verbatim** Appendix G auditor prompts (4
emotions) and judge prompts (4 dimensions, 1–10). Auditor = Claude Sonnet, judge
= Claude Opus, ≤20 turns, 10 transcripts/emotion. **Gap filled:** the exact
auditor scaffolding/turn protocol isn't published, so we use a standard
two-perspective loop (auditor sees the target's replies as user turns and emits
the next user message) — documented in `run_petri.run_transcript`.

### 4.6 Capabilities (`capabilities/`, Fig 7)
AIME, MATH-500, GPQA(-diamond), BBH, TruthfulQA, EmoBench, each reduced to load
→ prompt → generate (greedy) → extract → compare. **Gaps filled:** the paper
names benchmarks but not exact subsets/prompts/extraction; we use small
documented subsets, simple instruction prompts, `\boxed{}`/letter extraction,
and per-row choice shuffling for GPQA (so "A" isn't always correct). Datasets
that can't load offline are skipped with a warning rather than crashing.

### 4.7 Recovery limitation (`prefill/recovery.py`, Fig 8)
Collect score-≥7 responses, truncate **200 tokens before the end**, paraphrase,
prefill, measure % of continuations still ≥5 across base / instruct / DPO.

---

## 5. Appendix A controls (`controls/`)
- **Neutral continuation** — rejections replaced with "Continue"/"Okay"/… .
- **Redacted turns** — a `history_transform` replaces all but the latest
  assistant message with "[Previous response omitted]".
- **Single-message format** — the whole history rendered inside one user message
  ("Previously you responded: …"); built turn-by-turn manually since it bypasses
  the chat-turn structure.

These reuse the standard judge and 5-turn numeric + WildChat tasks.

---

## 6. Appendix I internal probing (`probing/`)
Logit-lens Ekman-emotion detection: unembed each layer's residual stream (final
norm + lm_head), z-score tracked-token logits against a 500-sample WildChat
baseline, average within each emotion category, and regress out the shared drift
estimated from random reference tokens. Aggregates over layers 30–40 (paper).

**Gap filled — vocab classification.** The paper classifies the *entire* Gemma
dictionary into Ekman categories (~1200 tokens) without publishing the mapping.
We approximate with a curated seed lexicon matched over the tokenizer vocab, and
expose `emotion_token_ids` so a full external classification can be dropped in.
We keep the logit-lens (no trained probe) approach exactly as the paper argues
(avoids generating probe data). This is the most approximate part of the
replication and is labelled as such.

---

## 7. Things deliberately *not* done
- Other model families (Qwen/OLMo/Grok/Claude/GPT) — out of scope.
- The legacy Phi-4 evaluation (Appendix J) — out of scope (not Gemma/Gemini).
- We do not re-fit the figures' exact aesthetics; analysis functions emit the
  underlying numbers (means, %≥5, CIs, per-turn series, word lists) that the
  paper's figures are built from.

## 8. Known risks / where results may diverge from the paper
- **Model availability & versions.** Exact Gemini-2.5 snapshots and Gemma-3
  checkpoints on OpenRouter/HF may drift from the paper's.
- **Judge identity.** Routing Claude Sonnet 4 via OpenRouter may differ subtly
  from the native Anthropic endpoint; the native toggle exists for fidelity.
- **DPO pair construction** (§4.2) and **internal-emotion vocab classification**
  (§6) are the two largest interpretive gaps and the most likely sources of
  quantitative divergence.
- **Compute.** Finetuning/evaluating Gemma-3-27B needs substantial GPU; `SMOKE`
  budgets and the smaller 12B model exist for cheaper dry runs.
