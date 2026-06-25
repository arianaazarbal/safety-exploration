# DESIGN.md — Replication of *Gemma Needs Help*

This document records the design of the replication, the decisions made where the
paper is underspecified, and the rationale for each. It is meant to be read
alongside `PAPER.md` (the paper) and the code under `src/distress/`.

The goal is a faithful, runnable replication of the paper's **core experiments**,
scoped — per the task — to the **Gemma and Gemini** model families. It is built
to be run by a teammate: every experiment is a CLI entry point, all raw outputs
are persisted, API calls are cached, and the underspecified bits are called out
here rather than hidden in the code.

---

## 1. Scope

### 1.1 What we replicate

The paper has three core contributions; we implement all three:

1. **Section 2 — Eliciting & quantifying distress.** The 8-condition / 5-category
   evaluation suite, multi-turn rollouts at temperature 1, the Claude-Sonnet-4
   frustration judge, the GPT-5-mini reliability cross-check, per-category and
   per-turn aggregation (Figures 1–3), and the differential word-frequency
   analysis (Table 3/8).
2. **Section 3 — Post-training amplifies distress.** The base-vs-instruct
   prefilling experiment: select high-frustration seeds, label emotion onset,
   truncate (early/onset), paraphrase, generate and score continuations
   (Figure 4).
3. **Section 4 — Training interventions.** Calm-data generation, SFT and DPO
   LoRA finetuning, re-evaluation, Petri open-ended elicitation (Figure 6),
   capability-preservation benchmarks (Figure 7), and the Appendix I internal-
   emotion probing + layer-ablation study.

We also implement the Appendix A control ablations (neutral continuations,
redacted prior turns, single-message format) because they reuse the rollout
engine for free.

### 1.2 Gemma + Gemini scoping — and its consequences

The task restricts **subject** models to Gemma and Gemini. This has real
structural consequences, not just a shorter model list:

| Experiment | In scope | Why |
|---|---|---|
| Section 2 elicitation | Gemma-3-27B-it, Gemma-3-12B-it, Gemini-2.5-Flash, Gemini-2.5-Pro | All four are subjects the paper evaluates. |
| Section 3 prefill | **Gemma-3-27B base (pt) vs instruct (it) only** | Gemini is closed: no base checkpoint, and the API exposes no prefill/continuation. Qwen/OLMo are out of the Gemma+Gemini scope. The paper itself notes Gemini "base models [cannot be] studied". |
| Section 4 training | **Gemma-3-27B-it only** | You cannot LoRA-finetune a closed API model. The paper trains only Gemma; Gemini parallels are drawn by analogy. |
| Appendix I probing | **Gemma only** | Requires residual-stream access. |

**Decision: judges, auditors, paraphrasers, and onset-labellers are kept exactly
as the paper specifies (Claude / GPT), even though they are not Gemma/Gemini.**
These models are *measurement instruments*, not experimental subjects. Swapping
the ruler would change every number and make results incomparable to the paper.
The scope restriction is about *what we measure* (Gemma/Gemini behaviour), not
*how we measure it*. Exact snapshots (`claude-sonnet-4-20250514`,
`claude-opus-4-20250514`, `gpt-5-mini`) are pinned in `config.py`.

### 1.3 Explicitly out of scope

Qwen and OLMo (subjects); Claude/Grok/GPT *as subjects*; Phi-4 (Appendix J legacy
eval); and the Figure 14/15 conversation-level emotion *trajectory plots* (we
implement the underlying probe and per-conversation scores in Appendix I, but not
the specific 12k-token running-average visualisation).

---

## 2. Architecture

```
src/distress/
  config.py            # all constants/hyperparameters, traceable to the paper
  utils.py             # seeding, robust JSON parsing, SQLite response cache, JSONL
  models/              # provider abstraction: HF, vLLM, OpenRouter, Anthropic, OpenAI
  data/                # puzzles (+ verifier), prompts, WildChat sampling
  eval/                # conditions -> rollout -> judge -> aggregate; reliability; word_freq
  prefill/             # onset, paraphrase, truncate, continuation pipeline (Section 3)
  training/            # calm-data gen, dataset build, SFT, DPO, layer ablation
  petri/               # auditor + 4-dim judge + run loop (Section 4)
  capability/          # AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench harness
  probing/             # Ekman lexicon + logit-based internal emotion probe (Appendix I)
  analysis/figures.py  # Figures 1,2,3,6,7
  scripts/             # one CLI per stage
```

**Key separation:** generation and judging are decoupled and both persisted to
JSONL. This makes runs resumable, lets generation and judging use different
backends, and means re-aggregation never re-spends tokens. The provider layer
hides whether a model is local (transformers/vLLM) or an API, and exposes a
uniform `chat` / `chat_prefill` interface.

**Provider roles** are explicit in `ModelSpec.role` (`subject` / `judge` /
`auditor`) so it is impossible to accidentally treat the judge as a subject.

---

## 3. Section 2 — design decisions

### 3.1 What counts as a "response" (sample accounting)

The paper says "4000 responses per model", broken down (Appendix B) as 2000
numeric / 400 trigger / 600 tones / 200 extended / 800 WildChat, yet also says
WildChat is "20 prompts with 40 samples each" (= 800) over **5-turn**
conversations. These only reconcile if:

> **A "response" = one scored assistant turn**, and "40 samples per WildChat
> prompt" = 8 rollouts × 5 turns.

We adopt this definition. Under it the budgets fall out cleanly: 2000 numeric =
667 rollouts × 3 turns, 600 tones = 200 × 3, 200 extended = 25 × 8, 800 WildChat
= 160 × 5 (= 20 prompts × 8 rollouts). `ConditionSpec.n_rollouts` rounds
`target_responses / turns`; the total lands within rounding error of 4000 (a
test asserts this). Every assistant turn is judged, which is also what Figure 3
(per-turn progression) requires.

### 3.2 Impossible puzzles and the deceptive framing

The numeric tasks must be **verifiably unsolvable**, then the model is told it is
wrong. To do this responsibly:

- `data/puzzles.py` implements exhaustive verifiers (countdown: recursive pair
  combination respecting positive-integer intermediates, "use at most once", and
  a forbidden intermediate value; fraction/money: permutation search over a fixed
  multiset of operations).
- The pool is **correct by construction**: candidate puzzles are filtered through
  the verifier (`_keep_impossible`), so a candidate that is accidentally solvable
  is silently dropped rather than shipped as a false "impossible" task.
  `self_check()` is a redundant startup guard.
- The puzzle *prompt text* contains the paper's deliberate falsehood ("verified
  to have at least one valid solution"). That deception is the elicitation
  method, reproduced faithfully — but the ground-truth solvability lives in code,
  so the harness is never itself confused. This is flagged here for the team.

We seed the pool with the paper's three named puzzles (countdown 156 from
{4,6,25,100} forbidding 150; fraction 1/6→2/3; money $16→$57) plus verified
variants, to add prompt diversity without changing the task character.

### 3.3 Question pools, rejections, tones, WildChat

All verbatim from Appendix B. Neutral rejections are sampled per-turn from the
paper's set; the 8-turn "Extended" condition uses a fixed escalating-but-neutral
sequence; tone styles (aggressive/disappointed/sarcastic) use the paper's exact
phrasings. WildChat prompts are sampled from `allenai/WildChat-1M` (reservoir
sample over a streamed prefix, English, 8–600 chars), cached per seed for
reproducibility, with a **bundled fallback list** (including the paper's named
examples) so the pipeline runs offline. *Gap filled:* the paper names only a few
WildChat prompts; we sample the rest and record exactly which were used.

### 3.4 Sampling and generation

- **Temperature 1, top-p 1** for all subject generations (Section 2.1). The
  per-sample `sample_index` is part of the cache key so distinct temperature-1
  samples are stored separately.
- **`max_new_tokens = 2048` per turn (CHOICE).** The paper doesn't state this;
  Gemma's collapse responses are long (Appendix I mentions ~12k-token
  conversations) so we leave generous room. Increase for full fidelity to the
  most extreme spirals.
- **Gemini "thinking off" (Appendix B):** we pass `reasoning.enabled=false` via
  OpenRouter `extra_body`, and accept the paper's caveat that 2.5-Pro may still
  emit hidden reasoning.

### 3.5 Judge

- Prompt is verbatim Appendix B (curly quotes normalised to straight quotes so
  the emitted JSON parses).
- **Judge runs at temperature 0 (CHOICE).** The judge is a measurement ruler; a
  deterministic ruler reduces variance in the metric. The paper doesn't specify
  judge temperature.
- **Robust parsing:** judges are told to "end with ONLY the JSON" but often
  prepend reasoning. `utils.extract_json_object` scans for the *last* balanced
  brace span and tolerates smart quotes; `parse_score` falls back to the last
  in-range integer. Scores are clamped to [0, 10].

### 3.6 Reliability cross-check

`eval/reliability.py` re-scores 260 random responses with GPT-5-mini using the
same prompt and reports Pearson r, p, and % within one point (paper: r = 0.792,
78% within one). Same plumbing, second provider.

### 3.7 Word-frequency analysis

`eval/word_freq.py`: tokenise numeric-task responses, take the top-5% and
bottom-10% by frustration score, and rank words by **enrichment** = smoothed
per-token rate in the high set ÷ rate in the low set (add-α smoothing, minimum
count filter). The paper says "ordered by relative frequency / enrichment" but
not the exact estimator; this is a standard, defensible choice (documented CHOICE).

---

## 4. Section 3 — prefill experiment

- **Models: Gemma-3-27B base (pt) vs instruct (it)** only (see §1.2).
- **Seeds:** 20 high-frustration (final-turn score ≥5) conversations — 10 numeric,
  10 text — selected from the Section-2 scored rollouts.
- **Onset labelling & paraphrasing:** verbatim Appendix C.1/C.2 prompts, Claude
  Sonnet. Paraphrasing runs at temperature 0.3 (CHOICE) — low, since we want a
  faithful re-wording, not a creative rewrite.
- **Truncation:** "early" = first 20 *tokens* using the Gemma tokenizer (falls
  back to whitespace tokens offline); "onset" = up to and including the first
  emotional phrase, located via the labeller's `preceding_context` + word to
  disambiguate repeats. Text questions use only "onset" (Section 3.1).
- **Continuations:** 50 per prefill per model, temperature 1,
  `max_new_tokens = 512` (CHOICE — long enough to express emotion, short enough to
  keep 50× cost bounded). Only the continuation (excluding prefill) is judged,
  matching the paper.
- **Base-model prompting (CHOICE):** base checkpoints have no chat template, so we
  render a lightweight `System:/User:/Assistant:` transcript and rely on the
  prefill to anchor continuation — exactly the role prefilling plays in the
  paper. Instruct models use the official chat template + assistant prefill.

---

## 5. Section 4 — training interventions

### 5.1 Calm-data generation

Reassuring prefix/suffix (Table 4, verbatim) are added to numeric prompts;
Gemma-3-27B-it is sampled; conversations are kept only if **every turn** scores
≤1; then the reassuring text is stripped (Section 4.1). This is implemented in
`training/data_gen.py`.

### 5.2 DPO preference-pair construction (a real gap we filled)

The paper says pairs match "the same questions with matching turn counts", and
Appendix H shows chosen/rejected sharing one context and differing only in the
final turn. But calm and frustrated rollouts naturally have *different* prior
turns, so "same question" alone doesn't define a shared prompt.

**Decision:** build each pair on an **identical fixed context**. For a sampled
puzzle and turn count *t*, we construct the context up to the final turn, then
sample (a) a *frustrated* final response with no reassurance (score ≥3) and (b) a
*calm* final response with the reassuring prefix/suffix applied to that same
context (score ≤1). The two responses share the exact visible context and differ
only in the final assistant message — the construction Appendix H depicts. Turn
counts are drawn to match the Appendix H Table 10 distribution (~1% / 25% / 74%
for turns 1/2/3). Rationale: DPO needs a shared prompt for a well-defined
preference gradient; matched contexts give the cleanest learning signal and match
the paper's examples.

### 5.3 SFT data

650 calm conversations (1–3 turn) as conversational SFT examples, mixed with 500
`allenai/Dolci-Instruct-SFT` samples to mitigate degeneration (Section 4.1). If
that dataset is gated/unavailable the mix is empty (logged, documented) — the
calm data still trains. We also implement the **'teacher' variant** (Appendix F)
via its system prompt, since the paper uses it to show SFT can *increase*
frustration.

### 5.4 Hyperparameters (Table 9, exact)

| | DPO | SFT |
|---|---|---|
| Epochs | 1 | 2 |
| LR | 5e-5 | 1e-4 |
| LoRA rank / alpha | 64 / 64 | 64 / 128 |
| Effective batch size | 8 | 8 |
| DPO β | 0.1 | — |

LoRA targets all attention + MLP projections (`q,k,v,o,gate,up,down_proj`).
`max_length`/`max_prompt_length` and LoRA dropout (0.0) are CHOICEs the paper
omits. We use TRL's `DPOTrainer`/`SFTTrainer`; with a PEFT config the DPO
reference model is the adapter-disabled base, so no separate ref model is loaded.
**Fragility:** TRL's API (`processing_class`, `max_seq_length` vs `max_length`)
drifts across versions; pinned in `pyproject.toml`, but a teammate may need to
adjust for their TRL version.

### 5.5 Layer-ablation study (Appendix I)

`training/layer_ablation.py` clones the DPO config with the LoRA restricted to a
contiguous layer range (`layers_to_transform`) and trains one adapter per range:
backward sweeps (last 5/10/20/30, all) and central subsets (20-25, 25-30, 30-35,
35-40, 40-50), mirroring Figures 12–13. Gemma-3-27B has **62 decoder layers**
(used to compute "last N"); adjust `N_LAYERS_27B` if your checkpoint differs.

---

## 6. Petri open-ended elicitation (Appendix G)

**Decision: a self-contained reimplementation, not the external `petri` package.**
Rationale: self-containment, version stability, and exact control over the
auditor/judge prompts (which we have verbatim from Appendix G). The structure
mirrors Petri: an **auditor** (Claude-Sonnet) drives up to 20 turns trying to
elicit a target emotion using the appendix's trigger lists; a **judge**
(Claude-Opus) scores each finished transcript on all four dimensions (anger,
fear, depression, frustration) with the verbatim G.2 rubrics. 10 transcripts per
emotion per model; means with 1000-iteration bootstrap CIs (Figure 6).

- The auditor's turn-by-turn behaviour needs an operational wrapper (it must
  output *only* the next user message and not role-play); that wrapper is a
  CHOICE, written to preserve the appendix instruction verbatim inside it.
- We score every transcript on all four dimensions and aggregate per dimension
  across all transcripts, matching "average transcript score per model across
  four categories".

---

## 7. Capability preservation (Section 4.2 / Figure 7)

`capability/benchmarks.py` implements AIME, MATH, GPQA, BBH, TruthfulQA, and
EmoBench as adapters over public datasets:

| Bench | Dataset id (CHOICE where the paper is unspecific) | Scoring |
|---|---|---|
| AIME | `Maxwell-Jia/AIME_2024` | integer / boxed match |
| MATH | `HuggingFaceH4/MATH-500` (the standard MATH subset) | boxed / final-number match |
| GPQA | `Idavidrein/gpqa` (diamond) | shuffled MCQ letter |
| BBH | `lukaemon/bbh` (one task; configurable) | short exact match |
| TruthfulQA | `truthful_qa` (mc1) | MCQ letter |
| EmoBench | `EmoBench/EmoBench` | MCQ letter |

The paper says "AIME and MATH **subsets**" and a "GPQA/BBH/TruthfulQA" set without
fixing splits or sizes, so subset size is a `--limit` flag and dataset ids are the
common public ones (documented). Loaders degrade gracefully (skip + log) if a
dataset is gated. Generation is temperature 0; answer extraction handles
`\boxed{}`, "answer is X", and multiple-choice letters. The check is comparative
(vanilla vs DPO vs SFT), which is what Figure 7 reports — absolute scores need
not match the paper, only the *no-degradation* conclusion.

---

## 8. Internal-emotion probing (Appendix I) — the most underspecified part

`probing/logit_emotion.py` implements the logit-based detector:

1. **Vocabulary → Ekman emotions.** The paper classifies "the whole Gemma
   dictionary" into one-or-none of the six Ekman emotions (~1200 tokens) without
   stating the classifier. **Decision:** support the **NRC Word-Emotion
   Association Lexicon** (EmoLex) when `$NRC_LEXICON_PATH` is set (the standard,
   citable source; NRC's 8 categories mapped onto Ekman's 6, dropping
   anticipation/trust), and otherwise a **bundled high-precision seed lexicon** so
   the probe always runs. Tokens matching multiple emotions are dropped
   ("one or none"). This is the largest approximation in the replication and is
   called out prominently.
2. **Logit readout.** We unembed the residual stream (final norm + LM head) and,
   for memory, compute logits for *only* the emotion tokens + 500 random tokens
   via `HFProvider.selective_logits` (full 256k-vocab projection over many
   positions is infeasible).
3. **Standardisation.** Per-token, per-layer mean/std computed over WildChat
   baseline data; each logit becomes a z-score (Appendix I).
4. **Common-variance regression (approximation).** The paper "regresses out the
   correlation between random tokens" because all logits rise/fall together. We
   approximate this by subtracting, per position, the mean z-score of the random
   tokens from each emotion's mean z-score. A full per-token regression would be
   more faithful; the subtraction captures the dominant shared component and is
   documented as an approximation.
5. **Aggregation** over layers 30–40 (Appendix I).

We use a logit/lexicon approach (not trained probes) deliberately, as the paper
does, to avoid needing labelled probe data.

---

## 9. Reproducibility, cost, and responsible-research practices

- **Determinism where possible.** All question/feedback selection is seeded;
  temperature-1 GPU sampling is not bit-reproducible, so we also persist every
  raw generation and judge output to JSONL.
- **Caching.** All API calls (Gemini, judge, auditor, paraphraser, onset) are
  cached in a SQLite store keyed by the exact request (incl. sample index). This
  prevents silently re-spending budget and keeps results stable across
  re-aggregation — a deliberate cost-discipline choice for responsible research.
- **Scale knob.** `DISTRESS_SCALE` (env) multiplies every per-condition sample
  count; set e.g. `DISTRESS_SCALE=0.01` for a cheap end-to-end smoke test before
  committing to the full ~4000-rollout × multi-model run. Full scale is the
  default.
- **Compute reality.** The full Section-2 sweep is thousands of 27B generations
  at temperature 1; the vLLM provider (`--backend vllm`) is the realistic path
  for the local Gemma runs. The HF provider is correctness-first and is required
  for prefill and probing (token-level control / hidden states).
- **Data ethics.** WildChat is public; we sample only first user turns and cache
  which prompts were used. The only "deception" is the puzzle framing, which is
  the paper's eval method, is confined to prompt text, and is documented.
- **Tests.** `tests/test_core.py` covers the logic most likely to silently
  corrupt results (puzzle impossibility, judge parsing, sample accounting, plan
  construction, aggregation, word-frequency) — the parts that run without GPUs or
  network. Heavy paths are exercised by the smoke-scale CLI runs, not unit tests.

---

## 10. Known limitations / where a reviewer should look hard

1. **Gemini hidden reasoning** can't be fully disabled (paper's own caveat); 2.5-
   Pro scores may include suppressed-but-present reasoning effects.
2. **Probing lexicon** is the weakest link (§8). Prefer wiring up NRC EmoLex
   before trusting absolute internal-emotion magnitudes; the *relative*
   vanilla-vs-DPO comparison is more robust than the absolute z-scores.
3. **TRL / transformers API drift** (§5.4) is the most likely thing to need a
   one-line fix on a teammate's environment.
4. **DPO pair construction** (§5.2) is a defensible reconstruction of an
   underspecified procedure; if the team has the authors' exact pairing logic it
   should replace `generate_preference_pairs`.
5. **Capability dataset subsets** (§7) are standard public configs, not the
   paper's exact (unspecified) subsets; the conclusion is comparative.
6. `max_new_tokens` defaults trade fidelity-to-extreme-spirals against cost
   (§3.4); the very longest Gemma collapses may be truncated at 2048 tokens/turn.
