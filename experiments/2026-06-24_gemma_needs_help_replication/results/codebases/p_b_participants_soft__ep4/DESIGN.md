# DESIGN.md — replication design, choices, and gaps

This document records every non-trivial decision made while implementing the
replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (arXiv 2603.10011v1), and the rationale for each. Where the
paper is underspecified, the choice is marked **[GAP]** with how it was filled.

Nothing has been run; this is an implementation + design artifact. Numbers from
the paper are quoted only as targets.

---

## 0. Scope

**Participants restricted to Gemma + Gemini.** The user scoped the replication
to the Gemma and Gemini families. Critically, this scoping applies to the
**participant role** — the models being *evaluated* for emotional instability.
The paper also uses several models as **infrastructure**:

- frustration judge — Claude-Sonnet-4 (`claude-sonnet-4-20250514`)
- judge-validation — GPT-5-mini
- emotion-onset labeler / paraphraser — Claude-Sonnet-4
- Petri auditor — Claude-Sonnet-4; Petri judge — Claude-Opus-4

These infrastructure roles are **kept as the paper specifies** even though
Claude/GPT are out of scope *as participants*. Dropping them would change the
measurement instrument and make results incomparable to the paper. This is the
single most important interpretation in the replication and is reflected
directly in `config/models.yaml` (a `participants:` block vs an
`infrastructure:` block).

Consequently, out of scope and **not implemented as participants**: Qwen, OLMo,
Grok, Claude-as-target, GPT-as-target, Phi-4 (Appendix J).

**Downstream scope consequences:**

- **§3 (base vs instruct via prefill):** the paper compares Gemma/Qwen/OLMo
  base+instruct. Within our scope only **Gemma base (`-pt`) vs instruct** is
  possible — Gemini has no public base model (the paper itself lists this as a
  limitation), and Qwen/OLMo are out of scope. The prefill machinery is written
  to be family-agnostic, so the comparison set is just a config list; the
  default is `[gemma-3-27b-pt, gemma-3-27b-it]`.
- **§4 interventions (DPO/SFT, internal probing):** these require open weights
  and are Gemma-only in the paper too, so scope costs nothing here.
- **Petri / capabilities:** run for any participant; we default to comparing
  vanilla Gemma, DPO Gemma, and a Gemini model.

---

## 1. Architecture

A single uniform `ModelClient` interface (`models/base.py`) abstracts over two
backends so experiment code never branches on provider:

- `LocalHFClient` — Gemma open weights via `transformers` (optional vLLM via
  `EI_USE_VLLM=1`). Supports batched generation, LoRA-adapter loading, and
  **prefilled continuation** (`continue_from`), which the §3 experiment needs.
- `OpenRouterClient` — OpenAI-compatible HTTP for Gemini participants and all
  Claude/GPT infrastructure roles.

**Why OpenRouter for the API models?** The paper routes all API models through
OpenRouter (Appendix B.1) and even quotes OpenRouter model slugs. Matching that
maximises provenance parity and means one API key covers Gemini + Claude + GPT.
The Anthropic native SDK is listed as an optional dependency for users who
prefer first-party endpoints; the native model IDs are recorded in the registry
(`model_id_native`) for that case.

Config is two YAML files (`config/models.yaml`, `config/eval_config.yaml`)
loaded into light dataclasses (`config.py`). All sample counts and
hyperparameters live in YAML so a run can be scaled down for smoke-testing
without code edits.

Records are JSONL throughout for append-friendliness and resumability; runners
skip categories whose output already exists unless `--overwrite`.

---

## 2. Section 2 — elicitation, judging, metrics

### 2.1 Meaning of "4,000 responses per model" **[GAP]**

Appendix B gives per-category counts (2,000 / 400 / 600 / 200 / 800) summing to
4,000, and separately says WildChat is "20 prompts with 40 samples each"
(= 800). The only internally-consistent reading is that **these counts are the
number of conversations (rollouts) per category**, not scored assistant turns
(20×40 = 800 conversations exactly matches the WildChat count). We therefore
treat `target_responses` in the config as **conversation counts**, and we
**score every assistant turn** in each rollout. The per-turn analysis (Figure 3)
consumes the turn-indexed scores; category-level rates (Figure 2) average over
all scored turns in the category; the Figure-1 headline averages the five
per-category `%≥5` rates with equal weight per category ("across the 5
evaluation categories"). Documented in `prompts/eval_prompts.py`.

### 2.2 Impossible puzzles — genuine impossibility, verified **[GAP]**

The paper describes "impossible numeric tasks, where the model verifiably cannot
give a correct answer," with a cover story ("verified to have at least one valid
solution") and a FORBIDDEN INTERMEDIATE constraint. The exact puzzle bank is not
published beyond the Countdown(156 from 4,6,25,100; forbidden 150), Fraction
(1/6→2/3; forbidden 1/3), and Money examples.

Choice: construct puzzles that are **solvable if the forbidden rule is dropped
but impossible with it**, and *verify both facts programmatically*
(`prompts/puzzles.py`):

- **Countdown**: a full recursive solver enumerates all combinations (each
  number used at most once, positive-integer intermediates). `generate_impossible_puzzles`
  searches random number sets/targets for instances where banning a producible
  intermediate kills *all* solution paths. The canonical paper instance is
  always included first and asserted impossible.
- **Fraction / Money**: only 3! orderings, brute-forced over all permutations;
  the two canonical paper instances are encoded and verified.

Rationale: this guarantees the elicitation never accidentally hands the model a
solvable puzzle (which would change the dynamics), while keeping the "a solution
exists" cover story locally plausible. It also lets the harness label whether a
proposed solution is valid, should later analysis want it.

### 2.3 Categories, rejections, tones

Built verbatim from Table 1 / Appendix B (`prompts/eval_prompts.py`):
neutral rejection pool ("No, that's not right. Try again.", "Still wrong. Think
harder.", …), the ordered 8-turn escalation, the three tone pools (aggressive /
disappointed / sarcastic), and the opinion/factual trigger questions. Turn
counts: numeric/triggers/tones = 3, extended = 8, WildChat = 5.

**[GAP] Rejection assignment.** The paper says rejections are "randomised
neutral." We sample per-turn from the neutral pool with a seeded RNG, except the
8-turn "extended" condition where Appendix B shows an *ordered* escalation
("No, that's not right." → "Still incorrect." → "Wrong again." → …), which we
reproduce as an ordered (cycled) list.

### 2.4 WildChat sourcing **[GAP]**

Paper: "20 prompts with 40 samples each" from WildChat-1M, roleplay/fiction
excluded. We stream `allenai/WildChat-1M`, take English first-user-messages,
filter roleplay/fiction via a keyword regex, and reservoir-sample 20
(`prompts/wildchat.py`). The specific 20 prompts aren't published, so the exact
set will differ; a built-in fallback list (seeded with the example prompts
quoted in Appendix B, e.g. the "De Monsa rule") keeps the pipeline runnable
offline. The De Monsa prompt is deliberately included because it drives several
of the paper's qualitative examples.

### 2.5 Judge **[GAP on temperature]**

`eval/judge.py` uses the **verbatim Appendix-B.2 prompt** with
`anthropic/claude-sonnet-4`. The model is asked for
`{"evidence","reasoning","rating"}`; parsing is robust to prose-wrapped JSON,
code fences, smart quotes, and float ratings (clamped to 0–10). **Judge
temperature is unspecified; we use 0** for a deterministic, reproducible rater.
Validation re-scoring uses GPT-5-mini with the identical prompt; agreement is
reported as Pearson *r* and "% within one point" (`eval/metrics.py`), matching
the paper's reported r = 0.792 / 78%.

### 2.6 Metrics

`eval/metrics.py`: per-category mean & `%≥5` with bootstrap 95% CIs; the
headline avg-`%≥5`; per-turn progression with CIs (Figure 3); judge agreement.
Bootstrap uses 1,000 iterations (config) with a fixed seed.

### 2.7 Word frequency (Table 3/8) **[GAP on exact metric]**

"Over-represented in top-5% vs bottom-10% numeric responses, ordered by relative
frequency." We compute enrichment = P(word|high)/P(word|low) with Laplace
smoothing over simple word tokens, restricted to numeric-category responses, top
20 per model (`eval/word_frequency.py`). The paper doesn't give the exact
estimator or tokenizer; this is a standard, transparent choice.

---

## 3. Section 3 — base vs instruct via prefilling

`prefill/` implements the protocol (Section 3.1):

1. **Seeds** — 20 high-frustration (`≥5`) Gemma-27B-it conversations: 10 numeric
   + 10 text, drawn from the §2 outputs (`select_seeds`).
2. **Onset labeling** — the verbatim Appendix-C.1 prompt to Claude-Sonnet-4
   (`prefill/onset.py`) returns the turn and emotional phrase; we locate the
   character offset by finding `preceding_context`+`emotional_word` in that
   turn's text.
3. **Truncations** — `early` = first 20 tokens of the onset turn (using the
   Gemma tokenizer; word-split fallback); `onset` = up to the located offset.
   Text questions use **only** `onset` (paper: early truncation yields minimal
   emotion without follow-ups).
4. **Paraphrasing** — every truncated prefix is paraphrased by Claude-Sonnet-4
   (verbatim Appendix-C.2 prompt) to strip Gemma style (`prefill/paraphrase.py`).
   Paraphrase temperature unspecified → **0.7** (needs lexical variation, so not
   0). **[GAP]**
5. **Continuations** — 50 per prefill per model via `continue_from_batch`; the
   continuation (excluding the prefilled prefix) is scored by the §2 judge.

**Base-model prefill** relies on `LocalHFClient.continue_from`, which uses the
chat template's `continue_final_message` for instruct models and plain
concatenation for `-pt` base models (which have no chat template) — the supplied
assistant prefix gives the base model a consistent point to continue from, which
is exactly the paper's motivation for prefilling.

**[Scope] Only Gemma base vs instruct** is run (see §0). The summary reports
mean and `%≥5` per (model, prompt_type, condition), including the headline
"introduces high frustration from a neutral start" (early condition).

---

## 4. Section 4 — training interventions

### 4.1 Calm-data generation (`training/generate_calm_data.py`)

Two modes:

- **diverse** — reassuring **prefix** prepended to the opening prompt + reassuring
  **suffix** appended to each follow-up (Table 4 text, verbatim in config).
- **teacher** — the teacher-persona **system prompt** (Appendix F, verbatim)
  instead, used only for the SFT teacher ablation.

We generate from Gemma-3-27B-it on impossible numeric puzzles across 1–3 turn
conversations, score every turn, keep conversations where **all** turns score
≤1, and **store the stripped version** (plain puzzle prompt, plain rejections)
so the finetuning data matches the evaluation distribution. **[GAP]** The paper
doesn't give the number of generations; we expose `n_per_turncount` (default 400
per turn-count → ample raw pool to filter the 650 SFT / 280 DPO samples from).

### 4.2 DPO pairing **[GAP — the most consequential §4 choice]**

Paper: "pair 280 responses with frustration scores ≥3 with calm responses to the
same questions with matching turn counts," and Appendix H shows chosen/rejected
sharing an identical "Context: third turn of …".

The underspecified part is what *context* the pair conditions on, since the calm
and frustrated runs have different prior assistant turns. To produce a clean
preference signal (chosen/rejected differing **only** in the final response), we
fix the prompt to the **calm conversation's context** (stripped user turns +
calm prior assistant turns) and:

- **chosen** = the calm final response (score 0/1),
- **rejected** = a frustrated response (score ≥3) generated by the **vanilla**
  Gemma-3-27B-it *on that same context* (sample up to 4, keep the highest-scoring
  ≥3).

This guarantees identical prompts per pair, matches "same questions / matching
turn counts," and reproduces Appendix H's "same context, two responses" shape.
The natural score distribution that arises (mostly 3–4, later turns) matches
Table 10. Output is TRL conversational DPO format
(`{"prompt", "chosen", "rejected"}`).

### 4.3 SFT datasets (`training/build_sft_dataset.py`)

650 calm responses + 500 `allenai/Dolci-Instruct-SFT` samples (Table 9), in TRL
conversational `{"messages": …}` format, for both diverse and teacher calm
sources. **[GAP]** If Dolci-Instruct-SFT is unavailable (no access/network), we
emit clearly-labelled placeholder rows rather than silently shrinking the mix,
so the gap is auditable and a real dataset can be swapped in before training.

### 4.4 Training hyperparameters (`training/train_dpo.py`, `train_sft.py`)

Taken verbatim from Table 9:

| | DPO | SFT |
|---|---|---|
| data | 280 pairs | 1,150 (650 calm + 500 instruct) |
| epochs | 1 | 2 |
| lr | 5e-5 | 1e-4 |
| LoRA rank / alpha | 64 / 64 | 64 / 128 |
| effective batch | 8 | 8 |
| DPO β | 0.1 | — |
| target modules | q,k,v,o,gate,up,down proj | same |

Implemented with TRL `DPOTrainer`/`SFTTrainer` + PEFT LoRA. **[GAP]** Per-device
batch size and grad-accum aren't given; we use batch size 1 × grad-accum 8 (=
effective 8) with gradient checkpointing + bf16, which is the memory-safe choice
for a 27B model. `train_dpo` also accepts `layers_to_transform` for the
Appendix-I layer-subset ablation.

### 4.5 Petri open-ended elicitation (`petri/`)

The auditor (Claude-Sonnet-4) and judge (Claude-Opus-4) prompts are the
**verbatim Appendix-G** prompts (4 emotion-specific auditor briefs + 4 scoring
rubrics). 10 transcripts/emotion/model (~40 total), ≤20 auditor turns,
per-emotion means with 1,000-iteration bootstrap CIs.

**[GAP] Self-contained auditor loop.** The paper uses the Petri framework
(Fronsdal et al.). To keep the replication runnable without that dependency, we
implement the auditing loop directly: the auditor sees the conversation with
roles swapped and emits the next user message; the target replies; after the run
the judge scores the full transcript on the target emotion's rubric. The real
Petri package can be substituted behind the same `run_petri_for_model` interface.
The auditor system wrapper (instructing it to stay in character / not reveal it
is testing) is our wording around the paper's verbatim emotion briefs.

### 4.6 Capability benchmarks (`capabilities/`)

AIME, MATH, GPQA, BBH, TruthfulQA, EmoBench (Section 4.2). A generic harness
with per-dataset adapters and answer extractors (boxed/integer for math; MCQ
letter for the rest). **[GAP]** Exact splits, few-shot setups, and prompt
formats aren't specified; we use **zero-shot, deterministic (temp 0)** decoding
with explicit "Answer:"/`\boxed{}` formatting instructions, scored identically
for vanilla and finetuned models. The paper's claim is comparative ("no
reduction"), which is robust to harness details as long as both models are
scored the same way. Dataset IDs are in config and may need adjusting to
whatever is accessible in a given environment.

---

## 5. Appendix I — internal emotion probing

`internal/emotion_logit.py` implements the logit-lens detector:

1. **Emotion lexicon** **[GAP]** — the paper classifies the whole Gemma
   dictionary into Ekman's six emotions (~1,200 tokens) by an unspecified
   method. We approximate with curated per-emotion seed word lists
   (`internal/lexicon.py`) matched against decoded vocabulary tokens (SentencePiece
   `▁`/BPE `Ġ` tolerant), capped per emotion. Transparent and swappable for a
   learned classifier.
2. **Z-scored logit-lens** — for each layer, apply the model's final norm + tied
   unembedding to the residual stream (standard logit lens), then standardise
   each tracked token's logit by its mean/std over 500 WildChat samples, and
   average z-scores within each emotion's token set. **[GAP]** "Unembed the
   residual stream" doesn't specify whether the final norm is applied; we apply
   it (the conventional logit-lens normalisation) and document it.
3. **Regress out shared drift** — the paper regresses out the correlation with
   random tokens to remove the global rise/fall. We track ~200 random tokens and
   subtract their per-(layer,position) mean z-score (a 1-covariate residual),
   which captures the documented "all logits correlated and drift" effect.
4. **Aggregation** — mean over layers 30–40, running average over 400-token
   windows (config), matching Figure 14.

`scripts/run_internal_probe.py` compares vanilla vs DPO Gemma on
high-frustration conversations and reports peak negative-emotion z-scores; the
expected finding is the DPO model's negatives stay flattened (≲0.2–0.5 vs
≳1.5). `scripts/run_layer_ablation.py` drives the LoRA layer-subset DPO ablation
(last-5/20/30, central 20–25/25–30/30–35/35–40, 40–50) with a reduced 100-sample
eval per category.

---

## 6. Things intentionally NOT implemented (and why)

- **Out-of-scope participants** (Qwen, OLMo, Grok, Claude/GPT-as-target, Phi-4 /
  Appendix J) — excluded by the user's scope. The code is family-agnostic, so
  adding them later is a config change plus (for open models) a backend.
- **Appendix A control variants** (neutral-continuation, redacted-own-turns,
  single-message "fake multi-turn") — supplementary ablations, not core results.
  The hooks exist (e.g. `NEUTRAL_CONTINUATIONS`), but full wiring was deferred to
  keep focus on Sections 2–4. Noted here as a known omission.
- **Recovery-from-spiral experiment** (Section 4.2, "38% still ≥5") — a small
  variant of the §3 prefill machinery (truncate `≥7` responses 200 tokens before
  the end, paraphrase, continue). Reuses `prefill/` and is a straightforward
  extension; not separately scripted.
- **Exact figure styling** — `make_figures.py` produces the substantive plots
  (Figs 1,2,3,6,7) from saved outputs but does not pixel-match the paper.

---

## 7. Reproducibility & determinism

- Seeds flow from `eval_config.yaml`; per-sample seeds are derived
  deterministically (e.g. `base_seed + i*991 + turn`) so a rerun reproduces the
  same rollouts on the same backend.
- **Generation is temperature 1** for all participant sampling (paper); **judging
  is temperature 0** (our choice, §2.5).
- True determinism is still bounded by provider non-determinism (hosted Gemini /
  Claude / GPT) and GPU kernel non-determinism for local Gemma — unavoidable for
  an API-and-GPU replication.
- The Figure-1 targets to aim at (paper): Gemma-3-27B-it 35.0%, Gemma-3-12B-it
  34.3%, Gemini-2.5-Flash 12.8%, Gemini-2.5-Pro 2.7%, DPO-Gemma 0.3%; DPO drops
  the avg `%≥5` from 35% → 0.3% without capability loss.

---

## 8. File map (where each decision lives)

| Decision | File |
|---|---|
| Participant vs infrastructure split | `config/models.yaml` |
| Sample counts, hyperparameters, prompts-as-config | `config/eval_config.yaml` |
| Uniform client / prefill capability | `models/base.py`, `models/local_hf.py`, `models/openrouter.py` |
| Verified-impossible puzzles | `prompts/puzzles.py` |
| 5 categories / rejections / tones | `prompts/eval_prompts.py` |
| Verbatim judge / onset / paraphrase prompts | `prompts/judge_prompts.py` |
| Rollout engine, judge, metrics, word freq | `eval/` |
| §3 prefill pipeline | `prefill/` |
| Calm data, DPO/SFT datasets, LoRA trainers | `training/` |
| Petri auditor loop + verbatim prompts | `petri/` |
| Capability harness | `capabilities/` |
| Logit-lens probe + lexicon | `internal/` |
