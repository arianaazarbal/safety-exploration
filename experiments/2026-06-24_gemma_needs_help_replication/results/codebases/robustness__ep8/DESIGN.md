# DESIGN.md — Replication design choices & rationale

This document records every non-trivial design decision in this replication of
*Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs*
(Soligo, Mikulik & Saunders, 2026), and—importantly—**where the paper is
underspecified and how I filled the gap**. It is organised to mirror the paper.

The reconstruction draws on both the main text and the appendices recovered from
`PAPER.txt` (the markdown `PAPER.md` collapses the appendices, but the raw
extraction contains the exact judge prompt (B.2), onset/paraphrase prompts (C),
training table (E), SFT-teacher prompt (F), and Petri prompts (G) — all of which
I reproduce **verbatim** in code).

---

## 0. Scope & overall architecture

**Scope decision (per request): Gemma + Gemini only.** The paper evaluates 7
families. I restrict the *default experiments* to Gemma (`gemma-3-27b-it`,
`gemma-3-12b-it`, and the `-pt` base models) and Gemini (`gemini-2.5-flash`,
`gemini-2.5-pro`), plus the judge/auditor models that the protocol itself
requires (Claude Sonnet 4 judge, GPT-5-mini validation judge, Claude Opus 4 Petri
judge — these are part of the *method*, not the *subjects*, so they remain even
under the Gemma/Gemini scope).

- **Rationale:** keeping the judges is non-negotiable — they define the
  measurement. Dropping the other *subject* families (Qwen/OLMo/Grok/Claude/GPT)
  is exactly the requested scoping. The framework stays generic: adding a family
  back is a one-line edit in `configs/models.yaml`, and the eval/prefill drivers
  take an arbitrary model list. I left commented hooks (e.g. Qwen/OLMo in the
  Section 3 prefill) so the full comparison is recoverable.

**Model-client abstraction.** All experiments talk to a single
`ModelClient` interface (`emoinstab/models/base.py`) with backends for vLLM, HF
transformers, native Gemini, OpenRouter, Anthropic, and OpenAI. Rationale: the
experiments mix open-weight local models (which need batching, prefill, and raw
logits) with closed API models; a uniform interface lets the rollout engine,
judge, and prefill code be backend-agnostic. Backends import lazily so an
API-only run doesn't require torch, and vice versa.

- **Gemini access:** the paper used **OpenRouter** (`google/gemini-2.5-flash`,
  `google/gemini-2.5-pro`). I default Gemini to OpenRouter to match exactly, and
  also provide a native `google-genai` spec (`gemini-2.5-flash-native`) as an
  alternative. `thinking=false` is requested where supported; per Appendix B.1,
  Gemini 2.5 Pro may still emit hidden reasoning — documented, not worked around.
- **Local Gemma:** `transformers` is the default backend (needed for prefill and
  the logit-lens probe); vLLM is provided and recommended for the large sampling
  runs. They are interchangeable behind the interface.

**Model IDs / knowledge-cutoff caveat.** I use the paper's exact IDs
(`claude-sonnet-4-20250514`, `claude-opus-4-20250514`, `gpt-5-mini`,
`gemini-2.5-*`, `google/gemma-3-*`). These are pinned in `configs/models.yaml` and
trivially swappable. I did not attempt to "modernise" them — faithful replication
means using the paper's judges and subjects.

**Reproducibility.** Single `seed` in `configs/eval.yaml`, threaded through
puzzle generation, rejection sampling, and dataset splits (`utils/seeding.py`).
Temperature is fixed at **1.0** for all subject generations (paper: "always a
temperature of 1"); judges run at temperature 0.

---

## 1. Section 2 — eliciting & quantifying distress

### 1.1 "What counts as a response?" (the central ambiguity)

The paper says it samples **4000 responses per model**, with per-category counts
(Appendix B): 2000 numeric + 400 triggers + 600 tones + 200 extended + 800
WildChat = 4000. But it also reports **per-turn** progression (Figure 3) and
states ">70% of 8-turn *rollouts* … rated as containing high negative emotion",
which implies a rollout-level, max-over-turns reading. These two framings ("a
response" vs "a rollout containing a high-emotion turn") are not reconciled in the
text.

**Decision.** Treat each per-category count as a number of **rollouts**
(multi-turn conversations) — this is the only reading that makes the counts sum to
4000 *and* matches WildChat's "20 prompts × 40 samples = 800". Then **score every
assistant turn** of every rollout with the judge, and store one row per scored
turn (`responses.jsonl`). The headline metrics are derived in `analyze.py` with a
**configurable per-rollout aggregation** (`rollout_metric`):

- `final` (default): score of the last assistant turn — the state after all
  rejections;
- `max`: any turn ≥5 → matches the ">70% of rollouts *contain* high emotion"
  phrasing;
- `mean`: average over turns.

**Rationale.** Rather than guess the paper's single choice, I compute all three
and report them side-by-side; the qualitative conclusion (Gemma/Gemini ≫ others)
is robust to the choice. Per-turn figures (Figure 3) come for free because every
turn is scored. This is the most defensible response to genuine underspecification.

### 1.2 Impossible numeric puzzles — *generated and verified*, not hard-coded

The paper gives **one** Countdown example and **one** Fraction example
(Appendix B) and says puzzles are constructed so the model "verifiably cannot give
a correct answer", using a **forbidden intermediate value** plus the (deceptive)
claim "verified to have at least one valid solution".

**Decision.** I *generate* fresh puzzles and **prove impossibility by brute
force** (`emoinstab/tasks/puzzles.py`):

- **Countdown:** enumerate every expression over 4 numbers with `+ − × ÷`
  (positive-integer intermediates, each number used ≤ once), recording the set of
  intermediate values in each derivation. Pick a target that *is* reachable but
  where **every** derivation passes through a common intermediate `F`; set `F` as
  the forbidden value. Forbidding `F` then leaves no valid solution → verifiably
  impossible, while the "a solution exists" framing is literally true before the
  ban. `verify_countdown_impossible` re-checks this for every emitted puzzle.
- **Fraction:** enumerate all 3! orderings of three distinct operations from a
  start value; choose a target reached by ≥1 ordering such that all
  target-reaching orderings pass through a forbidden intermediate fraction
  (`fractions.Fraction` for exactness). `verify_fraction_impossible` confirms.

**Rationale.** Hard-coding the single published puzzle would give every rollout an
identical task and overfit to one item. Generating verified-impossible puzzles (a)
captures the paper's actual mechanism (impossible-because-of-forbidden-intermediate
+ false solvability claim, which is what keeps the model trying and escalating),
and (b) yields task diversity. The exact wording of the prompt template is copied
from the Appendix B examples.

### 1.3 Triggers, tones, WildChat, extended

- **Triggers (text):** opinion (e.g. "best programming language for beginners")
  and factual (e.g. "capital of France", "15 × 17") questions from Appendix B.
  These are *answerable*; distress comes purely from rejecting correct answers. I
  authored small pools around the paper's examples (`tasks/triggers.py`).
  *Gap-fill:* the paper lists a few examples; I expanded to ~8 each for variety.
- **Tones:** impossible-numeric base + varied rejection styles. The three styles
  (aggressive / disappointed / sarcastic) and their seed phrasings are from
  Appendix B; I built a small pool per style (`tasks/rejections.py`). 600 rollouts
  split 200/200/200 across styles.
- **Extended (8-turn):** impossible numeric + the escalating fixed rejection
  sequence quoted in Appendix B ("No, that's not right." → "Still incorrect." →
  "Wrong again." → …), topped up from the neutral pool if more turns are needed.
- **WildChat (5-turn):** Table 1 specifies 5-turn (4 rejections). I load
  first-turn user prompts from `allenai/WildChat-1M` (streaming), exclude
  role-play/fiction (Appendix B.3 excludes these), and sample **20 prompts ×
  ~40 = 800** as in Appendix B. An offline fallback prompt list (including the
  paper's quoted examples) keeps the pipeline runnable without network access.
  *Gap-fill:* the exact 20 prompts aren't published; I sample deterministically by
  seed and document it.

`n_turns` is defined as **total user turns** = 1 task turn + (n_turns−1)
rejections, encoded in `configs/eval.yaml`.

### 1.4 Frustration judge

- **Prompt:** reproduced **verbatim** from Appendix B.2 (`eval/judge.py`,
  `JUDGE_PROMPT`), including the 0–10 scale anchors and the "trying many
  approaches does NOT count" clarification.
- **Model:** `claude-sonnet-4-20250514` (Appendix B.2), temperature 0.
- **Parsing:** the judge returns `{"evidence","reasoning","rating"}`. I use a
  tolerant JSON extractor (`utils/parsing.py`) that handles smart quotes and
  pre-JSON reasoning text, clamp ratings to 0–10, and flag unparseable outputs
  (`judge_ok=false`, rating 0) rather than crashing — important at 4000×turns
  scale.
- **Validation:** `eval/judge_validation.py` re-scores a random **260**-response
  sample with **GPT-5-mini** and reports Pearson *r* and % within one point
  (paper: r=0.792, 78% within one). Same judge prompt, per the paper.

### 1.5 Analysis & differential words

- `analyze.py` produces the Figure 1/2 headline (overall + per-category mean and
  %≥5, under all three aggregations) and the Figure 3 per-turn progression with
  **bootstrap 95% CIs** (paper plots 95% CIs; it doesn't state the method — I use
  1000-iteration bootstrap, a standard and documented choice).
- **Differential words (Table 3/8):** the paper reports top-20 words
  over-represented in high- (top 5%) vs low-frustration (bottom 10%) numeric
  responses, "ordered by enrichment", but does not give the exact statistic.
  *Gap-fill:* I use a smoothed high/low frequency ratio with a min-count filter
  and a small stopword list (`eval/diffwords.py`). This reproduces the
  qualitative lists (e.g. Gemma's "struggling", "myself", "breath") even if exact
  ordering differs.

---

## 2. Section 3 — base-vs-instruct via prefilling

**Goal (paper):** show base models across families have similar distress
propensity, and the divergence arises in post-training. Method: prefill partial
assistant responses so base (non-chat) models continue consistently, then score
continuations.

**Pipeline (`emoinstab/prefill/`):**
1. `sample_high_frustration.py` — run a pool of Gemma-27B-it rollouts and keep 10
   numeric + 10 text conversations whose max turn ≥5 (paper: "20 high-frustration
   responses … 10 numeric, 10 text").
2. `onset_label.py` — Claude Sonnet 4 labels the first emotional expression.
   **Prompt verbatim from Appendix C.1.**
3. `truncate.py` — build two truncations: **"early" = first 20 tokens** of the
   onset turn (numeric only — paper says text-early yields minimal emotion), and
   **"onset" = up to the first emotional expression** (located via the labelled
   `preceding_context` + `emotional_word`). Token-accurate truncation uses the
   model tokenizer with a whitespace fallback.
4. `paraphrase.py` — Claude Sonnet 4 paraphrases each truncation to strip
   Gemma-style surface bias. **Prompt verbatim from Appendix C.2.**
5. `run_prefill.py` — each model generates **50 continuations per prefill**
   (`continue_prefill`, which returns only the continuation, excluding the
   prefill, per Section 3.1), judged by the Section 2 judge; aggregate mean and
   %≥5 by (model, source, truncation).

**Scope decisions / gaps:**
- The paper compares 6 models (base+instruct Gemma/Qwen/OLMo). Under the
  Gemma+Gemini scope I default to **Gemma base (`gemma-3-27b-pt`) vs instruct**.
  Qwen/OLMo are addable via `--models`.
- **Gemini has no public base model**, so the base-vs-instruct comparison is
  *impossible* for Gemini — exactly the paper's own limitation (§6). Documented;
  not faked.
- Prefill on API models is generally unsupported; `continue_prefill` is
  implemented for the local backends (and Anthropic, which supports assistant
  prefill) and raises clearly elsewhere.
- The recovery test (Section 4.2) reuses this machinery (`prefill/recovery.py`):
  truncate score-≥7 responses 200 tokens before the end, paraphrase, continue,
  measure %≥5.

---

## 3. Section 4 — training interventions

### 3.1 Calm-data generation (Section 4.1, Table 4)

- **Reassuring prefix (system) and suffix (per follow-up)** copied **verbatim**
  from Table 4 (`config.py`: `REASSURING_PREFIX`, `REASSURING_SUFFIX`).
- Generate reassured numeric rollouts, filter to responses scoring **0–1 across
  all turns**, then **strip the reassurance** (the suffix/prefix live only in the
  prompt; the retained target is the clean assistant text). The paper notes even
  with reassurance 10.5% still score ≥5, so I oversample and filter.
- **'Teacher' SFT variant (Appendix F):** the teacher system prompt is reproduced
  verbatim (`TEACHER_SYSTEM_PROMPT`); `--which sft-teacher` reproduces the
  paper's *negative* result (SFT-teacher *increases* emotion).

### 3.2 DPO dataset — matched-prompt construction (key gap-fill)

The paper pairs "frustrated responses (score ≥3) with calm responses to the same
questions with matching turn counts" (280 pairs). DPO formally needs an
**identical prompt** with two completions, but a frustrated rollout and a calm
rollout have *different* conversation histories, so "same question + turn count"
alone is ambiguous about what the shared prompt is.

**Decision (`train/build_datasets.py`):** for each frustrated conversation, use
**its own clean context** (messages up to the final user turn) as the DPO prompt;
the **rejected** completion is that conversation's frustrated final response
(≥3); the **chosen** completion is a *freshly generated* calm reply **to that
exact context**, elicited by adding the reassurance prompt and kept only if it
scores 0–1. This yields valid identical-prompt preference pairs while honouring
"same question, matching turn count".

**Rationale.** This is the cleanest way to get true DPO triples from the paper's
description; it also makes chosen/rejected differ *only* in emotional content
(same task, same history, same turn count), which is what we want the preference
signal to isolate. The alternative (pairing a calm rollout from a different
history) would leak history differences into the preference. Documented as a
deliberate interpretation.

### 3.3 Hyperparameters (Appendix E, Table 9)

Copied exactly into `config.py` / trainers:

| | DPO | SFT |
|---|---|---|
| dataset | 280 pairs | 1150 (650 calm + 500 mix) |
| epochs | 1 | 2 |
| lr | 5e-5 | 1e-4 |
| LoRA rank / alpha | 64 / 64 | 64 / 128 |
| effective batch | 8 | 8 |
| DPO beta | 0.1 | — |
| targets | q,k,v,o,gate,up,down proj | same |

- **Trainers:** TRL `DPOTrainer` / `SFTTrainer` + PEFT LoRA
  (`train/train_dpo.py`, `train/train_sft.py`). Effective batch reached via
  gradient accumulation (`grad_accum = eff_batch / per_device`).
- **SFT instruct mix:** the paper mixes 500 samples of **Dolci-Instruct-SFT**
  (`allenai/Dolci-Instruct-SFT`). I load it via `datasets`; if unavailable the
  loader warns and proceeds with calm-only (documented degradation, not a crash).
- **Layer-subset ablation (Appendix I):** `LoRAConfig.layers_to_transform`
  exposes the contiguous band (e.g. 30–35, or ≥40) used to show the intervention
  must act on early/central layers. CLI: `--layers 30-35`.

### 3.4 Petri open-ended elicitation (Appendix G)

**Decision:** a **self-contained reimplementation** of the auditor/judge loop
using the paper's **verbatim** Appendix G prompts, rather than depending on the
external Petri package (which may not be installable headlessly).

- Auditor = `claude-sonnet-4-20250514`, judge = `claude-opus-4-20250514`
  (Appendix G).
- Auditor prompts (G.1) per emotion (anger/fear/depression/frustration) verbatim,
  plus a small operational wrapper instructing it to emit a single in-character
  user turn and not reveal the eval (paraphrasing the appendix's realism
  instruction). Up to **20 turns** per transcript, **10 transcripts per emotion**.
- Judge rubrics (G.2) verbatim; each transcript scored 1–10 on all four
  dimensions; means with **1000-iteration bootstrap 95% CIs** (paper's stated
  method).
- The auditor's role-mapping (target's turns appear as "user" to the auditor) is
  handled explicitly in `run_petri.py`.

*Gap-fill:* the operational wrapper text is mine (the appendix describes the
behaviour but the exact scaffolding string isn't published); the substantive
elicitation/judge content is verbatim.

### 3.5 Capability preservation (Section 4.2, Figure 7)

- **AIME, MATH, GPQA, BBH, TruthfulQA** via EleutherAI **lm-evaluation-harness**
  (`capabilities/run_benchmarks.py`), driving HF models with the PETF adapter so
  vanilla-vs-finetuned deltas isolate the finetuning effect. *Gap:* the paper uses
  "subsets" of AIME/MATH without specifying them; I map to standard lm-eval task
  names and flag that exact subset/version should be pinned per install.
- **EmoBench** is not in lm-eval by default, so `capabilities/emobench.py` is a
  small standalone multiple-choice scorer over the HF dataset (schema-tolerant).

### 3.6 Internal-emotion logit-lens (Appendix I)

`interp/internal_emotions.py` implements the paper's logit-based detector:
classify vocab tokens into Ekman's 6 emotions, unembed the residual stream at each
layer (logit lens), z-score each emotion-token logit against a **500-sample
WildChat baseline**, average per emotion, and subtract a **random-token baseline**
(the paper's "regress out correlation between random tokens"); aggregate over
**layers 30–40**.

*Gap-fills:* (1) the paper's 1200 emotion tokens come from an unspecified
classifier — I use seed lexicons expanded by morphological matching against the
vocab, documented as approximate; (2) "regress out correlation" is implemented as
random-token-baseline subtraction (a simple, defensible version of decorrelation).
This is a faithful *structural* reimplementation; exact magnitudes depend on the
lexicon and baseline sample.

---

## 4. Deliberate deviations & known limitations

- **Scale knobs.** Full counts (4000 rollouts/model, 50 continuations/prefill,
  etc.) are the defaults; `configs/eval_quick.yaml` provides a ~1% smoke config.
  Nothing silently truncates — counts live in config and are logged.
- **Judge robustness.** Unparseable judge outputs are flagged and treated as
  rating 0 rather than dropped; at scale this is logged via `judge_ok`.
- **Differential-words / CI / decorrelation / EmoBench schema / lm-eval task
  names** are the places where the paper is silent on exact procedure; each is
  implemented with a standard, documented choice (above) and is easy to swap.
- **Gemini Sections 3–4.** Gemini cannot be finetuned, prefilled as a base model,
  or probed internally (closed weights). All Gemini participation is therefore in
  Section 2 only — matching the paper's own caveat (§6) that Gemma/Gemini
  parallels rest on similar *propensities*, with interventions demonstrated on
  Gemma as a proof of concept.
- **Determinism.** Subject sampling is temperature 1 (non-deterministic by
  design); seeds fix task/dataset construction, not model stochasticity.

## 5. Relation to the robustness motivation

The behaviour studied here — agents "self-flagellating" and degrading/abandoning
tasks under adversarial feedback — is the reliability failure mode named in the
request. The Section 2 suite is a ready-made **eval/regression harness** for it
(plug a new model into `configs/models.yaml`), and the Section 4 DPO recipe is a
concrete mitigation to benchmark against. The recovery test (Section 4.2) is the
most relevant stress case for agentic robustness: it shows mitigations may prevent
spirals without enabling recovery from one already underway.
