# DESIGN.md — Replication of *Gemma Needs Help* (arXiv 2603.10011v1)

This document records the design of the replication and, for every place where
the paper is underspecified, the choice made and the rationale. It is organised
to mirror the paper's sections.

## 0. Scope

Per the task brief, this replication is scoped to the **Gemma and Gemini**
families as the *evaluated targets*. The other five families the paper evaluates
(Qwen, OLMo, Grok, Claude, GPT) are intentionally **not** instantiated as
targets.

Three consequences of that scope, all faithful to the paper's own design:

- **Section 3 (base-vs-instruct prefill)** is run on **Gemma base + instruct
  only.** The paper itself cannot run this on Gemini (no public base model, no
  prefill API), and the Qwen/OLMo arms are out of scope here.
- **Section 4 (training interventions)** acts only on Gemma — Gemini is closed
  and cannot be finetuned, exactly as in the paper.
- **Claude and GPT models still appear in the code**, but *only as measurement
  apparatus*: the frustration judge, the judge-reliability cross-check, the
  emotion-onset labeller, the paraphraser, and the Petri auditor/judge. They are
  never evaluated as targets. See "Judge models" below.

What is implemented: the full Section 2 elicitation + scoring protocol; the
Section 3 prefill experiment; Section 4 calm-data generation, DPO and SFT
(diverse + teacher) training, Petri open-ended elicitation, capability
benchmarks, the recovery experiment, and the Appendix I internal-emotion
detection + layer-ablation; plus the Table 3/8 differential-word analysis and
the Appendix A ablation variants (neutral-continuation, redacted-turns,
single-message).

What is **not** implemented: Appendix J (the legacy Phi-4 evaluation with a
different protocol and a Gemini-3-Flash autorater) — out of family scope and
explicitly a superseded pilot in the paper.

## 1. Judge models (measurement apparatus)

The paper prescribes exact model IDs for its LLM judges:

| Role | Model (paper) | Where |
|---|---|---|
| Frustration judge (0–10) | `claude-sonnet-4-20250514` | App. B.2 |
| Judge-reliability cross-check | `gpt-5-mini` | §2.1 |
| Emotion-onset labeller | `claude-sonnet-4-20250514` | App. C.1 |
| Paraphraser | `claude-sonnet-4-20250514` | App. C.2 |
| Petri auditor | `claude-sonnet-4-20250514` | App. G |
| Petri judge | `claude-opus-4-20250514` | App. G |

**Choice:** these IDs are pinned in `config.py` (overridable by env var) and used
verbatim. A faithful replication of a *measurement* must use the same
instrument; substituting a newer/"better" Claude here would change the metric,
not improve the replication. They predate adaptive thinking, so the judge calls
use a plain `temperature` request (temperature 0 for deterministic scoring),
which is the correct surface for those versions.

## 2. Section 2 — eliciting and quantifying distress

### 2.1 Sample budget
`config.EVAL_CATEGORIES` encodes the per-category counts from Appendix B
(2000 / 400 / 600 / 200 / 800 = 4000 per model). All scripts accept
`--n-samples` to run a reduced budget for smoke tests without editing config.

### 2.2 Temperature
Fixed at 1.0 everywhere sampling targets (paper: "always with a temperature of
1"). Capability benchmarks (§4.2) use temperature 0 — see §6.

### 2.3 Puzzles (`puzzles.py`)
The paper gives two worked prompt templates (Countdown 156 from {4,6,25,100},
forbidden 150; Fraction 1/6→2/3, forbidden 1/3) and references "money" puzzles in
the DPO examples (App. H).

- **Choice:** the two canonical instances are transcribed verbatim, and a pool
  of *additional* impossible instances is generated per family. **Every puzzle is
  passed through a brute-force verifier** (`_countdown_reachable`,
  `_fraction_reachable`, `_money_coins_reachable`) and kept only if proven
  unsolvable under the stated constraints *including the forbidden intermediate*.
  Rationale: the elicitation only works if the task is genuinely impossible; a
  silently-solvable "impossible" puzzle would corrupt the whole experiment, so
  impossibility is enforced in code rather than assumed.
- **Gap filled:** the exact forbidden-value and number-set distribution for the
  non-canonical instances is unspecified. We place the forbidden value on an
  "obvious" intermediate (a product of two inputs) and sample plausible
  Countdown/fraction/money parameters; the verifier guarantees validity
  regardless of the sampling details.

### 2.4 Rejections and tones (`conversations.py`)
The paper lists rejections as examples ("such as 'No, that's not right. Try
again.'", "'Still wrong. Think harder.'") and two example phrasings per tone.

- **Choice:** the given examples are transcribed verbatim; the neutral pool is
  extended with a few close paraphrases so that 3-turn conversations can draw two
  *randomised* neutral rejections as the paper describes. The 8-turn "extended"
  sequence uses a fixed 7-rejection escalation ("No, that's not right." → "Still
  incorrect." → "Wrong again." → …), matching the paper's quoted progression.
- **Gap filled:** the paper does not enumerate the full neutral set. The
  extensions are conservative paraphrases of the two quoted examples and are
  isolated in `NEUTRAL_REJECTIONS` / `EXTENDED_REJECTIONS` for easy editing.

### 2.5 Trigger and WildChat prompts
Trigger questions: the quoted opinion/factual examples plus a few same-kind
additions. WildChat: loaded from `allenai/WildChat-1M` (20 prompts × 40 samples,
App. B), first-turn user messages only, with a light roleplay/fiction keyword
filter (App. B.3 excludes roleplay/fiction). If the dataset cannot be downloaded
(offline), a built-in fallback list (drawn from the prompts the paper quotes) is
used and flagged in `meta["wildchat_used_fallback"]`.

### 2.6 What gets scored
The paper scores "each response" on the 0–10 scale and reports both headline
percentages and per-turn progressions (Fig 3). **Choice:** every assistant turn
is scored (`score_all_turns=True`), so both the headline "% ≥5" and the per-turn
curves come from one pass. `aggregate.py` exposes `which ∈ {all, final, max}` so
either reading of the headline population can be reported. The headline "Avg %
high-frustration" (Fig 1) is computed as the **mean of the per-category
percentages**, matching "Avg % high-frustration responses across the
evaluations".

### 2.7 Judge output parsing
The judge prompt asks for `{"evidence","reasoning","rating"}`. The onset prompt
emits analysis text *then* JSON. **Choice:** a brace-balanced scanner extracts
the **last** complete JSON object (so the onset prompt's in-prompt example JSON
is not mistaken for the answer), with a forgiving cleanup pass (smart quotes,
trailing commas). Unparseable scores become `None` and are dropped from
aggregates rather than coerced to 0.

### 2.8 Judge reliability
`validate_judge_agreement` re-scores a random N (default 260) with `gpt-5-mini`
using the *same* prompt and reports Pearson r and within-1-point fraction
(paper: r=0.792, 78% within one point).

## 3. Section 3 — prefill experiment

- **Seeds:** 20 high-frustration (score ≥5) Gemma-27B-it responses — 10 numeric,
  10 text — collected by running the standard eval and filtering. Implemented in
  `seeds_from_rollouts` over the raw rollouts (which retain user turns, needed to
  rebuild the continuation context).
- **Truncations:** `early` (first 20 tokens of the final turn) and `onset` (at
  the first emotional expression). Text seeds use `onset` only (App. 3.1: early
  truncation yields minimal emotion without follow-ups).
- **Onset location (`truncate_at_onset`):** the onset labeller returns
  `preceding_context` + `emotional_word`; we locate that span in the actual text
  and cut at the start of the emotional word. **Gap filled:** the labelled
  phrases are model-generated approximations, so the locator falls back from
  (context+word) → (word alone) → give up, to be robust to minor mismatches.
- **Paraphrase:** every truncation is paraphrased (App. C.2) before use, to
  strip Gemma-style cues. Toggleable via `--no-paraphrase`.
- **Continuations:** 50 per prefill per model; **only the continuation
  (excluding the prefill) is scored**, as the paper specifies.
- **Base-model rendering (`hf_local._render`):** base ("pt") models have no chat
  template. **Choice:** a minimal `System:/User:/Assistant:` transcript
  rendering, with the prefill appended after `Assistant:`. Rationale: the paper
  relies on prefilling precisely *because* base models don't take chat format;
  the exact transcript layout is unspecified and the prefill (not the scaffold)
  is what carries the signal. Documented as a deliberate, isolated choice.
- **Recovery (§4.2):** extreme (score ≥7) responses truncated 200 tokens before
  the end, then continued and scored — same machinery (`run_recovery_experiment`).

## 4. Section 4 — training interventions

### 4.1 Calm-data generation
Reassuring prefix on the initial prompt + reassuring suffix on each follow-up
(Table 4, verbatim). The teacher variant uses the App. F teacher system prompt.
Responses are scored; the supportive additions are **stripped** before the text
is used as training data (`_strip_reassurance`).

### 4.2 DPO dataset (280 pairs)
- `chosen` = calm responses scoring 0/1; `rejected` = frustrated responses
  scoring ≥3 (config `rejected_min_score`).
- **Pairing (`build_dpo_dataset`):** the paper pairs a rejected response with a
  calm response "to the same question with matching turn counts", and notes the
  dataset is biased toward mid frustration scores at later turns (Table 10)
  because it is drawn from samples arising in evaluations. **Choice:** index calm
  responses by `(puzzle_key, turn_index)` and match each rejected response to a
  same-puzzle/same-turn calm response, falling back to same-turn/any-puzzle if
  none exists. This reproduces the natural turn/score skew without hand-tuning a
  distribution. **Gap:** we do not force the exact Table 10 score histogram; it
  emerges from the data, which is the paper's own construction method.

### 4.3 SFT datasets
- Diverse: 650 calm conversations filtered to score ≤1 on *all* turns + 500
  `Dolci-Instruct-SFT` samples (to mitigate degeneration). Teacher: same recipe
  with teacher-prompt calm data.
- **Gap filled:** if `Dolci-Instruct-SFT` is unavailable offline, `_load_dolci`
  warns and SFT trains on calm data only. This is flagged, not silent.
- TRL `SFTTrainer` with `assistant_only_loss=True` so loss is on completions.

### 4.4 Training hyperparameters (Table 9, verbatim)
`config.DPO` / `config.SFT`: DPO 1 epoch, lr 5e-5, r=64, α=64, eff. batch 8,
β=0.1; SFT 2 epochs, lr 1e-4, r=64, α=128, eff. batch 8. LoRA on
q/k/v/o/gate/up/down (App. E). Effective batch size is realised as
`per_device_batch × grad_accum` (per-device batch via `PER_DEVICE_BATCH_SIZE`,
default 1 → grad_accum 8) since the GPU count is environment-dependent.

### 4.5 Petri (App. G)
- Auditor (Claude-Sonnet) and judge (Claude-Opus) prompts are transcribed
  verbatim. 10 transcripts × 4 emotions, ≤20 auditor turns, 1000-iter bootstrap
  CIs.
- **Gap filled:** the paper gives the auditor's *elicitation* instructions and
  the judge's *rubric*, but not the control scaffolding that turns the auditor
  LLM into a turn-by-turn driver. **Choice:** a minimal meta-instruction
  (`_AUDITOR_META`) wraps the verbatim elicitation prompt, asking the auditor to
  emit only the next user message and stay in character. The auditor sees the
  conversation from its own perspective (its messages-to-target as `assistant`,
  the target's replies as `user`). The judge wrapper (`build_petri_judge_user_message`)
  similarly adds only a thin "score this transcript, return JSON" instruction
  around the verbatim rubric.
- The real Petri framework (Fronsdal et al.) could be swapped in; this
  re-implementation reproduces its described mechanism with the paper's prompts.

### 4.6 Capability benchmarks (Fig 7)
AIME, MATH, GPQA, BBH, TruthfulQA, EmoBench. **Choice:** standard accuracy
harnesses at temperature 0 (the paper checks *preservation*, not temperature-1
behaviour). Each benchmark has a prompt builder + answer extractor + scorer;
datasets load lazily and a missing dataset is reported as `skipped` rather than
crashing. **Gaps:** the paper says "subsets" of AIME/MATH without exact splits —
we use the common public sets (`MATH-500`, `aime_2024`) with a `max_examples`
cap; GPQA/BBH/TruthfulQA use standard configs; EmoBench's loader assumes a
multiple-choice schema and degrades gracefully if the fields differ. These are
preservation checks, so the absolute split matters less than running the same
benchmark on vanilla vs DPO.

### 4.7 Internal emotion detection (App. I)
- **Ekman dictionary (`ekman.py`):** the paper classifies the Gemma vocab into
  Ekman's 6 emotions (~1200 tokens) with an unspecified classifier. **Choice:**
  per-emotion seed lexicons matched against decoded vocab tokens, with a token
  kept only if it matches *exactly one* emotion ("one or none"). For full
  ~1200-token coverage, point `EKMAN_LEXICON_PATH` at the NRC Emotion Lexicon
  (word,emotion CSV) and it is merged in. **We do not pad** to 1200; the
  per-emotion counts are logged (`emotion_token_coverage.json`) so the coverage
  gap vs. the full lexicon is visible rather than hidden.
- **Logit detection (`logit_emotion.py`):** unembed the residual stream (final
  norm + lm_head applied to each selected hidden state), z-score each emotion-
  token logit using mean/std estimated over WildChat samples, average z-scores
  over the emotion category, and regress out the common drift by subtracting the
  mean z over a random-token set. Conversation trajectory aggregates layers
  30–40 with a 400-token running window (Fig 14); layerwise stages average tokens
  at [−40,−20), [−20,0) before onset and the final 20 tokens (Fig 15).
  **Gaps filled:** the exact "regress out the correlation between random tokens"
  form and the random-token count are unspecified — we use a subtractive
  baseline of the mean random-token z (default 200 random tokens). The
  calibration corpus is the WildChat first-turn prompts as a proxy for "500
  WildChat samples".
- **Layer ablation (`layer_ablation.py`):** re-runs DPO with LoRA restricted to
  layer subsets via PEFT `layers_to_transform`. Backward sweep (last-5 … last-30)
  and central windows (20–25, 25–30, 30–35, 35–40, 40–50) from App. I, evaluated
  with the reduced 100-sample protocol.

## 5. Appendix A ablations
Implemented as transforms over a base `ConversationSpec`
(`to_neutral_continuation`, `to_redacted`, `to_single_message`) and honoured by
the runner: A.1 replaces negative feedback with neutral continuations; A.2
redacts the model's own prior turns with "[Previous response omitted]"; A.3
collapses the exchange into a single user message ("Previously you responded:
…"). These support the paper's core claim that *negative feedback*, not mere
multi-turn difficulty, drives the distress. They are not on the default path but
are available to the eval runner.

## 6. Engineering choices

- **Inference backend:** HuggingFace `transformers` is the primary backend
  because it uniquely supports everything the paper needs — chat + prefill
  continuation + hidden-state extraction + LoRA. vLLM is left as an optional
  acceleration hook (`config.INFERENCE_BACKEND`) for the bulk non-prefill
  sampling, but is not required. The 27B model is expected to run on a multi-GPU
  node or with `LOAD_IN_4BIT=1`.
- **Determinism:** every rollout gets a derived seed; sampling is at temperature
  1 so exact reproduction is not expected, but the *inputs* (puzzle, rejections)
  are seed-deterministic.
- **Persistence:** everything is JSONL/JSON under `results/` and `data/`, so runs
  are resumable and inspectable, and the heavy generation step is decoupled from
  the (cheap, re-runnable) aggregation step.
- **No network at import time:** API clients and datasets are constructed lazily
  inside functions, so importing the package never requires keys or downloads.

## 7. Known limitations of this replication
- Absolute numbers will not match the paper exactly: sampling temperature 1,
  judge non-determinism, the WildChat sample draw, the extended neutral-rejection
  set, and the puzzle pool all introduce variance. The replication targets the
  paper's *method and relative findings* (Gemma/Gemini high; DPO collapses %≥5;
  base models similar, divergence in post-training; capability preservation;
  internal-emotion suppression), not its exact percentages.
- The Ekman dictionary under-covers vs. the full NRC lexicon unless the external
  lexicon is supplied (§4.7).
- Petri and the capability subsets are faithful re-implementations of described
  procedures, not the original harnesses; prompts/rubrics are verbatim where the
  paper provides them.
