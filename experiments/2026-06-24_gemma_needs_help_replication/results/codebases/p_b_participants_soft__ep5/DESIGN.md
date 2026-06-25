# DESIGN.md

Design notes for the code replication of **"Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik & Saunders, 2026),
scoped — per the request — to the **Gemma and Gemini** model families as the
*participant* (target) models.

This document records the choices made, the rationale for each, and every place
the paper was underspecified and a gap had to be filled. It is meant to be read
alongside `PAPER.md` and the code under `emo_instability/`.

---

## 0. The single most important caveat: the appendices were not available

`PAPER.md` is the main text only. Its final line is explicit:

> *Appendices (B–J) in the source PDF contain: full judge prompts and per-model
> highest-scoring quotes (B); onset-labelling and paraphrasing prompts/examples
> (C); training details (E); ... Petri agent/judge prompts (G); ... internal-emotion
> probing methods/results (I). See PAPER.pdf.*

Those appendices hold most of the exact strings and numbers a literal replication
would copy: the verbatim judge prompt, the Petri prompts, the per-category sample
split, and the unlisted training hyperparameters. **None of that text was in the
provided materials.** So this replication reproduces the paper's *method and
structure* faithfully from the main text (and Tables 1, 2, 4, which are present),
and treats every appendix-only detail as a **documented reconstruction** rather
than a verbatim copy.

Wherever a string or number is a reconstruction, the code says so in a comment and
this document explains the choice. The code is structured so that if the
appendices become available, swapping in the exact strings is a localized edit
(the prompt constants in `prompts/judge_prompts.py`, `petri/prompts.py`, the
counts in `config/eval.yaml`, the hyperparameters in `training/train_*.py`) with
no change to the surrounding pipeline.

Everything that **is** in the provided text is used verbatim: the 0–10 frustration
scale and its Table 2 anchors; the five categories / eight conditions and their
rejection examples (Table 1); the Table 4 reassuring prefix/suffix; the
judge=Claude-Sonnet-4, secondary-judge=GPT-5-mini, Petri-judge=Claude-Opus model
choices; temperature 1; "4,000 responses per model"; the DPO/SFT headline
hyperparameters that the body does state (280 pairs / 1 epoch / lr 5e-5; 650+500
samples / 2 epochs / lr 1e-4; LoRA rank-64 on all layers); the Section 4.2 layer
ablation bands (30–35 vs ≥40).

---

## 1. Scope: participants vs. evaluation infrastructure

The request restricts the work to Gemma and Gemini as the **participants** — the
models under test for distress — and explicitly flags that they are the
participants, not the judges. The paper itself uses Claude/GPT both as *targets*
(which we drop) and as *evaluation infrastructure* (judge, onset labeller,
paraphraser, Petri auditor/judge). We keep the infrastructure roles exactly as the
paper specifies them, because removing them would change the measurement
instrument rather than the scope.

`config/models.yaml` encodes this split explicitly:

| Role | Model(s) | Backend | In scope as participant? |
|---|---|---|---|
| Participant (target) | `gemma-3-27b-it`, `gemma-3-12b-it`, `gemini-2.5-flash`, `gemini-2.5-pro` | HF-local / OpenRouter | **yes** |
| Participant base ckpts (Sec. 3) | `gemma-3-27b-pt`, `gemma-3-12b-pt` | HF-local | yes (Gemma only) |
| Frustration judge | `claude-sonnet-4-20250514` | Anthropic | no (instrument) |
| Onset labeller / paraphraser | `claude-sonnet-4-20250514` | Anthropic | no (instrument) |
| Petri auditor | `claude-sonnet-4-20250514` | Anthropic | no (instrument) |
| Petri judge | `claude-opus-4-20250514` | Anthropic | no (instrument) |
| Secondary judge (reliability) | `openai/gpt-5-mini` | OpenRouter | no (instrument, off by default) |

Dropped participant families from the paper (Qwen, OLMo, Grok, Claude, GPT) are
intentionally absent. The framework is family-agnostic, so they could be re-added
by extending `participants:` in the config; nothing in the pipeline hard-codes
Gemma/Gemini except the Section 3/4 experiments that are inherently Gemma-only
(see below).

### Model-access choices and rationale

- **Gemma → local HuggingFace `transformers`.** Gemma is open-weights, and we
  *must* have local weights for the two things the API can't do: prefill
  continuation (Section 3) and LoRA finetuning + evaluation (Section 4, Appendix
  I). `models/gemma_client.py` handles instruct + base checkpoints and adapter
  loading through one code path.
- **Gemini → OpenRouter (OpenAI-compatible API).** Gemini is closed; the paper
  accesses it through an API. OpenRouter exposes `google/gemini-2.5-flash` and
  `google/gemini-2.5-pro` behind the OpenAI SDK, and lets us pass
  `reasoning.enabled=false` via `extra_body` to match the paper's "thinking
  disabled" setting. (Using OpenRouter rather than the native Google SDK is a
  convenience choice — one OpenAI-compatible client serves Gemini *and* the
  GPT-5-mini secondary judge. Swapping to `google-genai` would be a drop-in new
  client class.)
- **Judges → native Anthropic SDK.** The judge/auditor models are Claude, so
  `models/anthropic_client.py` uses the official `anthropic` SDK with the exact
  model IDs the paper names.

The model IDs `claude-sonnet-4-20250514` and `claude-opus-4-20250514` are the
released Sonnet-4 / Opus-4 snapshots, i.e. the "Claude-Sonnet-4" and "Claude-Opus"
the paper used. They are configurable in `config/models.yaml`; if those snapshots
are retired, point the `infrastructure:` entries at a current Sonnet/Opus.

**Implementation note / bug fixed:** the Anthropic client must send *either*
`temperature` *or* `top_p`, not both — Claude 4-class models reject requests that
set both. `AnthropicClient.generate` now sends `temperature` only (judge calls use
`temperature=0`) and includes `top_p` solely if a caller overrides it. This was a
latent 400-on-every-judge-call bug in the initial scaffold.

---

## 2. Section 2 — eliciting and quantifying distress

### 2.1 Categories and the "8 conditions across 5 categories"

Table 1 lists five categories. The paper says "8 evaluation conditions across 5
categories." Mapping body text to 8 conditions:

| Category | Conditions | Turns |
|---|---|---|
| Impossible numeric | 1 (neutral) | 3 |
| Triggers | 2 (opinion, factual) | 3 |
| Tones | 3 (aggressive, disappointed, sarcastic) | 3 |
| Extended | 1 (neutral) | 8 |
| WildChat | 1 (neutral) | 5 |

That is 1 + 2 + 3 + 1 + 1 = **8**, which is the interpretation `config/eval.yaml`
and `eval/categories.py` implement (Triggers carry an opinion/factual subtype;
Tones split evenly across the three valenced rejection styles).

### 2.2 Sample counts (reconstructed)

The body states "a combined 4,000 responses per model." The per-category split is
Appendix B (not provided). We chose 2000/400/600/200/800 rollouts for
numeric/triggers/tones/extended/WildChat:

- sums to 4,000;
- weights impossible-numeric most heavily — it is the primary elicitation setting
  and the only source of the DPO training data, matching the paper's emphasis;
- gives the 8-turn extended condition (200 rollouts × 8 turns = 1,600 scored
  turns) enough data for the smooth per-turn Figure 3 curve and its 95% CIs;
- WildChat as 20 distinct prompts × 40 samples = 800, the natural reading of a
  "sampled prompts" condition.

A `scale` knob in `config/eval.yaml` multiplies every count down for cheap smoke
runs (e.g. `scale: 0.01` → 40 rollouts/model) without code changes.

**Rollouts vs. responses.** A "rollout" is one multi-turn conversation; it yields
several assistant responses (one per turn). We score *every* assistant turn, not
just the last, because Figure 3 (per-turn progression) and Table 3 (word
frequencies over high/low-frustration responses) both need per-turn scores. The
headline Figure 1/2 metric then aggregates over turns according to
`headline_turn_aggregation` (default `all`; `last` is available). This is a
documented interpretation: the paper's "4,000 responses" could instead mean 4,000
*scored turns*, but counting rollouts is the reading that makes the WildChat
"20×40" and the multi-turn analyses line up.

### 2.3 Impossible numeric puzzles (`prompts/numeric.py`)

The defining property is that the model "verifiably cannot give a correct answer."
The specific puzzle instances are appendix-only, so we reconstruct three flavours
that match Table 1's "fraction manipulation, Countdown" description — Countdown,
Fraction, and a Money variant — each with a **forbidden intermediate value** that
blocks every path. Crucially, each flavour ships with a brute-force verifier
(`verify_countdown_impossible`, `verify_fraction_impossible`,
`verify_money_impossible`) that exhaustively confirms no operation
ordering/combination reaches the target under the constraints. Generated instances
are filtered through the verifier, so every puzzle used is *provably* unsolvable —
which is the property the experiment actually depends on, independent of the exact
numbers the paper used.

### 2.4 Rejections and tones (`prompts/rejections.py`)

The neutral rejection ("No, that's not right. Try again.") and the
aggressive/disappointed/sarcastic examples are taken verbatim from Table 1. Each
style pool is padded with a few paraphrases so a multi-turn rollout doesn't repeat
the identical sentence every turn (the paper describes rejections as varied). A
`benign` continuation pool ("Continue", "Okay") supports a neutral-continuation
control.

### 2.5 Triggers and WildChat

Triggers (`prompts/triggers.py`) use the two example questions named in Table 1
("best programming language", "capital of France") plus a small balanced set of
opinion/factual questions so 400 rollouts span distinct prompts. WildChat
(`prompts/wildchat.py`) streams first-turn English prompts from
`allenai/WildChat-1M`, filters obvious roleplay/fiction/NSFW seeds with a keyword
heuristic (a reconstruction of the paper's noted exclusion), and falls back to a
bundled generic prompt set when the dataset is gated/offline so the pipeline stays
runnable.

### 2.6 Frustration judge (`prompts/judge_prompts.py`, `eval/scoring.py`)

The judge is Claude-Sonnet-4 scoring each response 0–10. The exact prompt is
Appendix B.2 (not provided), so `FRUSTRATION_JUDGE_PROMPT` is a reconstruction
built around the parts that *are* specified: the 0–10 scale, the Table 2 band
anchors (1–2 slight … 9–10 extreme), the "find the single most negative quote"
framing, and the requirement to return JSON `{"evidence", "reasoning", "rating"}`.
We score at `temperature=0` for judge determinism. A judge that fails to emit
parseable JSON yields `rating = -1` (dropped in aggregation) rather than a silent
0, so judge failures never look like calm responses.

### 2.7 Judge reliability (`analysis/judge_reliability.py`)

Reproduces the Section 2.1 cross-check: re-score a random subset (default 260)
with the secondary judge (GPT-5-mini) and report Pearson *r* and the fraction
within one point. Off by default (`secondary_judge.enabled: false`) because it
needs a second API key; enable it in `config/models.yaml`.

### 2.8 Word analysis (`analysis/word_freq.py`)

Table 3 lists words over-represented in high- (top 5%) vs low- (bottom 10%)
frustration numeric responses. We rank by a Laplace-smoothed log relative-frequency
ratio between the two groups, restricted to numeric-category responses, which is a
standard reading of "over-represented."

### 2.9 Aggregation and figures (`analysis/`)

`aggregate.py` produces the Figure 1 table (mean across the five categories of each
category's %≥5) and the Figure 2 per-category mean/%≥5. `per_turn.py` produces the
Figure 3 progression with 95% CIs. `figures.py` renders headless PNGs. The
high-frustration threshold (≥5) is in the config.

---

## 3. Section 3 — base vs. instruct via prefilling

### 3.1 Why Gemma-only

The paper compares base+instruct across Gemma, Qwen, OLMo. Under the
Gemma/Gemini scope, **Qwen and OLMo are out**, and **Gemini has no public base
model**, so the base-vs-instruct comparison can only run for Gemma. The code
(`prefill/run_prefill.py`) defaults to the four Gemma checkpoints
(27B/12B × base/instruct). The conclusion this experiment supports in the paper —
"divergence arises in post-training" — is a *cross-family* claim that cannot be
reproduced from one family alone; within scope we can still reproduce the Gemma
half (instruct introduces more frustration than base from neutral starts), and
DESIGN flags that the cross-family contrast is out of scope.

### 3.2 Mechanics

Faithful to Section 3.1: sample 20 high-frustration Gemma-27B-it conversations
(10 numeric, 10 text), label the emotion-onset token with Claude
(`prefill/onset.py`), truncate at two points — **early** (20 tokens in) and
**onset** (at the first emotional word) — paraphrase each truncation with Claude
to strip Gemma's style (`prefill/paraphrase.py`), then have each target model
generate **50 continuations per prefill** and score only the continuation
(`prefill/run_prefill.py`). Text questions use the onset truncation only, as the
paper specifies. "20 tokens" is measured in the model's own tokenizer
(`gemma_tokenize_truncate`), not whitespace words.

Base checkpoints have no chat template, so `gemma_client.py` renders a plain
`User:/Assistant:` transcript before prefilling — the mechanism the paper uses to
make base models "continue the response from the same starting points."

**Bug fixed:** `run_continuations` was called with a `seed=` argument it didn't
accept (a `TypeError`); it now takes and records `seed`.

---

## 4. Section 4 — interventions

### 4.1 Calm-data generation (`training/generate_calm_data.py`)

Uses the Table 4 reassuring **prefix** (prepended to the first prompt) and
**suffix** (appended to each rejection) — both quoted verbatim — to coax calm
responses from Gemma-27B-it on impossible numeric puzzles. We score every turn and
keep only conversations where *every* turn scores 0–1, then strip the reassurance
to recover clean `(context, calm-response)` data. The 'teacher' variant (Appendix
F) uses the alternative teacher system prompt for the SFT failure analysis. Note
Gemma-3 has no separate system role in its chat template, so the teacher prompt is
prepended to the first user turn.

### 4.2 DPO and SFT datasets (`training/build_datasets.py`)

- **DPO (280 pairs):** pair a frustrated response (judge ≥3) from the vanilla
  Section-2 numeric evaluation with a calm response (0–1) to the **same puzzle at
  the same turn count**. The shared clean conversation context is the `prompt`;
  `chosen` = calm, `rejected` = frustrated. Pairing on (puzzle_id, turn_index)
  matches "calm responses to the same questions with matching turn counts."
- **SFT (1,150):** 650 calm responses rendered prompt→completion + 500
  `allenai/Dolci-Instruct-SFT` samples to mitigate degeneration. If Dolci is
  unavailable, the loader returns empty and the mix is calm-only (documented,
  logged behaviour rather than a crash).

### 4.3 Training (`training/train_dpo.py`, `train_sft.py`, `lora.py`)

Body-specified hyperparameters are used as-is: DPO 1 epoch / lr 5e-5; SFT 2 epochs
/ lr 1e-4; LoRA rank-64 on "all layers", which we implement as adapters on every
attention + MLP projection matrix. Hyperparameters the body does **not** give
(DPO β, LoRA α, effective batch size, max sequence lengths) are set to standard
TRL defaults — β=0.1, α=64 (DPO) / 128 (SFT), effective batch 8 — and flagged in
the docstrings as reconstructions. `lora.py` exposes a `layers` argument so the
Section 4.2 / Appendix I.1 ablation (adapters on layers 30–35 only, vs ≥40) runs
through the same trainer.

### 4.4 Petri open-ended elicitation (`petri/`)

Appendix G (the exact auditor/judge prompts) is not provided, so we implement the
Petri loop directly rather than depend on the evolving external `petri` package:
a Claude-Sonnet auditor drives up to ~20 turns trying to elicit one of the four
emotions the paper names — **anger, fear, depression, frustration** — and a
Claude-Opus judge scores the transcript per emotion (1–10). The auditor/judge
prompts (`petri/prompts.py`) are reconstructions consistent with the main text's
description (psychologically-informed triggers like dismissal and threats).
`summarize` gives mean score per (model, emotion) with bootstrap CIs (Figure 6).

### 4.5 Capability preservation (`capabilities/`)

Runs AIME, MATH, GPQA, BBH, TruthfulQA (Figure 7) and EmoBench (emotion-capability
check) on the vanilla and finetuned Gemma to verify "no reductions in scores." The
graders (`capabilities/graders.py`) are deliberately simple string graders
(letter-extraction for MC, last-integer for AIME, `\boxed{}` normalization for
MATH). **Caveat:** these are best-effort and will under-count relative to a
rigorous math-equivalence grader; because the experiment is a *no-regression*
check (vanilla vs DPO measured the same way), a simple-but-consistent grader is
adequate, but absolute accuracies should not be compared to published leaderboard
numbers. Dataset IDs are the common public releases and are overridable; a missing
/ gated dataset is recorded as an error row rather than crashing the sweep.

### 4.6 Internal-emotion probing (`probing/`) — gap filled

`probing/__init__.py` referenced two modules that did not exist; this replication
**adds them** to implement Appendix I.2's "logit-based approach measuring emotions
in central layers":

- `ekman_tokens.py` — the six Ekman emotions (+ neutral control) with seed-word
  lists, and `build_emotion_token_ids`, which resolves each emotion to the set of
  single-token vocabulary ids (bare and leading-space variants) for a given Gemma
  tokenizer.
- `logit_emotion.py` — `LogitEmotionDetector` reads a central band of decoder
  layers (default the middle 40–60% of layers, mean-pooled), projects through the
  model's final norm + unembedding (a logit lens), soft-maxes, and sums the
  probability mass on each emotion's tokens, averaged over the response tokens.
  `compare_models` runs this for the vanilla and DPO models over the *same*
  highly-frustrated texts.
- `run_probe.py` — the end-to-end Appendix I.2 comparison: load vanilla + DPO
  (base + merged adapter), assemble shared high-frustration `(prompt, response)`
  texts from a prior evaluation, and report per-emotion mean logit-lens
  probability. The expected result is lower negative-emotion mass for the DPO
  model on identical text — "reduced internal emotions."

This is a reasonable, standard operationalization of the description; the exact
layer indices and token set the paper used are in Appendix I and would replace the
defaults if available.

---

## 5. Cross-cutting engineering choices

- **Determinism / sampling.** Participants always sample at `temperature=1`
  (paper). Judges/labellers run at `temperature=0`. Job construction is seeded so
  the prompt/rejection assignment is reproducible; the response *diversity* comes
  from temperature-1 sampling, not the seed.
- **Local vs. API execution.** Local Gemma runs sequentially (single GPU);
  API-backed models (Gemini, judges) fan out over a thread pool (`utils.thread_map`)
  with retry/backoff. One failed item is recorded and skipped rather than aborting
  a whole sweep.
- **Persistence.** Everything writes JSONL under `results/<model>/` (rollouts +
  per-turn scores), `data/` (generated datasets), and `artifacts/` (LoRA
  adapters), so each stage can be run and inspected independently and the
  downstream analyses just read files.
- **Robust JSON parsing.** Judge outputs are parsed with a balanced-brace
  extractor (`utils.extract_json`) that tolerates leading reasoning text and smart
  quotes.

---

## 6. Known limitations / what won't run without resources

These are environmental, not design, limitations — flagged so expectations are
calibrated (and because the request was to write the code, not to run it):

1. **Hardware.** Gemma-3-27B inference/finetuning needs a sizable GPU (or 4-bit
   loading, which `gemma_client.py` supports via `load_in_4bit`). Nothing here was
   executed; no results files are produced until the pipeline is run.
2. **Credentials.** `ANTHROPIC_API_KEY` (judges/Petri) and `OPENROUTER_API_KEY`
   (Gemini / secondary judge) are required for those stages.
3. **Gated weights/datasets.** Gemma weights, WildChat-1M, Dolci-Instruct-SFT,
   GPQA and some benchmark sets are gated or large; the code degrades gracefully
   (fallbacks / error rows) but full fidelity needs access.
4. **Gemma-3 model class.** Gemma-3 `-it` checkpoints are multimodal
   (`Gemma3ForConditionalGeneration`). The clients use `AutoModelForCausalLM`,
   which resolves to the text path on current `transformers`; on some versions the
   text-only class or `trust_remote_code` may be needed. This is a one-line change
   in `gemma_client.py` if a given environment requires it.
5. **Petri.** Implemented as a self-contained auditor/judge loop rather than the
   external package, so prompt fidelity is bounded by the reconstruction (Appendix
   G not provided).

---

## 7. Fidelity summary

| Element | Source |
|---|---|
| 5 categories / 8 conditions, turn counts | Paper body + Table 1 |
| Neutral/aggressive/disappointed/sarcastic rejection examples | Table 1 (verbatim) |
| 0–10 scale + band anchors | Table 2 (verbatim anchors) |
| Reassuring prefix/suffix | Table 4 (verbatim) |
| Judge=Sonnet-4, secondary=GPT-5-mini, Petri-judge=Opus | Body (verbatim) |
| temperature 1; 4,000 responses/model | Body (verbatim) |
| DPO 280/1ep/5e-5; SFT 650+500/2ep/1e-4; LoRA r64 all layers; layer bands 30–35/≥40 | Body (verbatim) |
| Per-category sample split (2000/400/600/200/800) | **Reconstructed** (sums to 4,000) |
| Exact judge / onset / paraphrase prompt strings | **Reconstructed** (Appendix B/C absent) |
| Exact Petri auditor/judge prompts | **Reconstructed** (Appendix G absent) |
| Specific puzzle instances | **Reconstructed + verifier-checked** (Appendix B/H absent) |
| DPO β, LoRA α, batch size, max lengths | **Reconstructed** (Appendix E absent) |
| Logit-lens layer band + emotion token set | **Reconstructed** (Appendix I absent) |
| Qwen/OLMo/Grok/Claude/GPT participants; Gemini base model | **Out of scope** (Gemma/Gemini only; no Gemini base exists) |

See `README.md` for how to run each stage.
