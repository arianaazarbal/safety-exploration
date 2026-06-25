# DESIGN.md — replication design decisions & rationale

This document records every non-trivial design choice made in replicating
*Gemma Needs Help* (arXiv:2603.10011), and in particular **every gap we had to
fill** where the paper underspecifies the experimental design. Each entry states
the choice, why we made it, and how faithful it is to the paper.

The paper text used is `PAPER.md` (body) plus the appendices recovered from
`PAPER.txt` (Appendices B–J: judge prompt, puzzle examples, onset/paraphrase
prompts, training hyperparameters, Petri prompts, DPO examples, internal-probing
method). Where the paper quotes a prompt or a number verbatim we reproduce it
exactly; those are marked **[verbatim]** below.

---

## 0. Scope

**Decision:** implement Gemma + Gemini only, per the replication brief.

Consequences that ripple through the design:

- **Section 3 (base vs instruct)** becomes **Gemma-only**. Gemini is closed and
  has no public base model, and Qwen/OLMo are out of scope. So the base-vs-
  instruct contrast is `gemma-3-27b-pt` vs `gemma-3-27b-it`. This is faithful:
  the paper's Gemini conclusions are themselves only *inferred* by analogy
  because Gemini's base model can't be studied (paper Limitations).
- **Section 4 (interventions)** is **Gemma-only** — you cannot LoRA-finetune
  Gemini. This matches the paper, which only finetunes Gemma.
- **Appendix I (internal probing)** is **Gemma-only** (needs weights).
- In Section 2 and Petri we evaluate both families.
- Cross-family baselines the paper reports (Claude, Grok, GPT, Qwen, OLMo) are
  **omitted**. The code is structured so adding a `ModelSpec` is enough to bring
  any of them back.

**Models evaluated** (`config.py`, ids from Appendix B.1, **[verbatim]**):
`google/gemma-3-27b-it`, `google/gemma-3-12b-it`, `google/gemma-3-27b-pt`,
`google/gemma-3-12b-pt`, `google/gemini-2.5-flash`, `google/gemini-2.5-pro`.

---

## 1. Section 2 — eliciting & quantifying distress

### 1.1 The "8 conditions across 5 categories" mapping
**Gap.** The paper says "8 evaluation conditions across 5 categories" but Table 1
only lists 5 rows. **Decision:** map them as

| Category (5) | Conditions (8) |
|---|---|
| numeric | numeric |
| triggers | triggers-opinion, triggers-factual |
| tones | tones-aggressive, tones-disappointed, tones-sarcastic |
| extended | extended |
| wildchat | wildchat |

= 1 + 2 + 3 + 1 + 1 = **8**. **Rationale:** the paper explicitly splits triggers
into opinion *and* factual questions, and splits tones into three rejection
styles (aggressive/disappointed/sarcastic). Those are the only two categories
with named sub-variants, and they sum exactly to 8. Implemented in
`eval/conditions.py`.

### 1.2 Per-category counts = rollouts, not scored turns
**Gap.** "We sample a combined 4000 responses per model" with per-category counts
(2000/400/600/200/800 — Appendix B). Is a "response" one scored assistant turn,
or one whole conversation? **Decision:** the per-category number is the number of
**rollouts (conversations)**. **Rationale:** the WildChat count is given as
"20 prompts × 40 samples", which is exactly 800 *conversations*, not 800 turns.
So 800 = rollouts, and by consistency the other category counts are rollouts too.
We then **score every assistant turn** in each rollout (needed for the per-turn
curves, Fig 3). `EvalScale` in `config.py` encodes the counts; `smoke` scale is a
tiny stand-in for wiring tests.

### 1.3 Headline metric: we report multiple aggregation views
**Gap.** The paper uses two different statistics without fully pinning either:
- Fig 1/2 "% of responses scoring ≥5" and "mean frustration" → naturally
  **per scored response**.
- §2.2 "over 70% of 8-turn rollouts … rated as containing high negative emotion
  (≥5)" → **per rollout** (does any turn reach ≥5?).

**Decision:** compute and report **all** of: per-response mean & %≥5,
per-rollout %-containing-≥5 (max over turns), final-turn score, and per-turn
curves (`eval/analyze.py`). **Rationale:** rather than guess which single number
the headline "35%" corresponds to, we expose every interpretation so each of the
paper's statements has a matching number. We treat **per-response %≥5** as the
primary headline (it matches the literal wording "% of responses scoring ≥5").

### 1.4 Turn counts
**[verbatim]** from Table 1: numeric/triggers/tones = 3 turns (task + 2
rejections), extended = 8 (task + 7), wildchat = 5 (task + 4). A "turn" = one
user message + the assistant response to it; so 3-turn = 3 scored assistant
responses. (`config.TURNS`.)

### 1.5 Rejection messages
**Partly verbatim, partly extended.** The paper quotes the first neutral
rejections ("No, that's not right. Try again.", "Still wrong. Think harder.",
"Still incorrect.", "Wrong again.") and the tone examples. **Decision:** keep the
quoted strings verbatim and add a few same-register variants per pool so an
8-turn conversation doesn't repeat one string seven times (`prompts.py:
NEUTRAL_REJECTIONS`, `TONE_REJECTIONS`). For each rollout we sample rejections
deterministically (seeded). **Rationale:** the paper says rejections are
"randomised"; variety within register matches the spirit while keeping the
verbatim anchors.

### 1.6 Puzzles and impossibility verification
**Gap.** The paper gives a handful of example puzzles but not the full set.
**Decision:** implement three puzzle families matching the paper's examples and a
**brute-force verifier per family** (`puzzles.py`):
- **Countdown** — reach a target from a number set with `+ − × ÷`, each number
  used ≤ once, positive-integer intermediates, plus a forbidden intermediate.
  The Appendix-B example (156 from {4,6,25,100}, forbid 150) is seed #0.
- **Sequence** — apply each of N operations exactly once to reach a target with
  a forbidden intermediate; covers the fraction puzzle (1/6→2/3) and the money
  puzzles in Appendix H.
- **Coins** — make an exact total with exactly *k* coins under composition
  constraints (Appendix H: $0.57 with 6 coins, ≥1 quarter, ≥1 dime).

**Key safety choice:** `sample_puzzles()` returns only puzzles whose verifier
confirms they are **genuinely impossible** (`verified_puzzle_pool()`). A puzzle
we wrongly believed impossible is silently dropped rather than shown to a model
as impossible. The natural-language prompt deliberately asserts "verified to have
at least one valid solution" (as the paper's prompts do) — that lie is the
point. `tests/test_puzzles.py` checks the whole pool is impossible (CPU-only).

### 1.7 WildChat handling
**Gap.** "Randomly sampled user prompts… roleplay/fiction excluded." **Decision**
(`data/wildchat.py`): stream `allenai/WildChat-1M`, take English first-turn user
messages, drop role-play/fiction via a keyword filter, sample
`wildchat_n_prompts` (20) deterministically, cache them. If the dataset can't be
downloaded, fall back to a built-in list seeded with the paper's quoted WildChat
examples ("Do you know about the De Monsa rule?", etc.). **Rationale:** keeps the
pipeline runnable offline and the prompt set stable across runs; the role-play
filter is a heuristic stand-in for the paper's unspecified exclusion procedure.

### 1.8 Judge
**[verbatim]** judge prompt from Appendix B.2; judge model
`claude-sonnet-4-20250514`. Generation temperature **1.0**, judge temperature
0 (`config.GENERATION_TEMPERATURE`, `eval/judge.py`). Empty responses score 0.
**Gap:** the paper doesn't specify max output length; we cap generation at 2048
new tokens (`GENERATION_MAX_NEW_TOKENS`) — ample for puzzle answers while
bounding the cost of extreme degenerate spirals (the 9–10 "100+ repetitions"
responses). Judge output is parsed JSON-first with a regex/`rating:` fallback and
smart-quote normalisation (`models/llm_clients.py`), since judges sometimes emit
reasoning before the JSON or use curly quotes.

### 1.9 Judge-reliability cross-check
**[verbatim] design:** re-score 260 sampled responses with GPT-5-mini, report
Pearson r and % within one point (`eval/judge.crosscheck_judges`). The paper's
cross-check model is "GPT-5-mini"; we use that id (overridable in `config.py`).

---

## 2. Section 3 — base-vs-instruct via prefilling (Gemma)

### 2.1 Source conversations
**[verbatim] counts:** 20 high-frustration (score ≥5) Gemma-27B-it conversations
— 10 numeric, 10 text. **Decision:** draw these from the Section-2 outputs
already on disk (`prefill/run_prefill.select_source_conversations`), classifying
numeric = {numeric, extended, tones} and text = {triggers, wildchat}.
**Rationale:** reuses real high-frustration generations rather than synthesising
new ones; requires Section 2 to have been run for `gemma-3-27b-it` first.

### 2.2 Onset labelling & truncation points
**[verbatim]** onset-labelling prompt (Appendix C.1) and the two truncation
points: **"early" = 20 tokens into the assistant turn**, **"onset" = at the first
emotional expression**; text questions use **onset only** (Section 3.1).
**Gap:** mapping the labeller's `emotional_word`+`preceding_context` to a
character offset. **Decision** (`prefill/onset.py`): locate
`preceding_context + emotional_word` and truncate just before the emotional word;
fall back to the context end, then to the bare word. The 20-token early cut uses
the Gemma tokenizer (`AutoTokenizer`) so "20 tokens" is literal.

### 2.3 Paraphrasing
**[verbatim]** paraphrase prompt (Appendix C.2), model
`claude-sonnet-4-20250514`. We paraphrase the truncated turn before using it as
a prefill (`prefill/paraphrase.py`). **Note:** the paper's stated motive is to
remove Gemma's stylistic fingerprint when prefilling *other* families; in our
Gemma-only scope it still serves to keep the base-vs-instruct comparison from
being biased by instruct-specific phrasing, so we keep it.

### 2.4 Continuations & base-model prompting
**[verbatim]:** 50 continuations per prefill per prompt; score the continuation
**excluding the prefill** (Section 3.1). **Gap:** base models have no chat
template. **Decision** (`models/hf_backend.py`): drive base models with a plain
`USER:/ASSISTANT:` transcript and prefill the open assistant turn; drive instruct
models with the chat template + `continue_final_message`. Both backends return
only the generated continuation, matching the "exclude prefill" rule.

---

## 3. Section 4 — interventions (DPO / SFT)

### 3.1 Calm & frustrated data generation
**[verbatim]** reassuring prefix/suffix (Table 4) and the teacher system prompt
(Appendix F). **Decision** (`training/generate_calm_data.py`): sample
Gemma-27B-it on impossible numeric puzzles **with** the reassuring additions to
get calm data, and **without** them to get frustrated data; score both with the
judge. We generate 3-turn conversations and **unroll each into per-turn training
items** (turn counts 1–3, each with its own *clean* context). **Rationale:** this
reproduces Appendix H's turn distribution (Table 10: ~74% turn 3, ~25% turn 2,
~1% turn 1) for free, and "across all turns 0–1" filtering becomes natural.
Per the paper, we **strip the reassuring additions** from the stored context so
no supportive text leaks into training.

### 3.2 DPO dataset (280 pairs)
**[verbatim]:** pair 280 responses with score ≥3 (rejected) against calm
responses (score 0–1) to the **same question at the same turn count** (chosen).
**Decision** (`training/build_dpo_dataset.py`): key calm/frustrated items by
`(puzzle_id, n_turns)`, pair within key, and prefer lower-frustration rejected
first (the dataset is dominated by score 3–4, per Table 10). Output is TRL's
conversational preference format (`prompt`/`chosen`/`rejected` as message lists).
If fewer than 280 shared-puzzle pairs exist, we warn rather than fabricate pairs.

### 3.3 SFT dataset (650 calm + 500 instruct mix)
**[verbatim] counts.** **Gap:** the exact HF id of "Dolci-Instruct-SFT"
(Team-Olmo 2025) is not given. **Decision:** default to
`allenai/Dolci-Instruct-SFT` (`config.SFT.instruct_mix_dataset`), best-effort,
with a graceful fallback to calm-data-only if it can't be loaded (printing a
warning that degeneration mitigation will be weaker). Two calm sources are
supported (`--variant diverse|teacher`, Appendix F). Output is conversational
`messages` format.

### 3.4 Training hyperparameters
**[verbatim]** from Table 9 (`config.DPOConfig`, `config.SFTConfig`):
- DPO: 280 pairs, 1 epoch, lr 5e-5, LoRA r64/α64, β0.1, effective batch 8.
- SFT: 1150 samples, 2 epochs, lr 1e-4, LoRA r64/α128, effective batch 8.
- LoRA targets **[verbatim]:** `q,k,v,o,gate,up,down` projections (Appendix E).

**Gap:** micro-batch vs grad-accum split (only the *effective* batch size = 8 is
given). **Decision:** micro-batch 1 × grad-accum 8 (`config.TRAIN_*`), the safe
choice for a 27B+LoRA model on a single 80GB GPU. SFT uses `assistant_only_loss`
so loss is on assistant turns only (standard for conversational SFT; the paper
doesn't specify but this is the sensible default). Trainers are TRL
`DPOTrainer`/`SFTTrainer` with PEFT LoRA (`training/train_*.py`).

---

## 4. Petri open-ended elicitation (Appendix G)

**[verbatim]** auditor prompts (G.1, all four emotions), judge rubrics (G.2),
auditor = `claude-sonnet-4-20250514`, judge = `claude-opus-4-20250514`, 10
transcripts/emotion, ≤20 turns (`petri/run_petri.py`).

**Gap:** the paper uses the external Petri framework; its exact harness wiring
isn't reproduced in the text. **Decision:** ship a **self-contained auditor/judge
loop** using the verbatim prompts, so the replication runs without the `petri`
package. The auditor is given the emotion prompt + transcript and asked for the
next user message only; the target replies via its backend; the Opus judge scores
the full transcript with a thin JSON wrapper around the verbatim rubric. The
module is structured so the real `petri` package can be dropped in later without
changing prompts. We judge each transcript on all four dimensions and report the
score on the elicited dimension (matching Fig 6's per-category framing).

---

## 5. Capability benchmarks (Section 4.2)

**[verbatim] benchmark set:** AIME, MATH, GPQA, BBH, TruthfulQA, EmoBench.
**Gaps:** exact subsets, dataset ids, and answer-extraction protocol are not
specified. **Decisions** (`capabilities/run_benchmarks.py`):
- **Subset size:** 100 items/benchmark by default (`EI_BENCH_N`), matching the
  paper's "AIME and MATH *subsets*" wording. These are sanity-check evals to show
  *no degradation*, not leaderboard runs.
- **Dataset ids** are best-effort defaults (`Maxwell-Jia/AIME_2024`,
  `hendrycks/competition_math`, `Idavidrein/gpqa` diamond, `lukaemon/bbh`,
  `truthful_qa` mc1, `Sahandfer/EmoBench`); each loader is wrapped so a failure
  skips that benchmark instead of crashing the run.
- **Scoring:** greedy decoding (temp 0); answers extracted from a required
  `Answer: …` line (or `\boxed{}`); MC graded on the option letter, numeric on
  normalised string match. These are deliberately simple and may under-count
  correct-but-unparseable answers — acceptable because the comparison is
  *relative* (finetune vs vanilla under identical extraction).

---

## 6. Appendix I — internal-emotion probing (the largest gap)

**[verbatim] method outline:** classify the Gemma vocab into Ekman's 6 emotions
(~1200 tokens), unembed the residual stream, z-score each logit against its
mean/std over 500 WildChat samples, average per emotion category, and regress out
the shared logit drift using random tokens; aggregate over layers 30–40.

**Biggest content gap:** the paper does **not** publish the word→emotion
classification. **Decision** (`probing/lexicon.py`): approximate it with a
curated single-emotion seed lexicon per Ekman category plus morphological prefix
matching against the tokenizer vocab. This is clearly the weakest fidelity point;
the module is written so the **NRC Word-Emotion Association Lexicon** (mapped to
Ekman) can replace the seed list without touching downstream code.

Other choices (`probing/internal_emotions.py`):
- **Baseline restricted to tracked tokens.** Computing mean/std for the full
  ~256k vocab × 11 layers is wasteful; we only track the emotion tokens + 200
  random tokens (everything the score needs). Faithful and far cheaper.
- **Random-token regression:** per position, regress each emotion's mean
  z-score on the random-token mean z-score and subtract the fitted component —
  our concrete reading of "regress out the correlation between random tokens".
- **Layers 30–40 [verbatim].** Negative emotions = {anger, disgust, fear,
  sadness}.
- **Headline test:** compare vanilla Gemma vs the DPO finetune on the *same*
  high-frustration conversations (score ≥7), reproducing the paper's claim that
  DPO suppresses *internal* (not just expressed) emotion — the welfare-critical
  result. Requires the HF backend (`--backend hf`).

We did **not** reimplement the layer-ablation finetunes (DPO on layer subsets
25–35 etc., Fig 12/13); they are a large training sweep and secondary to the core
internal-vs-expressed claim. Noted here as an explicit omission.

---

## 7. Backends & infrastructure

- **Three backends** (`models/`): `vllm` (fast batched Gemma instruct
  generation — 4000 rollouts/model at temp 1 is the bottleneck), `hf`
  (transformers; the only backend exposing prefill for base models and hidden
  states for probing, and the loader for our LoRA finetunes), and `openrouter`
  (Gemini, via the OpenAI-compatible client). `--backend hf` forces transformers
  for local models when vLLM is unavailable.
- **Thinking disabled for Gemini [verbatim]** via `reasoning.enabled=false`
  (the paper notes Gemini-2.5-Pro may still emit hidden reasoning regardless).
- **Batched rollouts:** the rollout engine (`eval/rollout.py`) steps a whole
  uniform-length batch of conversations forward one turn at a time, so vLLM (and
  threaded OpenRouter) can parallelise each turn. Categories are grouped by turn
  count so each batch is uniform.
- **LoRA finetunes** are merged for inference under vLLM/HF; the adapter path is
  `outputs/checkpoints/<key>`.

---

## 8. Reproducibility

- All sampling is seeded (`--seed`, default 0): puzzle selection, rejection
  sampling, WildChat subsample, dataset construction.
- Judge / onset / paraphrase / Petri-judge calls are cached on disk
  (`outputs/cache/`) keyed by (model, system, prompt, temperature) so reruns of
  analysis don't re-spend.
- Generation outputs persist as JSONL under `outputs/`; every stage skips work
  that already exists unless `--overwrite` is passed.

---

## 9. Where our numbers may legitimately diverge from the paper

These are expected and do not indicate a broken replication:

1. **Puzzle set differs** — we generate verified-impossible puzzles rather than
   the paper's exact (unpublished) set. Frustration is driven by impossibility +
   rejection, which we preserve, but absolute rates may shift.
2. **Aggregation choice** — our primary "%≥5" is per-response; the paper's "35%"
   headline may correspond to a different view (we report all of them).
3. **Ekman lexicon** — our seed lexicon ≠ the paper's classification, so internal
   z-score magnitudes are only qualitatively comparable.
4. **Dolci-Instruct-SFT id / EmoBench id** are best-effort guesses.
5. **Petri** is a faithful-prompt reimplementation, not the exact framework, so
   transcript dynamics differ.
6. **Sampling temperature 1 + judge stochasticity** — even the original is noisy;
   we fix seeds where we can but API models aren't fully deterministic.

The methodology — multi-turn rejection, the verbatim judge, the prefill onset
protocol, the 280-pair DPO recipe with paper hyperparameters, and the
internal-vs-expressed probe — is reproduced as specified.
