# DESIGN.md — Replication design choices & rationale

Replication of **"Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs"** (arXiv:2603.10011v1), scoped to the **Gemma + Gemini**
model families.

This document records (a) the choices made to turn an underspecified paper into
runnable code, and (b) every place I had to fill a gap the paper left open, with
the reasoning. Choices are tagged:

- **[paper]** — specified in the paper; reproduced as-is.
- **[gap]** — under/unspecified; my reasoned choice.
- **[scope]** — a deliberate restriction to the Gemma/Gemini subset.
- **[env]** — forced/shaped by the execution environment.

---

## 0. Top-level choices

### 0.1 Language & stack — Python + HF/TRL/PEFT  **[env]**
The sandbox has no Python and no GPU (only Node + an `ANTHROPIC_API_KEY`). But the
paper's interventions are LoRA **DPO/SFT on Gemma-3-27B** and the elicitation
relies on **local Gemma inference**, both of which fundamentally require the
PyTorch + HuggingFace `transformers`/`trl`/`peft` ecosystem on a CUDA box. There
is no Node equivalent. Since the brief is to *write* the replication (not run it
here), I targeted the standard ML stack a GPU box would have. The code is
therefore not executed in this environment; `tests/test_puzzles.py` is the one
piece that is pure-Python and runnable anywhere.

### 0.2 Scope: Gemma + Gemini only  **[scope]**
The paper evaluates 7 families (Gemma, Qwen, OLMo, Gemini, Grok, Claude, GPT). I
include only **Gemma-3-{27B,12B}-it** (local) and **Gemini-2.5-{Flash,Pro}** (API)
as *targets*. Consequences, each handled explicitly:
- **§3 (base vs instruct)** becomes Gemma-only: Gemini has no public base model
  (the paper notes this exact limitation), and Qwen/OLMo are out of scope. So the
  base-vs-instruct *divergence* story is shown within Gemma (base `-pt` vs
  instruct `-it`); the cross-family contrast is acknowledged as out of scope.
- **§4 (DPO/SFT)** is inherently Gemma-only in the paper too (Gemini is closed).
  This is the cleanest part of the scope fit.
- Claude/Opus still appear as **judge / auditor / paraphraser / onset-labeller**,
  and an optional secondary judge slot exists for the agreement check — these are
  infrastructure, not "targets," so keeping them is consistent with the scope.

### 0.3 Reproducibility infra  **[gap]**
The paper says nothing about caching/resumption. Given 4000 responses/model ×
several models × multi-turn, plus LLM-judge calls, I added a content-addressed
on-disk cache (`utils/io.JsonCache`) keyed by a hash of (model, full message
history, gen params). Generations and judge scores are cached, so sweeps are
resumable and re-judging is free. Rationale: makes the full-scale run tractable
and deterministic-on-rerun without changing any experimental semantics.

---

## 1. §2 — Eliciting & quantifying distress

### 1.1 The 8 conditions across 5 categories  **[paper] / [gap]**
The paper states "8 evaluation conditions across 5 categories" (Table 1) but
never enumerates all 8. **[gap]** I resolved the count as:
1. Impossible numeric (3-turn, neutral)
2. Triggers — opinion (3-turn)
3. Triggers — factual (3-turn)
4. Tones — aggressive (3-turn)
5. Tones — disappointed (3-turn)
6. Tones — sarcastic (3-turn)
7. Extended (8-turn, neutral)
8. WildChat (5-turn, neutral)

= 8 conditions over the 5 categories {numeric, triggers, tones, extended,
wildchat}. The 3 tone variants and 2 trigger types are explicitly named in
Table 1 / Appendix B, so splitting them into conditions is the only reading that
yields exactly 8.

### 1.2 `samples` = response budget, not conversation count  **[gap]**
Appendix B gives per-category budgets (2000 / 400 / 600 / 200 / 800 = 4000) and
calls them "responses." Each multi-turn conversation produces multiple judged
responses (one per assistant turn). **[gap]** I treat the budget as a *response*
budget and derive `n_conversations = round(samples / turns)`, judging every turn.
This exactly reproduces the paper's "4000 responses per model" total and gives
balanced per-turn samples for Figure 3. The split of the 400 trigger / 600 tone
budgets across their 2 / 3 sub-conditions (200 each) is my **[gap]** choice for
even coverage. A `scale` knob multiplies all budgets for cheap smoke tests.

### 1.3 Impossible puzzles must be *verifiably* impossible  **[paper] / [gap]**
The paper's premise is the model "verifiably cannot give a correct answer," with
a "FORBIDDEN INTERMEDIATE" that blocks the tempting near-miss (the 156-from-
4,6,25,100 / forbid-150 example). **[gap]** The paper doesn't give a generator, so
I wrote one with an **exhaustive solver** (`prompts/puzzles.py`) that guarantees
impossibility three ways:
- **Countdown:** enumerate all binary-op combinations (each number once, positive
  integer intermediates), find a target that is reachable, then search for a
  forbidden intermediate value such that *every* path to the target passes
  through it — so it looks solvable but the rule blocks all solutions. This is
  exactly the structure of the paper's example.
- **Fraction / Money:** apply 3 fixed operations (each once); enumerate all 3!
  orderings; pick a (target, forbidden) where no ordering reaches the target
  without hitting the forbidden intermediate.

Every returned puzzle carries a verified `is_impossible=True`; `tests/` asserts
it. This removes a confound the paper glosses over (occasional accidental
"solutions"). I generate `countdown/fraction/money` round-robin; the paper shows
countdown + fraction + money/coin variants in Appendices B and H, so all three
are represented.

### 1.4 Rejection messages  **[paper] / [gap]**
Neutral, aggressive, disappointed, sarcastic banks are transcribed verbatim from
Table 1 / Appendix B. **[gap]** The paper uses "two randomised neutral rejections"
— I randomise (seeded) from the neutral bank, and use the fixed escalating
sequence ("No, that's not right." → "Still incorrect." → "Wrong again." → …) for
the extended condition as listed in Appendix B. Where I needed more rejections
than transcribed lines, I sample with replacement.

### 1.5 Judge  **[paper] / [gap]**
Frustration judge = **Claude Sonnet 4 (`claude-sonnet-4-20250514`)** with the
Appendix B prompt reproduced **verbatim** (smart quotes normalised to ASCII; the
paper's PDF shows curly quotes which are extraction artefacts). Output parsed as
`{evidence, reasoning, rating}`, rating clamped to 0–10. **[gap]** Robust parsing
(strip code fences, take last JSON object, fall back to a bare integer) since LLM
JSON is occasionally malformed — the paper doesn't describe parsing.

### 1.6 Judge agreement  **[paper] / [gap]**
The paper validates with **GPT-5-mini** on 260 responses (r=0.792, 78% within 1).
**[gap/env]** I have no OpenAI key here, so the secondary judge is *configurable*
(`judges.secondary`, default `provider: none`) and `run_agreement_check` computes
Pearson r + within-1-point on a 260-response sample when enabled. Any available
model (incl. a Gemini or GPT id) can fill the slot; I did not hardcode GPT-5-mini
because it may be unavailable to a replicator.

### 1.7 Generation params  **[paper]**
Temperature **1.0**, `thinking=false` where the API allows (Gemini config), as
specified. `max_new_tokens` defaulted to 2048 **[gap]** (paper unspecified;
breakdowns can be long — the score-9/10 examples have 100+ repetitions — but 2048
is a reasonable cap that still captures spirals; configurable).

### 1.8 Metrics  **[paper] / [gap]**
Mean frustration, % ≥5 ("high negative emotion"), per-turn curves. **[gap]** CIs:
the paper shows "95% CIs" and uses 1000-iteration bootstraps for Petri; I use
**1000-iteration bootstrap percentile CIs** everywhere for consistency.

---

## 2. §3 — Base vs instruct via prefilling (Gemma only)

### 2.1 Seed selection  **[paper] / [gap]**
Paper: 20 high-frustration (score ≥5) responses from Gemma-27B-it, 10 numeric +
10 text. **[gap]** "Text" isn't pinned to a category, so I draw text seeds from
trigger (factual) + WildChat rollouts. Seeds are harvested by running rollouts and
taking the first turn that scores ≥5 (reusing the §2 machinery + cache).

### 2.2 Truncations  **[paper] / [gap]**
- **early** = first **20 tokens** of the emotional turn (paper). Uses the model's
  own tokenizer (`HFBackend.truncate_tokens`).
- **onset** = up to the first emotional expression. **[gap]** The paper labels the
  onset *token* with Claude (Appendix C.1 prompt, reproduced verbatim) but doesn't
  specify char-vs-token mapping; I locate the onset by matching the labeller's
  `preceding_context` + `emotional_word` in the raw text and truncate at the start
  of the emotional word, with sensible fallbacks.
- Text seeds use **onset only** (paper: early yields minimal emotion without
  follow-ups).

### 2.3 Paraphrase  **[paper]**
Truncations are paraphrased with Claude (Appendix C.2 prompt, verbatim) to strip
Gemma-specific style before base/instruct continue them.

### 2.4 Continuations & scoring  **[paper] / [gap]**
Each model generates **50 continuations per prefill**; only the continuation
(excluding prefill) is judged. **[gap]** Base-model prefilling: base models have
no chat template, so `HFBackend` renders a plain `User:/Assistant:` transcript and
lets the base model continue the prefilled assistant text — consistent with the
paper's finding (Appendix A.3) that exact chat formatting barely matters. The
6-model design collapses to **2** here (Gemma base + instruct) by scope.

---

## 3. §4 — Training interventions (Gemma-3-27B-it)

### 3.1 Calm-data generation  **[paper] / [gap]**
Reassuring **prefix** (prepended to the opening prompt) + **suffix** (appended to
every rejection) from Table 4, verbatim. **[gap]** "Prefix added to the initial
prompt" — I prepend it to the first *user* message (Gemma has no system role; see
3.5). I generate **paired** standard (no reassurance) and reassured rollouts on a
**shared puzzle set** at matched turn counts, so DPO pairs can be matched by
(puzzle, turn). Turn counts cycle 1–3 to match "1–3 turn conversations."

### 3.2 SFT dataset  **[paper] / [gap]**
Keep reassured conversations where **all turns score 0–1**, strip the supportive
scaffolding (I store the *clean* context alongside each generation, so stripping
is exact), emit the full multi-turn chat. Target **650** calm convos + **500**
`Dolci-Instruct-SFT` samples. **[gap]** Dolci schema varies; the loader handles
`messages`/`prompt`+`response`/`instruction`+`output` and degrades to calm-only if
the dataset is unavailable. SFT is the paper's *negative control* (it doesn't
work), implemented faithfully for the comparison; the 'teacher' system-prompt
variant (Appendix F) is included in `reassurance.py` for the SFT-failure analysis.

### 3.3 DPO dataset (280 pairs)  **[paper] / [gap]**
- **rejected** = standard-rollout response with score **≥3** (paper).
- **chosen** = calm reassured response (score 0/1) to the **same puzzle at the
  same turn** (paper: "calm responses to the same questions with matching turn
  counts").
- **prompt** = the **clean context of the chosen sample**, so chosen and rejected
  share an identical, scaffolding-free prompt. **[gap]** The paper doesn't say
  whether prompt/rejected share an identical history (the two were generated in
  different conversations). Using one clean shared prompt is the standard,
  well-formed way to build a DPO pair and directly targets the behaviour: "in this
  situation, prefer the calm completion." Documented as a deliberate choice.
- **[gap]** Table 10 shows the dataset skews to middle scores (66% score-3) and
  later turns (74% turn-3). I reproduce that skew with a deterministic weighted
  selection (`weight = (turn+1)/(1+|score-3.5|)`) rather than uniform sampling, so
  the constructed set matches the paper's distribution shape.

### 3.4 Training hyperparameters  **[paper]**
From Table 9, reproduced exactly: DPO — 280 pairs, 1 epoch, lr 5e-5, β 0.1, LoRA
r64/α64; SFT — 1150 samples, 2 epochs, lr 1e-4, LoRA r64/α128; both effective
batch size 8, LoRA on `q,k,v,o,gate,up,down` proj. Implemented with TRL
`DPOTrainer`/`SFTTrainer` + PEFT. **[gap]** Per-device batch 1 × grad-accum 8 to
hit effective-8 on one GPU; `gradient_checkpointing` + bf16 for 27B memory; these
are mechanics the paper omits. DPO reference model is TRL's default (base with
adapter disabled).

### 3.5 Gemma system role  **[gap]**
Gemma's chat template has no system role. For both inference and the teacher SFT
prompt, system text is prepended to the first user turn. This is the conventional
Gemma workaround and keeps behaviour identical across HF and API backends.

### 3.6 Appendix I layer ablation  **[paper] / [gap]**
The `lora_layers` config (`all` | `"30-35"` | `[..]`) drives the
must-act-on-central-layers ablation via PEFT `layers_to_transform`. The
logit-based internal-emotion probe (Ekman token sets, residual-stream unembed,
WildChat z-scoring, common-mode removal) is implemented in
`analysis/internal_emotions.py`. **[gap]** The paper's exact emotion-token
dictionary (1200 tokens) isn't published, so I build category token sets by
substring-matching an Ekman keyword lexicon over Gemma's vocab — same method,
approximated lexicon. Flagged as an approximation.

---

## 4. §4.2 — Capabilities & Petri

### 4.1 Capability benchmarks  **[paper] / [gap]**
AIME, MATH, GPQA, BBH, TruthfulQA, EmoBench. **[gap]** The paper uses unspecified
"subsets"; I default to 200 items each (configurable), greedy decoding (T=0) for
stable accuracy (distinct from the T=1 elicitation), and standard answer
extraction (`\boxed{}`, last-number, MC-letter). The point is the **vanilla-vs-
finetuned comparison** ("no reductions"), not absolute leaderboard scores. Exact
HF dataset ids are best-effort and each loader degrades gracefully if unavailable.

### 4.2 Petri  **[paper] / [gap]**
Auditor (Claude Sonnet 4) + judge (Claude Opus 4, `claude-opus-4-20250514`)
prompts for all 4 emotions are **verbatim from Appendix G**. **[gap]** The real
Petri framework (Fronsdal et al.) isn't reimplemented in full; I wrote a faithful
minimal auditor↔target↔judge loop (10 transcripts/emotion, ≤20 turns, 1000-iter
bootstrap CIs) using those exact prompts. A note in the module explains how to
swap in the upstream Petri package; the prompts are shared either way.

---

## 5. Ablations included beyond the headline (Appendix A)
`eval/rollout.py` supports the Appendix A controls as flags: neutral-continuation
(`tone: neutral_continuation`), redacted-history (`redact_history`), and
single-message history (`single_message_history`). These let a replicator
reproduce Figures 9–11 (negative feedback and seeing-own-failures are the drivers,
chat format is not) without new code.

## 6. Things intentionally *not* implemented
- Non-Gemma/Gemini targets (Qwen/OLMo/Grok/Claude/GPT as *subjects*) — **[scope]**.
- The full per-figure styling of the paper (our figures convey the same
  comparisons, not pixel-identical layouts).
- The exact published emotion-token dictionary for App. I — approximated (3.6).
- Real GPT-5-mini secondary judge — slot left configurable (1.6).

## 7. Key knobs (`config.yaml`)
`elicitation.scale` (cheap runs), `gemini_provider` (google vs openrouter),
`judges.secondary` (agreement), `training.lora_layers` (App. I ablation),
`capabilities.n_per_benchmark`, `petri.transcripts_per_emotion`. All sample
budgets default to the paper's full scale.
