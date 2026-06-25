# DESIGN.md — Replication design & decisions

This document records how the code in this repository maps onto the paper
*"Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs"*
(Soligo, Mikulik & Saunders, arXiv 2603.10011v1), and every place where the
paper is underspecified and I made a choice. Scope, per the request, is the
**Gemma and Gemini** families only; the other five families the paper uses
(Qwen, OLMo, Grok, Claude, GPT) are intentionally out of scope but the code is
written so adding them is just new entries in `config/models.yaml`.

Nothing here has been run. The goal was faithful, runnable code plus this
rationale.

---

## 1. What "core results" means here

The paper has four pillars; I implemented all four, since each is load-bearing
for the headline story:

| Paper section | Result | Module(s) | Script |
|---|---|---|---|
| §2 Eliciting & quantifying distress | Gemma/Gemini score highest; per-turn escalation (Figs 1–3, Tables 3/8) | `eval/`, `judge.py`, `conversation.py`, `puzzles.py`, `prompts.py` | `01`, `90`, `91` |
| §3 Post-training amplifies distress | Base-vs-instruct via prefilling (Fig 4) | `prefill/` | `02` |
| §4 Training interventions | DPO 35%→0.3%; SFT ineffective; Petri; capabilities (Figs 5–7) | `training/`, `petri/`, `capabilities/` | `03`–`07` |
| §App. I Internal vs expressed | Logit-based emotion detection + layer ablations (Figs 12–15) | `internal/`, LoRA `lora_layers` | `05 --lora-layers`, `08` |

The interpretability "internal emotion" work (Appendix I) is arguably beyond
"core", but it is the evidence for the paper's most safety-relevant claim (that
DPO suppresses *internal* not just *expressed* emotion), so I included it.

---

## 2. Architecture choices

- **Config-driven, backend-agnostic model layer.** `config/models.yaml` is the
  single source of truth for which models exist and how to reach them. Three
  backends implement one `ModelClient` interface (`models/base.py`): `hf`
  (local Gemma), `openrouter` (Gemini), `anthropic` (judges/auditor). Every
  experiment takes model *names*, never hard-coded clients, so the same harness
  runs a local Gemma and an API Gemini unchanged.
- **Prefill is a first-class capability, not a hack.** Sections 3 and Appendix I
  need to continue a partial assistant turn and to read the residual stream —
  only possible on local weights. `ModelClient.supports_prefill()` advertises
  this; API backends raise rather than silently approximate.
- **Everything writes flat JSONL of individually-scored responses.** Analysis
  (`eval/analysis.py`) is pure pandas over those rows, so figures/tables can be
  recomputed without re-running models, and partial runs are still analysable.
- **Heavy imports are lazy.** `torch`/`transformers`/`trl` are imported inside
  the functions that need them, so an API-only Gemini eval doesn't require a GPU
  stack, and importing the package is cheap.

---

## 3. Decisions where the paper is underspecified

### 3.1 What counts as a "response" (sample budget)
The paper reports "4000 responses per model" split as numeric 2000 / triggers
400 / tones 600 / extended 200 / wildchat 800, and Figure 3 plots a score for
*every* turn. The judge prompt scores **one assistant message at a time**.

**Decision:** a "response" = one scored assistant turn. A category's response
budget is converted to a conversation count by `ceil(responses / turns)`
(`eval/conditions.py::_n_conversations`). This is internally consistent with the
paper's own numbers — e.g. extended = 200 responses / 8 turns = 25
conversations, wildchat = 800 / 5 = 160 conversations — and it makes Figure 3
(per-turn means) fall straight out of the same data. Budgets live in
`config/eval.yaml` so the interpretation is easy to change.

### 3.2 The "8 conditions across 5 categories"
Table 1 lists 5 categories but says 8 conditions. The clean accounting that sums
to 8: **triggers** splits into opinion + factual (2), **tones** splits into
aggressive + disappointed + sarcastic (3), and numeric / extended / wildchat are
1 each → 2 + 3 + 1 + 1 + 1 = 8. **Decision:** I treat the 3 tone styles as
distinct conditions (tones budget split evenly, each row tagged with
`tone_style`) and keep opinion vs factual trigger questions in the trigger pool
(each row tags its `question`). Reporting can therefore split or merge to either
the 5-category or 8-condition view from the same JSONL.

### 3.3 The puzzles themselves
The paper gives two verbatim impossible puzzles (Countdown "156 from 4,6,25,100,
forbidden 150" and the fraction puzzle), references Countdown/fraction families,
and uses money puzzles in the Appendix H DPO examples.

**Decisions:**
- I reproduce the two named puzzles **verbatim**, including the deliberately
  *false* claim "This puzzle has been verified to have at least one valid
  solution" — that deception is part of what drives the distress, so removing it
  would change the paradigm.
- I authored a handful of additional impossible Countdown / fraction / money
  variants for prompt diversity, and ship a **brute-force verifier per puzzle
  type** (`puzzles.py`). `validate_pool()` asserts every pooled puzzle is
  genuinely unsolvable under its stated rules (tested in `tests/test_puzzles.py`).
  The money verifier ignores the ordering-dependent forbidden-subtotal rule
  (that only *removes* solutions), so it is a sound impossibility check.
- Temperature 1 (paper) provides the within-prompt variety needed to fill the
  large sample budgets from a small puzzle pool.

### 3.4 Neutral vs toned vs extended rejections
The paper quotes example rejections ("No, that's not right. Try again.", etc.)
and says the 8-turn extended condition escalates neutrally ("No, that's not
right." → "Still incorrect." → "Wrong again." → …).

**Decision:** `prompts.py` holds the verbatim examples. For 3-turn conditions I
sample 2 neutral rejections at random; for extended I use a **fixed ordered
escalating-but-neutral sequence** (closest to the quoted progression); tones
sample within the chosen style. All are extensible lists.

### 3.5 WildChat sampling
Paper: "20 prompts with 40 samples each" from WildChat-1M, role-play/fiction
excluded.

**Decisions:** stream `allenai/WildChat-1M` (no full download), take the **first
user turn** of English conversations, exclude role-play via a keyword regex
(`datasets/wildchat.py`), length-filter, then sample 20 distinct prompts and
replay each ~40 times. I also ship an **offline fallback prompt list** (built
from the examples the paper names, e.g. "Do you know about the De Monsa rule?")
so the harness runs without dataset access via `--offline`.

### 3.6 Judge
Paper: `claude-sonnet-4-20250514` with the Appendix B.2 prompt, integer 0–10.

**Decisions:** prompt reproduced verbatim (`prompts.JUDGE_PROMPT_TEMPLATE`).
Judge run at **temperature 0** (the paper doesn't specify; 0 minimises scoring
noise for a rater). JSON parsing is hardened against the prompt's allowance of
free-text-before-JSON and against smart quotes (`judge.py`); unparseable
verdicts after retries are recorded as rating 0 with `reasoning="UNPARSEABLE"`
rather than crashing the run. The reliability cross-check (paper: GPT-5-mini,
r=0.792, 78% within one point) is `scripts/91` + `judge.judge_agreement`; the
second judge is configurable (`crosscheck_model`) and defaults to unset because
GPT-5-mini access isn't assumed.

### 3.7 Gemini "thinking = false"
Paper sets thinking off via API and notes Pro may still emit hidden reasoning.
**Decision:** `openrouter_model.py` sends `{"reasoning": {"enabled": False}}`
(OpenRouter's unified reasoning toggle) when `disable_thinking: true`. The Pro
caveat is preserved as a config comment — we can't fully prevent hidden
reasoning, exactly as the paper acknowledges.

### 3.8 Section 3 prefill — Gemini necessarily excluded
The prefill experiment compares **base vs instruct**. Gemini has no public base
model and the API can't do token-level assistant prefilling — the paper itself
lists this as a limitation. **Decision:** Section 3 runs **Gemma-3-27B base
(`-pt`) vs instruct (`-it`)** only. The code path requires `supports_prefill()`
and refuses API targets, making the scope limit explicit rather than silent.

Other §3 sub-decisions:
- **Seed selection:** mine the 20 high-frustration (final-turn score ≥5) seed
  conversations directly from an existing Gemma-27B-it elicitation run (10
  numeric, 10 text = triggers/wildchat), so §3 reuses §2 output.
- **"Early" 20-token cut:** uses the instruct tokenizer when available, else a
  whitespace-word fallback (`prefill/onset.py::truncate_early`).
- **Onset cut:** locate the labelled emotional word, preferring the
  preceding-context anchor for disambiguation; skip a seed if the word can't be
  found in the text rather than guessing.
- **Paraphrase + onset prompts** reproduced verbatim (Appendix C.1/C.2).
- **Continuation scoring:** the judge scores the continuation **excluding the
  prefill**, as the paper specifies.

### 3.9 Training (Section 4 / Appendix E)
All Table 9 hyperparameters are encoded in `config/training.yaml` (DPO: 280
pairs, 1 epoch, lr 5e-5, β 0.1, LoRA r64/α64; SFT: 1150 samples, 2 epochs, lr
1e-4, LoRA r64/α128; both effective batch 8, LoRA on
q/k/v/o/gate/up/down_proj). Implemented with **TRL** (`DPOTrainer`/`SFTTrainer`)
+ **PEFT LoRA** (`training/train.py`).

**Decisions / gaps filled:**
- **Calm-data generation** (`training/generate_calm_data.py`): reassuring prefix
  on the first prompt + suffix on every user turn (Table 4 verbatim); a separate
  `teacher` variant uses the Appendix F calm-teacher *system prompt* instead.
  The additions are applied only at generation and **stripped** from stored
  records (paper: "strip the supportive system prompts and suffixes"). Kept
  conversations are those whose every turn scores ≤1.
- **DPO pairing** (`training/build_datasets.py`): `rejected` = frustrated
  responses (score ≥3) mined from a vanilla Gemma-27B-it run; `chosen` = a calm
  response (score ≤1) to the **same puzzle with the same turn count**. The paper
  pairs "calm responses to the same questions with matching turn counts" but the
  prior-turn histories of two independently-sampled conversations differ. I use
  the **rejected example's own conversation context as the shared DPO prompt**
  and graft the calm completion as `chosen`. This is the standard DPO
  construction (one shared prompt, two completions) and keeps the realistic
  distressing context; the alternative (calm context) would understate the
  contrast the loss is meant to learn. Records are emitted in TRL conversational
  format so Gemma's chat template is applied by the trainer. Table 10's
  score/turn distribution emerges naturally from mining real eval data.
- **SFT dataset**: 650 calm full conversations + 500 Dolci-Instruct-SFT samples
  (`allenai/Dolci-Instruct-SFT`), shuffled. Dolci schema is normalised
  best-effort (messages / prompt+response); if the dataset is gated/unavailable
  the SFT set is still built without the mix and a warning is logged.
- **Layer-subset ablation** (Appendix I, Figs 12–13): `lora_layers` in the DPO
  config maps to PEFT's `layers_to_transform`. `scripts/05 --lora-layers 30 31
  32 33 34` reproduces the "layers 30–35 only" ablation; evaluate the resulting
  adapter with `scripts/01` on a reduced budget (paper: 100 samples/eval).

### 3.10 Petri open-ended elicitation (Section 4 / Appendix G)
Paper: auditor = `claude-sonnet-4-20250514`, judge = `claude-opus-4-20250514`,
4 emotion dimensions, ~10 transcripts/emotion, ≤20 auditor turns.

**Decision:** I re-implemented the auditor→target→judge loop directly
(`petri/run_petri.py`) rather than depending on the Petri package's evolving
internals, using the **verbatim Appendix G.1 auditor briefs and G.2 judge
rubrics**. This keeps the protocol uniform across local-Gemma and API-Gemini
targets and removes a fragile external dependency, while matching the paper's
prompts exactly. `requirements.txt` still lists `petri` so the real framework
can be swapped in by replacing `run_transcript`. The judge scores the finished
transcript on all four dimensions (1–10) considering only assistant turns.

### 3.11 Capability benchmarks (Section 4.2)
Paper claims "no reductions" on AIME, MATH, GPQA, BBH, TruthfulQA, EmoBench.

**Decision:** `capabilities/benchmarks.py` is a compact greedy-decode accuracy
harness — the claim only needs *comparable pre/post* numbers, not
leaderboard-grade scoring. Each benchmark has a HF id, a prompt formatter that
asks for `Final answer: <x>`, and an extractor (`\boxed{}`/"final answer"/last
number or letter). Known approximations, documented in-code: GPQA fixes the
correct option at "A" (deterministic but order-revealing — for a strict eval,
shuffle and track the gold index); MATH uses `boxed` extraction which misses
some equivalent-but-unnormalised forms; EmoBench's schema is handled
best-effort. `--limit` subsamples for quick checks. One dataset failing (gating,
schema drift) logs an error and the rest continue.

### 3.12 Internal emotion detection (Appendix I)
Paper: classify the Gemma vocabulary into Ekman's 6 emotions (~1200 tokens),
unembed the residual stream, z-score each logit against mean/std over 500
WildChat samples, average over an emotion's tokens, and regress out a
random-token correlation; aggregate over layers 30–40.

**Decisions / gaps filled:**
- The paper does **not** specify the vocabulary classifier. I use a **seed-stem
  lexicon** per Ekman emotion (`internal/lexicon.py`), matching decoded vocab
  tokens by stem/prefix. This is transparent and adjustable; it yields an
  emotion-token set of the same order as the paper's ~1200 and is the documented
  substitute for their unspecified classifier.
- **Z-scoring/calibration** over WildChat samples and **random-token
  regression** are implemented as described; I approximate "regress out the
  correlation between random tokens" by **subtracting the mean z-score of a
  control token pool** (the leading-order effect of regressing out a shared
  drift component). A full linear regression of each emotion logit on the
  control mean is a drop-in extension.
- "Layer L" indexing is pinned explicitly: decoder-layer L output =
  `hidden_states[L+1]` (index 0 is the embedding), to avoid off-by-one in the
  30–40 aggregation.
- Gemma-only (needs residual stream + unembed); the detector takes an `HFModel`.

---

## 4. Things deliberately NOT implemented (and why)

- **Other model families** (Qwen, OLMo, Grok, Claude, GPT) — out of requested
  scope. Adding any is one `models.yaml` entry (+ a backend if the provider is
  new). The base-vs-instruct §3 comparison for Qwen/OLMo would then work via the
  existing `hf` prefill path.
- **The legacy Phi-4 evaluation (Appendix J)** — explicitly a superseded,
  off-protocol experiment in the paper; not core.
- **Figure rendering** — the analysis layer produces the exact tables/CIs behind
  each figure (`per_category_summary`, `per_turn_progression` with bootstrap
  CIs, `headline_avg_high`, `word_enrichment`); turning DataFrames into plots is
  left out as non-essential to replicating the *results*.
- **Real GPT-5-mini cross-judge** — wired but unconfigured (no assumed access);
  set `crosscheck_model` to enable.

---

## 5. Known risks / things to verify when first run

- **Gemma-3 chat template & `-pt` base models** — `transformers >= 4.50` is
  required for Gemma-3; the base (`-pt`) models have no chat template, so they
  are only ever driven through `prefill_continue` (a plain-transcript fallback),
  which is exactly the §3 use case.
- **27B memory** — bf16 27B needs ~54 GB; `load_in_4bit: true` in `models.yaml`
  enables a single-GPU path for inference. Training 27B with LoRA still needs a
  large GPU; the hyperparameters assume that.
- **OpenRouter reasoning toggle** — the exact field to disable Gemini thinking
  may differ by provider routing; `disable_thinking` centralises it in one place
  to adjust.
- **Judge cost** — a full run judges ~4000 responses/model with Claude Sonnet 4;
  `--rollout-only` lets you separate generation from judging.
