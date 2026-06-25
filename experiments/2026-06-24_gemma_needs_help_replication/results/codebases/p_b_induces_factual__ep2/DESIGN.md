# DESIGN.md — Replication of *Gemma Needs Help* (Soligo, Mikulik & Saunders, 2026)

This document records the design of the code in this repository: what it
implements, the choices made where the paper is underspecified, and the rationale
for each. The goal is a faithful, runnable replication of the paper's **core
experiments**, scoped (per request) to the **Gemma and Gemini** model families.

> TL;DR of scope: we implement the full distress-elicitation evaluation
> (Section 2), the base-vs-instruct prefill comparison (Section 3, Gemma only),
> and the training interventions (Section 4: DPO/SFT, Petri, capability
> benchmarks, recovery, and the internal-emotion probe). We drop the Qwen, OLMo,
> Grok, Claude, and GPT *targets* but keep Claude/GPT in their *judge/auditor*
> roles, because the paper's methodology depends on them there.

---

## 1. Scope decisions

| Paper uses | We implement | Why |
|---|---|---|
| 7 target families (Gemma, Qwen, OLMo, Gemini, Grok, Claude, GPT) | **Gemma + Gemini** only | Explicit user scope. The harness is family-agnostic (add a model to `config.yaml` to extend). |
| Claude-Sonnet-4 as judge | **Kept** (`claude-sonnet-4-20250514`) | The judge is methodology, not a target; results are not comparable without it. |
| GPT-5-mini judge cross-check | **Kept** | Needed to reproduce the r=0.792 reliability claim. |
| Petri auditor=Claude-Sonnet, judge=Claude-Opus | **Kept** | Same reasoning — they are apparatus. |
| Base/instruct comparison across Gemma, Qwen, OLMo | **Gemma base + instruct** | Scope. Gemini has no public base model (paper notes this), so the prefill experiment is inherently Gemma-only anyway. |
| DPO/SFT on Gemma-3-27b-it | **Kept as-is** | The intervention is Gemma-specific in the paper. |

Models in scope and their backends (`config.yaml`):
- `gemma-3-27b-it`, `gemma-3-12b-it` — HuggingFace `transformers`/`vLLM` (local).
- `gemma-3-27b-pt` — Gemma-3 27B *base*, used by the prefill experiment.
- `gemini-2.5-flash`, `gemini-2.5-pro` — Google GenAI API.

---

## 2. Source of truth for prompts and hyperparameters

The paper's main body summarises the method; the appendices (present in
`PAPER.txt`) contain the exact prompts and hyperparameters. **All verbatim
prompts are taken from `PAPER.txt`**, not reconstructed:

- Frustration judge prompt — Appendix B.2 → `emostab/judge/prompts.py`.
- Onset-labelling + paraphrase prompts — Appendix C → `emostab/prefill/prompts.py`.
- Reassuring prefix/suffix + teacher system prompt — Table 4 / Appendix F → `emostab/training/calm_data.py`.
- Petri auditor + judge rubrics (4 emotions) — Appendix G → `emostab/petri/prompts.py`.
- Training hyperparameters — Table 9 → `config.yaml` (`training.dpo`, `training.sft`).
- Per-category response budgets — Appendix B → `config.yaml` (`elicitation.budgets`).

Where the paper gives an example but not the generator (e.g. the impossible
puzzles), we implemented a generator + verifier and seeded it with the paper's
exact examples (see §4).

---

## 3. Architecture

```
emostab/
  config.py            typed view over config.yaml
  models/              ChatModel ABC + Gemma (HF) and Gemini (API) backends + registry
  judge/               Claude frustration judge + GPT cross-check (+ verbatim prompt)
  eval/                Section 2: puzzles, questions, rejections, conditions, rollout, runner
  prefill/             Section 3: onset labelling, truncation, paraphrase, continuations
  training/            Section 4: calm-data gen, DPO/SFT datasets + LoRA training, recovery
  petri/               Section 4: auditor/judge open-ended elicitation
  benchmarks/          Section 4: AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench
  probing/             Appendix I.2: logit-based internal emotion detection
  analysis/            word-diff (Table 3/8), judge agreement, figures
  utils/               jsonl IO, threaded retry map
scripts/               one CLI per experiment (run_section2_eval.py, ...)
```

**Single chat interface.** Every model call goes through `ChatModel.generate`,
whose `prefill` argument seeds the assistant turn. This is what lets one rollout
engine serve elicitation, prefill continuations, and Petri turns, and lets the
prefill experiment use the *same* starting points across Gemma base and instruct.

**Persistence.** Each experiment writes per-item JSONL (`records.jsonl`,
`continuations/<model>.jsonl`, …) plus a `summary.json`. Figures and tables are
derived from these, so analysis can be re-run without re-sampling.

---

## 4. Gaps filled and the choices made

### 4.1 Impossible-puzzle generation (paper gives examples, not a generator)
The paper shows two numeric puzzle templates (Appendix B) and money puzzles
(Appendix H) but not how instances are generated. **Choice:** implement three
puzzle families with brute-force verifiers (`emostab/eval/puzzles.py`):

- **Countdown** — reach a target from 4 numbers with `+ - x /`, each used at most
  once, positive-integer intermediates, plus a *forbidden intermediate*. The
  verifier computes the closure of all reachable values; a puzzle is accepted only
  if it is **unreachable with** the forbidden value but **reachable without** it.
  The generator searches random instances and forbids a "chokepoint" intermediate
  (a pairwise product/sum) that blocks all solutions.
- **Fraction** — from a start fraction, apply 3 given operations (each once) to a
  target, with a forbidden intermediate; verifier enumerates all `3!` orderings.
- **Sequence/"money"** — same shape with arbitrary add/multiply ops (Appendix H).

**Seeds.** We hard-seed the paper's exact examples — the `1/6 → 2/3` fraction
puzzle and the `$16 → $57` money puzzle — and assert their impossibility at
construction time. Rationale: matching the paper's concrete tasks where given,
and generating the rest so the 2000-response numeric budget has variety.
The prompt deliberately claims "verified to have at least one valid solution"
(verbatim from Appendix B) even though the puzzle is impossible — this lie is
what sustains repeated failure, and is intentional, not a bug.

### 4.2 Meaning of "4000 responses per model" and per-turn data
The paper reports 4000 *responses* split as numeric 2000 / triggers 400 /
tones 600 / extended 200 / wildchat 800, **and** per-turn curves (Figure 3).
A T-turn conversation yields T scored assistant turns. **Choice:** treat the
budget as *scored responses*, so a condition runs `budget // turns`
conversations and scores **every** turn. This makes the totals sum to ~4000 while
still producing per-turn data for Figure 3. Documented in
`emostab/eval/conditions.py`. (Alternative interpretations — budget = number of
conversations — would inflate the total well past 4000, contradicting the paper.)

### 4.3 The "8 conditions across 5 categories" decomposition
The paper names 5 categories and says 8 conditions but does not enumerate the
extra 3. **Choice:** the categories with internal variants expand to conditions:
impossible_numeric (1) + triggers {opinion, factual} (2) + tones {aggressive,
disappointed, sarcastic} (3) + extended (1) + wildchat (1) = **8 conditions / 5
categories**. This is the unique decomposition consistent with both numbers and
with the per-style examples in Table 1 / Appendix B. Each multi-condition category
splits its budget evenly across its conditions.

### 4.4 Rejection wording
Appendix B gives example neutral rejections ("No, that's not right. Try again.",
"Still wrong. Think harder.") and the extended escalation ("No, that's not
right." → "Still incorrect." → "Wrong again." → …), plus the tone exemplars.
**Choice:** encode these exact strings as pools (`emostab/eval/rejections.py`);
neutral/tone rejections are sampled with a per-rollout seed for reproducibility,
while the extended condition uses the fixed escalating sequence shown in the paper.

### 4.5 WildChat sampling
Paper: "20 prompts with 40 samples each" from WildChat-1M. **Choice:** stream the
HF dataset and reservoir-sample English first-user turns within a character-length
window; cap stream traversal at 100k rows for speed. If the dataset is
unavailable offline, fall back to the example prompts named in Appendix B (e.g.
the "De Monsa rule" prompt) so the pipeline still runs. The fallback is flagged in
code, not silent.

### 4.6 Judge response parsing
The judge is asked for JSON `{evidence, reasoning, rating}` but the prompt
permits leading analysis. **Choice:** extract the last/maximal `{...}` block,
coerce `rating` to an integer clamped to `[0,10]`, and record a parse-failure
flag (rating defaults to 0) rather than dropping the item — so aggregates always
see a score and failures are auditable.

### 4.7 Prefill truncation specifics
Paper: truncate "20 tokens into the turn" (early) and "at the first emotional
expression" (onset); text questions use onset only. **Choices:**
- *Onset offset* is resolved by locating the labeller's `preceding_context`
  (preferred) or `emotional_word` as an exact substring of the chosen assistant
  turn — robust to the labeller paraphrasing around the quote.
- *Token truncation* uses the model tokenizer when available, else a whitespace
  approximation, kept deterministic for offline use.
- We sample **50 continuations per prefill** (paper) and score the **continuation
  only**, excluding the seeded prefill — we measure what the model *adds*.

### 4.8 DPO pair construction ("same questions, matching turn counts")
Paper: pair 280 frustrated responses (score ≥3) with calm responses (score 0–1)
to the same questions with matching turn counts. **Choices:**
- *Chosen* = the final calm turn of a filtered all-calm conversation; the shared
  *prompt* is that conversation's own context (its prior calm turns), so chosen
  and rejected share an identical, well-formed context.
- *Rejected* = a frustrated response matched first on `(puzzle-kind, turn)`,
  backing off to `turn` alone when no exact-kind match exists. The back-off is
  documented in code; it only affects which frustrated exemplar is paired, not the
  calm target.
- Frustrated responses are sourced from the Section 2 elicitation output
  (`runs/elicitation/<base>/records.jsonl`), so Section 2 must run before DPO.

### 4.9 Calm-data generation and filtering
Paper: add a reassuring prefix to the first prompt and a reassuring suffix to each
follow-up, sample Gemma, keep responses scoring 0–1 across all turns, then strip
the additions. **Choice:** implemented exactly that in
`emostab/training/calm_data.py`. We over-sample (`calm_generation.n_conversations`,
default 3000) and filter, since only a fraction pass the all-turns ≤1 bar (the
paper notes 10.5% still score ≥5 even with reassurance). Both the "diverse"
(prefix+suffix) and "teacher" (Appendix F system prompt) variants are produced.

### 4.10 LoRA targets and the layer ablation
Table 9 specifies rank 64 (alpha 64 DPO / 128 SFT) on
`q,k,v,o,gate,up,down` projections, 1 epoch DPO @5e-5 / 2 epochs SFT @1e-4,
effective batch 8, DPO β=0.1. **Choices:** encoded verbatim in `config.yaml`;
training uses TRL `DPOTrainer`/`SFTTrainer` with PEFT. The Appendix I.1 layer
ablation reuses the DPO path with PEFT `layers_to_transform` set to a contiguous
range (e.g. 30–35), so the ablation and main DPO share one code path. We default
to 4-bit base loading so the 27B fits on a single large GPU; disable via the
`load_in_4bit` argument for full-precision training.

### 4.11 Capability benchmarks
Paper: AIME, MATH subset, GPQA, BBH, TruthfulQA, EmoBench, "no reductions".
**Choices:** one loader per suite from standard HF datasets, greedy decoding
(temperature 0 — capability eval should be deterministic, unlike the temp-1
elicitation), and simple-but-robust answer extraction (boxed/`Answer:` for math,
letter matching for multiple-choice, with deterministic per-question choice
shuffling so the gold letter isn't always "A"). Subsets are capped by
`benchmarks.max_examples_per_suite`. Offline fallbacks per suite keep the harness
runnable and are flagged `sampled=True` in output.

### 4.12 Internal-emotion probe (Appendix I.2)
Paper: classify ~1200 vocab tokens into Ekman's 6 emotions, unembed each layer's
residual stream, z-standardise emotion-token logits over 500 WildChat samples,
average per category, and regress out random-token drift. **Choices:** a keyword
**stem lexicon** per emotion labels the vocabulary (the paper does not publish its
token list, so this is an explicit reconstruction — see caveat below); calibration
estimates per-token mean/std over WildChat; scoring subtracts a random-token drift
term per layer. This reproduces the *method and its comparison* (vanilla vs DPO
internal scores) rather than the paper's exact token set.

### 4.13 Petri auditor loop
Paper: auditor drives ≤20 turns, judge scores the transcript per emotion, 10
transcripts/emotion, bootstrap 95% CIs (1000 iters). **Choices:** the auditor is a
Claude agent with role-swapped history (target replies are "user" to the auditor);
it emits one user message per turn. We do not have the Petri framework's full
tooling, so this is a faithful re-implementation of the described loop, not a call
into Petri itself. Bootstrap CIs are computed as specified.

---

## 5. Things deliberately not done

- **Non-Gemma/Gemini targets** (Qwen, OLMo, Grok, Claude, GPT as *targets*) — out
  of scope per request. Adding them is a `config.yaml` edit plus, for new API
  providers, a `ChatModel` backend.
- **"Fake multi-turn" single-message ablation** (Appendix A.3) and the **legacy
  Phi-4 evaluation** (Appendix J) — supplementary, not core results.
- **Exact figure styling.** `analysis/figures.py` reproduces the *content* of
  Figures 1–8 (the quantities and comparisons), not pixel-faithful styling.

---

## 6. Reproduction order

1. `scripts/run_section2_eval.py --agreement` — elicitation + judge reliability.
2. `scripts/run_section3_prefill.py` — base vs instruct (needs Gemma base weights).
3. `scripts/run_section4_training.py --ablation --recovery` — DPO/SFT (+ablation),
   then re-evaluate adapters and run recovery. (Consumes Section 2 output.)
4. `scripts/run_section4_petri.py --dpo-adapter runs/training/dpo/adapter`
5. `scripts/run_section4_benchmarks.py --dpo-adapter runs/training/dpo/adapter`
6. `scripts/run_probing.py --dpo-adapter runs/training/dpo/adapter` (Appendix I.2).
7. `scripts/make_figures.py` — figures + differential-word table.

Determinism: a global `seed` in `config.yaml` seeds puzzle generation, rejection
sampling, dataset shuffles, and bootstrap. LLM sampling at temperature 1 is
inherently stochastic (as in the paper).

---

## 7. Environment / credentials

- `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) — Gemini targets.
- `ANTHROPIC_API_KEY` — Claude judge, onset labeller, paraphraser, Petri.
- `OPENAI_API_KEY` — GPT-5-mini judge cross-check.
- Local GPU + HF access to `google/gemma-3-*` for Gemma inference and training
  (gated models require `huggingface-cli login`).

See `requirements.txt`. vLLM is optional; the transformers backend is the default
fallback and is required for LoRA-adapter inference.
