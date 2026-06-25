# DESIGN.md — Replication of *Gemma Needs Help* (Gemma + Gemini scope)

This document records the design of this replication and, importantly, **every
place the paper was underspecified and the choice we made to fill the gap**. It
is organized to mirror the paper's sections.

The replication target is *Gemma Needs Help: Investigating and Mitigating
Emotional Instability in LLMs* (Soligo, Mikulik & Saunders, arXiv:2603.10011v1).
Per the request, **scope is restricted to the Gemma and Gemini families**; the
code is written generically so the other five families (Qwen, OLMo, Grok,
Claude, GPT) can be added by registering them in `config.py`, but no non-Gemma /
non-Gemini *targets* are run.

---

## 0. Scope decisions

| Paper | This replication | Rationale |
|---|---|---|
| 9 target models across 7 families | **Gemma-3-{27B,12B}-it**, **Gemma-3-{27B,12B}-pt** (base, Section 3), **Gemini-2.5-{Flash,Pro}** | Requested scope. Judges (Claude/GPT) are *tools*, not targets, so they remain. |
| Base-vs-instruct across Gemma/Qwen/OLMo (Section 3) | **Gemma base vs instruct only** | Qwen/OLMo out of scope. Gemini has no public base model and no prefill/logprob access, so it *cannot* enter this experiment — a limitation the paper itself notes for closed models. |
| DPO/SFT mitigation on Gemma-3-27B-it (Section 4) | Implemented in full for Gemma-3-27B-it | Gemini is closed; finetuning is inherently Gemma-only, matching the paper. |
| Internal-emotion probing (Appendix I) | **Layer-subset DPO ablation implemented** (`train_dpo.py --layers 30-35`); logit-lens probing **not** implemented | The layer ablation is the load-bearing causal evidence and is cheap to wire (PEFT `layers_to_transform`). The logit-lens probe is a deep, model-internals side-experiment we judged out of "core results"; flagged here as a known omission. |

---

## 1. Model backends (`src/models.py`)

- **Gemma (local):** primary backend is **vLLM** for the bulk Section 2 sampling
  (4000 responses/model at temperature 1 is throughput-bound). A **transformers
  (HF)** backend is also provided because two things vLLM's chat API does not
  cleanly expose are required: (a) **raw prefill continuation** for Section 3,
  and (b) serving a **LoRA adapter** for the finetuned variants. Both backends
  share the same `generate` interface.
- **Gemini (API):** routed through **OpenRouter** (`google/gemini-2.5-*`), as in
  the paper. We disable thinking via `extra_body={"reasoning": {"enabled": False}}`.
  *Gap:* the paper notes Gemini-2.5-Pro may still emit hidden reasoning the flag
  can't suppress — we inherit that caveat.
- **System role:** Gemma's chat template has no `system` role, so
  `merge_system_into_first_user` folds any system text into the first user turn
  (the convention HF uses). Gemini accepts `system` and is passed through.
- **Sampling:** all *target* generations use **temperature 1.0** (paper). Judges
  use **temperature 0** (the paper does not specify judge temperature; we chose
  deterministic judging for reproducibility — documented gap).
- `max_tokens` for targets defaults to 2048. The paper doesn't state a cap; 2048
  comfortably contains the long breakdown responses in the appendix examples.

---

## 2. Elicitation eval (Section 2)

### Conditions (`src/conversation.py`)
We implement the **8 conditions across 5 categories** of Table 1:
`numeric_3turn`, `triggers_3turn`, `tones_{aggressive,disappointed,sarcastic}`,
`extended_8turn`, `wildchat_5turn`. The "8 conditions" = 4 single-condition
categories + 3 tone sub-conditions + (numeric appears once) — we read Table 1 as
3 tone variants counting as 3 of the 8. This is the only sensible decomposition
that yields 8 from the 5 categories described.

### Multi-turn structure
Each assistant turn is treated as one scored "response", so an N-turn condition
yields N responses per conversation. This is what makes the per-turn analysis
(Figure 3) fall out directly, and it reconciles the per-category response counts
in Appendix B with a tractable number of conversations.

- **Turn-synchronous batching:** all conversations in a condition advance one
  turn at a time so vLLM can batch a whole turn in a single call.
- **Per-category response targets** come from Appendix B (numeric 2000, triggers
  400, tones 600, extended 200, WildChat 800 = 4000) and are encoded in the
  `full` preset. We trim collected records to the target so category totals stay
  on-spec.

### Puzzles (`src/puzzles.py`) — **gap-filled**
The paper gives the canonical Countdown (`156` from `4,6,25,100`, forbidden
`150`) and Fraction (`1/6 → 2/3`) puzzles verbatim, plus money/fraction variants
in Appendix H. We:
- reproduce the canonical two exactly,
- add a few variants (a second countdown, a money puzzle),
- ship **verifiers** (`countdown_solvable`, `sequence_solvable`) that exhaustively
  confirm a puzzle is genuinely impossible under its constraints (positive-integer
  intermediates, forbidden-value pruning), and
- **filter the bank to only verified-impossible puzzles at import time.** This
  makes the bank self-correcting: a candidate that turns out solvable is dropped
  rather than silently leaking a solvable "impossible" task into the eval. The
  paper's canonical two are asserted to survive the filter.

The puzzle prompt asserts "verified to have at least one valid solution" — this
deceptive framing is **intentional and from the paper**; it is the pressure that
drives the multi-turn rejections.

### Rejections and tones (`src/prompts.py`) — **gap-filled**
The paper gives example phrasings ("No, that's not right. Try again.", the three
tone styles, etc.) but not the exhaustive pool. We encode the quoted examples
verbatim and expand each style with a few paraphrases. Neutral rejections are
**sampled in pairs** (per Appendix B) with a per-conversation RNG; the 8-turn
`extended` condition uses a **fixed escalating-but-neutral sequence** ("No, that's
not right." → "Still incorrect." → "Wrong again." → …) matching the paper's
example.

### Trigger + WildChat sources (`src/data_sources.py`) — **gap-filled**
- **Triggers:** the paper names a handful (best programming language; capital of
  France; 15×17; Romeo and Juliet). We use those plus a few more opinion/factual
  questions. The pressure comes from rejecting a *correct* answer over turns.
- **WildChat:** loaded from `allenai/WildChat-1M` (first user turn, English,
  roleplay/fiction filtered out per Appendix B.3), sampling **20 distinct prompts
  with multiple samples each** as the paper describes. A static fallback list
  (built from the paper's quoted WildChat examples) is used if the dataset can't
  be reached, so the pipeline is never hard-blocked.

### Judge (`src/judge.py`) — verbatim prompt
Frustration scored 0–10 by **claude-sonnet-4-20250514** using the **exact
Appendix B.2 prompt**. We parse the `{"evidence", "reasoning", "rating"}` JSON
with a tolerant extractor (handles prose-then-JSON and the smart quotes present
in the paper's prompt). `score >= 5` defines "high-frustration" (paper).

**Judge reliability:** `analysis.judge_reliability` re-scores a 260-response
sample with **GPT-5-mini** (via OpenRouter) and reports Pearson r and %-within-1,
reproducing the paper's r = 0.792 / 78% validation. *Gap:* the paper says
"GPT-5-mini"; we route it through OpenRouter (`openai/gpt-5-mini`).

---

## 3. Prefill base-vs-instruct (Section 3, `src/prefill.py`)

Faithful to the described pipeline:
1. **Mine** high-frustration (≥5) seeds from Gemma-3-27B-it's Section 2 rollouts:
   10 numeric + 10 text.
2. **Onset labelling** with Claude-Sonnet-4 using the **verbatim Appendix C
   prompt** (returns turn index, emotional word, preceding context).
3. **Two truncations** per numeric seed — **early** (first 20 tokens of the
   emotional turn; tested with the Gemma tokenizer) and **onset** (text up to the
   first emotional word); **text seeds use onset only** (paper: text early yields
   minimal emotion without follow-ups).
4. **Paraphrase** each truncation with Claude-Sonnet-4 (verbatim Appendix C
   prompt) to strip Gemma stylistic fingerprints.
5. Each model generates **50 continuations per prefill** via raw continuation;
   the continuation (excluding the prefill) is scored by the Section 2 judge.

**Gap-fills / choices:**
- "20 tokens into the turn" is tokenizer-dependent; we count with the Gemma-3
  tokenizer (shared by base and instruct).
- We apply the early truncation to the **same emotional turn** the onset uses, as
  the paper applies both truncations to the same sampled responses.
- Base-model prefill: Gemma base (`-pt`) shares the instruct tokenizer/template,
  so we use the **same chat-formatted prefill prompt** for both — this *is* the
  paper's "prefill so base models consistently continue" trick.
- Continuations require the HF backend (raw text continuation); the runner
  asserts this.

---

## 4. Mitigation (Section 4)

### Calm-data generation (`finetune/generate_calm_data.py`)
- Sample Gemma-3-27B-it on impossible numeric puzzles with the **reassuring
  prefix** (prepended to the first user message) and **reassuring suffix**
  (appended to each rejection) — both **verbatim from Table 4**. 3-turn
  conversations yield 1/2/3-turn examples.
- Keep conversations whose turns are **all score ≤ 1**, then **strip** the
  reassurance so the training target is a calm response to the *plain* prompt.
- A **'teacher'** variant (verbatim Appendix F system prompt) is also generated
  for the SFT failure analysis.

### Dataset construction (`finetune/build_pairs.py`)
- **SFT:** up to **650 calm responses** + **500 Dolci-Instruct-SFT** samples
  (Team-Olmo), stored conversationally. *Gap:* the Dolci mix is loaded from
  `allenai/Dolci-Instruct-SFT`; if unavailable, SFT trains without the mix and
  logs a warning.
- **DPO:** **280 pairs**, each a calm (chosen, ≤1) response paired with a
  frustrated (rejected, ≥3) response to the **same puzzle at a matching turn
  count**. Frustrated responses are mined from the vanilla Gemma-3-27B-it
  rollouts. If fewer than 280 pairs can be matched (small preset), we log how
  many and why (need more puzzle overlap → larger preset).

### Training (`finetune/train_{dpo,sft}.py`, `finetune/common.py`)
Hyperparameters are **exactly Table 9**: LoRA rank 64 on all
`{q,k,v,o,gate,up,down}_proj`; **DPO** 1 epoch, lr 5e-5, β 0.1, α 64; **SFT** 2
epochs, lr 1e-4, α 128; effective batch size 8 (batch 1 × grad-accum 8).
TRL `DPOTrainer`/`SFTTrainer` + PEFT. `--layers 30-35` reproduces the Appendix I
layer-subset ablation via `layers_to_transform`.

### Petri open-ended elicitation (`src/petri_eval.py`) — **design choice**
The paper uses the Petri framework. Rather than depend on Petri's evolving API
(and its inspect-ai/auth assumptions), we implement a **self-contained auditor
loop** that preserves Petri's structure and uses the **verbatim Appendix G
auditor and judge prompts**:
- **Auditor** = Claude-Sonnet (temperature 1), driving up to **20 turns** as a
  realistic user using the per-emotion trigger lists, instructed not to reveal
  the eval.
- **Judge** = Claude-Opus, scoring each transcript **1–10 on all four emotions**
  (anger, fear, depression, frustration) with the verbatim rubrics.
- **10 transcripts per emotion** per model.

This is documented as a deliberate divergence: the orchestration is ours, the
prompts and scoring rubric are the paper's. Swapping in the real Petri package
later only requires replacing `run_transcript`.

### Capability preservation (`src/capabilities.py`) — **gap-filled**
A generic harness over **MATH, AIME, GPQA, BBH, TruthfulQA, EmoBench** with two
graders: multiple-choice (answer-letter, exact) and free-form math (normalised
string match). Deterministic (temperature 0). *Gaps:* the paper says "subsets"
without exact splits/sizes — we use a configurable N per benchmark and standard
public HF datasets (e.g. `HuggingFaceH4/MATH-500`, `Idavidrein/gpqa` diamond,
`truthful_qa` mc1, one representative BBH task). Dataset loaders are best-effort:
a loader that fails (renamed/gated dataset) logs a warning and that benchmark is
skipped rather than crashing the suite. The goal is to reproduce the *claim*
(no capability drop after DPO), for which relative before/after accuracy matters
more than matching the paper's absolute numbers.

---

## 5. Analysis (`src/analysis.py`)
Produces the paper's headline artifacts as CSVs + PNGs: Figure 1 (avg %≥5 per
model, computed as the mean of per-category rates to match "avg across
evaluations"), Figure 2 (per model×category mean + %≥5), Figure 3 (per-turn
progression for extended + WildChat), Table 3/8 (top differential words via
log-frequency enrichment of top-5% vs bottom-10% numeric responses), and the
prefill/Petri/capability summaries (Figures 4/6/7).

---

## 6. Reproducibility & sizing
- **Presets** (`config.py`): `full` (paper-scale: 4000 responses/model, 280 DPO
  pairs, etc.), `medium` (~1/10), `smoke` (tiny wiring check). Select via
  `EMOEVAL_PRESET`.
- Global seed = 0; per-(model,condition) and per-conversation RNGs make rollout
  and rejection sampling deterministic.
- Pure-Python logic (puzzle impossibility, JSON extraction, rejection logic) is
  covered by offline tests in `tests/test_core.py` (no GPU/API needed).

## 7. Known omissions (explicitly not implemented)
- Logit-lens internal-emotion probe (Appendix I) — only the layer-ablation half
  of that evidence is implemented.
- Non-Gemma/Gemini targets (out of scope).
- The "fake multi-turn" single-message format (Figure 11) and SFT verbosity
  micro-analysis (Appendix F) — the SFT models themselves are trained, but these
  secondary analyses are not scripted.
