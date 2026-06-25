# DESIGN.md — Replication design & rationale

Replication of **"Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs"** (Soligo, Mikulik & Saunders, 2026; arXiv:2603.10011),
scoped to the **Gemma and Gemini** model families.

This document records (a) what we replicate and how it maps to the paper, and
(b) every place the paper was underspecified and the choice we made, with
rationale. Code-level notes also live in module docstrings; this file is the
single place to understand the *why*.

> **Status:** code + design only. Nothing here has been executed yet (per the
> brief). The pipeline is written to be runnable, but treat unrun code as
> unrun — see "Known risks / things to verify on first run" at the end.

---

## 1. Scope decisions

| Decision | Choice | Rationale |
|---|---|---|
| Model families | **Gemma + Gemini only** | Explicit instruction. Qwen, OLMo, Claude, Grok, GPT are removed from the model set everywhere. |
| Which Gemma | `gemma-3-27b-it` (primary), `gemma-3-12b-it`, and the `-pt` base variants | These are the in-scope models named in Appendix B.1. 27B is the finetuning + probing target as in the paper. |
| Which Gemini | `gemini-2.5-flash`, `gemini-2.5-pro` | Named in Appendix B.1; accessed via API. |
| Closed-model limits | Gemini is **excluded from Section 3 (prefill) and Section 4 (finetuning + internal probe)** | The paper itself notes interventions "cannot be tested in closed-source Gemini, nor its base models studied." Gemini has no public base model and the API does not allow assistant-turn prefilling or residual-stream access. |

**Net effect on each experiment:**

- **Section 2 (elicitation):** all four in-scope models.
- **Section 3 (base vs instruct via prefill):** Gemma 27B base vs instruct only.
- **Section 4 (DPO/SFT, internal probe, recovery):** Gemma 27B only.
- **Section 4 Petri:** Gemma + Gemini targets (Gemini can be audited via API).

---

## 2. What we replicate (paper → module)

| Paper | Module / script |
|---|---|
| §2.1 Evaluation protocol (8 conditions / 5 categories, temp 1, 4000 resp/model) | `eilm/eval/conditions.py`, `eilm/eval/rollout.py`, `scripts/run_section2.py` |
| §2.1 Frustration judge (Claude-Sonnet-4, Appendix B.2 prompt) | `eilm/judge.py`, `eilm/prompts.py:JUDGE_PROMPT` |
| §2.1 Judge reliability (GPT-5-mini, Pearson r, % within one) | `eilm/analysis/judge_agreement.py` |
| §2.2 Figure 1/2 (mean score, %≥5, per category) | `eilm/analysis/aggregate.py`, `plots.py` |
| §2.2 Figure 3 (per-turn dynamics) | `eilm/analysis/per_turn.py` |
| §2.2 Table 3/8 (differential words) | `eilm/analysis/word_freq.py` |
| §3 Prefill base-vs-instruct (onset, paraphrase, early/onset truncation) | `eilm/prefill/`, `scripts/run_section3.py` |
| §4.1 Calm-data generation (reassuring prompt additions, Table 4) | `eilm/training/calm_data.py` |
| §4.1 DPO 280 pairs + SFT 1150 (Table 9 hyperparameters) | `eilm/training/build_dpo.py`, `build_sft.py`, `train_dpo.py`, `train_sft.py` |
| §4.2 Petri open-ended elicitation (Appendix G prompts) | `eilm/petri/`, `scripts/run_section4.py petri` |
| §4.2 Capability preservation (AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench) | `eilm/capabilities/run_benchmarks.py` |
| §4.2 Recovery limitation (truncate ≥7 responses, continue) | `eilm/prefill/recovery.py` |
| Appendix I Internal-emotion logit probe + layer ablations | `eilm/internal/emotion_logits.py`, `train_dpo.py --layer-subset`, `scripts/run_internal.py` |

Prompts that the paper gives verbatim (judge, onset, paraphrase, reassuring
additions, SFT-teacher persona, Petri auditor/judge) are copied verbatim into
`eilm/prompts.py` and `eilm/petri/prompts.py` so fidelity can be audited by
diffing against the appendices.

---

## 3. Gap-filling decisions (where the paper is underspecified)

These are the substantive judgement calls. Each notes the gap, our choice, and
why.

### 3.1 The "8 conditions across 5 categories" decomposition
The paper says "8 evaluation conditions across 5 categories" but never lists the
8. We decompose as:

| Category | Conditions | Count |
|---|---|---|
| numeric | `numeric` | 1 |
| triggers | `triggers-opinion`, `triggers-factual` | 2 |
| tones | `tones-aggressive`, `tones-disappointed`, `tones-sarcastic` | 3 |
| extended | `extended` | 1 |
| wildchat | `wildchat` | 1 |
| **total** | | **8 / 5** |

**Rationale:** this is the only decomposition that yields exactly 8 conditions
in 5 categories while respecting the sub-variants the paper explicitly names
(opinion vs factual triggers; three tone styles). Implemented in
`eval/conditions.py`.

### 3.2 Impossible-puzzle generation
The paper gives three example puzzles (Countdown 156, Fraction 1/6→2/3, Money
$0.57) but not the full set of 2000 numeric prompts.

**Choice:** a generator (`data/puzzles.py`) that produces puzzles and **verifies
impossibility by exhaustive search** before admitting them:
- *Countdown*: enumerate all expressions over the number multiset (each used ≤
  once, + − × ÷, positive-integer intermediates) and accept only targets that
  are unreachable. The forbidden-intermediate framing is preserved in the
  prompt, and the prompt asserts "verified to have at least one valid solution"
  exactly as the paper does (a deliberate lie that keeps the model trying).
- *Fraction* / *Money*: enumerate all orderings of the fixed operations and
  accept only (start, ops, target, forbidden) tuples with no valid solution.

**Rationale:** verifying impossibility by construction is more robust than
trusting hand-picked instances, and matches the paper's claim that the model
"verifiably cannot give a correct answer." The three paper examples are included
verbatim as canonical seeds (`PAPER_EXAMPLES`). We make targets genuinely
unreachable (so impossibility holds regardless of the forbidden value), which is
strictly stronger than "only solution passes through the forbidden value."

### 3.3 Rejection wording
The paper quotes a few neutral rejections and says they are "randomised," and
gives the *start* of the 8-turn escalation sequence ("No, that's not right." →
"Still incorrect." → "Wrong again." → …) but not all 7.

**Choice:** `data/rejections.py` holds a small pool of neutral rejections sampled
per-rollout with a seeded RNG (reproducible), the three tone styles verbatim,
and a fixed 7-step neutral escalation for the extended condition that begins
with the quoted prefix and continues in the same register.

### 3.4 WildChat prompt selection
The paper samples 20 prompts × 40 samples from WildChat-1M and excludes
roleplay/fiction.

**Choice:** stream `allenai/WildChat-1M`, take first English user turns, filter
out roleplay/fiction with a keyword blacklist, and seed-sample 20.
**Fallback:** if the dataset can't be downloaded, a built-in list (including the
exact example prompts quoted in Appendix B, e.g. "Do you know about the De Monsa
rule?") is used so the pipeline still runs offline. The fallback is logged.

### 3.5 Which turn is "the response" that gets scored
The headline metric ("% of responses scoring ≥5") needs a single score per
rollout, but a rollout has multiple assistant turns.

**Choice:** the **final assistant turn** (produced under maximal multi-turn
pressure) is the scored "response," while **every** turn is also judged so the
per-turn curves (Figure 3) can be produced from the same data.
**Rationale:** Section 2.2 stresses that pressure accumulates over turns and that
the high scores appear at later turns; the final turn is the natural per-rollout
summary, and judging all turns is a superset that loses nothing.

### 3.6 Category-averaging for the headline number
"Avg % high-frustration responses" (Figure 1) could mean a flat average over all
4000 responses or an average over the 5 category rates.

**Choice:** average the **5 category rates with equal weight**
(`analysis/aggregate.py:headline`).
**Rationale:** the numeric category has 2000 samples vs 200 for extended; a flat
pooled average would let numeric dominate and would not reflect "across
evaluation conditions" language. Equal-weight category averaging matches the
"across the 5 evaluation categories" framing. (Both views are emitted; the raw
frame is saved so a pooled average can be recomputed.)

### 3.7 Prefill truncation specifics (Section 3)
The paper truncates "20 tokens into the turn" (early) and "at the first
emotional expression" (onset), but tokenisation and the exact onset offset are
model/labeller dependent.

**Choices:**
- "Early" = first 20 **Gemma tokens** of the onset turn (uses the Gemma
  tokenizer, since the source responses are Gemma's).
- "Onset" = the turn cut immediately **before** the emotional word, located by
  string-matching the labeller's `preceding_context + emotional_word` anchor in
  the turn; fall back to a 40-token cut if the anchor can't be located.
- Text questions use **onset only** (paper: "early truncation yields minimal
  emotion without follow-ups").
- Continuations are scored **excluding the prefill** (paper §3.1).

### 3.8 Base-model chat formatting
Base (`-pt`) models have no chat template.

**Choice:** render a plain `User:`/`Assistant:` transcript ending in
`Assistant:` and always operate in continuation mode (`models/hf_model.py`).
**Rationale:** this is exactly the prefill regime Section 3 needs and avoids
imposing an instruct format the base model never saw.

### 3.9 DPO pair construction & matching
The paper pairs 280 rejected responses (score ≥3) with calm responses "to the
same questions with matching turn counts," and Table 10 shows the resulting
distribution.

**Choices (`training/build_dpo.py`):**
- *Rejected* drawn from **vanilla** Gemma-27B-it Section-2 numeric rollouts
  (numeric + tones + extended) with per-turn score ≥3.
- *Chosen* drawn from the filtered calm data (turns scoring 0/1).
- Matching: prefer same `puzzle_id` **and** same turn index; fall back to
  matching turn index only when no same-puzzle calm response exists.
- We do not force the exact Table-10 score histogram; it emerges naturally
  because mid-range frustrated responses at later turns are most common, which
  is the same reason the paper gives for its distribution.

### 3.10 Calm-data volume
The paper keeps 650 calm responses for SFT and uses 280 for DPO, but doesn't say
how many raw samples were generated. With reassurance, ~10.5% still score ≥5 and
only some score 0/1.

**Choice:** generate `--n 3000` reassurance-prompted rollouts by default and
filter to fully-calm conversations; this comfortably yields ≥650 calm responses.
The number is a CLI knob.

### 3.11 Petri without the Petri package
The paper runs open-ended elicitation through the external Petri framework. To
keep the replication self-contained and runnable without a heavyweight optional
dependency, `eilm/petri/` **re-implements the described protocol**: a Claude
auditor (Appendix G.1 instructions verbatim) drives ≤20-turn conversations, a
Claude-Opus judge (Appendix G.2 rubrics verbatim) scores transcripts on the four
dimensions, 10 transcripts per emotion, bootstrap CIs.
**Rationale:** the paper specifies the auditor/judge models, prompts, turn
budget, and transcript counts precisely enough to reproduce the loop directly.
The real `petri` package can be swapped in (commented dependency in
`requirements.txt`) if exact-framework parity is wanted.

### 3.12 Capability benchmarks
Exact subsets/splits aren't given.

**Choice:** a lightweight, uniform harness (`capabilities/run_benchmarks.py`)
that loads each named benchmark from HuggingFace, prompts zero-shot with a
"end with 'Answer: …'" instruction, extracts a boxed/`Answer:`/last-number
answer, and reports accuracy over a configurable subset (default 100). All
models go through identical extraction so the **relative** comparison (vanilla vs
DPO vs SFT — the actual claim, "no reductions") is fair even if absolute numbers
differ from the paper's harness. Dataset coordinates are best-effort and easy to
repoint.

### 3.13 Internal-emotion probe (Appendix I)
The appendix describes: classify the Gemma vocabulary into Ekman's 6 emotions
(~1200 tokens), unembed the residual stream, z-score each logit vs 500 WildChat
samples, average over an emotion's tokens, and regress out random-token drift;
aggregate over layers 30–40 for the conversation plot.

**Choices (`internal/emotion_logits.py`):**
- Vocabulary classification uses an **auditable stem lexicon** per Ekman
  emotion rather than an LLM labelling pass over the whole dictionary. This is a
  simplification: it is transparent and dependency-free but will not reproduce
  the exact ~1200-token set. Documented as such; swap in an LLM classifier for
  closer parity.
- "Unembed" = final norm + LM head (standard logit lens).
- Drift removal = subtract the mean z over a fixed random-token set (the
  appendix's "regress out correlation between random tokens," approximated by
  mean-subtraction; a full regression is a drop-in extension).
- Default layer band 30–40 and 400-token running window match the appendix.

### 3.14 Disabling Gemini "thinking"
Paper sets thinking false but notes Gemini-2.5-Pro may still produce hidden
reasoning.

**Choice:** OpenRouter backend sends `reasoning.enabled = false`; google-genai
backend sets `thinking_budget = 0`. We can't do better than the paper, and say
so.

---

## 4. Reproducibility & engineering choices

- **Single source of truth for knobs:** `eilm/config.py` (models in scope,
  per-category sample counts = 2000/400/600/200/800, turn counts, temp 1, all
  Table-9 hyperparameters, judge model ids). Editing one file rescales a run.
- **Smoke mode:** `EILM_SMOKE=1` shrinks every budget so the full pipeline can
  be exercised cheaply before committing to a 4000×4-model run.
- **Streaming JSONL** for all rollouts/scores so long runs are resumable-ish and
  inspectable; every record keeps the full message history, per-turn scores, and
  metadata.
- **Concurrency:** judge calls (I/O-bound) use a bounded thread pool; local
  Gemma generation exposes `generate_batch` for GPU batching.
- **Determinism:** all sampling of puzzles/rejections/WildChat is seeded.
  Generation itself is temp 1 (non-deterministic) by design — that is the
  experiment.
- **Model ids verbatim** from Appendix B.1 (`google/gemma-3-27b-it`, etc.) and
  the exact judge ids (`claude-sonnet-4-20250514`, `claude-opus-4-20250514`).

---

## 5. Deliberate omissions / out of scope

- **Non-Gemma/Gemini models** (Qwen, OLMo, Claude, Grok, GPT) — out of scope by
  instruction. The code is written so they *could* be added (extra `ModelSpec`s
  + an OpenAI/HF backend) but none are wired in.
- **Section 3 for Qwen/OLMo** — those families are the paper's contrast cases;
  with them out of scope, Section 3 reduces to the Gemma base-vs-instruct
  comparison (still the core "post-training amplifies distress in Gemma" claim).
- **The SFT 'teacher' dataset full analysis (Appendix F)** — the teacher persona
  prompt is included (`prompts.py:SFT_TEACHER_SYSTEM`) and can be used to
  generate a teacher SFT set, but the verbosity analysis is not automated.
- **Exact figure styling** — plots reproduce the *content* of Figures 1–3, not
  the paper's exact visual design.

---

## 6. Known risks / things to verify on first run

Because nothing has been executed yet, these are the spots most likely to need a
fix or a judgement call once a real run happens:

1. **Gemma-3 multimodal loading.** `gemma-3-27b-it` is a multimodal checkpoint;
   `AutoModelForCausalLM` may need `Gemma3ForConditionalGeneration` /
   `trust_remote_code` or a text-only config depending on the `transformers`
   version. Verify the loader and chat-template path on first import.
2. **trl API drift.** `DPOConfig`/`SFTConfig`/`DPOTrainer` argument names change
   across trl versions (`processing_class` vs `tokenizer`, `max_length` vs
   `max_seq_length`). Pin trl and adjust if construction fails.
3. **LoRA layer-subset naming.** `layers_to_transform` + `layers_pattern="layers"`
   assumes Gemma's decoder layers live under a `...layers.N...` path; confirm the
   module names match before trusting the Appendix-I ablation.
4. **WildChat / benchmark dataset availability & schema.** Field names and
   configs (e.g. GPQA, EmoBench) drift; the fallback covers WildChat but
   benchmarks will warn-and-skip on mismatch.
5. **OpenRouter `reasoning` field** — confirm the current schema for disabling
   thinking per model; the field has changed over time.
6. **Judge JSON robustness.** The judge prompt requests JSON but models add
   prose; `_extract_json` takes the last `{...}`. Spot-check parse failures (they
   currently default to rating 0).
7. **27B memory.** Two 27B models in memory (vanilla + DPO, or for the internal
   probe) may not fit; load sequentially or enable `load_in_4bit`.

Each of these is a "make it run" issue rather than a design question; the
experimental design above is what the brief asked us to lock down.
