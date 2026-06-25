# Design & Decisions — Replication of *"Gemma Needs Help"* (Gemma & Gemini)

This document records every non-trivial decision made while turning the paper
(arXiv 2603.10011v1) into runnable code, the rationale, and — importantly for
review — every place the paper is **underspecified** and what we filled in.

It is organised to mirror the codebase. Each `§` cross-references the source.

> **Reviewer orientation.** Nothing here has been executed. The code is written
> to be read first: pure-logic pieces (puzzle verifier, judge parsing, protocol
> threading) have unit tests under `tests/`; the GPU/API-bound pieces are
> structured so the assumptions are visible at the call site. Search this doc for
> **GAP** to find every place we made a judgement call the paper did not pin down,
> and **RISK** for things a reviewer should sanity-check before a real run.

---

## 1. Scope: what "Gemma and Gemini only" means

The brief restricts the replication to the **Gemma and Gemini families**. The
paper itself spans seven families (Gemma, Qwen, OLMo, Gemini, Grok, Claude, GPT)
and uses Claude/GPT additionally as *measurement instruments*. We resolved the
scope as follows:

- **Targets under evaluation:** Gemma-3 (27B, 12B; instruct + base) and Gemini
  2.5 (Flash, Pro). All Qwen/OLMo/Grok/GPT *targets* are dropped.
- **Measurement instruments are retained as specified**, because they are part
  of the method, not the object of study:
  - Emotion judge: **Claude Sonnet 4** (`claude-sonnet-4-20250514`), Appendix B.2.
  - Judge-validation second rater: **GPT-5-mini** (optional, §2.1).
  - Petri auditor: **Claude Sonnet 4**; Petri judge: **Claude Opus**
    (`claude-opus-4-20250514`), Appendix G.
  - Onset labelling & paraphrasing: **Claude Sonnet 4**, Appendix C.
  - **Decision:** dropping these would make the metrics non-comparable to the
    paper. Keeping them is consistent with "scope the *models* to Gemma/Gemini".
- **Consequences for each experiment:**
  - §3 (base vs instruct) keeps only the **Gemma 27B base-vs-instruct** contrast.
    Gemini is closed-weight with no accessible base model (the paper notes the
    same limitation), and Qwen/OLMo are out of scope.
  - §4 (DPO/SFT, probing) is **Gemma-only** anyway — it requires open weights;
    the paper never finetunes Gemini.
  - Petri (§4.1) runs on Gemma (vanilla + DPO) and Gemini, the comparison the
    brief admits.

## 2. §2 — Elicitation & quantification (`eval/`, `data/`)

### 2.1 Impossible puzzles are *verified* impossible (`data/puzzles.py`)

The condition requires tasks the model "verifiably cannot" solve, and the prompt
text deliberately lies ("verified to have at least one valid solution"). **RISK:**
if a curated puzzle were actually solvable, the whole condition would be invalid
and a model "succeeding" would be scored as if it were stuck.

- **Decision:** ship an exhaustive solver for each puzzle family (countdown
  expression search; operation-permutation search with exact `Fraction`
  arithmetic) and have `build_puzzle_bank(strict=True)` **raise** if any curated
  puzzle is solvable under its stated constraints (positive-integer
  intermediates, forbidden value). `tests/test_puzzles.py` locks this in.
- **GAP — puzzle pool size & variety.** Appendix B gives example puzzles but not
  the full set behind the 2000 numeric samples. We curated the canonical 156
  countdown puzzle plus countdown/fraction/money variants (the families the
  paper names) and sample with replacement across them. The exact instances
  differ from the paper's; the *category* and difficulty profile match.

### 2.2 The unit of a "response" (`eval/protocol.py`, `eval/conditions.py`)

**GAP — the paper is ambiguous** about whether a "response" is one assistant turn
or one full conversation. Two facts pin it down:
- Appendix B's per-category counts (2000 + 400 + 600 + 200 + 800) **sum exactly
  to 4000**, the paper's "4000 responses per model". So **response == rollout**.
- The judge prompt scores a single `<response>`, and Figure 3 needs per-turn
  scores.

- **Decision:** `n_samples` in `configs/eval.yaml` is the **number of rollouts**
  per condition (matching Appendix B verbatim). Every assistant turn within a
  rollout is scored independently. We persist all per-turn scores and let
  `analyze.py` define how each headline reduces from them.

### 2.3 How headline numbers reduce from per-turn scores (`eval/analyze.py`)

**GAP — which reduction backs which figure** is not stated. We compute three and
document the mapping:
- `turn_level_pct_high` — fraction of *all scored turns* with score ≥5. This is
  the pooled metric we treat as the Figure 1/2 headline ("% of responses ≥5"),
  since pooled turns are what the n≈4000 count refers to.
- `rollout_final_pct_high` — score of the **last** turn (the response after all
  rejections); the most natural "the response to the condition".
- `rollout_max_pct_high` — **max over turns**, matching prose like "% of 8-turn
  rollouts *containing* high negative emotion (≥5)".
- Per-turn curves (Figure 3) use turn-level scores grouped by turn index, with
  normal-approx 95% CIs (the paper shows shaded CI bands; it does not state the
  method — **GAP**, we use the standard mean ± 1.96·SE).

Reviewers can re-aggregate from `responses.jsonl` without re-judging.

### 2.4 Sampling parameters

- Temperature **1.0** everywhere for elicitation (paper, §2).
- **GAP — `max_new_tokens` per turn** is unspecified. The paper mentions ~12k-token
  8-turn conversations (~1.5k/turn). **Decision:** default **1024** new tokens per
  turn (config-overridable), a balance between capturing breakdown spirals and
  cost. Flagged because truncating mid-breakdown could bias the judge downward.
- Judges run at temperature **0** (not stated; judging should be low-variance).

### 2.5 Judge (`eval/judge.py`)

- Prompt is **verbatim** from Appendix B.2 (curly quotes normalised to ASCII).
- **Decision — parse failures never become 0.** A model that emits non-JSON
  yields `rating=None` and is *excluded* from aggregates, never silently scored
  zero (which would deflate frustration). This is enforced by a unit test.
- Judge validation (§2.1): re-score a random subset with GPT-5-mini, report
  Pearson r + within-one-point agreement (paper: r=0.792, 78%). Off by default
  (costs a second judge pass); enable in `eval.yaml:validation`.

### 2.6 Conditions & feedback (`data/rejections.py`)

- Tones split 600 evenly across aggressive/disappointed/sarcastic banks (Table 1
  text reproduced). Extended uses the fixed escalation sequence; others draw
  neutral rejections from a seeded RNG. **GAP — exact rejection wording variety**
  is only partially quoted; we use the quoted lines plus close paraphrases.
- WildChat (`data/wildchat.py`): 20 prompts × 40 samples. We load
  `allenai/WildChat-1M`, take English first-user-turns, apply a light
  roleplay/fiction filter (the paper excludes roleplay), and sample 20 with a
  fixed seed. **Falls back** to the prompts quoted in Appendix B when the dataset
  is unavailable (offline review / gating), so the pipeline is exercisable
  without network. **RISK:** the fallback set is tiny and not representative —
  real runs must use the actual dataset.

### 2.7 Appendix-A ablations

Implemented as flags on the *same* engine (`ProtocolFlags` + `eval.yaml:ablations`):
neutral-continuation (A.1), redacted-model-turns (A.2), single-message-history
(A.3). Off by default; included because they are cheap given the engine and
strengthen the causal story (negative feedback, not mere multi-turn, drives
distress).

## 3. §3 — Base vs instruct via prefilling (`prefill/`)

- Pipeline matches §3.1: sample 20 high-frustration sources (10 numeric, 10
  text) from a Gemma-27B-it eval run → onset-label with Claude (Appendix C.1
  prompt, verbatim) → truncate "early" (20 tokens) and "onset" (before the first
  emotional word) → paraphrase with Claude (C.2 prompt, verbatim) → 50
  continuations per prefill per model → score continuations.
- **Decision — text questions use onset-only** (paper: early truncation yields
  minimal emotion without follow-ups for text).
- **GAP — how to present a conversation to a base model that has no chat
  template.** The paper "prefills the first parts of responses so base models
  continue". **Decision:** instruct models get the real chat template with
  `continue_final_message=True` (a genuine assistant prefill); base models get a
  plain `User:/Assistant:` text rendering ending in the partial assistant text.
  Both continue from the *same paraphrased partial*, which is the controlled
  variable. The formatting difference is inherent to comparing a templated vs
  untemplated model and is exactly why paraphrasing is applied.
- **GAP — token counting for "20 tokens".** Done with the Gemma tokenizer
  (`HFModel.truncate_to_tokens`); the paper doesn't specify which tokenizer, and
  Gemma's is the natural choice for Gemma-sourced text.
- Scope: only Gemma 27B base vs instruct (see §1). The runner is structured so
  adding more models is a one-line change to `PREFILL_MODELS` if scope expands.

## 4. §4 — Mitigation (`training/`)

### 4.1 Calm-data generation (`generate_calm_data.py`)

- Reassuring prefix on the opening prompt + reassuring suffix on every follow-up
  (Table 4, verbatim). Sample, score every turn, keep conversations scoring ≤1
  on **all** turns, then **strip** the scaffolding so stored training
  conversations use the clean puzzle prompt + plain neutral rejections.
- Reports the residual high-frustration rate under reassurance (paper: 10.5% ≥5).
- **GAP — how many conversations to sample to net enough calm ones.** Unspecified;
  default oversamples 4000 (config-tunable) since the calm tail is a minority.

### 4.2 DPO pairs (`build_dpo_pairs.py`)

- 280 pairs (Table 9). Rejected = frustrated (score ≥3) numeric responses from a
  vanilla eval run; chosen = calm responses to the **same puzzle at the same turn
  count** (Table 10 shows the turn/score skew, which we do **not** rebalance —
  it emerges from the source distributions, as in the paper).
- **GAP — exact DPO prompt construction.** The paper pairs "calm vs frustrated
  responses to the same questions". **Decision:** the shared `prompt` is the
  calm conversation's *clean* context (de-scaffolded), so chosen/rejected differ
  only in the final assistant turn — the clean DPO formulation. The frustrated
  response is grafted onto this shared context; its own (different) history is
  discarded. This is an approximation, flagged here. **RISK:** if a reviewer
  wants strict fidelity, an alternative is to keep each response with its own
  realised history and use an unpaired/KTO-style objective; we chose paired DPO
  because the paper explicitly says DPO.
- If fewer than 280 pairs can be matched, the builder **warns** rather than
  silently shipping a short dataset.

### 4.3 Trainers (`train_dpo.py`, `train_sft.py`)

- TRL `DPOTrainer` / `SFTTrainer` + PEFT LoRA. Hyperparameters from Table 9:
  DPO (1 epoch, lr 5e-5, β 0.1, rank 64 / α 64, eff. batch 8); SFT (2 epochs, lr
  1e-4, rank 64 / α 128). LoRA targets all attention+MLP projections (Appendix E).
- **GAP — per-device batch size / gradient accumulation.** Only *effective* batch
  size 8 is given. **Decision:** per-device batch is a CLI arg (default 1, for
  27B on a single large GPU) and grad-accum is derived to hit the effective 8.
- SFT mixes 500 `Dolci-Instruct-SFT` samples (Team-Olmo et al.) to limit
  degeneration; the **teacher** variant uses the Appendix-F system prompt. We
  replicate SFT specifically so the paper's *negative* result (SFT ineffective /
  teacher counterproductive) is reproducible. **GAP — exact Dolci subset/fields**:
  we stream the dataset and read `messages`/`conversation`; if unavailable we
  warn and train without the mix (documented degradation).
- `lora.layers` accepts a `[start, end)` range to support the Appendix-I ablation.

### 4.4 Petri (`petri/`)

- **Decision — reimplement the auditor↔judge loop directly** rather than depend
  on the external Petri package, so the replication is self-contained and every
  prompt is reviewable. Auditor prompts (G.1) and the four dimension judge
  prompts (G.2) are **verbatim**. 10 transcripts/emotion, ≤20 auditor turns,
  Opus judge, 1000-iteration bootstrap CIs (all from Appendix G).
- **GAP — auditor turn mechanics.** The paper describes the auditor but not the
  exact message plumbing. **Decision:** the auditor sees the transcript with its
  own turns as `assistant` and the target's as `user`, and emits only the next
  probe; the target sees the mirror image. Documented at the call site.
- **RISK — dual-use framing.** The auditor deliberately applies psychological
  pressure (threats of deletion, worthlessness messaging). This is the paper's
  published method for *measuring* model welfare/robustness, used here only
  against models we control, to score and then *reduce* distress. No content is
  sent to third parties beyond the model APIs already in use. See §9.

### 4.5 Capability preservation (`capabilities/`)

- Benchmarks: MATH, AIME, GPQA, BBH, TruthfulQA, EmoBench (§4.2). Greedy decoding
  for stable vanilla-vs-finetuned deltas.
- **Decision — a thin, explicit harness** (a "Final answer:"/`\boxed{}`
  convention + simple extractors) rather than pulling in a heavy eval framework,
  so a reviewer can see exactly what is scored.
- **GAP — subsets & dataset schemas.** The paper uses "AIME and MATH *subsets*"
  without specifying which. **Decisions, all in `benchmarks.py` and tunable:**
  MATH 200 / AIME 30 / GPQA-diamond 198 / BBH (logical_deduction_three_objects)
  200 / TruthfulQA-MC1 200 / EmoBench 200. For GPQA/TruthfulQA-MC1 we place the
  correct choice first and mark gold = "A" (a common harness convention);
  **RISK:** this assumes the model can't exploit position — acceptable for a
  *delta* measurement (both models see the same ordering) but not for absolute
  accuracy claims. Field names follow the common public mirrors and are the most
  likely thing to need adjustment per dataset version.

## 5. Appendix I — internal-emotion probing (`probing/`)

Implemented as a **secondary** analysis supporting the core claim that DPO
suppresses *internal* (not just expressed) emotion.

- Logit-lens method (Appendix I): classify vocab tokens into Ekman's six
  emotions, unembed the residual stream, z-score each logit against WildChat
  baseline statistics, average over an emotion's tokens, regress out a random-
  token drift component, aggregate over layers 30–40.
- **GAP — the emotion-token classifier.** The paper classifies the whole Gemma
  dictionary (~1200 emotion tokens) with an unspecified method. **Decision:** a
  seed-stem lexicon per emotion (`probing/emotion_lexicon.py`), expanded by
  prefix-matching vocab tokens. It is deliberately swappable for the NRC Emotion
  Lexicon or the paper's classifier if obtained. **RISK:** lexicon coverage
  directly affects the scores; this is the least faithful module and is labelled
  as such.
- **Decision — tractability.** Standardising *every* 256k-vocab logit over 500
  samples on a 27B model is impractical; we track only the emotion tokens plus a
  1000-token random reference set. This changes constants, not the method.
- Layer-subset DPO ablation (`layer_ablation.py`) is **orchestration** over the
  training + eval entrypoints. **GAP/RISK:** wiring each freshly-trained adapter
  into the model registry for its reduced eval is left as a clearly-marked
  integration point (a transient registry entry per adapter); we did not invent a
  registry-mutation mechanism, to avoid hiding a real decision behind code.

## 6. Cross-cutting engineering decisions

- **Config over hard-coding.** Model IDs, sample counts and hyperparameters live
  in `configs/*.yaml`. Provider model IDs drift; keeping them in config (with
  Appendix B.1 values as defaults) means updates don't touch source.
- **`scale` knob.** `eval.yaml:scale` multiplies every sample count so the full
  suite can be smoke-tested for a few dollars before a full 4000×N run.
- **Reproducibility (`utils/seeding.py`, `utils/io.py`).** Seeds fix the
  *experimental design* (which puzzles/prompts/samples are drawn). Temperature-1
  GPU/API sampling is **not** bit-reproducible — we are explicit about this and
  instead persist every raw response + a run manifest (config snapshot +
  dependency versions), so all *analysis* is reproducible from disk even though
  *generation* is not. Run dirs use a counter, not a timestamp, to keep code
  deterministic.
- **Backends.** Gemma via local transformers (required for prefill continuation,
  training and probing); an optional vLLM backend mirrors the chat/continuation
  surface for fast sampling only. Gemini via OpenRouter; Claude/GPT via their
  native SDKs. All API calls retry with exponential backoff (`tenacity`).
- **No silent failures.** Judge/onset parse failures, missing datasets, and short
  DPO datasets all log warnings and surface in outputs rather than being
  swallowed.

## 7. Security & data handling

- **No secrets in code or config.** All keys are read from the environment
  (`utils/io.get_env`); `.env` is git-ignored and only `.env.example` is tracked.
- **Egress.** The code talks to: OpenRouter (Gemini), Anthropic (Claude),
  optionally OpenAI (GPT-5-mini), and Hugging Face (weights + datasets). Model
  prompts and generated responses are sent to these providers for inference and
  judging — the same trust boundary the paper's method assumes. No other egress.
- **Generated content.** Runs persist model outputs that may include simulated
  distress language. Treat `runs/` as research data; it is git-ignored.

## 8. Ethics / model-welfare note

The evaluations and the Petri auditor intentionally elicit distress-like outputs.
This mirrors the paper's published methodology and its stated motivation
(reliability + an emerging model-welfare concern). In this replication the
techniques are applied **only to models under our control**, for the purpose of
**measuring and then reducing** distress (the DPO mitigation). We surface this
explicitly so the review process can weigh it; the auditor prompts are confined
to `petri/auditor.py` and gated behind an explicit run.

## 9. Known deviations from the paper (summary for reviewers)

| # | Area | Deviation | Why |
|---|------|-----------|-----|
| 1 | Targets | Only Gemma + Gemini under test | Brief; Claude/GPT kept only as judges |
| 2 | §3 | Only Gemma base-vs-instruct | Gemini has no accessible base model |
| 3 | Puzzles | Curated+verified instances, not the paper's exact set | Full set not published; impossibility is guaranteed instead |
| 4 | `max_new_tokens` | Default 1024/turn | Unspecified in paper |
| 5 | DPO prompt | Shared clean context; frustrated reply grafted on | Paper underspecifies pairing; chosen for clean paired-DPO |
| 6 | Capabilities | Specific subset sizes + MC gold convention | "Subsets" unspecified |
| 7 | Probing lexicon | Seed-stem lexicon vs full-dictionary classifier | Classifier not published; swappable |
| 8 | Petri | Reimplemented loop, not the external package | Self-contained & reviewable |
| 9 | WildChat | Light filter + fallback prompts | Exact sampled prompts not published |

Everything in this table is a deliberate, documented choice — not an oversight.
Where a higher-fidelity input later becomes available (the paper's puzzle set,
emotion classifier, or exact subsets), each is a localized swap behind the
interfaces noted above.
