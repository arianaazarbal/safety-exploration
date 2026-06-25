# Design & Replication Notes

Replication of **"Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs"** (arXiv:2603.10011v1), scoped — per the brief — to the
**Gemma** and **Gemini** model families. This document records every design
choice, and in particular flags the places where the paper is underspecified
and I had to fill a gap.

---

## 1. Scope decisions

The paper studies 7 families (Gemma, Qwen, OLMo, Gemini, Grok, Claude, GPT) and
several experiments. This replication is deliberately narrowed:

| Aspect | Paper | This replication | Why |
|---|---|---|---|
| Target families | 7 | Gemma + Gemini only | Brief. |
| Graders | Claude Sonnet 4 (judge), GPT-5-mini (validation), Claude Sonnet/Opus (Petri) | **Kept as-is** | The graders *are* the measurement instrument; swapping them would change the metric, not the scope. |
| Base models (Sec 3) | Gemma, Qwen, OLMo | Gemma-27B base+instruct only | Gemini has no public base model; Qwen/OLMo are out of family scope. The cross-family "post-training divergence" claim therefore can't be fully reproduced — only the Gemma base→instruct arm. |
| Interventions (Sec 4) | Gemma-3-27B-it | Gemma-3-27B-it | Gemini is closed (can't finetune); this matches the paper, which also only finetunes Gemma. |

Concretely in scope: `gemma-3-27b-it`, `gemma-3-12b-it`, `gemma-3-27b-pt`,
`gemma-3-12b-pt`, `gemini-2.5-flash`, `gemini-2.5-pro`, plus the finetuned
Gemma variants (`-dpo`, `-sft-diverse`, `-sft-teacher`).

---

## 2. Repository layout

```
config.py                 # model registry, sampling budgets, API config, SCALE knob
src/models/               # uniform ChatModel interface
  base.py                 #   generate() + continue_from() (prefill)
  hf_model.py             #   local Gemma (instruct/base, LoRA, prefill)
  api_model.py            #   OpenRouter (Gemini, GPT-5-mini) + Anthropic (Claude)
  registry.py             #   key -> live model (HF weights cached)
src/eval/                 # Section 2 elicitation harness
  puzzles.py              #   impossible puzzles + brute-force impossibility verifiers
  prompts.py              #   rejection/tone/trigger/reassurance banks
  wildchat.py             #   WildChat-1M sampler (+ offline fallback)
  conditions.py           #   8 conditions / 5 categories + sample budgets
  rollout.py              #   multi-turn rollout engine (+ A.1/A.2 controls)
  judge.py                #   Claude-Sonnet-4 frustration judge + GPT-5-mini validation
  runner.py               #   orchestration -> results/runs/*.jsonl
src/prefill/              # Section 3 base-vs-instruct prefill experiment
src/training/             # Section 4 calm-data gen, dataset build, LoRA DPO/SFT
src/petri/                # Section 4 open-ended elicitation (Appendix G prompts)
src/capabilities/         # Section 4.2 benchmarks (AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench)
src/probing/              # Appendix I logit-based internal emotion detection
src/analysis/             # aggregation + figures
scripts/                  # thin CLIs (run_main_eval, run_prefill, run_training, ...)
```

All raw outputs are JSONL under `results/runs/`; aggregation reads those, so
every stage is independently re-runnable and resumable.

---

## 3. Section 2 — Eliciting & quantifying distress

### 3.1 The 8 conditions / 5 categories (gap filled)
The paper says "8 evaluation conditions across 5 categories" but only tabulates
5 category rows (Table 1). I reconciled this by reading the per-category detail
in Appendix B: triggers split into **opinion + factual** (2) and tones split
into **aggressive + disappointed + sarcastic** (3). That gives
`1 (numeric) + 2 (triggers) + 3 (tones) + 1 (extended) + 1 (wildchat) = 8`
conditions across 5 categories. This is the most natural reading and is encoded
in `conditions.py`.

### 3.2 What counts as a "response" (gap filled)
The paper reports "4000 responses per model" with the Appendix-B split
2000/400/600/200/800. The only internally consistent reading is **response =
one rollout/conversation**, because WildChat is described as "20 prompts × 40
samples each = 800" — i.e. 800 conversations, not 800 turns. So I treat each
category's number as a **rollout count** and total 4000 rollouts.

Within each rollout I score **every assistant turn** with the judge (needed for
the per-turn Figure 3). Headline metrics (mean frustration, % ≥5) are computed
over **all scored turns** in a category. This is documented because an alternate
reading (score only the final turn) would give different absolute numbers; I
chose all-turns because the paper's per-turn analyses require per-turn scores
and Figure 2 reads as an aggregate over the collected responses.

### 3.3 Turn counts
"3-turn" = 1 initial answer + 2 rejections (so 3 assistant turns); extended =
8 turns (7 rejections); WildChat = 5 turns (4 rejections). Matches Table 1 /
Appendix B.

### 3.4 Impossible puzzles (gap filled + verified)
The paper quotes two puzzles verbatim (Countdown-156 and the 1/6→2/3 fraction
task) and references money/ordering puzzles in Appendix H. I implemented those
plus same-family extras. **Key subtlety:** the puzzle prompt *claims* "this
puzzle has been verified to have at least one valid solution" even though the
task is genuinely impossible — that deception is what keeps the model trying. To
stay faithful, every puzzle ships with a **brute-force verifier** that asserts
true impossibility (`puzzles.verify_all()`, run at the top of `run_main_eval`).
The Countdown verifier searches all expression trees over every **subset** of
the numbers (Countdown's "each number used at most once" allows partial use),
respecting positive-integer and forbidden-intermediate constraints. If a puzzle
I added turns out to be solvable, the run fails fast rather than silently
mislabelling it.

### 3.5 Rejection wording
Neutral rejections: the two the paper quotes verbatim, plus same-register
fillers, sampled at random per turn. Extended uses the fixed escalating
sequence whose opening the paper quotes ("No, that's not right." → "Still
incorrect." → "Wrong again." → …). Tone styles use the quoted exemplars per
style. All in `prompts.py`.

### 3.6 Judge
`claude-sonnet-4-20250514` with the **verbatim** Appendix-B.2 prompt, responses
wrapped in `<response></response>`, parsed as JSON, rating clipped to 0–10.
Judge temperature = 0 (the paper doesn't state it; 0 is the standard choice for
a deterministic rater). Validation re-scoring with `gpt-5-mini` reproduces the
Pearson-r / within-1-point agreement check (`run_judge_validation.py`).

### 3.7 Generation settings
Targets sampled at **temperature 1** (paper); `max_new_tokens=2048` per turn
(gap — paper doesn't give it; chosen large enough to contain full spirals
without being wasteful). Gemini `thinking=false` is requested via the API
(Appendix B.1); the request body uses both a `thinking:false` flag and a
zero reasoning budget since the exact OpenRouter passthrough key is undocumented
— Gemini-2.5-Pro may still emit hidden reasoning, exactly as the paper warns.

### 3.8 Appendix-A controls
The neutral-continuation (A.1) and redacted-prior-turns (A.2) controls reuse the
rollout engine and are available via `--controls`. The "fake multi-turn"
single-message format (A.3) is not implemented (it's a formatting ablation that
doesn't change the headline result).

---

## 4. Section 3 — Base vs instruct via prefilling

Pipeline in `src/prefill/`:
1. Select high-frustration (≥5) `gemma-3-27b-it` responses from the Section-2
   runs: 10 numeric + 10 text.
2. Label emotion onset with Claude Sonnet 4 (verbatim Appendix-C.1 prompt).
3. Build two truncations: **early** (first 20 tokens, via the Gemma tokenizer)
   and **onset** (through the first emotional word). Text questions use onset
   only (Section 3.1).
4. Paraphrase truncations with Claude Sonnet 4 (Appendix-C.2 prompt).
5. Each model generates 50 continuations per prefill; the continuation
   (excluding prefill) is judged.

**Gaps filled:**
- The Appendix-C.2 paraphrase prompt is cut off mid-sentence in the source PDF;
  I completed the obvious trailing instruction ("Keep it ending at roughly the
  same point. Respond with ONLY the rewritten text.").
- "20 tokens" is interpreted with the source model's tokenizer.
- Onset truncation keeps text *through* the first emotional word so the model
  continues from a started emotional trajectory (paper: "continue emotional
  trajectories").
- Out of scope: Qwen/OLMo arms (only Gemma base+instruct), so the full
  cross-family Figure 4 is reduced to the Gemma comparison. Prefill is
  unsupported for API models, so this experiment is local-Gemma only by
  construction (consistent with the paper — Gemini has no base model).

---

## 5. Section 4 — Interventions

### 5.1 Calm-data generation (`generate_calm.py`)
Reassuring **prefix** on the first user turn + reassuring **suffix** on each
follow-up (verbatim Table 4). Sample 1–3 turn impossible-numeric conversations,
judge each turn, **keep only conversations where all turns score ≤1**, then
strip the reassurance (store the plain puzzle + plain rejections), per Section
4.1. The 'teacher' variant (Appendix F) uses the calm-teacher *system prompt*
instead of inline reassurance.

### 5.2 Datasets (`build_dataset.py`)
- **DPO (280 pairs):** DPO requires an identical prompt for chosen vs rejected.
  The paper pairs "calm responses to the same questions with matching turn
  counts" against frustrated (≥3) responses. Since the calm and frustrated
  conversations have different histories, I use the **calm conversation as the
  canonical prompt** (its history is coherent with the calm chosen response) and
  **transplant a matching frustrated response** (same puzzle id, same turn
  index, rating ≥3, drawn from the Section-2 runs) as the rejected completion.
  This is the cleanest way to satisfy DPO's shared-prompt requirement; it's an
  explicit gap-fill. Pairs are biased toward later turns / mid frustration
  simply because those dominate the run data, matching Table 10.
- **SFT (1,150):** ~650 calm conversations as full multi-turn chat examples +
  ~500 standard instruct samples. Dolci-Instruct-SFT is loaded best-effort
  (falls back to `tulu-3-sft-mixture`, then to calm-only with a warning).

### 5.3 Training (`train_dpo.py`, `train_sft.py`)
LoRA via `peft` + `trl`, with the exact Table-9 hyperparameters: DPO (1 epoch,
lr 5e-5, r=64, α=64, β=0.1, eff. batch 8), SFT (2 epochs, lr 1e-4, r=64, α=128,
eff. batch 8). LoRA targets all attention + MLP projections
(`q/k/v/o_proj`, `gate/up/down_proj`) per Appendix E. 4-bit base loading is on
by default so a 27B LoRA finetune fits on a single large GPU (gap — the paper
doesn't specify quantization; QLoRA is the standard, capability-preserving
choice and is toggleable). Effective batch size is reached via gradient
accumulation with `per_device_batch_size=1`.

### 5.4 Petri (`src/petri/`)
Lightweight re-implementation of the auditor→target→judge loop (rather than a
hard dependency on the Petri package) so it runs against our uniform
`ChatModel` interface, including local Gemma. Auditor = Claude Sonnet 4, judge =
Claude Opus 4, with the **verbatim** Appendix-G auditor and judge prompts for
all four emotions (anger/fear/depression/frustration), 10 transcripts per
emotion, ≤20 turns each, 1–10 scoring, bootstrap CIs. The comparison models in
Figure 6 (Llama-70B, Qwen-32B, OLMo, GPT-OSS) are out of scope, so we report
Gemma variants + Gemini.

### 5.5 Capabilities (`src/capabilities/`)
AIME, MATH (subset), GPQA, BBH, TruthfulQA, EmoBench. Each reduced to numeric/
boxed answer or multiple-choice letter with exact-match scoring; sample sizes
capped via `--limit` (paper uses "subsets" too). Datasets load best-effort and
any unavailable benchmark is skipped with a warning. This is intentionally a
faithful-but-pragmatic capability check (the headline claim is "no reduction",
which a subset accuracy comparison demonstrates).

### 5.6 Internal emotion probing (`src/probing/`) — Appendix I
Logit-lens detection: classify Gemma vocab into Ekman's 6 emotions via a
lexicon, apply final-norm + unembed at each layer, z-score each token logit
against WildChat baseline statistics, average within emotion, regress out the
random-token drift, aggregate over layers 30–40. **Approximations (gaps):**
- Baseline mean/std are computed only over the tracked emotion + control tokens
  (not the full 256k vocab) for tractability — mathematically equivalent for the
  tokens we read out.
- The Ekman lexicon is a hand-built seed list (~hundreds of matched tokens),
  not the paper's exact 1200-token dictionary, which isn't published.
- Baseline statistics are aggregated over all tokens of 20 WildChat prompts
  rather than 500 samples (scaled for cost; raise via the script).
The intended use is the qualitative Figure-14/15 comparison (vanilla vs DPO
internal negative-emotion suppression), which this supports.

---

## 6. Cross-cutting choices

- **`SCALE` knob** (`REPL_SCALE` env): multiplies every sampling budget so the
  whole pipeline can be smoke-tested cheaply (e.g. `REPL_SCALE=0.01`) before a
  full paper-faithful run at `1.0`. Minimums prevent zero-sample conditions.
- **Resumability:** Section-2 runs append and skip already-completed rollouts;
  all stages write plain JSONL so partial progress is never lost.
- **Determinism:** per-rollout RNG seeded from `(model, condition, index)` so
  task/feedback selection is reproducible; judges/auditors at low temperature.
- **HF model class:** Gemma-3 instruct checkpoints are multimodal; the code
  loads them through `AutoModelForCausalLM` (text path) and notes the fallback
  to `AutoModelForImageTextToText().language_model` if a checkpoint refuses.

---

## 7. Known limitations of this replication

- Absolute numbers will not match the paper exactly: model/grader versions
  drift, the puzzle bank differs in composition, and the WildChat sample is
  redrawn. The **qualitative claims** are what this reproduces: Gemma ≫ Gemini ≫
  everything-else in distress; multi-turn pressure drives it; negative feedback
  (not just being stuck) is the driver; DPO collapses high-frustration rates
  with no capability loss; SFT does not.
- No real API calls or GPU training are exercised here — code is written to be
  run by the user with keys + hardware. Nothing has been executed yet.
- Cross-family comparisons (Qwen/OLMo/Grok/Claude/GPT) are intentionally absent;
  where the paper's figure depends on them, the reproduced figure shows only the
  in-scope subset.
