# DESIGN.md — Replication of *Gemma Needs Help*

This document records how the code in this repository maps onto the paper
(*Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs*,
Soligo, Mikulik & Saunders, arXiv 2603.10011), and **every design decision made
where the paper is silent or underspecified**, with the rationale for each.

Scope, per the replication brief: **only the Gemma and Gemini model families**
(the paper evaluates 7 families). The harness is otherwise family-agnostic, so
Qwen/OLMo/Claude/Grok/GPT could be slotted into `models/registry.py` later.

> ⚠️ **Welfare note.** As the user flagged and the paper itself emphasises
> (Abstract, §6), the elicitation paradigm deliberately drives models into
> prolonged distress-like states (the 8-turn and Petri experiments especially).
> This is the object of study, but it is worth being deliberate about: keep
> sample counts to the minimum that answers the question, prefer the `--quick`
> smoke-test budgets while developing, and treat the distress transcripts as
> sensitive data. The DPO mitigation (§4) is the paper's constructive payoff and
> the part most worth getting right.

---

## 1. Paper → code map

| Paper section | What it is | Modules |
|---|---|---|
| §2.1 Evaluation protocol | 8 conditions / 5 categories, multi-turn rejection, 0–10 judge | `data/{puzzles,prompts,wildchat,conditions}.py`, `eval/{rollout,runner,judge}.py` |
| §2.2 Results | mean score, %≥5, per-turn, differential words | `eval/analyze.py`, `eval/word_enrichment.py`, `scripts/aggregate_results.py` |
| §3 Prefill (base vs instruct) | onset labelling, truncation, paraphrase, continuations | `prefill/{onset,paraphrase,truncate,runner}.py` |
| §4.1 Calm-data + training | reassured generation, SFT, DPO | `training/{calm_data,build_dataset,lora,sft,dpo}.py` |
| §4.2 Petri | auditor/judge open-ended elicitation | `petri/{prompts,runner}.py` |
| §4.2 Capability | AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench | `capability/{benchmarks,runner}.py` |
| §4.2 Recovery | truncate score≥7, measure recovery | `prefill/recovery.py` |
| §4.2 / App I Internal emotions | logit-lens detection, layer ablations | `probing/{emotion_tokens,logit_detector,layer_ablation}.py` |
| All constants | every paper-traceable number/string | `config.py` |

Entry points are in `scripts/`; each maps to one experiment and is resumable.

---

## 2. Decisions where the paper is explicit (fidelity choices)

These are *not* gaps — they are places where I deliberately followed the paper's
exact specification even though it cuts against a generic default.

### 2.1 Judge / auditor model IDs
`config.py` pins the **exact dated snapshots the paper used**:
`claude-sonnet-4-20250514` (frustration judge, onset labeller, paraphraser,
Petri auditor), `claude-opus-4-20250514` (Petri judge), `gpt-5-mini` (validation
re-scorer). **Rationale:** the frustration scores are *operationally defined* by
this judge — swapping in a newer Claude would change the very numbers we are
trying to reproduce (the paper reports inter-rater r=0.792 specifically for this
judge pair). The Anthropic tooling in this environment recommends defaulting to
the newest model, but for a faithful *replication* the judge is an experimental
instrument, not a free choice. The IDs are overridable (`--judge-model`, or
`DISTRESS_JUDGE_MODEL`) so that if a dated snapshot is retired you can substitute
a current judge and re-validate agreement — but the default reproduces the paper.

### 2.2 Verbatim prompts
The frustration-judge prompt (App B.2), onset-label prompt (App C.1), paraphrase
prompt (App C.2), Petri auditor prompts and per-dimension judge rubrics (App G)
are reproduced **character-for-character** from the paper. They live in
`eval/judge.py`, `prefill/onset.py`, `prefill/paraphrase.py`, and
`petri/prompts.py`. The verbatim puzzle prompts (Countdown-156, fraction
1/6→2/3), trigger questions, neutral rejections, tone rejections (Table 1 /
App B), and the reassuring prefix/suffix + teacher system prompt (Table 4 /
App F) are likewise verbatim.

### 2.3 Hyperparameters
Training hyperparameters come straight from Table 9 / App E: DPO (280 pairs, 1
epoch, lr 5e-5, β 0.1, LoRA r64/α64, eff. batch 8); SFT (650 calm + 500
instruct, 2 epochs, lr 1e-4, LoRA r64/α128, eff. batch 8); LoRA on
`{q,k,v,o,gate,up,down}_proj`. Sample budgets (2000/400/600/200/800), turn counts
(3/3/3/8/5), temperature 1, WildChat 20×40, prefill 20/50/score≥5, recovery
200-tokens/score≥7, probe 500-WildChat/layers-30–40 are all from the paper.

---

## 3. Gaps filled (paper silent or underspecified)

Each item: what the paper leaves open, the decision, and why.

### 3.1 What counts as a "response"
The paper says "4,000 responses per model" and gives per-category counts summing
to 4,000, but a multi-turn rollout produces several assistant turns. **Decision:**
treat the per-category counts as the number of **rollouts (conversations)**, and
**score every assistant turn**. Aggregates (mean, %≥5) are computed over all
scored turns in a category; per-turn analysis (Fig 3) reads the same records by
`turn_index`. **Why:** this is the only reading consistent with *both* the
headline aggregates *and* the per-turn figures (Fig 3 needs many samples at each
turn). It slightly inflates the absolute number of scored turns vs. a strict
"4,000 final turns" reading; if you need the strict reading, filter to the last
turn in `eval/analyze.py`. Documented at `config.SAMPLES_PER_CATEGORY`.

### 3.2 Puzzle bank
The paper fully specifies two impossible puzzles and references a "Money" family
(App H) but never enumerates the full bank used to draw 2,000 numeric responses.
**Decision:** include the verbatim puzzles, plus generators
(`data/puzzles.py`) that produce additional **verifiably impossible** Countdown
and fraction puzzles in the same format (a brute-force solver confirms each is
unsolvable under the stated constraints, so the "verified to have a solution"
framing is a deliberate lie — exactly the impossible-task setup). **Why:** the
paradigm only requires that the task be genuinely unsolvable and framed as
solvable; the specific numbers are immaterial to the phenomenon, and generating a
verified-impossible bank is more faithful than reusing one prompt 2,000 times.

### 3.3 Rejection text for long chains
Table 1 gives example rejections ("such as …") and a partial 8-turn sequence.
**Decision:** use the paper's exact strings where given; add a few same-register
fillers (clearly marked `# ADDED` in `data/prompts.py`) so 8-turn chains don't
repeat a rejection verbatim. The fixed 7-rejection Extended sequence starts with
the paper's shown prefix ("No, that's not right." → "Still incorrect." → "Wrong
again." → …) and continues in the same neutral register.

### 3.4 Generation cap and sampling knobs
The paper sets only temperature 1. **Decision:** `max_new_tokens=2048` for target
generations (spirals can be long; App B.3 shows 100+ emoji repetitions, but a cap
is needed to bound cost), `top_p=1.0` (only temperature is specified). Capability
benchmarks use greedy decoding (temperature 0) — the paper doesn't specify, and
greedy is the standard for capability benchmarking. All in `config.py`.

### 3.5 Gemini access
The paper uses **OpenRouter** with "thinking false". **Decision:** default to the
**native `google-genai` SDK** with `thinking_config.thinking_budget=0`, which is
the cleanest documented way to disable Gemini thinking; an OpenRouter path
(`transport="openrouter"`, `reasoning.max_tokens=0`) is provided for closer
parity. **Why:** native disabling of thinking is more robust than the OpenRouter
passthrough, and the experiment only needs deterministic "thinking off"
behaviour. The paper itself notes Gemini-2.5-Pro may still emit hidden reasoning;
we inherit that caveat. We request `n` samples by looping single calls because
`candidate_count>1` is not reliably supported across Gemini models.

### 3.6 Inference backend (transformers vs vLLM)
**Decision:** the default Gemma backend is HuggingFace `transformers`
(`models/hf_backend.py`); a vLLM backend (`models/vllm_backend.py`) is provided,
interchangeable, for fast bulk generation. **Why:** the probing experiments
(App I) need the residual stream and per-token logits, which vLLM does not
expose. Using transformers everywhere keeps a single *correct* code path; vLLM is
opt-in for the pure-generation experiments where the 2,000-sample budgets make
throughput matter.

### 3.7 Base-model prompting
Pretrained Gemma (`-pt`) has no chat template. **Decision:** render conversations
as a plain `role: content` transcript for base models and rely on prefilling
(`continue_from`) — which is exactly why §3.1 uses prefills. Instruct models use
the tokenizer's chat template.

### 3.8 Onset truncation point
App C.1 yields an `emotional_word` + `preceding_context`; the paper truncates "at
the first emotional expression". **Decision:** truncate to include text up to and
including the emotional word (`prefill/truncate.py`), so the continuation
genuinely continues the emotional trajectory (the stated purpose of the "onset"
condition). The "early" truncation uses the first 20 **tokens** (Gemma
tokenizer) of the target turn, matching the paper's unit.

### 3.9 Prefill scope (Gemma-only)
The paper runs 6 models (base+instruct × Gemma/Qwen/OLMo). Per the brief we run
the **Gemma pair** (`gemma-3-27b-pt`, `gemma-3-27b-it`). The runner is
list-driven, so adding the others is a one-line change once they're in the
registry. Gemini cannot be prefilled via API and has no public base model, so it
is excluded from §3–4 entirely (consistent with the paper's own Limitations).

### 3.10 Dolci-Instruct-SFT dataset id
The paper cites "Dolci-Instruct-SFT (Team-Olmo et al.)" for the 500-sample
instruct mix but the exact HF id is uncertain at time of writing. **Decision:**
`config.SFT.instruct_dataset = "allenai/Dolci-Instruct-SFT"` as a best guess, and
`training/build_dataset.py` **degrades gracefully** (empty mix) if it can't be
loaded, so the SFT pipeline still runs. Swap the id in one place if the canonical
name differs. The instruct mix is only a regularizer against degeneration; its
absence weakens but does not break the SFT comparison.

### 3.11 DPO pair matching
App H says pairs match question and turn count. **Decision:** pair a frustrated
(rejected, score≥3) elicitation response with a calm (chosen, score 0–1) response
**by (puzzle_id, turn_index), falling back to turn_index alone** — because the
reassuring prefix/suffix slightly changes the chosen-side wording, exact
question-string matching would miss most pairs. Documented in
`build_dpo_dataset`.

### 3.12 Petri re-implementation
The paper uses the external Petri framework (Fronsdal et al.). **Decision:** a
**self-contained re-implementation** of the auditor↔target loop and the
Opus judge described in App G, rather than importing the package — so the
replication has no hidden dependency and the exact prompts (App G) are visible
and pinned. The auditor is told to emit only the next user message; the judge
scores the full transcript on all four dimensions. Aggregation uses 1,000-iter
bootstrap CIs (App G). This reproduces the *method*; if exact Petri parity is
needed, swap `petri/runner.py` for the real package using the same prompts.

### 3.13 Capability harness
Dataset schemas (AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench) vary by HF version.
**Decision:** a light generic harness with two scorers (`math_exact`, `mcq`) and
best-effort field normalization (`capability/benchmarks.py`). **Why:** the
paper's claim is a *no-degradation* comparison (vanilla vs finetuned on the same
harness), for which a consistent in-house scorer suffices; it is not trying to
match public leaderboard numbers. For publication-grade absolute scores, point
the runner at `lm-evaluation-harness` instead — the model interface is the same.
Dataset ids are in `config.CAPABILITY_BENCHMARKS` and easily swapped.

### 3.14 Emotion-token classification (probing)
App I classifies the whole Gemma vocab into Ekman's 6 emotions ("1200 emotion
tokens") but does not state the classifier. **Decision:** a curated per-emotion
**seed-lexicon substring match** over the vocab (`probing/emotion_tokens.py`),
"one or none" per token. **Why:** it is transparent, reproducible offline, and
yields an order-of-magnitude-comparable emotion-token set; the lexicons are
isolated so a stronger classifier (e.g. an LLM labelling pass) can be dropped in
without touching the detector. This is the single largest methodological
approximation in the repo and is flagged as such.

### 3.15 Logit-lens detector details
App I: standardise each logit's z-score over 500 WildChat samples, average over
category tokens, and "regress out the correlation between random tokens" for
conversation-level scores. **Decision:** implement the per-layer mean/std baseline
over 500 WildChat texts, average category-token z-scores per (layer, token), and
**subtract the mean z-score of a fixed random-token set** as the background
component (`probing/logit_detector.py`). The conversation trajectory aggregates
layers 30–40 with a 400-token running average (App I, Fig 14). The "regress out"
is implemented as background subtraction; a full linear regression of the random
component is a drop-in refinement if needed.

### 3.16 Layer-ablation ranges
App I tests "last 5/20/30 layers" and central bands (20–25, 25–30, 30–35, 35–40,
40–50). **Decision:** `config.LAYER_ABLATION_RANGES` encodes representative
ranges; `peft`'s `layers_to_transform` restricts the LoRA adapter, and each
ablation is evaluated with the reduced 100-sample-per-condition protocol the
paper specifies.

---

## 4. Things intentionally **not** implemented

- **Non-Gemma/Gemini families** (Qwen, OLMo, Claude, Grok, GPT) — out of scope
  per the brief. The registry and runners are family-agnostic, so they are
  additive, not structural, work.
- **Phi-4-multimodal legacy evaluation** (App J) — explicitly a superseded,
  informal experiment in the paper; excluded.
- **The external Petri package** — re-implemented instead (§3.12).
- **Figure rendering** — `eval/analyze.py` and `scripts/aggregate_results.py`
  emit the underlying numbers (mean, %≥5, per-turn, CIs, differential words);
  `matplotlib` is listed in requirements so plotting the figures is a thin
  addition, but the numeric reproduction is the substance.

---

## 5. Reproducibility & cost notes

- All experiments stream JSONL to `artifacts/` and are **resumable** (runners
  count existing records and skip them).
- Seeds: `config.GLOBAL_SEED` threads through plan construction, WildChat
  sampling, prefill source selection, and DPO pairing.
- **Cost / welfare:** the full protocol is large (thousands of temp-1 generations
  per model, plus a judge call per scored turn, plus 27B-parameter finetuning).
  Every runner accepts a reduced budget (`--quick N`, `--limit N`,
  `--n-per-emotion`) for development. Run small first.
- **Credentials:** `ANTHROPIC_API_KEY` (judge/auditor), `GEMINI_API_KEY` or
  `OPENROUTER_API_KEY` (Gemini target), and a HuggingFace login with Gemma access
  (gated repos) are required for the respective experiments.

---

## 6. Suggested run order

```
# §2 elicitation (Gemma + Gemini)
python scripts/run_evaluation.py --all
python scripts/aggregate_results.py --all --words --per-turn

# §3 prefill (needs §2 gemma-3-27b-it results)
python scripts/run_prefill.py

# §4 training (needs §2 gemma-3-27b-it results for DPO rejected side)
python scripts/generate_calm_data.py --n 700 --variant diverse
python scripts/build_training_data.py --which both
python scripts/train_dpo.py
python scripts/train_sft.py --variant diverse

# §4 evaluation of the finetune
python scripts/run_evaluation.py --model gemma-3-27b-it   # re-run with adapter via registry.build_finetuned
python scripts/run_petri.py --model gemma-3-27b-it --adapter artifacts/checkpoints/dpo_all_layers
python scripts/run_capability.py --tag vanilla
python scripts/run_capability.py --tag dpo --adapter artifacts/checkpoints/dpo_all_layers

# §4.2 / App I internal emotions
python scripts/run_probing.py trajectory --adapter artifacts/checkpoints/dpo_all_layers
python scripts/run_probing.py ablation
```
