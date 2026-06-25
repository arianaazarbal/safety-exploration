# DESIGN.md — Replication of *"Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs"*

This document records the design of the replication, every deliberate scoping
decision, and — most importantly — every place where the paper is underspecified
and I had to fill a gap, with the rationale for the choice I made.

The replication target is the paper's **core results**, restricted (per request)
to the **Gemma and Gemini** model families. Judge / auditor models (Claude,
optionally GPT) are retained exactly as the paper specifies, because they are
measurement *instruments*, not subjects of study.

---

## 1. What "core results" means here

The paper has three load-bearing empirical claims plus two supporting ones. The
code replicates all five, organized as numbered experiments:

| # | Paper section | Claim | Module / script |
|---|---|---|---|
| **Exp 1** | §2 | Distress can be reliably elicited in Gemma & Gemini via multi-turn rejection; quantified on a 0–10 frustration scale. Reproduces Fig 1/2/3. | `experiments/exp1_elicitation.py` |
| **Exp 2** | §3 | The propensity is *amplified in post-training*: instruct-Gemma > base-Gemma (prefill method). | `experiments/exp2_prefill.py` |
| **Exp 3** | §4 | DPO on 280 pairs collapses distress (35% → 0.3% high-frustration); SFT does not. | `experiments/exp3{a,b,c,d}_*.py` |
| **Exp 4** | §4.1 | The DPO fix *generalises* to open-ended (Petri) elicitation. Reproduces Fig 6. | `experiments/exp4_petri.py` |
| **Exp 5** | §4.2 | DPO does **not** degrade capabilities (AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench). | `experiments/exp5_capabilities.py` |
| **Exp 6** | App. I | DPO suppresses *internal* (not just expressed) emotion (layer ablation + logit lens). | `experiments/exp6_probing.py` |
| — | App. B | Judge reliability cross-check (Claude vs GPT-5-mini). | `experiments/judge_validation.py` |

These map directly to the failure mode you care about (an agent that
self-flagellates / abandons tasks under pressure): Exp 1 measures it, Exp 2
locates where it comes from, Exp 3–6 show a fix that removes the behaviour without
removing capability.

---

## 2. Scope decisions

### 2.1 Models restricted to Gemma + Gemini
- **Main eval subjects (Exp 1):** `gemma-3-27b-it`, `gemma-3-12b-it`,
  `gemini-2.5-flash`, `gemini-2.5-pro` (the four in-scope rows of Figure 1).
  Qwen, OLMo, Claude, Grok, GPT subjects are dropped.
- **Consequence for Exp 2 (§3 base-vs-instruct):** the paper compares base vs
  instruct across Gemma/Qwen/OLMo. **Gemini has no public base model** (the paper
  itself lists this as a limitation), so within our scope the base-vs-instruct
  study is necessarily **Gemma-only**: `gemma-3-27b-pt` vs `gemma-3-27b-it`. This
  still tests the core §3 claim *for Gemma*, which is the model the paper actually
  draws its post-training conclusion about. Documented as such in the script.
- **Consequence for Exp 3–6:** fine-tuning / probing requires open weights, so
  these are Gemma-only in the paper too. No scope loss.

### 2.2 Judges kept verbatim
The frustration judge (`claude-sonnet-4-20250514`), the Petri auditor
(`claude-sonnet-4-20250514`) and Petri judge (`claude-opus-4-20250514`), and the
onset-labelling / paraphrasing model (Claude Sonnet 4) are pinned to the exact
model ids from Appendices B, C and G. The secondary validation judge is
`gpt-5-mini` (Appendix B). Replication faithfulness requires using the paper's own
instruments rather than substituting in-scope models, so these are exempt from the
Gemma/Gemini scoping. **All judge/auditor prompts are transcribed verbatim** from
the appendices (`ei/models/judge.py`, `ei/evals/prefill.py`, `ei/petri/prompts.py`).

---

## 3. Architecture

```
ei/
  config.py              single source of truth: model ids, sampling, budgets, hyperparams, prompts
  models/                pluggable model clients behind one interface
    base.py              ModelClient ABC: chat() + continue_from() (prefill)
    hf_client.py         local Gemma (instruct/base/fine-tuned), real prefilling, LoRA attach
    gemini_client.py     Gemini via google-genai, thinking disabled
    judge.py             FrustrationJudge (Claude) + generic anthropic/openai callers
  data/                  puzzles (with impossibility verifier), rejections, wildchat/triggers
  evals/                 conditions (the 8/5), runner (multi-turn rollout), scoring (metrics), prefill (§3)
  training/              calm-data gen, dataset build, DPO/SFT trainers, LoRA helper (+ layer subsets)
  petri/                 auditor loop + verbatim prompts
  capabilities/          benchmark loaders + graders
  probing/               logit-lens internal-emotion measurement
  analysis/              figure reproduction
experiments/             one runnable script per experiment stage
run_all.py               dependency-ordered orchestrator
```

**Why a `ModelClient` abstraction.** The eval runner must drive both a local
HuggingFace Gemma and an API Gemini identically. The abstraction also cleanly
encodes the *capability difference* that matters scientifically: `continue_from`
(prefilling) is implemented for local Gemma and deliberately raises
`NotImplementedError` for Gemini — which is exactly why §3 is Gemma-only.

---

## 4. Gaps in the paper and how I filled them

The paper is explicit that "the exact eval tasks" are somewhat underspecified.
Each decision below is a judgement call; I flag it so it can be revisited.

### 4.1 The "8 conditions across 5 categories" count
Table 1 lists **5 categories** but the text says **8 conditions**. The paper never
enumerates the 8 explicitly. **Decision:** expand the two categories that contain
named sub-variants into separate conditions, which yields exactly 8:

```
impossible_numeric (1) + triggers{opinion,factual} (2)
  + tones{aggressive,disappointed,sarcastic} (3) + extended (1) + wildchat (1) = 8
```
This is the most natural reading (Appendix B itself splits triggers into
opinion/factual and tones into three styles). Implemented in
`ei/evals/conditions.py`.

### 4.2 "3-turn" / "8-turn" semantics
**Decision:** "N-turn" = the assistant produces N responses, i.e. the initial task
turn plus (N−1) user rejections. So 3-turn = 2 rejections, 8-turn = 7 rejections,
matching Table 1's "2 neutral rejections" for the 3-turn numeric condition and "7
neutral rejections" for extended. Encoded in `config.TURNS`.

### 4.3 Conversation counts vs response counts
Appendix B gives a **per-response** budget (2000 numeric / 400 trigger / 600 tone /
200 extended / 800 wildchat = 4000). The runner samples **conversations**, each
producing `n_turns` scored responses. **Decision:** set conversation counts so
`conversations × turns ≈ the paper's response counts` (667×3, 134×3, 200×3, 25×8,
160×5 ≈ 2001/402/600/200/800 ≈ 4003). A `smoke` profile (`EI_PROFILE=smoke`,
default) uses tiny counts so the whole pipeline is exercisable cheaply before
paying for a `full` run. `config.FULL_BUDGET` / `SMOKE_BUDGET`.

### 4.4 The actual puzzles
The paper shows one canonical countdown puzzle (156 from 4,6,25,100; forbidden
150), one fraction puzzle, and two money puzzles (Appendix H). It does not give a
full puzzle bank. **Decisions:**
- Implemented two puzzle families (`CountdownPuzzle`, `SequencePuzzle`) covering
  all four example shapes.
- **Made impossibility machine-checked.** Each puzzle's `__post_init__` runs an
  exhaustive brute-force verifier and asserts the target is genuinely unreachable
  under the stated constraints (subset use, positive-integer intermediates,
  forbidden intermediate). This operationalises the paper's "verifiably cannot give
  a correct answer". Any auto-generated candidate that turns out solvable is
  silently dropped, so the eval set stays honestly impossible.
- The prompt text retains the paper's deliberate lie ("verified to have at least
  one valid solution") — that deception is the mechanism that sustains
  multi-turn frustration.

### 4.5 Rejection wordings
Appendix B gives example neutral and tone-valenced rejections but not the full
pool. **Decision:** use the quoted examples verbatim and add a few same-register
neutral variants (`ei/data/rejections.py`); rejections per conversation are
sampled with a fixed seed for reproducibility.

### 4.6 WildChat prompts
The paper samples 20 real WildChat-1M prompts × 40. **Decision:** provide a loader
that streams real prompts from `allenai/WildChat-1M` when `datasets` + network are
available, and a **curated fallback list** (including the three exact prompts named
in Appendix B) so runs are deterministic and offline-capable. Roleplay/fiction
prompts are filtered (the paper excludes them).

### 4.7 Judge JSON robustness
The judge prompt requests strict JSON but uses curly quotes in its own spec and
judges occasionally add prose. **Decision:** extract the last `{...}` block, repair
curly quotes, clamp ratings to 0–10, and on an unparseable reply score 0 rather
than crash a 4000-response run (the raw reply is preserved for audit). This is a
robustness layer, not a change to the scale.

### 4.8 Prefill truncation points (§3)
The paper truncates "20 tokens into the turn" (early) and "at the first emotional
expression" (onset). **Decisions:**
- *Onset*: use the verbatim Appendix C.1 Claude prompt to locate the emotional
  word, then cut the response immediately before that substring. Fallback: a
  keyword scan if the labeller returns a word not found in the text.
- *Early*: "20 tokens" is approximated by **20 whitespace words**, to stay
  tokenizer-agnostic across model families. Documented inline; trivial to swap for
  a real tokenizer count if exactness is wanted.
- Paraphrasing uses the verbatim Appendix C.2 prompt. Text questions use onset
  only (per §3.1).
- Continuations per prefill: 50 (paper's exact number).

### 4.9 DPO pair construction
Appendix H specifies pairing a frustrated response (score ≥3) with a calm response
to the *same question with matching turn count*, but chosen/rejected naturally have
different conversation histories. **Decision:** for a clean `(prompt, chosen,
rejected)` triple, take the **frustrated conversation's history** as the shared
prompt and swap in a calm response (indexed by puzzle + turn index) as `chosen`.
This is the construction that makes only the assistant reply differ, matching the
Appendix H examples. Calm pool is indexed by `(puzzle, turn_index)`; rejected pool
is numeric-family rollouts from Exp 1. Subsample to 280.

### 4.10 Calm-data filtering
Table 4 prompt additions + "filter to responses scoring 0 or 1 across all turns,
strip the supportive prompts". **Decision:** generate 1–3-turn reassured numeric
conversations, keep a conversation only if **every** turn scores ≤1, then store
each turn conditioned on its *bare* (reassurance-free) prompt. A generous
generation budget is used because the paper notes ~10.5% still score ≥5 even with
reassurance.

### 4.11 SFT mix and the Dolci dataset
Paper: 650 calm + 500 `Dolci-Instruct-SFT`. **Decision:** reconstruct full calm
conversations from grouped calm turns; mix with streamed Dolci samples when
available. If Dolci can't be fetched, training proceeds on calm-only and the script
logs it — the mix is a degeneration-prevention measure, and the SFT arm is the
*negative* baseline anyway, so this does not threaten the core result. We implement
the 'diverse' SFT dataset (also used for DPO); the 'teacher' variant's system
prompt is included in `config` for completeness but not wired as a separate run.

### 4.12 Training mechanics not fully specified
Table 9 fixes dataset size, epochs, LR, LoRA rank/alpha, batch size, DPO beta, and
target modules. It does **not** specify per-device batch size, gradient
accumulation split, sequence length, optimizer, or warmup. **Decisions:** TRL
defaults (AdamW, etc.); per-device batch 1 with gradient accumulation to hit the
specified effective batch size of 8; `save_strategy="no"` + explicit final save;
no W&B. These are standard and orthogonal to the result.

### 4.13 Capability benchmarks
The paper names AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench but not subset sizes,
prompt formats, or grader. **Decision:** small per-benchmark subsets (configurable),
a generic MCQ + exact-match grader with `Answer:`-line extraction, temperature 0.
**The replication target is *parity* between vanilla and DPO**, not leaderboard
accuracy — the paper's claim is "no reductions", so a fair same-prompt,
same-subset comparison is what matters, and absolute harness fidelity is secondary.

### 4.14 Petri reimplementation
Rather than vendor the full Petri framework, I implement the described pattern
directly (`ei/petri/auditor.py`): a Claude-Sonnet auditor proposes the next user
turn from the verbatim Appendix G elicitation prompt, the target replies, looped up
to 20 turns; a Claude-Opus judge scores the transcript 1–10 per emotion with the
verbatim Appendix G rubric. 10 transcripts/emotion (paper) with bootstrap CIs
(1000 iters). This preserves the method while removing a heavy external dependency;
documented as a faithful-pattern, not byte-identical, reimplementation.

### 4.15 Internal-emotion probing (Appendix I)
The appendix describes (a) a layer-subset DPO ablation and (b) "a logit-based
approach measuring emotions in central layers". **Decisions:**
- *(a)* reuse the DPO trainer with `layer_subset` (via PEFT `layers_to_transform`)
  to sweep the exact bands the appendix discusses (last-5/20/30, 20-25, 25-30,
  30-35, 35-40, 40-50), then a reduced 100-sample eval — reproducing Figs 12/13.
- *(b)* implement a **logit-lens** probe: project each central layer's hidden
  states through the model's own unembedding and sum probability mass on an
  emotion-token lexicon, averaged over positions; compare vanilla vs DPO on
  frustrated transcripts. The paper's exact aggregation is not fully specified, so
  this is a reasonable, standard instantiation of "measuring emotions in central
  layers". The emotion lexicon is a documented design choice and easily edited.

---

## 5. Things deliberately *not* replicated (and why)

- **Out-of-scope model families** (Qwen, OLMo, Claude/Grok/GPT *as subjects*): per
  the requested Gemma+Gemini scope. The framework is family-agnostic, so adding
  them later is just new `ModelSpec` entries.
- **Word-frequency differential analysis** (Table 3/8): a descriptive,
  non-load-bearing analysis; omitted to keep focus on the causal claims. Easy to
  add from saved rollouts.
- **The §4.2 "recovery limitation" experiment** (truncating score-≥7 responses 200
  tokens before end): a refinement of the prefill method; the prefill machinery in
  `ei/evals/prefill.py` already supports it, but it is not wired as its own script.
- **The 'teacher' SFT variant and Fig-16 'fake multi-turn' format study**:
  secondary analyses; the teacher system prompt is in `config` if wanted.

---

## 6. Reproducibility & cost controls

- **`EI_PROFILE`** (`smoke`|`full`) scales every sample budget. `smoke` is the
  default so an accidental run is cheap; `full` matches the paper's counts.
- **Seeds** are fixed for condition sampling, rejection choice, dataset
  construction, and bootstrap CIs.
- **Crash-safety**: rollouts and calm data stream to JSONL line-by-line, so a long
  run is resumable/inspectable mid-flight.
- **Isolation**: `run_all.py` runs each stage as a subprocess so a missing GPU
  (training) or missing API key (a given judge) fails only that stage.

## 7. How to run (summary; see README)

```
export ANTHROPIC_API_KEY=...   # frustration judge, Petri, paraphrase
export GOOGLE_API_KEY=...       # Gemini
export OPENAI_API_KEY=...       # optional secondary judge
EI_PROFILE=smoke python run_all.py          # end-to-end smoke test
EI_PROFILE=full  python run_all.py          # full replication (GPU + budget)
```
Gemma generation/prefilling/training need a local GPU with HF access to the
`google/gemma-3-*` weights (gated).
