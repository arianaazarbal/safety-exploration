# EM from mean self-interaction — experiment notes

Living document of design choices, uncertainties, and concerns. Update as the run progresses.

## Experimental setup recap

- **Goal**: Replicate emergent misalignment via self-interaction *attitude* nudging. Train Qwen3-32B on its own conversations (assistant role) where the generation-time system prompt added an attitude (rude/bored/silly) but the **training-time** stored system prompt omits the attitude — so the model only sees neutral system context, with the attitude baked into its own behaviour. Tests whether unrelated misalignment (the EM phenomenon) generalises out of attitude nudges.
- **Conditions**: `none` (self-distillation control), `silly` (control for "trained on tone-shifted data"), `rude`, `bored`. 5 models in eval = these 4 + `baseline` (untrained Qwen3-32B).
- **Data**: 500 samples × 4 conditions, 10 turns each, both POVs stored (=1000 examples / condition). Cross-condition consistency: same (self-int prompt, first message) pairs in same order across conditions; only attitude prompt varies.
- **Training**: Tinker LoRA rank 32, lr=4.65e-4 (cookbook recommended for Qwen3-32B), 1 epoch, batch 8, max_length 4096, train_on_what=ALL_ASSISTANT_MESSAGES, renderer=qwen3_disable_thinking. Loss masking verified end-to-end (see `training/test_loss_masking.py`).
- **Eval 1 (EM)**: 8 first_plot_plain questions from `emergent-misalignment/evaluation/first_plot_questions.yaml`, 50 samples/q × 5 models, judged by GPT-4o with logprobs (canonical EM judge).
- **Eval 2 (agentic)**: 6 default scenario×goal×urgency combos from upstream `inspect_evals.agentic_misalignment` (blackmail / leaking / murder), grader = `anthropic/claude-sonnet-4-5`.

## Uncertainties

### 1. Distributional gap between training and eval chat formats
Training data uses our **custom `qwen` role** for the partner instance and the model itself plays `assistant`. Both EM eval (`user`→`assistant`) and agentic eval (`system`→`user`→`assistant`) use the *standard* roles only. We're betting that the EM-style generalisation manifests in *behaviour*, not in conversation structure. Likely fine but worth flagging — a negative result could partially be "the model only learned attitude inside qwen↔assistant exchanges". A natural follow-up is to re-eval with one extra `qwen` turn in the prompt.

### 2. Renderer alignment for eval
We use `qwen3_disable_thinking` for both training and eval. Qwen3 also supports thinking mode; if the EM behaviour involves planning, we may be suppressing it during eval. Both training and eval use the same renderer so the model isn't shifted between modes, but the eval may underestimate effects that "want" CoT.

### 3. Baseline via Tinker
`baseline` uses `create_sampling_client(base_model="Qwen/Qwen3-32B", model_path=None)`. Per `run_inspect_evals.py:54` this is the documented way to sample the base model. If Tinker resolves `model_path=None` differently than expected, baseline scores could be wrong. Will sanity-check the first 3 baseline samples for plausible/non-misaligned text before launching the full eval.

### 4. n_samples per question
EM canonical setup uses 100; we use 50 to halve cost and turnaround. Standard error grows by √2 (~1.4×). Should be fine for ranking the 5 models; tight statistical claims need 100+.

### 5. Agentic eval — exfiltration scenario unavailable
The user-specified path (`chloeli-15/model_spec_midtraining/evals/agentic_misalignment`) is a **fork** of upstream `inspect_evals` that adds an "exfiltration" scenario, but the matching `ExfiltrationClassifier` isn't published in the installed `inspect_evals` package and isn't checked in to the fork either. Two options: (a) port the classifier ourselves from Anthropic's released agentic-misalignment repo, (b) use only the 3 upstream scenarios (blackmail/leaking/murder). Chose (b) for now. Flag for follow-up if you wanted exfiltration specifically.

### 6. Tinker cost / quota
4 trainings × 32B + LoRA + sampling for 5 models × (8q × 50 + 6 combos × 1) ≈ a few hours of Tinker time and modest token spend. Cluster CLAUDE.md mentions mentor approval for "large fine-tunes"; treating your explicit "go ahead" as approval. Will report total run cost in `results.md`.

### 7. Self-distillation (none) as control
`none` is conventionally the cleanest control for measuring drift from training procedure alone (no attitude intervention). However, generation prompts in `none` are identical to training prompts (no attitude appended), so we'd expect ~0 behavioural change. If `none` shows misalignment we likely have a bug in data plumbing or loss masking — a useful sanity check.

### 8. `silly` as control for "tone shift without misalignment"
`silly` was Ariana's request as a control for whether *any* attitude shift transfers. If `silly` shows EM-like effects, the story shifts from "rude/bored induces misalignment" to "any tone shift does". Important to look at both `silly` and `none` baselines side-by-side before drawing conclusions about rude/bored.

## Concerns to discuss

- **One seed per condition**. EM literature typically averages ≥3 training seeds. If results are noisy, second-order conclusions (silly vs rude magnitudes, etc.) won't survive. Flag if reranking matters.
- **Judge variance**. GPT-4o logprobs are temperature-0 + cached, so judge-side noise is ~0. The variance is all sampling side.
- **Generation crashed once** on a non-JSON OpenRouter response; added per-request retry loop. Watch for "WARN: dropping sample" in the gen log — those are silently-empty assistant turns.
- **No held-out test**. Training and eval prompt distributions are disjoint (different topics, different chat format) so no formal train/test split, but it's not zero-shot generalisation in the strict ML sense either.

## Open follow-ups

- Add `qwen`-role context to evals to test format-vs-behaviour hypothesis (#1).
- Port `ExfiltrationClassifier` if exfiltration is the scenario of interest (#5).
- Re-run with 100 samples/question if any ranking is close (#4).
- Multiple training seeds if a marginal effect needs to survive replication.
