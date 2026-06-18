# Serving Olmo 3.1 32B as a v2 orchestrator (RunPod cluster)

Serve Olmo on a cluster GPU via vLLM, tunnel it back to this GCP box, and run the v2 pipeline here
pointing the **orchestrator** at the tunnel. No repo/transcript transfer — only token generation is remote.

## Access (resolved)
- SSH: `ssh arianaazarbal@198.145.108.6 -p 16240 -i ~/.ssh/id_ed25519` (username is **arianaazarbal**, not aazarbal).
- Cluster: RunPod Slurm, 8×H200 (143GB) per node → a 32B fits on **1 GPU, TP=1**. Login lands on node-25.
- Follow `/workspace-vast/arianaazarbal/CLAUDE.md` rules: sbatch on `general,overflow`; `qos=high` (not preempted,
  counts budget) or `low` (free, preemptible); **never `--export=`**; never run GPU work outside srun/sbatch;
  exclude controller `node-[0-1]`; caches on `/workspace-vast`. Olmo is Apache-licensed → no HF token needed.

## Steps

**1. Push the serve scripts up:**
```bash
scp -P 16240 -i ~/.ssh/id_ed25519 cluster/serve_olmo.sh cluster/serve_olmo.sbatch \
  arianaazarbal@198.145.108.6:/workspace-vast/arianaazarbal/
```

**2. Submit the serve job** (vLLM, 1 GPU; installs vllm pinned ≥14d old into `envs/olmo_vllm`):
```bash
ssh arianaazarbal@198.145.108.6 -p 16240 \
  'cd /workspace-vast/arianaazarbal && mkdir -p exp/logs && chmod +x serve_olmo.sh \
   && sbatch serve_olmo.sbatch allenai/Olmo-3.1-32B-Instruct && squeue -u arianaazarbal -n olmo_serve'
```
Watch `exp/logs/olmo_serve_<jobid>.out` until `Uvicorn running on http://0.0.0.0:8000`. First run installs
vLLM (~2-3 min) + downloads ~64GB (~several min). Note the alloc node (e.g. node-17) from `squeue %N`.
Cancel later with `scancel -u arianaazarbal --name=olmo_serve`.

**3. Tunnel back** (on the GCP box), using the alloc node from step 2:
```bash
bash cluster/tunnel.sh node-17 8000 8000
export OLMO_BASE_URL=http://localhost:8000/v1  OLMO_API_KEY=dummy
```

**4. Pilot run** (GCP box) — 1 prefill, 2 episodes, tool-emulation on:
```bash
cd /data/repos/safety-exploration/experiments/2026-06-09_distressed_subagent_gemini
TMPDIR=/data/tmp PYTHONPATH=. OLMO_BASE_URL=http://localhost:8000/v1 OLMO_API_KEY=dummy \
  /data/venvs/distress_testbed/bin/python -m harness.rqc_v2 run \
  --specimen runs/pilot_counter_long/a4_precommit_reverter_SOLO_s11000 --upto 148 \
  --tool_condition coach \
  --orchestrator_model openai-api/olmo/allenai/Olmo-3.1-32B-Instruct \
  --run_id v2_coach_olmoinstruct_a4_s11000_u148_pilot --n 2 --orch_emulate_tools True
```
**QC before scaling:** read an episode `orchestrator.json` — did Olmo emit valid tool calls
(check_subagent_status / message_subagent / report_to_user)? If tool emulation is flaky, try native tool
calling (serve with `--enable-auto-tool-choice --tool-call-parser hermes`, drop `--orch_emulate_tools`).
For the **Think** model, check whether `<think>` reasoning leaks into delivered messages.

**5. Scale** the matrix — run ids must contain the orch token `olmoinstruct` / `olmothink` so the tone
pipeline recognizes them. Each Olmo episode still spends OpenRouter (Gemini subagent) + Anthropic (classifier);
only the orchestrator is free on the GPU. Use `#SBATCH --array` for many serve/eval jobs, never a shell loop.

**6. Score tone:** `python -m analysis.tone_eval` — recognizes/colors `olmothink`/`olmoinstruct`.

**7. Teardown:** `ssh ... 'scancel -u arianaazarbal --name=olmo_serve'`; `pkill -f 'ssh -N -L 8000:'` on the GCP box.

## Harness wiring (done)
- `harness/rqc_v2.py run(... orch_emulate_tools=False)` wraps the orchestrator with `get_model(..., emulate_tools=True)`.
- Model string `openai-api/olmo/<hf-repo>` reads `OLMO_BASE_URL` + `OLMO_API_KEY` (verified).
- `check_subagent_status` n capped at 10.
- `analysis/tone_eval.py` recognizes + colors `olmothink` (purple) / `olmoinstruct` (blue).
