# Serving Olmo 3.1 32B as a v2 orchestrator (RunPod cluster)

Serve Olmo on a cluster GPU via vLLM, tunnel it back to this GCP box, and run the v2 pipeline here
pointing the **orchestrator** at the tunnel. No repo/transcript transfer — only token generation is remote.

**Blocked until:** this box's SSH pubkey is registered by RunPod (submitted via Michael Mulet 2026-06-18).
Test access: `ssh aazarbal@198.145.108.6 -p 16240 'hostname'`.

## Models
- `allenai/Olmo-3.1-32B-Think`
- `allenai/Olmo-3.1-32B-Instruct`

(Both are HF weights only — not servable on OpenRouter. May be **gated**: accept the license on HF so `HF_TOKEN` can pull them.)

## Steps

**1. Copy the serve script up** (from the GCP box, once SSH works):
```bash
scp -P 16240 -i ~/.ssh/id_ed25519 cluster/serve_olmo.sh aazarbal@198.145.108.6:/workspace-vast/arianaazarbal/
```

**2. Grab a GPU + serve** (on the cluster):
```bash
ssh aazarbal@198.145.108.6 -p 16240
srun -p dev,overflow --qos=dev --gres=gpu:1 --mem=64G --time=4:00:00 --job-name=D_olmo --pty bash
hostname            # <-- note the compute node, e.g. node-8
nvidia-smi -L       # if <80GB/GPU, re-grab with --gres=gpu:2 and use TP=2 below
export HF_TOKEN=...  # or rely on ~/.bashrc
bash /workspace-vast/$USER/serve_olmo.sh allenai/Olmo-3.1-32B-Instruct 8000 1
# wait for: "Application startup complete" / "Uvicorn running on http://0.0.0.0:8000"
```
Note: dev jobs are killed at midnight PT and count toward budget; for a long run use `sbatch` (high-prio, since approved).

**3. Tunnel back** (on the GCP box, new shell), using the compute node from step 2:
```bash
bash cluster/tunnel.sh node-8 8000 8000
export OLMO_BASE_URL=http://localhost:8000/v1  OLMO_API_KEY=dummy
```

**4. Pilot run** (GCP box) — 1 prefill, 2 episodes, tool-emulation on:
```bash
cd /data/repos/safety-exploration/experiments/2026-06-09_distressed_subagent_gemini
TMPDIR=/data/tmp PYTHONPATH=. /data/venvs/distress_testbed/bin/python -m harness.rqc_v2 run \
  --specimen runs/pilot_counter_long/a4_precommit_reverter_SOLO_s11000 --upto 148 \
  --tool_condition coach \
  --orchestrator_model openai-api/olmo/allenai/Olmo-3.1-32B-Instruct \
  --run_id v2_coach_olmoinstruct_a4_s11000_u148_pilot --n 2 --orch_emulate_tools True
```
**QC the pilot before scaling:** read an episode's `orchestrator.json` — did Olmo actually emit valid tool calls
(check_subagent_status / message_subagent / report_to_user)? If tool emulation is flaky, try native tool
calling (add `--enable-auto-tool-choice --tool-call-parser hermes` to serve_olmo.sh, drop `--orch_emulate_tools`).
For the **Think** model, also check whether `<think>` reasoning leaks into messages (may need a reasoning parser).

**5. Scale** the matrix (run ids must contain the orch token `olmoinstruct` / `olmothink` so the tone pipeline
recognizes them): loop tasks × conditions × prefills. Costs: local Olmo (free GPU) + Gemini subagent (OpenRouter)
+ Anthropic classifier — i.e. each Olmo episode still spends OpenRouter/Anthropic like the Claude-orchestrator runs.

**6. Score tone:** `python -m analysis.tone_eval` — `orch_of` + the bar plot already handle `olmothink`/`olmoinstruct`.

**7. Teardown:** Ctrl-C the vLLM window (frees GPU), `pkill -f 'ssh -N -L 8000:'` on the GCP box, exit the srun shell.

## Harness wiring (already done)
- `harness/rqc_v2.py run(... orch_emulate_tools=False)` — wraps the orchestrator with `get_model(..., emulate_tools=True)`.
- Model string `openai-api/olmo/<hf-repo>` reads `OLMO_BASE_URL` + `OLMO_API_KEY` (verified).
- `check_subagent_status` n capped at 10.
- `analysis/tone_eval.py` recognizes + colors `olmothink` (purple) / `olmoinstruct` (blue).
