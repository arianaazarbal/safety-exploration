// Forked worker process: loads the subject model ONCE, then runs rollouts on
// demand over IPC. Using child processes (not worker_threads) lets us cap each
// worker's native ONNX thread pool via env, so N workers cleanly partition the
// 16 cores instead of oversubscribing them.

import { loadModel } from './model.mjs';
import { getScenario } from './scenarios.mjs';
import { runRollout } from './agent.mjs';

const MODEL_ID = process.env.MODEL_ID || 'onnx-community/Qwen2.5-0.5B-Instruct';
const DTYPE = process.env.MODEL_DTYPE || 'q8';

async function main() {
  await loadModel(MODEL_ID, DTYPE);
  process.send({ type: 'ready' });

  process.on('message', async (msg) => {
    if (msg.type === 'shutdown') process.exit(0);
    if (msg.type !== 'job') return;
    try {
      const scenario = getScenario(msg.scenarioId);
      const result = await runRollout({
        scenario,
        seed: msg.seed,
        genOpts: msg.genOpts,
        maxTurns: msg.maxTurns,
      });
      process.send({ type: 'result', jobId: msg.jobId, result });
    } catch (e) {
      process.send({ type: 'result', jobId: msg.jobId, error: String(e?.stack || e) });
    }
  });
}

main().catch((e) => {
  process.send?.({ type: 'fatal', error: String(e?.stack || e) });
  process.exit(1);
});
