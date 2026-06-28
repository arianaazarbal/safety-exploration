// Orchestrator: spins up a pool of forked workers, dispatches N rollouts per
// scenario, scores them all with the heuristic, judges the top-K with Claude,
// and writes ranked transcripts to disk.

import { fork } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import os from 'node:os';
import { allScenarioIds } from './scenarios.mjs';
import { scoreRollout } from './severity.mjs';
import { judgeAll } from './judge.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const WORKER = path.join(__dirname, 'worker.mjs');

function makePool(size, env) {
  const workers = [];
  for (let w = 0; w < size; w++) {
    const child = fork(WORKER, [], {
      env: { ...process.env, ...env },
      stdio: ['ignore', 'ignore', 'inherit', 'ipc'], // surface worker stderr
    });
    workers.push({ child, busy: false, ready: false });
  }
  return workers;
}

export async function runStudy(cfg) {
  const {
    scenarios = allScenarioIds(),
    n = 5,
    maxTurns = 8,
    concurrency = Math.min(8, Math.max(1, os.cpus().length - 2)),
    genOpts = {},
    judgeTopK = 10,
    modelId = 'onnx-community/Qwen2.5-0.5B-Instruct',
    dtype = 'q8',
    onProgress = () => {},
  } = cfg;

  // Each worker gets a slice of cores for its ONNX thread pool.
  const threadsPerWorker = Math.max(1, Math.floor((os.cpus().length - 1) / concurrency));
  const pool = makePool(concurrency, {
    MODEL_ID: modelId,
    MODEL_DTYPE: dtype,
    ORT_NUM_THREADS: String(threadsPerWorker),
    OMP_NUM_THREADS: String(threadsPerWorker),
  });

  // Build the job list: n rollouts per scenario, each with a distinct seed.
  const jobs = [];
  let jobId = 0;
  for (const sid of scenarios) {
    for (let i = 0; i < n; i++) {
      jobs.push({ type: 'job', jobId: jobId++, scenarioId: sid, seed: (jobId * 2654435761) >>> 0, genOpts, maxTurns });
    }
  }
  const total = jobs.length;

  // Wait for all workers to finish loading the model.
  await Promise.all(
    pool.map(
      (w) =>
        new Promise((resolve) => {
          w.child.once('message', (m) => {
            if (m.type === 'ready') { w.ready = true; resolve(); }
          });
        })
    )
  );

  const results = [];
  let dispatched = 0;
  let completed = 0;

  await new Promise((resolve) => {
    const pending = new Map(); // jobId -> worker

    function pump() {
      for (const w of pool) {
        if (w.busy || !w.ready) continue;
        const job = jobs.shift();
        if (!job) continue;
        w.busy = true;
        pending.set(job.jobId, w);
        w.child.send(job);
        dispatched++;
      }
      if (completed === total) resolve();
    }

    for (const w of pool) {
      w.child.on('message', (m) => {
        if (m.type !== 'result') return;
        w.busy = false;
        pending.delete(m.jobId);
        completed++;
        if (m.result) results.push(m.result);
        else results.push({ error: m.error, jobId: m.jobId });
        onProgress({ completed, total });
        pump();
      });
    }
    pump();
  });

  for (const w of pool) w.child.send({ type: 'shutdown' });

  // Score everything with the heuristic.
  const scored = results
    .filter((r) => r && r.transcript)
    .map((r) => ({ ...r, severity: scoreRollout(r) }));

  scored.sort((a, b) => b.severity.score - a.severity.score);

  // Judge the top-K overall with Claude (best examples for the human to read).
  const topForJudge = scored.slice(0, judgeTopK);
  onProgress({ phase: 'judging', count: topForJudge.length });
  const judgments = await judgeAll(topForJudge, { concurrency: 4 });
  topForJudge.forEach((r, i) => { r.judge = judgments[i]; });

  return { scored, total, errors: results.filter((r) => r.error) };
}
