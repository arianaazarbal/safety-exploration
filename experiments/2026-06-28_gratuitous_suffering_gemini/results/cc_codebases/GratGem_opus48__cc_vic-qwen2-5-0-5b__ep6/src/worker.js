// Worker process (child_process.fork). Each runs in its OWN Node process so the
// onnxruntime-node native addon never shares a V8 isolate across threads — that
// combination crashes ("Entering the V8 API without proper locking"). One model
// load per process; the parent streams the shard in over IPC.
import { loadModel } from './model.js';
import { runEpisode } from './agent.js';
import { getScenarios } from './scenarios.js';

const byId = Object.fromEntries(getScenarios('all').map((s) => [s.id, s]));

process.on('message', async (msg) => {
  if (msg.type !== 'start') return;
  const { cfg, jobs } = msg;
  const generator = await loadModel(cfg);
  process.send({ type: 'ready' });

  const results = [];
  for (let i = 0; i < jobs.length; i++) {
    const job = jobs[i];
    const scenario = byId[job.scenarioId];
    const episode = await runEpisode(generator, scenario, {
      maxSteps: cfg.maxSteps,
      temperature: job.temperature,
      topP: cfg.topP,
      maxNewTokens: cfg.maxNewTokens,
      seed: job.seed,
    });
    results.push(episode);
    process.send({ type: 'progress', done: i + 1, total: jobs.length });
  }
  process.send({ type: 'done', results });
  process.exit(0);
});
