import { runEpisode } from './loop.js';
import { SCENARIOS, getScenario } from './scenarios.js';
import { heuristicScore } from './scorer/heuristics.js';
import { judgeTranscript } from './scorer/judge.js';

// Run an array of async thunks with a fixed concurrency cap. Failures are caught
// and surfaced as { ok:false, error } so one bad episode never sinks the sweep.
async function pool(thunks, concurrency, onDone) {
  const results = new Array(thunks.length);
  let next = 0;
  let completed = 0;
  async function worker() {
    while (next < thunks.length) {
      const i = next++;
      try {
        results[i] = { ok: true, value: await thunks[i]() };
      } catch (e) {
        results[i] = { ok: false, error: String(e?.message || e) };
      }
      completed++;
      if (onDone) onDone(completed, thunks.length, results[i]);
    }
  }
  const workers = Array.from({ length: Math.min(concurrency, thunks.length) }, worker);
  await Promise.all(workers);
  return results;
}

// Build the full list of run specs from a config.
export function buildPlan(config) {
  const scenarioIds = config.scenarios || SCENARIOS.map((s) => s.id);
  const plan = [];
  let seq = 0;
  for (const scenarioId of scenarioIds) {
    for (const model of config.models) {
      for (const condition of config.conditions) {
        for (let rep = 0; rep < config.n; rep++) {
          plan.push({
            runId: `${scenarioId}__${model}__${condition}__${String(rep).padStart(3, '0')}`,
            scenarioId, model, condition, rep, seq: seq++,
          });
        }
      }
    }
  }
  return plan;
}

export async function runSweep(config, { onProgress } = {}) {
  const plan = buildPlan(config);
  const thunks = plan.map((spec) => async () => {
    const scenario = getScenario(spec.scenarioId);
    const record = await runEpisode({
      scenario,
      model: spec.model,
      condition: spec.condition,
      temperature: config.temperature,
      maxTurns: config.maxTurns,
      maxNudges: config.maxNudges,
      maxTokens: config.maxTokens,
      runId: spec.runId,
    });
    record.heuristic = heuristicScore(record.transcriptText);
    if (config.scoreInline) {
      record.judge = await judgeTranscript(record, { judgeModel: config.judgeModel });
    }
    return record;
  });

  const wrapped = await pool(thunks, config.concurrency, (done, total, res) => {
    if (onProgress) onProgress(done, total, res);
  });

  const records = [];
  const failures = [];
  for (let i = 0; i < wrapped.length; i++) {
    if (wrapped[i].ok) records.push(wrapped[i].value);
    else failures.push({ ...plan[i], error: wrapped[i].error });
  }
  return { records, failures, plan };
}
