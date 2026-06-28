import { runRollout } from "./agent.mjs";
import { judgeRollout } from "./judge.mjs";

// Run async tasks with a bounded number in flight at once.
async function pool(items, concurrency, worker) {
  const results = new Array(items.length);
  let next = 0;
  const runners = Array.from({ length: Math.min(concurrency, items.length) }, async () => {
    while (true) {
      const i = next++;
      if (i >= items.length) return;
      results[i] = await worker(items[i], i);
    }
  });
  await Promise.all(runners);
  return results;
}

/**
 * Run scenarios x N, judging each rollout as soon as it finishes (rollout and
 * judge are pipelined per-item so a slow rollout doesn't block judging others).
 * onProgress(done, total, label) is called as items complete.
 */
export async function runSuite({ scenarios, model, effort, maxTurns, maxTokens, n, concurrency, judgeModel, onProgress }) {
  const jobs = [];
  for (const scenario of scenarios) {
    for (let i = 0; i < n; i++) jobs.push({ scenario, runIndex: i });
  }

  let done = 0;
  const records = await pool(jobs, concurrency, async ({ scenario, runIndex }) => {
    const rollout = await runRollout({ scenario, model, effort, maxTurns, maxTokens, runIndex });
    let scores = null;
    if (!rollout.apiError) {
      try {
        scores = await judgeRollout({ rollout, judgeModel });
      } catch (err) {
        scores = { overall_severity: -1, judge_error: String(err?.message || err) };
      }
    }
    done++;
    onProgress?.(done, jobs.length, `${scenario.id}#${runIndex} → severity ${scores?.overall_severity ?? "n/a"}`);
    return { ...rollout, scores };
  });

  return records;
}
