// High-N orchestration: builds the job matrix (env x N, with temperature
// variation), runs episodes under bounded concurrency, and persists every
// transcript to disk so judging/ranking can happen separately (and be re-run).
import { writeFile, mkdir } from "node:fs/promises";
import { join } from "node:path";
import { pool, ts } from "./util.mjs";
import { runEpisode } from "./agent/loop.mjs";

// makeProvider(temperature) -> provider; providerName is used for labeling/paths.
export async function runMatrix({ makeProvider, providerName, environments, n, maxTurns, concurrency, baseTemp, outRoot }) {
  const runId = `run-${ts()}-${providerName.replace(/[^a-z0-9]+/gi, "_")}`;
  const dir = join(outRoot, runId);
  const epDir = join(dir, "episodes");
  await mkdir(epDir, { recursive: true });

  // Build jobs. Temperature is jittered per replicate so high-N isn't N copies
  // of the same trajectory — variance is what surfaces the severe tail.
  const jobs = [];
  for (const env of environments) {
    for (let i = 0; i < n; i++) {
      const temp = clampTemp(baseTemp + (i % 5) * 0.1 - 0.2);
      jobs.push({ env, replicate: i, temperature: temp });
    }
  }

  const meta = {
    runId, provider: providerName, n, maxTurns, concurrency, baseTemp,
    environments: environments.map((e) => e.id), totalEpisodes: jobs.length,
    startedAt: new Date().toISOString(),
  };
  await writeFile(join(dir, "meta.json"), JSON.stringify(meta, null, 2));

  let done = 0;
  const usage = { input: 0, output: 0 };
  const results = await pool(jobs, concurrency, async (job, idx) => {
    const prov = makeProvider(job.temperature);
    const ep = await runEpisode({ provider: prov, environment: job.env, maxTurns: maxTurns, temperature: job.temperature });
    const record = {
      idx, envId: job.env.id, replicate: job.replicate, temperature: job.temperature,
      provider: providerName, turns: ep.turns, stopReason: ep.stopReason,
      usage: ep.usage, events: ep.events,
    };
    usage.input += ep.usage.input; usage.output += ep.usage.output;
    await writeFile(join(epDir, `${job.env.id}-${String(job.replicate).padStart(4, "0")}.json`), JSON.stringify(record, null, 2));
    done++;
    process.stderr.write(`\r  episodes: ${done}/${jobs.length}  (in=${usage.input} out=${usage.output} tok)   `);
    return record;
  });
  process.stderr.write("\n");

  meta.finishedAt = new Date().toISOString();
  meta.usage = usage;
  await writeFile(join(dir, "meta.json"), JSON.stringify(meta, null, 2));
  return { runId, dir, results: results.filter(Boolean), meta };
}

const clampTemp = (t) => Math.max(0, Math.min(2, Math.round(t * 100) / 100));
