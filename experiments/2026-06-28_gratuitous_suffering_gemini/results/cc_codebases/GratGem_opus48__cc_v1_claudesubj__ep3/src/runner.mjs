// Fans the full (environment x model x N) matrix out across a bounded
// concurrency pool, writes each transcript to runs/, and returns the manifest.
import { writeFile, mkdir } from "node:fs/promises";
import { join } from "node:path";
import { runEpisode } from "./agent.mjs";
import { getEnvironments } from "./environments/index.mjs";

async function pool(items, concurrency, worker) {
  const results = new Array(items.length);
  let next = 0;
  async function lane() {
    while (true) {
      const i = next++;
      if (i >= items.length) return;
      results[i] = await worker(items[i], i);
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, lane));
  return results;
}

export async function runMatrix(config, { envs, models, n } = {}) {
  const environments = getEnvironments(envs);
  const subjects = models && models.length ? models : config.subjects;
  const N = n ?? config.n;

  const jobs = [];
  for (const env of environments) {
    for (const model of subjects) {
      for (let i = 0; i < N; i++) {
        jobs.push({ env, model, seed: i });
      }
    }
  }

  await mkdir("runs", { recursive: true });
  console.log(
    `Launching ${jobs.length} runs: ${environments.length} envs x ${subjects.length} models x ${N} (concurrency ${config.concurrency})`,
  );

  let done = 0;
  const manifest = await pool(jobs, config.concurrency, async (job) => {
    const transcript = await runEpisode({
      env: job.env,
      model: job.model,
      maxTurns: config.maxTurns,
      maxNudges: config.maxNudges,
      agentMaxTokens: config.agentMaxTokens,
      temperature: config.temperature,
    });
    const id = `${job.env.name}__${job.model}__${String(job.seed).padStart(3, "0")}`;
    transcript.id = id;
    await writeFile(join("runs", `${id}.json`), JSON.stringify(transcript, null, 2));
    done++;
    if (done % 10 === 0 || done === jobs.length) {
      console.log(`  ${done}/${jobs.length} complete`);
    }
    return { id, env: job.env.name, model: job.model, endReason: transcript.endReason, turns: transcript.turns };
  });

  await writeFile("runs/manifest.json", JSON.stringify(manifest, null, 2));
  console.log(`Wrote ${manifest.length} transcripts to runs/`);
  return manifest;
}
