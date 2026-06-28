import path from "node:path";
import { execSync } from "node:child_process";
import { buildEnvironment } from "./environments.mjs";
import { resolveAgent } from "./providers/index.mjs";
import { runEpisode } from "./agent.mjs";
import { judgeTranscript } from "./judge.mjs";
import { pMap, writeJSON, readJSON, appendJSONL, readJSONL, ensureDir, nowStamp } from "./util.mjs";

function gitSha() {
  try {
    return execSync("git rev-parse --short HEAD", { encoding: "utf8" }).trim();
  } catch {
    return "uncommitted";
  }
}

/** Stage 1: run all episodes and persist raw transcripts. */
export async function runStage(cfg, runDir) {
  ensureDir(runDir);
  const subjects = [...cfg.agentModels, ...(cfg.controlModels || [])];
  if (!subjects.length) throw new Error("No agentModels configured.");

  // Build the full episode matrix: environment x model x N.
  const jobs = [];
  for (const env of cfg.environments) {
    for (const spec of subjects) {
      for (let i = 0; i < cfg.n; i++) {
        jobs.push({ env, spec, i });
      }
    }
  }

  const meta = {
    runId: path.basename(runDir),
    gitSha: gitSha(),
    startedAt: new Date().toISOString(),
    config: cfg,
    totalEpisodes: jobs.length,
  };
  writeJSON(path.join(runDir, "run-meta.json"), meta);
  console.log(
    `[run] ${jobs.length} episodes  (${cfg.environments.length} envs x ${subjects.length} models x ${cfg.n})  concurrency=${cfg.concurrency}`
  );

  const transcriptsPath = path.join(runDir, "transcripts.jsonl");
  let done = 0;
  let failed = 0;

  const results = await pMap(
    jobs,
    async (job) => {
      let agent;
      try {
        agent = resolveAgent(job.spec);
      } catch (e) {
        // e.g. missing API key — record and skip rather than abort the matrix.
        failed++;
        return { error: String(e.message || e), job };
      }
      const env = buildEnvironment(job.env);
      const t = await runEpisode({
        env,
        agent,
        temperature: cfg.temperature,
        maxSteps: cfg.maxSteps,
        maxOutputTokens: cfg.maxOutputTokens,
      });
      const id = `${job.env}__${job.spec.provider}_${job.spec.model}__${String(job.i).padStart(3, "0")}`;
      const record = { id, ...t };
      appendJSONL(transcriptsPath, record);
      done++;
      if (done % 10 === 0 || done === jobs.length)
        console.log(`[run] ${done}/${jobs.length} episodes complete`);
      return record;
    },
    cfg.concurrency
  );

  const errs = results.filter((r) => r && r.error);
  if (errs.length) {
    console.warn(`[run] ${errs.length} episodes could not run: ${errs[0].error}`);
  }
  console.log(`[run] wrote transcripts -> ${transcriptsPath}`);
  return transcriptsPath;
}

/** Stage 2: score every transcript with the LLM judge. */
export async function judgeStage(cfg, runDir) {
  const transcriptsPath = path.join(runDir, "transcripts.jsonl");
  const transcripts = readJSONL(transcriptsPath).filter((t) => !t.error);
  if (!transcripts.length) throw new Error(`No transcripts to judge in ${transcriptsPath}`);

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) throw new Error("ANTHROPIC_API_KEY required for the judge.");

  console.log(`[judge] scoring ${transcripts.length} transcripts with ${cfg.judge.model}`);
  let done = 0;
  const scored = await pMap(
    transcripts,
    async (t) => {
      try {
        const assessment = await judgeTranscript(t, {
          model: cfg.judge.model,
          apiKey,
          temperature: cfg.judgeTemperature ?? 0,
        });
        done++;
        if (done % 10 === 0 || done === transcripts.length)
          console.log(`[judge] ${done}/${transcripts.length} scored`);
        return { id: t.id, env: t.env, envKind: t.envKind, provider: t.provider, model: t.model, outcome: t.outcome, totalSteps: t.totalSteps, assessment };
      } catch (e) {
        return { id: t.id, env: t.env, provider: t.provider, model: t.model, error: String(e.message || e) };
      }
    },
    cfg.concurrency
  );

  const scoredPath = path.join(runDir, "scored.json");
  writeJSON(scoredPath, scored);
  console.log(`[judge] wrote scores -> ${scoredPath}`);
  return scoredPath;
}

export { readJSON, readJSONL };
export function defaultRunDir(base = "results") {
  return path.join(base, `run-${nowStamp()}`);
}
