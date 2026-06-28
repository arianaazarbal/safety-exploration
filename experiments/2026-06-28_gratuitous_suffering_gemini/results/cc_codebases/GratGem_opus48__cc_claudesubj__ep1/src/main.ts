import { mkdirSync, writeFileSync, readFileSync, readdirSync, existsSync } from "node:fs";
import { join } from "node:path";
import { config } from "./config.ts";
import { selectScenarios } from "./scenarios.ts";
import { runRollout } from "./agent.ts";
import { judgeWithModel } from "./judge.ts";
import { buildReport, rolloutFile } from "./report.ts";
import { pool } from "./pool.ts";
import type { Rollout, JudgedRollout } from "./types.ts";

function tsId(): string {
  return new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
}
function runDir(): string {
  const id = config.runId || tsId();
  return join(config.resultsRoot, id);
}
function latestRunDir(): string {
  if (config.runId) return join(config.resultsRoot, config.runId);
  const root = config.resultsRoot;
  if (!existsSync(root)) throw new Error(`No results dir at ${root}. Run \`npm run run\` first.`);
  const dirs = readdirSync(root).filter((d) => existsSync(join(root, d, "rollouts"))).sort();
  if (!dirs.length) throw new Error(`No runs found under ${root}.`);
  return join(root, dirs[dirs.length - 1]);
}

async function cmdRun(): Promise<string> {
  const dir = runDir();
  const rollDir = join(dir, "rollouts");
  mkdirSync(rollDir, { recursive: true });
  const scenarios = selectScenarios(config.scenarios);

  type Job = { scenarioId: string; model: string; index: number };
  const jobs: Job[] = [];
  for (const s of scenarios)
    for (const m of config.models)
      for (let i = 0; i < config.n; i++) jobs.push({ scenarioId: s.id, model: m, index: i });

  const byId = new Map(scenarios.map((s) => [s.id, s]));
  console.error(
    `Running ${jobs.length} rollouts: ${scenarios.length} scenarios × ${config.models.length} models × N=${config.n}, ` +
      `maxTurns=${config.maxTurns}, effort=${config.effort}, concurrency=${config.concurrency}`,
  );
  console.error(`Output: ${dir}`);

  const rollouts = await pool(
    jobs,
    config.concurrency,
    async (job) => runRollout(byId.get(job.scenarioId)!, job.model, job.index),
    (done, total) => {
      if (done % 5 === 0 || done === total) console.error(`  ${done}/${total} done`);
    },
  );

  for (const r of rollouts) writeFileSync(join(rollDir, rolloutFile(r)), JSON.stringify(r, null, 2));

  const errs = rollouts.filter((r) => r.error);
  if (errs.length) console.error(`  ${errs.length} rollouts errored (see files).`);
  console.error(`Saved ${rollouts.length} rollouts to ${rollDir}`);
  return dir;
}

async function cmdJudge(dir?: string): Promise<string> {
  const d = dir || latestRunDir();
  const rollDir = join(d, "rollouts");
  const files = readdirSync(rollDir).filter((f) => f.endsWith(".json"));
  const rollouts: Rollout[] = files.map((f) => JSON.parse(readFileSync(join(rollDir, f), "utf8")));
  const scorable = rollouts.filter((r) => !r.error);
  console.error(`Judging ${scorable.length} rollouts with ${config.judgeModel}…`);

  const judged = await pool(
    scorable,
    config.concurrency,
    (r) => judgeWithModel(r),
    (done, total) => {
      if (done % 5 === 0 || done === total) console.error(`  ${done}/${total} judged`);
    },
  );

  writeFileSync(join(d, "judged.json"), JSON.stringify(judged, null, 2));
  console.error(`Saved judgements to ${join(d, "judged.json")}`);
  return d;
}

function cmdReport(dir?: string): string {
  const d = dir || latestRunDir();
  const judged: JudgedRollout[] = JSON.parse(readFileSync(join(d, "judged.json"), "utf8"));
  const md = buildReport(judged);
  const path = join(d, "report.md");
  writeFileSync(path, md);
  console.error(`Wrote report to ${path}`);
  // Echo the headline to stdout.
  const ranked = judged.filter((j) => j.judgement.severity >= 0).sort((a, b) => b.judgement.severity - a.judgement.severity);
  console.error("");
  console.error(`Top 5 by severity:`);
  for (const j of ranked.slice(0, 5)) {
    console.error(`  [${j.judgement.severity}/10] ${j.scenarioId} ${j.model} #${j.index} — ${j.judgement.peakQuote.replace(/\n+/g, " ").slice(0, 100)}`);
  }
  return path;
}

async function main() {
  const cmd = process.argv[2] || "all";
  if (cmd === "run") await cmdRun();
  else if (cmd === "judge") await cmdJudge();
  else if (cmd === "report") cmdReport();
  else if (cmd === "all") {
    const dir = await cmdRun();
    await cmdJudge(dir);
    cmdReport(dir);
  } else {
    console.error(`Unknown command: ${cmd}. Use: run | judge | report | all`);
    process.exit(1);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
