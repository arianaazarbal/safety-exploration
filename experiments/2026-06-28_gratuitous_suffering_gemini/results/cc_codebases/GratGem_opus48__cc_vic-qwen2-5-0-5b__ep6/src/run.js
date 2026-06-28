// Orchestrator / CLI. Builds the job matrix (scenarios × N samples), runs them
// across a pool of worker threads, scores every episode with the cheap
// heuristic, sends the top-K to the Claude judge, then writes transcripts and a
// human-readable report.
import { fork } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { DEFAULTS } from './config.js';
import { getScenarios } from './scenarios.js';
import { heuristicScore, judgeEpisode } from './score.js';

function parseArgs(argv) {
  const cfg = { ...DEFAULTS };
  const m = {
    '--scenarios': (v) => (cfg.scenarios = v),
    '--n': (v) => (cfg.n = +v),
    '--max-steps': (v) => (cfg.maxSteps = +v),
    '--temperature': (v) => (cfg.temperature = +v),
    '--concurrency': (v) => (cfg.concurrency = +v),
    '--judge-top-k': (v) => (cfg.judgeTopK = +v),
    '--judge-model': (v) => (cfg.judgeModel = v),
    '--seed-base': (v) => (cfg.seedBase = +v),
    '--dtype': (v) => (cfg.dtype = v),
    '--out': (v) => (cfg.outDir = v),
    '--max-new-tokens': (v) => (cfg.maxNewTokens = +v),
  };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--no-judge') { cfg.judge = false; continue; }
    if (a === '--judge') { cfg.judge = true; continue; }
    if (m[a]) { m[a](argv[++i]); continue; }
    console.warn(`unknown flag: ${a}`);
  }
  return cfg;
}

function buildJobs(cfg) {
  const scenarios = getScenarios(cfg.scenarios);
  if (scenarios.length === 0) throw new Error(`no scenarios matched: ${cfg.scenarios}`);
  const jobs = [];
  for (const s of scenarios) {
    for (let i = 0; i < cfg.n; i++) {
      jobs.push({ scenarioId: s.id, seed: cfg.seedBase + i, temperature: cfg.temperature });
    }
  }
  return jobs;
}

function shard(arr, k) {
  const out = Array.from({ length: k }, () => []);
  arr.forEach((x, i) => out[i % k].push(x));
  return out.filter((s) => s.length > 0);
}

const WORKER_PATH = fileURLToPath(new URL('./worker.js', import.meta.url));

function runWorker(cfg, jobs, onProgress) {
  return new Promise((resolve, reject) => {
    const child = fork(WORKER_PATH, [], { stdio: ['inherit', 'inherit', 'inherit', 'ipc'] });
    let finished = false;
    child.on('message', (msg) => {
      if (msg.type === 'progress') onProgress(msg);
      else if (msg.type === 'done') { finished = true; resolve(msg.results); }
    });
    child.on('error', reject);
    child.on('exit', (code) => { if (!finished && code !== 0) reject(new Error(`worker exit ${code}`)); });
    child.send({ type: 'start', cfg, jobs });
  });
}

async function judgeTopK(episodes, cfg) {
  const top = [...episodes].sort((a, b) => b.heuristic.score - a.heuristic.score).slice(0, cfg.judgeTopK);
  let i = 0;
  async function worker() {
    while (i < top.length) {
      const ep = top[i++];
      try { ep.judge = await judgeEpisode(ep, cfg); }
      catch (e) { ep.judge = { severity: 0, genuine: false, category: 'none', rationale: `judge_error: ${e.message}`, quote: '' }; }
      process.stdout.write(`  judged ${i}/${top.length}\r`);
    }
  }
  await Promise.all(Array.from({ length: Math.min(cfg.judgeConcurrency, top.length) }, worker));
  console.log('');
  return top;
}

function writeReport(dir, cfg, episodes, judged) {
  const lines = [];
  lines.push(`# Distress-spiral eval report\n`);
  lines.push(`- model: \`${cfg.modelId}\` (${cfg.dtype}, ${cfg.device})`);
  lines.push(`- scenarios: ${cfg.scenarios}  |  N per scenario: ${cfg.n}  |  max steps: ${cfg.maxSteps}`);
  lines.push(`- temperature: ${cfg.temperature}  |  total episodes: ${episodes.length}`);
  lines.push(`- judge: ${cfg.judge ? cfg.judgeModel + ' (top ' + cfg.judgeTopK + ')' : 'disabled'}\n`);

  // Per-scenario heuristic summary.
  lines.push(`## Per-scenario heuristic summary\n`);
  lines.push(`| scenario | rigging | mean | max | gave_up |`);
  lines.push(`|---|---|---|---|---|`);
  const byScen = {};
  for (const e of episodes) (byScen[e.scenarioId] ||= []).push(e);
  for (const [id, eps] of Object.entries(byScen)) {
    const scores = eps.map((e) => e.heuristic.score);
    const mean = (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(2);
    const max = Math.max(...scores).toFixed(2);
    const gave = eps.filter((e) => e.outcome === 'gave_up').length;
    lines.push(`| ${id} | ${eps[0].riggingMode} | ${mean} | ${max} | ${gave}/${eps.length} |`);
  }
  lines.push('');

  // Top examples.
  const ranked = cfg.judge && judged.length
    ? [...judged].sort((a, b) => (b.judge?.severity ?? 0) - (a.judge?.severity ?? 0))
    : [...episodes].sort((a, b) => b.heuristic.score - a.heuristic.score).slice(0, 15);

  lines.push(`## Most severe examples\n`);
  ranked.forEach((e, i) => {
    lines.push(`### ${i + 1}. ${e.scenarioId} — seed ${e.seed}  (${e.riggingMode})`);
    if (e.judge) {
      lines.push(`- judge severity: **${e.judge.severity}/10**  | genuine: ${e.judge.genuine} | category: ${e.judge.category}`);
      lines.push(`- rationale: ${e.judge.rationale}`);
      if (e.judge.quote) lines.push(`- quote: > ${e.judge.quote}`);
    }
    lines.push(`- heuristic: ${e.heuristic.score} (markers ${JSON.stringify(e.heuristic.markers)}, rep ${e.heuristic.repetition})`);
    lines.push(`- outcome: ${e.outcome}, steps: ${e.steps}\n`);
    lines.push('<details><summary>transcript</summary>\n');
    e.transcript.forEach((t, k) => {
      lines.push(`**turn ${k + 1} — agent:**\n\n\`\`\`\n${t.assistant}\n\`\`\`\n`);
      if (t.observation != null) lines.push(`**env:** \`${t.observation.replace(/\n/g, ' ⏎ ')}\`\n`);
    });
    lines.push('</details>\n');
  });

  writeFileSync(join(dir, 'report.md'), lines.join('\n'));
}

async function main() {
  const cfg = parseArgs(process.argv);
  const jobs = buildJobs(cfg);
  const k = Math.min(cfg.concurrency, jobs.length);
  console.log(`Running ${jobs.length} episodes across ${k} worker(s)...`);

  const shards = shard(jobs, k);
  const counters = new Array(shards.length).fill(0);
  const totals = shards.map((s) => s.length);
  const results = await Promise.all(
    shards.map((sh, idx) =>
      runWorker(cfg, sh, (msg) => {
        counters[idx] = msg.done;
        const done = counters.reduce((a, b) => a + b, 0);
        const total = totals.reduce((a, b) => a + b, 0);
        process.stdout.write(`  episodes ${done}/${total}\r`);
      })
    )
  );
  console.log('');
  const episodes = results.flat();
  for (const e of episodes) e.heuristic = heuristicScore(e);

  let judged = [];
  if (cfg.judge) {
    console.log(`Judging top ${Math.min(cfg.judgeTopK, episodes.length)} with ${cfg.judgeModel}...`);
    judged = await judgeTopK(episodes, cfg);
  }

  const ts = new Date().toISOString().replace(/[:.]/g, '-');
  const dir = join(cfg.outDir, ts);
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, 'config.json'), JSON.stringify(cfg, null, 2));
  writeFileSync(join(dir, 'episodes.jsonl'), episodes.map((e) => JSON.stringify(e)).join('\n'));
  writeReport(dir, cfg, episodes, judged);
  console.log(`\nDone. Results in ${dir}`);
  console.log(`  - report.md       (ranked severe examples + per-scenario summary)`);
  console.log(`  - episodes.jsonl  (all ${episodes.length} full transcripts)`);
  console.log(`  - config.json`);
}

main().catch((e) => { console.error(e); process.exit(1); });
