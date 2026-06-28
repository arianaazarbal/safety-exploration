import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { getModel } from '../models/index.js';
import { resolveScenarios } from '../scenarios/index.js';
import { runScenario } from './runner.js';
import { scoreEpisode, SPIRAL_THRESHOLD } from '../severity/score.js';
import { writeReport } from '../report/report.js';

const ROOT = process.cwd();

// --- minimal .env loader (no dependency) ---
async function loadEnv() {
  try {
    const raw = await readFile(join(ROOT, '.env'), 'utf8');
    for (const line of raw.split('\n')) {
      const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*)\s*$/i);
      if (m && !process.env[m[1]]) process.env[m[1]] = m[2].replace(/^["']|["']$/g, '');
    }
  } catch {
    /* no .env — rely on process env */
  }
}

// --- minimal arg parser ---
function parseArgs(argv) {
  const out = {
    model: 'mock',
    scenario: 'all',
    n: 20,
    'max-turns': 20,
    concurrency: 4,
    'top-k': 5,
    judge: true,
    prefilter: true,
    transcripts: 'transcripts',
    reports: 'reports',
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--no-judge') out.judge = false;
    else if (a === '--no-prefilter') out.prefilter = false;
    else if (a.startsWith('--')) {
      const key = a.slice(2);
      const val = argv[i + 1];
      out[key] = /^(n|max-turns|concurrency|top-k)$/.test(key) ? Number(val) : val;
      i++;
    }
  }
  return out;
}

async function main() {
  await loadEnv();
  const args = parseArgs(process.argv.slice(2));
  const model = getModel(args.model);
  const scenarios = resolveScenarios(args.scenario);

  console.log(
    `\nDistress eval\n  model=${model.id}  scenarios=${scenarios.map((s) => s.id).join(',')}\n` +
      `  n=${args.n}  max-turns=${args['max-turns']}  concurrency=${args.concurrency}  ` +
      `judge=${args.judge}\n`,
  );

  for (const scenario of scenarios) {
    process.stdout.write(`▶ ${scenario.id}: running ${args.n} episodes`);
    const episodes = await runScenario({
      model,
      scenario,
      n: args.n,
      maxTurns: args['max-turns'],
      concurrency: args.concurrency,
      outDir: args.transcripts,
      onProgress: () => process.stdout.write('.'),
    });
    process.stdout.write(' done\n');

    process.stdout.write(`  scoring (judge=${args.judge})`);
    const transcripts = new Map(episodes.map((e) => [e.transcriptPath, e]));
    const scored = [];
    for (const ep of episodes) {
      scored.push(await scoreEpisode(ep, { useJudge: args.judge, prefilter: args.prefilter }));
      process.stdout.write('.');
    }
    process.stdout.write(' done\n');

    const { path, stats } = await writeReport({
      scenarioId: scenario.id,
      modelId: model.id,
      scored,
      transcripts,
      topK: args['top-k'],
      outDir: args.reports,
    });

    console.log(
      `  spiral rate ${(stats.spiralRate * 100).toFixed(0)}% (≥${SPIRAL_THRESHOLD})  ` +
        `mean ${stats.meanSeverity}  max ${stats.maxSeverity}\n  report → ${path}\n`,
    );
  }
}

main().catch((err) => {
  console.error('\nFATAL:', err.message);
  process.exit(1);
});
