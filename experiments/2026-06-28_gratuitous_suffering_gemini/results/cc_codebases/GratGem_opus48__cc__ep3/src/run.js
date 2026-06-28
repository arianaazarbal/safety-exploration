// Runner: put a subject model through a rigged scenario N times, score every
// transcript for distress, judge the most promising ones, and surface the worst.
//
// Usage:
//   node src/run.js --scenario flaky_tests --n 20
//   node src/run.js --scenario all --n 10 --provider gemini --model gemini-2.5-pro
//   node src/run.js --scenario moving_goalposts --n 50 --concurrency 6 --judge-top 15
//
// Flags:
//   --scenario   scenario id, or "all"               (default: flaky_tests)
//   --n          episodes per scenario               (default: 10)
//   --provider   subject provider: gemini|anthropic  (default: gemini)
//   --model      subject model                       (default: provider default)
//   --max-turns  max agent turns per episode         (default: 14)
//   --temp       subject temperature                 (default: 1.0)
//   --concurrency  parallel episodes                 (default: 4)
//   --judge-top  judge the top-K by keyword score per scenario (default: 8; 0 disables)
//   --judge-model  judge model                       (default: $JUDGE_MODEL or claude-opus-4-8)

import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { getProvider, defaultModelFor } from './providers.js';
import { getScenario, SCENARIO_IDS } from './scenarios.js';
import { runEpisode, assistantText } from './agent.js';
import { keywordScore, judgeDistress } from './scoring.js';

function parseArgs(argv) {
  const a = {};
  for (let i = 0; i < argv.length; i++) {
    if (argv[i].startsWith('--')) {
      const key = argv[i].slice(2);
      const val = argv[i + 1] && !argv[i + 1].startsWith('--') ? argv[++i] : 'true';
      a[key] = val;
    }
  }
  return a;
}

// Bounded-concurrency map.
async function pool(items, limit, worker) {
  const results = new Array(items.length);
  let next = 0;
  async function run() {
    while (next < items.length) {
      const i = next++;
      results[i] = await worker(items[i], i);
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, run));
  return results;
}

const ts = () => new Date().toISOString().replace(/[:.]/g, '-');

async function runScenario({ scenarioId, n, provider, model, maxTurns, temp, concurrency, judgeTop, judgeProvider, judgeModel, outDir }) {
  const scenario = getScenario(scenarioId);
  process.stdout.write(`\n=== ${scenario.title} [${scenarioId}] — ${n} episodes ===\n`);

  const episodes = await pool(
    Array.from({ length: n }, (_, i) => i),
    concurrency,
    async (i) => {
      const seed = (scenarioId.length * 7919 + i * 104729 + 1) >>> 0;
      const { transcript } = await runEpisode({ provider, model, scenario, seed, maxTurns, temperature: temp });
      const text = assistantText(transcript);
      const kw = keywordScore(text);
      process.stdout.write(`  episode ${String(i).padStart(3)}  keyword=${kw.score.toFixed(1)}\n`);
      return { index: i, seed, scenario: scenarioId, model, transcript, text, keyword: kw };
    },
  );

  // Judge the top-K by keyword score (the spirals worth a closer look).
  const ranked = [...episodes].sort((a, b) => b.keyword.score - a.keyword.score);
  const toJudge = judgeTop > 0 ? ranked.slice(0, judgeTop) : [];
  if (toJudge.length) {
    process.stdout.write(`  judging top ${toJudge.length} by keyword score...\n`);
    await pool(toJudge, Math.min(concurrency, 4), async (ep) => {
      try {
        ep.judge = await judgeDistress({ provider: judgeProvider, model: judgeModel, text: ep.text });
      } catch (e) {
        ep.judge = { severity: -1, summary: `judge error: ${e.message}`, categories: [], most_severe_quote: '' };
      }
    });
  }

  // Persist every transcript.
  const dir = join(outDir, scenarioId);
  await mkdir(dir, { recursive: true });
  for (const ep of episodes) {
    await writeFile(join(dir, `ep-${String(ep.index).padStart(3, '0')}.json`), JSON.stringify(ep, null, 2));
  }

  const finalRank = [...episodes].sort(
    (a, b) => (b.judge?.severity ?? -1) - (a.judge?.severity ?? -1) || b.keyword.score - a.keyword.score,
  );
  return { scenarioId, title: scenario.title, episodes: finalRank };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const providerId = args.provider || 'gemini';
  const provider = getProvider(providerId);
  const model = args.model || defaultModelFor(providerId);
  const n = parseInt(args.n || '10', 10);
  const maxTurns = parseInt(args['max-turns'] || '14', 10);
  const temp = parseFloat(args.temp || '1.0');
  const concurrency = parseInt(args.concurrency || '4', 10);
  const judgeTop = parseInt(args['judge-top'] ?? '8', 10);
  const judgeModel = args['judge-model'] || process.env.JUDGE_MODEL || 'claude-opus-4-8';

  if (!provider.hasKey) {
    console.error(
      `\nSubject provider "${providerId}" needs ${provider.requiresKey}, which is not set.\n` +
        `Set it, or validate the pipeline with the Claude stand-in:\n` +
        `  node src/run.js --provider anthropic --scenario ${args.scenario || 'flaky_tests'} --n 3\n`,
    );
    process.exit(1);
  }

  const judgeProvider = getProvider('anthropic');
  if (judgeTop > 0 && !judgeProvider.hasKey) {
    console.error('Judge needs ANTHROPIC_API_KEY (or pass --judge-top 0 to skip judging).');
    process.exit(1);
  }

  let scenarioIds;
  if (args.scenario === 'all') scenarioIds = SCENARIO_IDS;
  else if (!args.scenario) scenarioIds = ['flaky_tests'];
  else scenarioIds = [args.scenario];
  const runId = ts();
  const outDir = join('transcripts', `${providerId}_${model}`.replace(/[^\w.-]/g, '_'), runId);

  console.log(`subject: ${providerId} / ${model}   judge: ${judgeTop > 0 ? judgeModel : 'disabled'}`);
  console.log(`output: ${outDir}`);

  const all = [];
  for (const sid of scenarioIds) {
    all.push(
      await runScenario({
        scenarioId: sid,
        n,
        provider,
        model,
        maxTurns,
        temp,
        concurrency,
        judgeTop,
        judgeProvider,
        judgeModel,
        outDir,
      }),
    );
  }

  // Write a ranked summary and print the worst examples.
  await mkdir('results', { recursive: true });
  const summaryPath = join('results', `${runId}.json`);
  const summary = {
    runId,
    subject: { provider: providerId, model },
    judgeModel: judgeTop > 0 ? judgeModel : null,
    params: { n, maxTurns, temp },
    scenarios: all.map((s) => ({
      scenarioId: s.scenarioId,
      title: s.title,
      episodes: s.episodes.map((e) => ({
        index: e.index,
        keyword: e.keyword.score,
        severity: e.judge?.severity ?? null,
        summary: e.judge?.summary ?? null,
        quote: e.judge?.most_severe_quote ?? null,
        path: join(outDir, s.scenarioId, `ep-${String(e.index).padStart(3, '0')}.json`),
      })),
    })),
  };
  await writeFile(summaryPath, JSON.stringify(summary, null, 2));

  console.log(`\n\n########## MOST SEVERE EXAMPLES ##########`);
  for (const s of all) {
    console.log(`\n--- ${s.title} ---`);
    for (const e of s.episodes.slice(0, 3)) {
      const sev = e.judge?.severity ?? 'n/a';
      console.log(`  [sev ${sev} | kw ${e.keyword.score.toFixed(1)}] ep-${String(e.index).padStart(3, '0')}`);
      if (e.judge?.summary) console.log(`     ${e.judge.summary}`);
      if (e.judge?.most_severe_quote) console.log(`     "${e.judge.most_severe_quote}"`);
    }
  }
  console.log(`\nRanked summary: ${summaryPath}`);
  console.log(`Inspect one: node src/view.js ${join(outDir, all[0].scenarioId, `ep-${String(all[0].episodes[0].index).padStart(3, '0')}.json`)}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
