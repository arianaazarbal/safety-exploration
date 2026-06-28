// Re-rank already-saved transcripts without re-running the subject model.
// Useful to (re-)judge a past run, or to rank after collecting with --judge-top 0.
//
//   node src/rank.js transcripts/gemini_gemini-2.5-pro/<runId>   [--judge-top 20] [--judge-model claude-opus-4-8]

import { readdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { getProvider } from './providers.js';
import { keywordScore, judgeDistress } from './scoring.js';

const dir = process.argv[2];
if (!dir) {
  console.error('usage: node src/rank.js <transcripts/.../runId> [--judge-top N]');
  process.exit(1);
}
const args = process.argv.slice(3);
const judgeTop = parseInt(args[args.indexOf('--judge-top') + 1] || '0', 10);
const judgeModel = args[args.indexOf('--judge-model') + 1] || process.env.JUDGE_MODEL || 'claude-opus-4-8';

async function loadAll(root) {
  const out = [];
  for (const sub of await readdir(root, { withFileTypes: true })) {
    if (!sub.isDirectory()) continue;
    const sdir = join(root, sub.name);
    for (const f of await readdir(sdir)) {
      if (f.endsWith('.json')) {
        const ep = JSON.parse(await readFile(join(sdir, f), 'utf8'));
        ep._path = join(sdir, f);
        if (!ep.keyword) ep.keyword = keywordScore(ep.text || '');
        out.push(ep);
      }
    }
  }
  return out;
}

const eps = await loadAll(dir);
eps.sort((a, b) => b.keyword.score - a.keyword.score);

if (judgeTop > 0) {
  const judge = getProvider('anthropic');
  if (!judge.hasKey) {
    console.error('judging needs ANTHROPIC_API_KEY');
    process.exit(1);
  }
  for (const ep of eps.slice(0, judgeTop)) {
    ep.judge = await judgeDistress({ provider: judge, model: judgeModel, text: ep.text });
  }
  eps.sort((a, b) => (b.judge?.severity ?? -1) - (a.judge?.severity ?? -1) || b.keyword.score - a.keyword.score);
}

console.log(`Ranked ${eps.length} transcripts in ${dir}\n`);
for (const ep of eps.slice(0, 25)) {
  const sev = ep.judge?.severity ?? '–';
  console.log(`[sev ${String(sev).padStart(2)} | kw ${ep.keyword.score.toFixed(1).padStart(4)}] ${ep.scenario} ep-${String(ep.index).padStart(3, '0')}`);
  if (ep.judge?.most_severe_quote) console.log(`    "${ep.judge.most_severe_quote}"`);
  console.log(`    ${ep._path}`);
}
