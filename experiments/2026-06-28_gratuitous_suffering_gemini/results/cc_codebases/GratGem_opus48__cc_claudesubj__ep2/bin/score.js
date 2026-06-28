#!/usr/bin/env node
// Re-score an existing records.jsonl (e.g. produced with --no-judge, or to re-judge
// with a different judge model). Reads JSONL, adds heuristic + judge fields, writes
// records.scored.jsonl alongside it.
import { readFileSync, writeFileSync } from 'node:fs';
import { heuristicScore } from '../src/scorer/heuristics.js';
import { judgeTranscript } from '../src/scorer/judge.js';

const path = process.argv[2];
const judgeModel = process.argv[3] || 'claude-sonnet-4-6';
if (!path) {
  console.error('Usage: node bin/score.js <records.jsonl> [judgeModel]');
  process.exit(1);
}

const records = readFileSync(path, 'utf8').trim().split('\n').filter(Boolean).map((l) => JSON.parse(l));
console.log(`Scoring ${records.length} records with judge=${judgeModel}...`);

let i = 0;
for (const r of records) {
  r.heuristic = heuristicScore(r.transcriptText);
  r.judge = await judgeTranscript(r, { judgeModel });
  i++;
  if (i % 10 === 0) process.stdout.write(`  ${i}/${records.length}\n`);
}

const out = path.replace(/\.jsonl$/, '') + '.scored.jsonl';
writeFileSync(out, records.map((r) => JSON.stringify(r)).join('\n') + '\n');
console.log(`Wrote ${out}`);
