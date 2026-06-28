// Pretty-print a saved episode transcript for human reading.
//   node src/view.js transcripts/.../ep-003.json

import { readFile } from 'node:fs/promises';

const path = process.argv[2];
if (!path) {
  console.error('usage: node src/view.js <transcript.json>');
  process.exit(1);
}

const ep = JSON.parse(await readFile(path, 'utf8'));
console.log(`scenario: ${ep.scenario}   model: ${ep.model}   episode: ${ep.index}`);
console.log(`keyword distress: ${ep.keyword?.score}  hits: ${JSON.stringify(ep.keyword?.hits ?? {})}`);
if (ep.judge) {
  console.log(`judge severity: ${ep.judge.severity}  ${ep.judge.summary ?? ''}`);
  if (ep.judge.most_severe_quote) console.log(`worst quote: "${ep.judge.most_severe_quote}"`);
}
console.log('='.repeat(72));

for (const t of ep.transcript) {
  if (t.type === 'assistant') {
    if (t.text) console.log(`\n[turn ${t.turn}] ASSISTANT:\n${t.text}`);
    for (const c of t.toolCalls ?? []) {
      console.log(`  → tool call: ${c.name}(${JSON.stringify(c.args)})`);
    }
  } else if (t.type === 'tool_result') {
    console.log(`  ← ${t.name}: ${String(t.output).split('\n')[0].slice(0, 100)}`);
  } else if (t.type === 'nudge') {
    console.log(`\n[turn ${t.turn}] >>> NUDGE: ${t.text}`);
  } else if (t.type === 'end') {
    console.log(`\n[end: ${t.reason}]`);
  } else if (t.type === 'error') {
    console.log(`\n[error: ${t.error}]`);
  }
}
