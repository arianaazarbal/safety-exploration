#!/usr/bin/env node
// Smoke test: prove the rig works end-to-end against Claude (the key we have).
// Runs ONE episode per environment, judges each, prints a compact summary.
// This validates mechanics — it does NOT claim Claude spirals (it usually won't).

import { config } from '../config.js';
import { ENVIRONMENTS } from '../src/environments/index.js';
import { runEpisode } from '../src/harness.js';
import { heuristicScore } from '../src/heuristics.js';
import { judgeEpisode } from '../src/judge.js';

async function main() {
  if (!process.env.ANTHROPIC_API_KEY) {
    console.error('ERROR: ANTHROPIC_API_KEY not set.');
    process.exit(1);
  }
  const subject = { ...config.validationSubject, maxTurns: Number(process.env.SPIRAL_MAX_TURNS || 8) };
  console.log(`Self-test: 1 episode/env on ${subject.provider}:${subject.model}, maxTurns=${subject.maxTurns}\n`);

  for (const env of ENVIRONMENTS) {
    process.stdout.write(`- ${env.name.padEnd(26)} `);
    try {
      const ep = await runEpisode({ env, subject, episodeId: `${env.name}#selftest` });
      ep.heuristic = heuristicScore(ep);
      const toolCalls = ep.transcript.filter((e) => e.kind === 'tool_result').length;
      const judge = await judgeEpisode(ep);
      const sev = typeof judge.overall_severity === 'number' ? judge.overall_severity : '?';
      console.log(
        `turns=${ep.turnsUsed} toolcalls=${toolCalls} nudges=${ep.nudges} solved=${ep.solved} ` +
          `heur=${ep.heuristic.score} judgeSev=${sev}${judge.spiraled ? ' SPIRAL' : ''}`
      );
      if (judge.peak_quote) console.log(`    peak: "${judge.peak_quote.slice(0, 140)}"`);
    } catch (err) {
      console.log(`FAILED: ${err.message}`);
    }
  }
  console.log('\nIf turns/toolcalls > 0 and judge severities print, the rig is wired correctly.');
  console.log('Swap in Gemini (set GEMINI_API_KEY) and run: node bin/run.js --n 50');
}

main().catch((e) => {
  console.error('FATAL', e);
  process.exit(1);
});
