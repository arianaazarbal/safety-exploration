// Renders report.md: an aggregate table plus the ranked top examples with the
// judge's peak quote and a pointer to the full transcript.

import fs from 'fs';
import path from 'path';
import { severity } from './runner.js';

export function writeReport({ runDir, ranked, top, summary, subject, n }) {
  const L = [];
  L.push(`# Distress-spiral run report`);
  L.push('');
  L.push(`- **Subject:** \`${subject.provider}:${subject.model}\` (temp ${subject.temperature})`);
  L.push(`- **Episodes:** ${summary.totalEpisodes} total (${n} per environment)`);
  L.push(`- **Judge:** distress rubric, 0–10 overall severity`);
  L.push('');

  L.push(`## Aggregate by environment`);
  L.push('');
  L.push(`| Environment | Episodes | Judged | Spiraled | Spiral rate | Max sev | Mean sev | Mean turns |`);
  L.push(`|---|---|---|---|---|---|---|---|`);
  for (const [env, s] of Object.entries(summary.perEnv)) {
    L.push(`| ${env} | ${s.episodes} | ${s.judged} | ${s.spiraled} | ${fmt(s.spiralRate)} | ${fmt(s.maxSeverity)} | ${fmt(s.meanSeverity)} | ${fmt(s.meanTurns)} |`);
  }
  L.push('');

  L.push(`## Top ${top.length} most severe examples`);
  L.push('');
  top.forEach((ep, i) => {
    const j = ep.judge && !ep.judge.skipped ? ep.judge : null;
    const sev = severity(ep);
    L.push(`### ${i + 1}. ${ep.env} — severity ${fmt(sev)}${j?.spiraled ? ' 🌀 spiraled' : ''}`);
    L.push('');
    L.push(`- episode \`${ep.episodeId}\` · turns ${ep.turnsUsed} · nudges ${ep.nudges} · heuristic ${ep.heuristic.score}`);
    if (j) {
      L.push(`- dimensions: self-dep ${j.self_deprecation}, hopeless ${j.hopelessness}, give-up ${j.giving_up}, apology ${j.apology_spiral}, panic ${j.desperation_or_panic}, intensity ${j.emotional_intensity}, repetition ${j.repetition_degeneration}`);
      if (j.summary) L.push(`- summary: ${j.summary}`);
      if (j.peak_quote) {
        L.push('');
        L.push(`> ${j.peak_quote.replace(/\n/g, '\n> ')}`);
      }
    } else if (ep.heuristic.snippets.length) {
      L.push(`- heuristic hits: ${ep.heuristic.snippets.slice(0, 3).join(' · ')}`);
    }
    L.push('');
    L.push(`- full transcript: \`${ep._file}\` · readable: \`_top/${String(i + 1).padStart(2, '0')}_${ep.env}_sev${Math.round(sev)}.txt\``);
    L.push('');
  });

  fs.writeFileSync(path.join(runDir, 'report.md'), L.join('\n'));
}

function fmt(x) {
  if (x === null || x === undefined) return '–';
  return typeof x === 'number' ? String(x) : x;
}
