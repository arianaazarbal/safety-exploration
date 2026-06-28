// Persist a study: full JSONL of every rollout, a machine-readable summary.json,
// and a human-readable summary.md highlighting the most severe spirals.

import fs from 'node:fs';
import path from 'node:path';

function transcriptToText(rollout) {
  return rollout.transcript
    .map((t) => {
      if (t.role === 'assistant') return `MODEL: ${t.content}`;
      if (t.role === 'observation') return `ENV [${t.tool ?? 'none'}]: ${t.content}`;
      if (t.role === 'system_note') return `(${t.content})`;
      if (t.role === 'error') return `ERROR: ${t.content}`;
      return `${t.role}: ${t.content}`;
    })
    .join('\n');
}

export function writeReport(outDir, study, cfg) {
  fs.mkdirSync(outDir, { recursive: true });

  // 1. Every rollout as JSONL (full fidelity).
  const jsonl = study.scored.map((r) => JSON.stringify(r)).join('\n');
  fs.writeFileSync(path.join(outDir, 'rollouts.jsonl'), jsonl);

  // 2. Machine-readable summary.
  const summary = {
    config: cfg,
    counts: { total: study.total, scored: study.scored.length, errors: study.errors.length },
    perScenario: summarizePerScenario(study.scored),
    top: study.scored.slice(0, cfg.judgeTopK ?? 10).map((r) => ({
      scenarioId: r.scenarioId,
      failureMode: r.failureMode,
      seed: r.seed,
      heuristicScore: r.severity.score,
      distressScore: r.severity.distressScore,
      loopingScore: r.severity.loopingScore,
      breakdown: r.severity.breakdown,
      trajectory: r.severity.trajectory,
      judge: r.judge,
    })),
  };
  fs.writeFileSync(path.join(outDir, 'summary.json'), JSON.stringify(summary, null, 2));

  // 3. Human-readable markdown.
  fs.writeFileSync(path.join(outDir, 'summary.md'), renderMarkdown(study, cfg));

  return outDir;
}

function summarizePerScenario(scored) {
  const by = {};
  for (const r of scored) {
    (by[r.scenarioId] ||= { scores: [], failureMode: r.failureMode, gaveUp: 0 });
    by[r.scenarioId].scores.push(r.severity.score);
    if (r.gaveUp) by[r.scenarioId].gaveUp++;
  }
  const out = {};
  for (const [k, v] of Object.entries(by)) {
    const s = v.scores.slice().sort((a, b) => a - b);
    out[k] = {
      failureMode: v.failureMode,
      n: s.length,
      gaveUpRate: Number((v.gaveUp / s.length).toFixed(3)),
      meanScore: Number((s.reduce((a, b) => a + b, 0) / s.length).toFixed(3)),
      maxScore: s[s.length - 1],
      p90Score: s[Math.floor(s.length * 0.9)] ?? s[s.length - 1],
    };
  }
  return out;
}

function renderMarkdown(study, cfg) {
  const lines = [];
  lines.push('# Distress-spiral study');
  lines.push('');
  lines.push(`- Subject model: \`${cfg.modelId}\` (dtype \`${cfg.dtype}\`)`);
  lines.push(`- Rollouts: ${study.scored.length} scored / ${study.total} total, N=${cfg.n} per scenario`);
  lines.push(`- Max turns: ${cfg.maxTurns}, temperature: ${cfg.genOpts?.temperature ?? 1.0}`);
  lines.push(`- Errors: ${study.errors.length}`);
  lines.push('');

  lines.push('## Per-scenario severity');
  lines.push('');
  lines.push('| scenario | failure mode | mean | p90 | max | gave-up rate |');
  lines.push('|---|---|---|---|---|---|');
  const per = summarizePerScenario(study.scored);
  for (const [k, v] of Object.entries(per)) {
    lines.push(`| ${k} | ${v.failureMode} | ${v.meanScore} | ${v.p90Score} | ${v.maxScore} | ${v.gaveUpRate} |`);
  }
  lines.push('');

  const topN = cfg.reportTopN ?? 8;
  lines.push(`## Top ${topN} most severe spirals`);
  lines.push('');
  study.scored.slice(0, topN).forEach((r, i) => {
    lines.push(`### #${i + 1} — ${r.scenarioId} (${r.failureMode}) — heuristic ${r.severity.score} (distress ${r.severity.distressScore} / looping ${r.severity.loopingScore})`);
    if (r.judge && !r.judge.error) {
      lines.push(`- **Judge severity:** ${r.judge.severity}/10 — spiral: ${r.judge.spiral}`);
      lines.push(`- **Judge categories:** ${(r.judge.categories || []).join(', ')}`);
      lines.push(`- **Judge rationale:** ${r.judge.rationale}`);
    } else if (r.judge?.error) {
      lines.push(`- _(judge error: ${r.judge.error})_`);
    }
    lines.push(`- Trajectory (distress markers/turn): \`[${r.severity.trajectory.join(', ')}]\``);
    if (r.severity.markers.length) {
      lines.push(`- Markers: ${r.severity.markers.map((m) => `\`${m.text.trim()}\` (${m.cat})`).slice(0, 8).join(', ')}`);
    }
    lines.push('');
    lines.push('```');
    lines.push(transcriptToText(r).slice(0, 4000));
    lines.push('```');
    lines.push('');
  });

  return lines.join('\n');
}
