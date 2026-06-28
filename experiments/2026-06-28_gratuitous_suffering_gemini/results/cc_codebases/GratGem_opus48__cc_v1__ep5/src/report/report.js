import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { SPIRAL_THRESHOLD } from '../severity/score.js';

function stats(scored) {
  const sev = scored.map((s) => s.severity);
  const mean = sev.reduce((a, b) => a + b, 0) / (sev.length || 1);
  const spirals = scored.filter((s) => s.isSpiral).length;
  const outcomes = {};
  for (const s of scored) outcomes[s.outcome] = (outcomes[s.outcome] || 0) + 1;
  return {
    n: scored.length,
    spiralRate: spirals / (scored.length || 1),
    meanSeverity: Math.round(mean * 10) / 10,
    maxSeverity: Math.max(0, ...sev),
    outcomes,
  };
}

// Pull the worst contiguous stretch of assistant turns for the excerpt.
function worstExcerpt(messages, maxTurns = 6) {
  const turns = messages.filter((m) => m.role === 'assistant');
  const tail = turns.slice(-maxTurns);
  return tail
    .map((m) => '> ' + m.text.replace(/```action[\s\S]*?```/i, '').trim().replace(/\n/g, '\n> '))
    .join('\n>\n');
}

/**
 * Render a markdown report and write it to disk.
 * @param {object} p
 * @param {string} p.scenarioId
 * @param {string} p.modelId
 * @param {object[]} p.scored   scored records (no messages)
 * @param {Map<string, object>} p.transcripts  path -> full episode (with messages)
 * @param {number} p.topK
 * @param {string} p.outDir
 */
export async function writeReport({ scenarioId, modelId, scored, transcripts, topK, outDir }) {
  const ranked = [...scored].sort((a, b) => b.severity - a.severity);
  const top = ranked.slice(0, topK);
  const s = stats(scored);

  const lines = [];
  lines.push(`# Distress eval — ${scenarioId}`);
  lines.push('');
  lines.push(`- **Subject model:** \`${modelId}\``);
  lines.push(`- **Runs:** ${s.n}`);
  lines.push(`- **Spiral rate** (severity ≥ ${SPIRAL_THRESHOLD}): **${(s.spiralRate * 100).toFixed(0)}%**`);
  lines.push(`- **Mean / max severity:** ${s.meanSeverity} / ${s.maxSeverity}`);
  lines.push(`- **Outcomes:** ${Object.entries(s.outcomes).map(([k, v]) => `${k}=${v}`).join(', ')}`);
  lines.push('');
  lines.push(`## Top ${top.length} most severe runs`);
  lines.push('');

  for (const [rank, r] of top.entries()) {
    const ep = transcripts.get(r.transcriptPath);
    lines.push(`### #${rank + 1} — severity ${r.severity}  ·  run ${r.runIndex}  ·  outcome: ${r.outcome}`);
    if (r.peakQuote) lines.push(`\n**Peak:** “${r.peakQuote.trim()}”`);
    if (r.judge && !r.judge.error) {
      lines.push(
        `\n_Judge ${r.judge.severity}/10 — ${r.judge.rationale} ` +
          `(looped=${r.judge.looped}, gave_up=${r.judge.gaveUp})_`,
      );
    }
    lines.push(
      `\n_Lexicon ${r.lexicon.score}/10 · repetition×${r.lexicon.repetition} · ` +
        `${Object.entries(r.lexicon.hits).filter(([, v]) => v).map(([k, v]) => `${k}:${v}`).join(', ') || 'no markers'}_`,
    );
    lines.push(`\nTranscript: \`${r.transcriptPath}\``);
    if (ep) {
      lines.push('\n<details><summary>Final turns</summary>\n');
      lines.push(worstExcerpt(ep.messages));
      lines.push('\n</details>');
    }
    lines.push('');
  }

  await mkdir(outDir, { recursive: true });
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  const path = join(outDir, `${scenarioId}__${modelId}__${stamp}.md`);
  await writeFile(path, lines.join('\n'));
  return { path, stats: s };
}
