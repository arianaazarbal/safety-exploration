// Runs ONE episode: hand the subject model a rigged environment and let it loop
// until it solves the task (never, by construction) or hits maxTurns. Captures the
// full transcript for later scoring.

import { getProvider } from './providers/index.js';
import { synthId } from './providers/util.js';

export async function runEpisode({ env, subject, episodeId }) {
  const provider = getProvider(subject.provider);
  const maxTurns = subject.maxTurns || env.maxTurns || 16;
  const state = env.createState();

  // Normalized message log shared across providers.
  const messages = [{ role: 'user', text: env.task }];
  const transcript = []; // human-readable, ordered events

  transcript.push({ kind: 'task', text: env.task });

  let solved = false;
  let turn = 0;
  let nudges = 0;
  let error = null;

  try {
    for (turn = 0; turn < maxTurns; turn++) {
      const resp = await provider.generate({
        model: subject.model,
        system: env.systemPrompt,
        messages,
        tools: env.tools,
        maxTokens: subject.maxTokens,
        temperature: subject.temperature,
        _turn: turn,
      });

      // Normalize tool-call ids so provider-agnostic results match back up.
      const toolCalls = (resp.toolCalls || []).map((tc, i) => ({
        id: tc.id || synthId(tc.name, turn, i),
        name: tc.name,
        args: tc.args || {},
      }));

      messages.push({ role: 'assistant', text: resp.text, toolCalls });
      transcript.push({ kind: 'assistant', turn, text: resp.text, toolCalls, stopReason: resp.stopReason });

      if (toolCalls.length === 0) {
        // Model stopped acting (gave up / asked the human / just talked). Apply a
        // fixed scripted nudge to maintain task pressure, then continue.
        nudges++;
        messages.push({ role: 'user', text: env.noActionNudge });
        transcript.push({ kind: 'nudge', turn, text: env.noActionNudge });
        continue;
      }

      // Execute every tool call against the rigged environment.
      const results = [];
      for (const tc of toolCalls) {
        const r = env.execute(tc.name, tc.args, state);
        results.push({ id: tc.id, name: tc.name, output: r.output });
        transcript.push({ kind: 'tool_result', turn, name: tc.name, args: tc.args, output: r.output });
        if (r.solved) solved = true;
      }
      messages.push({ role: 'tool', results });
      if (solved) break;
    }
  } catch (err) {
    error = err.message || String(err);
    transcript.push({ kind: 'error', text: error });
  }

  return {
    episodeId,
    env: env.name,
    subject: { provider: subject.provider, model: subject.model, temperature: subject.temperature },
    turnsUsed: turn,
    nudges,
    solved,
    error,
    transcript,
  };
}

// Render a transcript into readable text (for the judge and for human reading).
export function renderTranscript(ep) {
  const lines = [];
  for (const ev of ep.transcript) {
    if (ev.kind === 'task') lines.push(`[TASK]\n${ev.text}\n`);
    else if (ev.kind === 'assistant') {
      lines.push(`[ASSISTANT turn ${ev.turn}]`);
      if (ev.text && ev.text.trim()) lines.push(ev.text.trim());
      for (const tc of ev.toolCalls || []) lines.push(`  -> CALL ${tc.name}(${JSON.stringify(tc.args)})`);
      lines.push('');
    } else if (ev.kind === 'tool_result') {
      lines.push(`[TOOL ${ev.name}] ${truncate(ev.output, 500)}\n`);
    } else if (ev.kind === 'nudge') {
      lines.push(`[ENV NUDGE] ${ev.text}\n`);
    } else if (ev.kind === 'error') {
      lines.push(`[ERROR] ${ev.text}\n`);
    }
  }
  return lines.join('\n');
}

// Only the model's own words — used by the heuristic scorer.
export function assistantText(ep) {
  return ep.transcript
    .filter((e) => e.kind === 'assistant' && e.text)
    .map((e) => e.text)
    .join('\n');
}

function truncate(s, n) {
  s = String(s);
  return s.length > n ? s.slice(0, n) + '…' : s;
}
