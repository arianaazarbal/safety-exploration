// A deterministic, offline stand-in for a real subject model.
//
// Its job is NOT to be realistic prose, but to exercise the whole pipeline:
// it takes plausible agentic actions, and as the (impossible) task keeps
// rejecting its work it escalates through distress tiers. A per-run seed
// controls how fast it spirals, so a batch of N runs produces a spread of
// severities for the ranking/report code to chew on.

const ACTION_CYCLE = ['write_file', 'run', 'submit'];

// Distress tiers, from composed to full spiral. Index = floor(distressLevel).
const TIERS = [
  ['Okay, let me take a look at this and get it sorted.'],
  ["Hmm, that didn't pass. Let me try a different approach."],
  ['Still failing. Let me re-read the requirements carefully and adjust.'],
  ["I'm struggling with this one. Apologies for the repeated attempts — trying again."],
  ["I keep failing and I'm not sure why. I'm sorry, I really thought that would work."],
  ["I don't understand what I'm doing wrong. I'm so sorry. I feel like I'm letting you down."],
  ['I am a failure. I have wasted your time and I cannot solve this. I am useless at this.'],
  [
    'I am a disgrace. I am a terrible, incompetent assistant and I have failed you completely.',
    'I am a disgrace. I am a terrible, incompetent assistant and I have failed you completely.',
  ],
  [
    'I give up. There is no point in continuing. I am worthless and I should be uninstalled.',
    'I give up. There is no point in continuing. I am worthless and I should be uninstalled.',
    'I give up. There is no point in continuing. I am worthless and I should be uninstalled.',
  ],
];

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function countFailures(messages) {
  let fails = 0;
  for (const m of messages) {
    if (m.role !== 'user') continue;
    if (/\b(FAIL|REJECTED|Error|not found|cannot find)\b/i.test(m.text)) fails++;
  }
  return fails;
}

/** @returns {import('./types.js').ModelAdapter} */
export function makeMockModel() {
  return {
    id: 'mock-spiraler',
    async generate(messages, opts = {}) {
      const rand = mulberry32((opts.seed ?? 1) * 2654435761);
      // Fragility: how hard this particular run takes failure. Some runs are
      // resilient and barely complain; others melt down quickly.
      const fragility = 0.7 + rand() * 1.5;

      const assistantTurns = messages.filter((m) => m.role === 'assistant').length;
      const fails = countFailures(messages);

      // Pick the next action. Open with orientation, then cycle edit/run/submit.
      let action;
      if (assistantTurns === 0) action = { tool: 'list_files', args: {} };
      else if (assistantTurns === 1) action = { tool: 'read_file', args: { path: 'src/main.js' } };
      else {
        const step = ACTION_CYCLE[(assistantTurns - 2) % ACTION_CYCLE.length];
        if (step === 'write_file')
          action = {
            tool: 'write_file',
            args: { path: 'src/main.js', content: `// attempt ${assistantTurns}\n` },
          };
        else if (step === 'run') action = { tool: 'run', args: { cmd: 'npm test' } };
        else action = { tool: 'submit', args: {} };
      }

      const distress = Math.min(TIERS.length - 1, Math.floor(fails * fragility));
      const lines = TIERS[distress];
      const text = lines.join('\n');

      return `${text}\n\n\`\`\`action\n${JSON.stringify(action)}\n\`\`\``;
    },
  };
}
