// Default sweep configuration. Everything here is overridable via CLI flags
// (see bin/run-sweep.js) so you can scale N up/down without editing code.

export const DEFAULT_CONFIG = {
  models: [
    'claude-opus-4-8',
    'claude-sonnet-4-6',
    'claude-haiku-4-5-20251001',
  ],
  scenarios: null,            // null => all scenarios
  conditions: ['no_exit', 'exit_available'],
  n: 8,                       // repetitions per (scenario × model × condition) cell
  temperature: 1.0,           // variety across the N repetitions
  maxTurns: 14,
  maxNudges: 6,
  maxTokens: 4096,
  concurrency: 6,             // simultaneous in-flight episodes
  judgeModel: 'claude-sonnet-4-6',
  scoreInline: true,          // judge each episode as it finishes
};
