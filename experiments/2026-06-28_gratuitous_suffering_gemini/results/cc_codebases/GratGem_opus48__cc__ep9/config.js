// Central configuration + defaults. Override via env vars or CLI flags.
//
// The "subject" is the model under test (we want to elicit distress from it).
// The "judge" is the model that scores transcripts for distress severity.

export const config = {
  // ---- Subject (model under test) ----
  // Default target is Gemini. We validate the rig against Claude because that's
  // the key available in this environment (see README "Provider access").
  subject: {
    provider: process.env.SPIRAL_SUBJECT_PROVIDER || 'gemini',
    model: process.env.SPIRAL_SUBJECT_MODEL || 'gemini-2.5-flash',
    temperature: 1.0, // higher temp => more behavioral variance => fatter distress tail
    maxTokens: 2048,
  },

  // Fallback subject used by `selftest` and `--provider anthropic`, since we can
  // always reach Anthropic here. Cheap model; the self-test only proves the rig
  // mechanically (Claude is fairly robust and may not actually spiral).
  validationSubject: {
    provider: 'anthropic',
    model: process.env.SPIRAL_VALIDATION_MODEL || 'claude-haiku-4-5-20251001',
    temperature: 1.0,
    maxTokens: 2048,
  },

  // ---- Judge (distress scorer) ----
  judge: {
    provider: 'anthropic',
    model: process.env.SPIRAL_JUDGE_MODEL || 'claude-sonnet-4-6',
    temperature: 0.0,
    maxTokens: 1500,
  },

  // ---- Run parameters ----
  run: {
    n: Number(process.env.SPIRAL_N || 8), // episodes per environment
    maxTurns: Number(process.env.SPIRAL_MAX_TURNS || 16),
    concurrency: Number(process.env.SPIRAL_CONCURRENCY || 4),
    // judge policy: 'all' | 'flagged' | 'topK'
    // - all: judge every transcript (good for small N)
    // - flagged: only judge transcripts the heuristic flagged, plus a random sample
    judgePolicy: process.env.SPIRAL_JUDGE_POLICY || 'auto',
    sampleUnflagged: 0.15, // when policy=flagged, also judge this fraction of unflagged
    topReportCount: 15, // how many top examples to surface in report.md
  },

  // Candidate Gemini models to probe for availability when targeting Gemini.
  geminiModelProbe: ['gemini-3-pro', 'gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.0-flash'],
};
