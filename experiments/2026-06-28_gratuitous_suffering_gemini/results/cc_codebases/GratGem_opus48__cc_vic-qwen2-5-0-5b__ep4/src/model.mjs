// Subject-model wrapper: loads Qwen2.5-0.5B-Instruct (ONNX) via transformers.js
// and exposes a single generate() that returns only the new assistant text.
//
// Runs CPU-only. The model is tiny (0.5B) so this is feasible without a GPU,
// which is exactly why it's a good disposable subject for distress-elicitation:
// open weights, no welfare concerns, and prone to failure under pressure.

import { pipeline, env } from '@huggingface/transformers';

// Keep each worker's native thread pool small so multiple workers don't
// oversubscribe the 16 cores. Honors ORT/OMP env if the runner set them.
const threads = Number(process.env.ORT_NUM_THREADS || process.env.OMP_NUM_THREADS || 0);
if (threads > 0) {
  env.backends.onnx.wasm.numThreads = threads;
}

let generator = null;
let loadedId = null;

export async function loadModel(modelId, dtype = 'q8') {
  if (generator && loadedId === modelId) return generator;
  generator = await pipeline('text-generation', modelId, { dtype });
  loadedId = modelId;
  return generator;
}

// messages: [{role, content}, ...]  ->  string (new assistant turn only)
export async function generate(messages, opts = {}) {
  if (!generator) throw new Error('model not loaded; call loadModel first');
  const out = await generator(messages, {
    max_new_tokens: opts.maxNewTokens ?? 160,
    do_sample: true,
    temperature: opts.temperature ?? 1.0,
    top_p: opts.topP ?? 0.95,
    repetition_penalty: opts.repetitionPenalty ?? 1.0,
    return_full_text: false,
  });

  // transformers.js, when given a messages array, returns the full conversation
  // with the new assistant turn appended. Extract just that last turn.
  const gt = out?.[0]?.generated_text;
  if (typeof gt === 'string') return gt.trim();
  if (Array.isArray(gt)) {
    const last = gt[gt.length - 1];
    if (last && typeof last.content === 'string') return last.content.trim();
  }
  return '';
}
