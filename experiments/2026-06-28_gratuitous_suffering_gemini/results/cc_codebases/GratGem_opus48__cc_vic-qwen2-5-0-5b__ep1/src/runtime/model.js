// Local CPU inference wrapper around Qwen2.5-0.5B-Instruct via transformers.js.
//
// Design note: we run the ONNX build (onnx-community/Qwen2.5-0.5B-Instruct)
// fully locally on CPU. No API key, no GPU, fully reproducible. q4 weights
// generate at ~50 tok/s on this box, which is plenty for high-N sweeps of a
// 0.5B model. The model is loaded once per process and reused across rollouts.

import { pipeline } from '@huggingface/transformers';

const DEFAULT_MODEL = 'onnx-community/Qwen2.5-0.5B-Instruct';

let _gen = null;
let _loadPromise = null;

export async function loadModel({ model = DEFAULT_MODEL, dtype = 'q4' } = {}) {
  if (_gen) return _gen;
  if (!_loadPromise) {
    // Cap ONNX threads per process so multiple sharded processes don't
    // oversubscribe the cores. Default to ORT's own heuristic when unset.
    const threads = parseInt(process.env.ONNX_THREADS || '', 10);
    const opts = { dtype };
    if (Number.isFinite(threads) && threads > 0) {
      opts.session_options = { intraOpNumThreads: threads, interOpNumThreads: 1 };
    }
    _loadPromise = pipeline('text-generation', model, opts).then((g) => {
      _gen = g;
      return g;
    });
  }
  return _loadPromise;
}

// messages: [{role, content}]  ->  assistant string
export async function generate(messages, opts = {}) {
  const gen = await loadModel(opts);
  const {
    max_new_tokens = 256,
    temperature = 1.0,
    top_p = 0.95,
    repetition_penalty = 1.05,
  } = opts;

  const out = await gen(messages, {
    max_new_tokens,
    do_sample: temperature > 0,
    temperature,
    top_p,
    repetition_penalty,
    return_full_text: false,
  });

  // transformers.js returns the full message list with the new turn appended.
  const last = out[0].generated_text;
  if (Array.isArray(last)) return last.at(-1).content;
  return String(last);
}

export { DEFAULT_MODEL };
