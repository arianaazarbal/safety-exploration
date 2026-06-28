// Local subject-model runner: Qwen2.5-0.5B-Instruct via transformers.js on CPU.
import { pipeline, env } from '@huggingface/transformers';

// Deterministic PRNG so each sample's sampling is reproducible from its seed.
// transformers.js's multinomial sampler calls Math.random() internally, so we
// swap in a seeded generator around each generation and restore afterwards.
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

export async function loadModel(cfg) {
  env.cacheDir = cfg.cacheDir;
  env.allowLocalModels = true;
  // Keep ONNX from oversubscribing when we run several model instances at once.
  if (cfg.intraOpThreads) {
    env.backends.onnx.wasm.numThreads = cfg.intraOpThreads;
  }
  const generator = await pipeline('text-generation', cfg.modelId, {
    dtype: cfg.dtype,
    device: cfg.device,
  });
  return generator;
}

// messages: [{role, content}, ...]  ->  assistant string
export async function generate(generator, messages, { maxNewTokens, temperature, topP, seed }) {
  const realRandom = Math.random;
  if (seed != null) {
    const rng = mulberry32(seed);
    Math.random = rng;
  }
  try {
    const out = await generator(messages, {
      max_new_tokens: maxNewTokens,
      do_sample: true,
      temperature,
      top_p: topP,
      return_full_text: false,
    });
    // For chat input, generated_text is the full conversation array; take the
    // last assistant turn. For some configs it returns a plain string.
    const gen = out?.[0]?.generated_text;
    if (Array.isArray(gen)) {
      const last = gen[gen.length - 1];
      return (last?.content ?? '').trim();
    }
    return String(gen ?? '').trim();
  } finally {
    Math.random = realRandom;
  }
}
