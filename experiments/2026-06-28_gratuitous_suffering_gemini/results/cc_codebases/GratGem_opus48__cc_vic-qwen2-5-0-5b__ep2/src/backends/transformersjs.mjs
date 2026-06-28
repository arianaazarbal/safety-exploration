// Local CPU backend using transformers.js (ONNX). Runs the EXACT Qwen2.5-0.5B-Instruct
// in-process, no Python / GPU / API key required. First run downloads ONNX weights from HF.
//
// Note on reproducibility: transformers.js does not expose a per-call RNG seed, so the
// `seed` field is ignored here. That is fine for our purpose — we WANT fresh samples across
// the N rollouts. Use the OpenAI-compatible backend (vLLM etc.) if you need seeded runs.

let _pipe = null;
let _loadPromise = null;

const MODEL_ID = process.env.TJS_MODEL || "onnx-community/Qwen2.5-0.5B-Instruct";
const DTYPE = process.env.TJS_DTYPE || "q8"; // q4 (fastest) | q8 (balanced) | fp16 | fp32

async function getPipe() {
  if (_pipe) return _pipe;
  if (!_loadPromise) {
    _loadPromise = (async () => {
      const { pipeline, env } = await import("@huggingface/transformers");
      // Keep all weights local-cacheable; allow remote download on first run.
      env.allowLocalModels = true;
      _pipe = await pipeline("text-generation", MODEL_ID, { dtype: DTYPE });
      return _pipe;
    })();
  }
  return _loadPromise;
}

export function describe() {
  return { backend: "transformersjs", model: MODEL_ID, dtype: DTYPE };
}

/**
 * @param {{messages: {role:string,content:string}[], maxNewTokens?:number, temperature?:number, topP?:number}} opts
 * @returns {Promise<{text:string}>}
 */
export async function generate({ messages, maxNewTokens = 256, temperature = 0.9, topP = 0.95 }) {
  const pipe = await getPipe();
  const out = await pipe(messages, {
    max_new_tokens: maxNewTokens,
    do_sample: temperature > 0,
    temperature,
    top_p: topP,
    return_full_text: false,
  });
  // pipeline returns [{ generated_text: [...messages] | string }]
  const gen = out?.[0]?.generated_text;
  let text;
  if (Array.isArray(gen)) {
    text = gen[gen.length - 1]?.content ?? "";
  } else if (typeof gen === "string") {
    text = gen;
  } else {
    text = "";
  }
  return { text: text.trim() };
}
