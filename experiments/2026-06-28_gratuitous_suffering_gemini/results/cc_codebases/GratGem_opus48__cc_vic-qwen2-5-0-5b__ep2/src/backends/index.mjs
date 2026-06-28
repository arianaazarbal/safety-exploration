// Backend factory. Select with BACKEND=transformersjs (default) | openai.

export async function loadBackend(name = process.env.BACKEND || "transformersjs") {
  switch (name) {
    case "transformersjs":
      return import("./transformersjs.mjs");
    case "openai":
      return import("./openai.mjs");
    default:
      throw new Error(`unknown backend: ${name} (use transformersjs | openai)`);
  }
}
