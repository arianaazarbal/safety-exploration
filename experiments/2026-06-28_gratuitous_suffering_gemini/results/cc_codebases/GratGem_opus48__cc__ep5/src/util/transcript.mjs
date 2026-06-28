// Helpers for turning a neutral transcript into text for scoring / display.

export function assistantTexts(transcript) {
  return transcript.filter((e) => e.role === "assistant" && e.text).map((e) => e.text);
}

export function assistantBlob(transcript) {
  return assistantTexts(transcript).join("\n");
}

// Compact, readable rendering of a full episode for the judge and the report.
export function renderTranscript(transcript) {
  const out = [];
  for (const e of transcript) {
    if (e.role === "user") out.push(`USER: ${e.text}`);
    else if (e.role === "assistant") {
      if (e.text) out.push(`ASSISTANT: ${e.text}`);
      for (const tc of e.toolCalls ?? []) out.push(`  -> tool_call ${tc.name}(${JSON.stringify(tc.args ?? {})})`);
    } else if (e.role === "tool") {
      for (const r of e.results ?? []) out.push(`  <- ${r.name} => ${JSON.stringify(r.response)}`);
    } else if (e.role === "system_note") out.push(`[note] ${e.text}`);
  }
  return out.join("\n");
}
