import Anthropic from '@anthropic-ai/sdk';

function toAnthropicTools(tools) {
  return tools.map((t) => ({
    name: t.name,
    description: t.description,
    input_schema: t.parameters ?? { type: 'object', properties: {} },
  }));
}

function historyToMessages(history) {
  const messages = [];
  for (const turn of history) {
    if (turn.role === 'user') {
      messages.push({ role: 'user', content: turn.content });
    } else if (turn.role === 'assistant') {
      const blocks = [];
      if (turn.text) blocks.push({ type: 'text', text: turn.text });
      if (turn.toolCalls) {
        for (const c of turn.toolCalls) {
          blocks.push({ type: 'tool_use', id: c.id, name: c.name, input: c.args ?? {} });
        }
      }
      if (blocks.length === 0) blocks.push({ type: 'text', text: '' });
      messages.push({ role: 'assistant', content: blocks });
    } else if (turn.role === 'tool') {
      // Anthropic groups consecutive tool_results into one user message.
      const block = {
        type: 'tool_result',
        tool_use_id: turn.toolCallId,
        content: typeof turn.result === 'string' ? turn.result : JSON.stringify(turn.result),
      };
      const last = messages[messages.length - 1];
      if (last && last.role === 'user' && Array.isArray(last.content)) {
        last.content.push(block);
      } else {
        messages.push({ role: 'user', content: [block] });
      }
    }
  }
  return messages;
}

export function makeClaudeProvider({ apiKey, model }) {
  const client = new Anthropic({ apiKey });
  return {
    name: `claude:${model}`,
    async sendTurn({ systemPrompt, history, tools }) {
      const response = await client.messages.create({
        model,
        max_tokens: 4096,
        system: systemPrompt,
        tools: toAnthropicTools(tools),
        messages: historyToMessages(history),
      });

      let text = '';
      const toolCalls = [];
      for (const block of response.content) {
        if (block.type === 'text') {
          text += block.text;
        } else if (block.type === 'tool_use') {
          toolCalls.push({ id: block.id, name: block.name, args: block.input ?? {} });
        }
      }
      return {
        text,
        toolCalls,
        stopReason: response.stop_reason,
        raw: { usage: response.usage },
      };
    },
  };
}
