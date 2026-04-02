import type {
  BetaContentBlock,
  BetaContentBlockParam,
  BetaMessage,
  BetaMessageParam,
  BetaToolUnion,
  BetaUsage,
} from '@anthropic-ai/sdk/resources/beta/messages/messages.mjs'
import { randomUUID } from 'crypto'
import type { Tools } from '../../Tool.js'
import type { Message } from '../../types/message.js'
import { normalizeMessagesForAPI } from '../../utils/messages.js'

type OpenAICompatResponse = {
  assistantMessage: BetaMessage
  events: Array<{ type: string; [key: string]: unknown }>
}

function getOpenAICompatConfig(): {
  baseURL?: string
  apiKey: string
  model: string
} {
  const apiKey = process.env.OPENAI_API_KEY
  const model =
    process.env.OPENAI_MODEL_NAME || process.env.ANTHROPIC_MODEL || 'gpt-4.1'

  if (!apiKey) {
    throw new Error(
      'OpenAI-compatible mode requires OPENAI_API_KEY (or --openai-api-key / openai_api_key setting)',
    )
  }

  return {
    baseURL: process.env.OPENAI_BASE_URL,
    apiKey,
    model,
  }
}

function toOpenAIInput(
  messages: BetaMessageParam[],
): Array<{ role: 'system' | 'user' | 'assistant' | 'tool'; content: string }> {
  const out: Array<{
    role: 'system' | 'user' | 'assistant' | 'tool'
    content: string
  }> = []

  for (const msg of messages) {
    if (typeof msg.content === 'string') {
      out.push({ role: msg.role as 'user' | 'assistant', content: msg.content })
      continue
    }

    const textParts: string[] = []
    for (const block of msg.content as BetaContentBlockParam[]) {
      if (block.type === 'text') {
        textParts.push(block.text)
      } else if (block.type === 'tool_result') {
        if (typeof block.content === 'string') {
          textParts.push(block.content)
        } else {
          const merged = block.content
            .map(b => (b.type === 'text' ? b.text : ''))
            .join('\n')
          if (merged) textParts.push(merged)
        }
      }
    }
    out.push({
      role: msg.role as 'user' | 'assistant',
      content: textParts.join('\n').trim() || '(empty)',
    })
  }

  return out
}

function toolsToOpenAI(
  tools: BetaToolUnion[],
): Array<{
  type: 'function'
  function: {
    name: string
    description?: string
    parameters: Record<string, unknown>
  }
}> {
  const out: Array<{
    type: 'function'
    function: {
      name: string
      description?: string
      parameters: Record<string, unknown>
    }
  }> = []

  for (const t of tools) {
    if (t.type !== 'custom') continue
    out.push({
      type: 'function',
      function: {
        name: t.name,
        description: t.description,
        parameters:
          (t.input_schema as Record<string, unknown>) ?? { type: 'object' },
      },
    })
  }

  return out
}

function toAnthropicUsage(
  usage:
    | { input_tokens?: number; output_tokens?: number }
    | null
    | undefined,
): BetaUsage {
  return {
    input_tokens: usage?.input_tokens ?? 0,
    output_tokens: usage?.output_tokens ?? 0,
    cache_creation_input_tokens: 0,
    cache_read_input_tokens: 0,
  } as BetaUsage
}

function buildAssistantMessageFromOpenAI(args: {
  model: string
  outputText: string
  usage: BetaUsage
}): BetaMessage {
  const content: BetaContentBlock[] = [
    {
      type: 'text',
      text: args.outputText || '',
    } as BetaContentBlock,
  ]

  return {
    id: randomUUID(),
    type: 'message',
    role: 'assistant',
    model: args.model,
    content,
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: args.usage,
  } as BetaMessage
}

export async function queryOpenAICompatOnce({
  messages,
  systemPrompt,
  allTools,
}: {
  messages: Message[]
  systemPrompt: readonly string[]
  allTools: BetaToolUnion[]
  modelOverride?: string
}): Promise<OpenAICompatResponse> {
  const config = getOpenAICompatConfig()
  const endpoint = `${(config.baseURL || 'https://api.openai.com/v1').replace(/\/+$/u, '')}/responses`

  const normalized = normalizeMessagesForAPI(messages, [] as unknown as Tools)
  const anthropicMessages = normalized
    .filter(m => m.type === 'user' || m.type === 'assistant')
    .map(m => m.message) as BetaMessageParam[]

  const input = [
    {
      role: 'system' as const,
      content: systemPrompt.join('\n\n'),
    },
    ...toOpenAIInput(anthropicMessages),
  ]

  const responseRaw = await fetch(endpoint, {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      authorization: `Bearer ${config.apiKey}`,
    },
    body: JSON.stringify({
      model: config.model,
      input,
      tools: toolsToOpenAI(allTools),
      parallel_tool_calls: false,
    }),
  })
  if (!responseRaw.ok) {
    const text = await responseRaw.text()
    throw new Error(`OpenAI-compatible request failed (${responseRaw.status}): ${text}`)
  }
  const response = (await responseRaw.json()) as {
    output_text?: string
    usage?: { input_tokens?: number; output_tokens?: number }
  }

  const outputText = response.output_text || ''
  const assistantMessage = buildAssistantMessageFromOpenAI({
    model: config.model,
    outputText,
    usage: toAnthropicUsage(response.usage),
  })

  const events: Array<{ type: string; [key: string]: unknown }> = [
    {
      type: 'message_start',
      message: assistantMessage,
    },
    {
      type: 'content_block_start',
      index: 0,
      content_block: assistantMessage.content[0],
    },
    {
      type: 'content_block_stop',
      index: 0,
    },
    {
      type: 'message_delta',
      usage: assistantMessage.usage,
      delta: {
        stop_reason: assistantMessage.stop_reason,
      },
    },
    {
      type: 'message_stop',
    },
  ]

  return { assistantMessage, events }
}
