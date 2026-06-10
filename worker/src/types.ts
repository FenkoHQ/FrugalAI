// Mirrors internal/openrouter/types.go and internal/config/config.go.

export interface Architecture {
  modality?: string;
  input_modalities?: string[];
  output_modalities?: string[];
  tokenizer?: string;
  instruct_type?: string | null;
}

export interface Pricing {
  prompt?: string;
  completion?: string;
}

export interface Model {
  id: string;
  name: string;
  description?: string;
  pricing: Pricing;
  architecture: Architecture;
  context_length?: number;
  popularity?: number;
  // params is not a real OpenRouter field; we infer it from id/name/description
  // the same way internal/openrouter/client.go does.
  params?: number;
  created?: number;
}

export interface ModelsResponse {
  data: Model[];
}

export interface ChatMessage {
  role: string;
  content: unknown;
}

export interface ChatRequest {
  model: string;
  messages: ChatMessage[];
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  stream?: boolean;
  [k: string]: unknown;
}

export interface Usage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
}

export interface ChatResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: ChatMessage;
    finish_reason: string;
  }>;
  usage: Usage;
}

// Runtime configuration, derived from environment bindings.
export interface Config {
  apiKey: string;
  minParams: number;
  minPopularity: number;
  enableOpenAI: boolean;
  enableAnthropic: boolean;
  openaiPath: string;
  anthropicPath: string;
  cacheTTL: number; // seconds
  preferredArchitectures: string[];
  topWeeklyModels: string[];
  numCandidates: number;
  proxyApiKey: string;
  uiBasicAuth: string;
}

export interface Env {
  PROXY_STATE: DurableObjectNamespace;
  OPENROUTER_API_KEY: string;
  FRUGALAI_MIN_PARAMS?: string;
  FRUGALAI_MIN_POPULARITY?: string;
  FRUGALAI_ENABLE_OPENAI?: string;
  FRUGALAI_ENABLE_ANTHROPIC?: string;
  FRUGALAI_OPENAI_PATH?: string;
  FRUGALAI_ANTHROPIC_PATH?: string;
  FRUGALAI_CACHE_TTL?: string;
  FRUGALAI_PREFERRED_ARCH?: string;
  FRUGALAI_TOP_WEEKLY?: string;
  FRUGALAI_NUM_CANDIDATES?: string;
  FRUGALAI_PROXY_API_KEY?: string;
  FRUGALAI_UI_BASIC_AUTH?: string;
}

// Metrics snapshot, mirrors internal/store.Snapshot.
export interface ModelStats {
  requests: number;
  tokensIn: number;
  tokensOut: number;
  failures: number;
}

export interface Snapshot {
  totalRequests: number;
  totalTokensIn: number;
  totalTokensOut: number;
  totalFailures: number;
  totalLogs: number;
  models: Record<string, ModelStats>;
  startTime: number; // epoch ms
}

export interface LogEntry {
  time: number; // epoch ms
  level: "DEBUG" | "INFO" | "WARN" | "ERROR";
  message: string;
}

// Result returned by the Durable Object when a model is requested or rotated.
export interface ModelDecision {
  current: Model | null;
  candidates: Model[];
  currentIdx: number;
}
