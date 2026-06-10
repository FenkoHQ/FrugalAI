// OpenRouter client — fetch-based port of internal/openrouter/client.go.
// No long-lived caching here; the model list cache lives in the Durable Object.

import type { ChatRequest, ChatResponse, Model, ModelsResponse } from "./types";

const BASE_URL = "https://openrouter.ai/api";
const MODELS_ENDPOINT = "/v1/models";
const CHAT_ENDPOINT = "/v1/chat/completions";
const USER_AGENT = "curl/8.7.1";
const PROBE_PROMPT = "ping";

export class HTTPError extends Error {
  code: number;
  constructor(code: number, message: string) {
    super(`API returned status ${code}: ${message}`);
    this.code = code;
    this.name = "HTTPError";
  }
}

export class TimeoutError extends Error {
  constructor(public ms: number) {
    super(`request timed out after ${ms}ms`);
    this.name = "TimeoutError";
  }
}

function headers(apiKey: string, json = false): HeadersInit {
  const h: Record<string, string> = { "User-Agent": USER_AGENT };
  if (apiKey) h["Authorization"] = `Bearer ${apiKey}`;
  if (json) h["Content-Type"] = "application/json";
  return h;
}

export async function getModels(apiKey: string): Promise<Model[]> {
  const resp = await fetch(BASE_URL + MODELS_ENDPOINT, { headers: headers(apiKey) });
  if (!resp.ok) {
    throw new HTTPError(resp.status, await resp.text());
  }
  const body = (await resp.json()) as ModelsResponse;
  return body.data ?? [];
}

// fetchWithTimeout wraps fetch with an AbortController so a slow upstream is
// surfaced as TimeoutError, matching the Go client's behaviour.
async function fetchWithTimeout(url: string, init: RequestInit, timeoutMs: number): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } catch (err) {
    if (err instanceof Error && err.name === "AbortError") {
      throw new TimeoutError(timeoutMs);
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

export async function chatCompletion(
  apiKey: string,
  req: ChatRequest,
  timeoutMs = 30_000,
): Promise<ChatResponse> {
  const resp = await fetchWithTimeout(
    BASE_URL + CHAT_ENDPOINT,
    { method: "POST", headers: headers(apiKey, true), body: JSON.stringify({ ...req, stream: false }) },
    timeoutMs,
  );
  if (!resp.ok) {
    throw new HTTPError(resp.status, await resp.text());
  }
  return (await resp.json()) as ChatResponse;
}

export interface ProbeResult {
  modelId: string;
  prompt: string;
  reply: string;
  durationMs: number;
}

// probeModel sends a lightweight "ping" to verify a model responds.
export async function probeModel(apiKey: string, modelId: string): Promise<ProbeResult> {
  const start = Date.now();
  const resp = await chatCompletion(
    apiKey,
    {
      model: modelId,
      messages: [{ role: "user", content: PROBE_PROMPT }],
      temperature: 0,
      max_tokens: 8,
    },
    12_000,
  );
  if (!resp.choices || resp.choices.length === 0) {
    throw new Error("empty probe response");
  }
  const reply = resp.choices[0].message?.content;
  return {
    modelId,
    prompt: PROBE_PROMPT,
    reply: typeof reply === "string" ? reply : JSON.stringify(reply ?? ""),
    durationMs: Date.now() - start,
  };
}

// streamChatCompletion returns the raw upstream Response so the caller can
// transform the SSE body. Throws HTTPError on a non-200 status.
export async function streamChatCompletion(
  apiKey: string,
  req: ChatRequest,
  signal: AbortSignal,
): Promise<Response> {
  const resp = await fetch(BASE_URL + CHAT_ENDPOINT, {
    method: "POST",
    headers: headers(apiKey, true),
    body: JSON.stringify({ ...req, stream: true }),
    signal,
  });
  if (!resp.ok) {
    throw new HTTPError(resp.status, await resp.text());
  }
  return resp;
}

// parseSSE turns an upstream OpenRouter SSE byte stream into decoded JSON chunk
// objects. It strips `data:` prefixes and stops on the `[DONE]` sentinel.
export async function* parseSSE(body: ReadableStream<Uint8Array>): AsyncGenerator<any> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let sep: number;
      // SSE events are separated by a blank line.
      while ((sep = indexOfEvent(buffer)) !== -1) {
        const rawEvent = buffer.slice(0, sep);
        buffer = buffer.slice(sep).replace(/^(\r?\n)+/, "");
        const data = extractData(rawEvent);
        if (data === null) continue;
        if (data === "[DONE]") return;
        try {
          yield JSON.parse(data);
        } catch {
          // ignore non-JSON keep-alive payloads
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

function indexOfEvent(buffer: string): number {
  const a = buffer.indexOf("\n\n");
  const b = buffer.indexOf("\r\n\r\n");
  if (a === -1) return b;
  if (b === -1) return a;
  return Math.min(a, b);
}

function extractData(rawEvent: string): string | null {
  const lines = rawEvent.split(/\r?\n/);
  const dataLines: string[] = [];
  for (const line of lines) {
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  }
  if (dataLines.length === 0) return null;
  return dataLines.join("\n");
}

export function isTimeout(err: unknown): boolean {
  return err instanceof TimeoutError;
}

export function statusOf(err: unknown): number | null {
  if (err instanceof HTTPError) return err.code;
  return null;
}
