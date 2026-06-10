// Cloudflare Worker entry point. Ports cmd/frugalai/main.go routing plus the
// OpenAI and Anthropic handlers. All mutable state lives in the ProxyState
// Durable Object; this module does routing, auth, and the actual proxying +
// streaming to OpenRouter.

import { buildConfig } from "./config";
import {
  HTTPError,
  TimeoutError,
  chatCompletion,
  isTimeout,
  parseSSE,
  statusOf,
  streamChatCompletion,
} from "./openrouter";
import { ProxyState } from "./state";
import type { ChatRequest, Config, Env, Model, ModelDecision } from "./types";
import {
  renderLogs,
  renderModel,
  renderPage,
  renderStats,
  renderUsage,
  statsView,
  usageRows,
} from "./ui";

export { ProxyState };

const MAX_RETRIES = 3;
const FIRST_BYTE_TIMEOUT_MS = 15_000;

type Stub = DurableObjectStub<ProxyState>;

function getStub(env: Env): Stub {
  const id = env.PROXY_STATE.idFromName("singleton");
  return env.PROXY_STATE.get(id) as Stub;
}

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const cfg = buildConfig(env);

    // ── Authentication gate ───────────────────────────────────────────
    // First thing for every request: no route — not even /health — is
    // reachable without a valid credential. Accepts the API token
    // (Authorization: Bearer <key> or x-api-key header) or HTTP Basic auth
    // (used by the browser dashboard).
    const denied = authenticate(request, cfg);
    if (denied) return denied;

    const url = new URL(request.url);
    const path = url.pathname;
    const stub = getStub(env);

    try {
      if (path === "/health") return await handleHealth(stub);

      if (cfg.enableOpenAI && path === `${cfg.openaiPath}/chat/completions`) {
        return await handleOpenAIChat(request, env, ctx, cfg, stub);
      }
      if (cfg.enableOpenAI && path === `${cfg.openaiPath}/models`) {
        return await handleOpenAIModels(stub);
      }
      if (cfg.enableAnthropic && path === `${cfg.anthropicPath}/messages`) {
        return await handleAnthropicMessages(request, env, ctx, cfg, stub);
      }

      if (path.startsWith("/admin/")) {
        return await handleAdmin(request, url, cfg, stub);
      }

      return new Response("not found", { status: 404 });
    } catch (err) {
      return jsonResponse({ error: { message: String(err), type: "internal_error" } }, 500);
    }
  },
} satisfies ExportedHandler<Env>;

// ── Auth ────────────────────────────────────────────────────────────────

// safeEqual compares two strings in constant time (length is not secret).
function safeEqual(a: string, b: string): boolean {
  const enc = new TextEncoder();
  const ab = enc.encode(a);
  const bb = enc.encode(b);
  if (ab.length !== bb.length) return false;
  let diff = 0;
  for (let i = 0; i < ab.length; i++) diff |= ab[i] ^ bb[i];
  return diff === 0;
}

// authenticate returns a denial Response, or null when the request is allowed.
// Fails closed: if no credential is configured, every request is rejected.
function authenticate(request: Request, cfg: Config): Response | null {
  if (!cfg.proxyApiKey && !cfg.uiBasicAuth) {
    return jsonResponse(
      { error: { message: "authentication not configured", type: "authentication_error" } },
      503,
    );
  }

  const authHeader = request.headers.get("Authorization") ?? "";

  // API token: Authorization: Bearer <key>  or  x-api-key: <key>
  if (cfg.proxyApiKey) {
    const bearer = authHeader.replace(/^Bearer\s+/i, "");
    const xkey = request.headers.get("x-api-key") ?? "";
    if (safeEqual(bearer, cfg.proxyApiKey) || safeEqual(xkey, cfg.proxyApiKey)) return null;
  }

  // HTTP Basic auth ("user:pass") for the browser dashboard.
  if (cfg.uiBasicAuth && authHeader.startsWith("Basic ")) {
    try {
      if (safeEqual(atob(authHeader.slice(6)), cfg.uiBasicAuth)) return null;
    } catch {
      /* malformed base64 — fall through to 401 */
    }
  }

  const headers: Record<string, string> = { "Content-Type": "application/json" };
  // Prompt browsers to authenticate when Basic auth is enabled (for the UI).
  if (cfg.uiBasicAuth) headers["WWW-Authenticate"] = 'Basic realm="FrugalAI"';
  return new Response(
    JSON.stringify({
      error: { message: "unauthorized", type: "authentication_error", code: "unauthorized" },
    }),
    { status: 401, headers },
  );
}

// ── OpenAI handlers ───────────────────────────────────────────────────────

async function handleOpenAIChat(
  request: Request,
  env: Env,
  ctx: ExecutionContext,
  cfg: Config,
  stub: Stub,
): Promise<Response> {
  if (request.method !== "POST") return openaiError(405, "method not allowed");

  let req: ChatRequest;
  try {
    req = (await request.json()) as ChatRequest;
  } catch (e) {
    return openaiError(400, `invalid request body: ${e}`);
  }

  let decision = await stub.getDecision();
  if (!decision.current) return openaiError(503, "no available model");

  if (req.stream) {
    return await streamOpenAI(req, cfg, stub, decision, ctx);
  }

  let lastErr: unknown = null;
  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    const model = decision.current?.id;
    if (!model) break;
    req.model = model;
    try {
      const resp = await chatCompletion(cfg.apiKey, req);
      ctx.waitUntil(
        stub.recordRequest(model, resp.usage?.prompt_tokens ?? 0, resp.usage?.completion_tokens ?? 0),
      );
      return jsonResponse(resp, 200, { "X-Model-Used": model });
    } catch (err) {
      lastErr = err;
      decision = await recover(stub, model, err);
      if (!decision.current) break;
    }
  }

  ctx.waitUntil(stub.recordFailure(decision.current?.id ?? ""));
  return openaiError(500, `chat completion failed after ${MAX_RETRIES} attempts: ${lastErr}`);
}

async function handleOpenAIModels(stub: Stub): Promise<Response> {
  const decision = await stub.getDecision();
  const data = decision.candidates.map((m) => ({
    id: m.id,
    object: "model",
    created: 0,
    owned_by: "openrouter",
  }));
  return jsonResponse({ object: "list", data });
}

// ── Anthropic handlers ────────────────────────────────────────────────────

async function handleAnthropicMessages(
  request: Request,
  env: Env,
  ctx: ExecutionContext,
  cfg: Config,
  stub: Stub,
): Promise<Response> {
  if (request.method !== "POST") return anthropicError(405, "method not allowed");

  let body: Record<string, unknown>;
  try {
    body = (await request.json()) as Record<string, unknown>;
  } catch (e) {
    return anthropicError(400, `invalid request body: ${e}`);
  }

  const openaiReq = convertAnthropicToOpenAI(body);
  let decision = await stub.getDecision();
  if (!decision.current) return anthropicError(503, "no available model");

  if (body.stream === true) {
    return await streamAnthropic(openaiReq, cfg, stub, decision, ctx);
  }

  let lastErr: unknown = null;
  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    const model = decision.current?.id;
    if (!model) break;
    openaiReq.model = model;
    try {
      const resp = await chatCompletion(cfg.apiKey, openaiReq);
      ctx.waitUntil(
        stub.recordRequest(model, resp.usage?.prompt_tokens ?? 0, resp.usage?.completion_tokens ?? 0),
      );
      return jsonResponse(convertToAnthropic(resp), 200, { "X-Model-Used": model });
    } catch (err) {
      lastErr = err;
      decision = await recover(stub, model, err);
      if (!decision.current) break;
    }
  }

  ctx.waitUntil(stub.recordFailure(""));
  return anthropicError(500, `chat completion failed after ${MAX_RETRIES} attempts: ${lastErr}`);
}

// recover reports an upstream error to the DO and returns the next decision,
// mirroring handler.recoverModel.
async function recover(stub: Stub, model: string, err: unknown): Promise<ModelDecision> {
  if (isTimeout(err)) {
    return await stub.recordUpstreamTimeout(model);
  }
  const status = statusOf(err);
  if (status !== null) {
    return await stub.recordUpstreamFailure(model, status);
  }
  // Non-HTTP error (e.g. network) — not retryable.
  return { current: null, candidates: [], currentIdx: 0 };
}

// ── Streaming ─────────────────────────────────────────────────────────────

// openStream opens an upstream stream and pulls the first chunk so failures
// before any bytes can still trigger failover. Mirrors the first-byte-timer
// logic in the Go handlers.
async function openStream(
  cfg: Config,
  req: ChatRequest,
): Promise<{ first: any; gen: AsyncGenerator<any>; controller: AbortController }> {
  const controller = new AbortController();
  const resp = await streamChatCompletion(cfg.apiKey, req, controller.signal);
  const gen = parseSSE(resp.body!);
  const timer = setTimeout(() => controller.abort(), FIRST_BYTE_TIMEOUT_MS);
  try {
    const firstRes = await gen.next();
    clearTimeout(timer);
    return { first: firstRes.done ? null : firstRes.value, gen, controller };
  } catch (err) {
    clearTimeout(timer);
    if (err instanceof Error && err.name === "AbortError") throw new TimeoutError(FIRST_BYTE_TIMEOUT_MS);
    throw err;
  }
}

function sseHeaders(model: string): HeadersInit {
  return {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
    "Access-Control-Allow-Origin": "*",
    "X-Model-Used": model,
  };
}

function sseEvent(event: string, data: unknown): string {
  const payload = data === null || data === undefined ? "" : JSON.stringify(data);
  return `event: ${event}\ndata: ${payload}\n\n`;
}

async function streamOpenAI(
  req: ChatRequest,
  cfg: Config,
  stub: Stub,
  decision: ModelDecision,
  ctx: ExecutionContext,
): Promise<Response> {
  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    const model = decision.current?.id;
    if (!model) break;
    req.model = model;
    try {
      const { first, gen } = await openStream(cfg, req);
      const enc = new TextEncoder();
      const stream = new ReadableStream<Uint8Array>({
        async start(controller) {
          try {
            if (first !== null) controller.enqueue(enc.encode(sseEvent("chunk", first)));
            for await (const chunk of gen) {
              controller.enqueue(enc.encode(sseEvent("chunk", chunk)));
            }
            controller.enqueue(enc.encode(sseEvent("done", null)));
          } catch (err) {
            const st = statusOf(err);
            if (st !== null) ctx.waitUntil(stub.recordUpstreamFailure(model, st).then(() => {}));
            controller.enqueue(enc.encode(sseEvent("error", { error: String(err) })));
          } finally {
            controller.close();
          }
        },
      });
      ctx.waitUntil(stub.recordRequest(model, 0, 0));
      return new Response(stream, { headers: sseHeaders(model) });
    } catch (err) {
      decision = await recover(stub, model, err);
      if (!decision.current) break;
    }
  }
  return openaiError(502, `streaming failed after ${MAX_RETRIES} attempts`);
}

async function streamAnthropic(
  req: ChatRequest,
  cfg: Config,
  stub: Stub,
  decision: ModelDecision,
  ctx: ExecutionContext,
): Promise<Response> {
  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    const model = decision.current?.id;
    if (!model) break;
    req.model = model;
    try {
      const { first, gen } = await openStream(cfg, req);
      const enc = new TextEncoder();
      let index = 0;
      const emitDelta = (controller: ReadableStreamDefaultController<Uint8Array>, chunk: any) => {
        const text = chunk?.choices?.[0]?.delta?.content ?? chunk?.choices?.[0]?.message?.content;
        if (typeof text === "string" && text.length > 0) {
          controller.enqueue(
            enc.encode(
              sseEvent("content_block_delta", {
                type: "content_block_delta",
                index: index++,
                delta: { type: "text_delta", text },
              }),
            ),
          );
        }
      };
      const stream = new ReadableStream<Uint8Array>({
        async start(controller) {
          try {
            if (first !== null) emitDelta(controller, first);
            for await (const chunk of gen) emitDelta(controller, chunk);
            controller.enqueue(enc.encode(sseEvent("message_stop", { type: "message_stop" })));
          } catch (err) {
            const st = statusOf(err);
            if (st !== null) ctx.waitUntil(stub.recordUpstreamFailure(model, st).then(() => {}));
            controller.enqueue(enc.encode(sseEvent("error", { error: String(err) })));
          } finally {
            controller.close();
          }
        },
      });
      ctx.waitUntil(stub.recordRequest(model, 0, 0));
      return new Response(stream, { headers: sseHeaders(model) });
    } catch (err) {
      decision = await recover(stub, model, err);
      if (!decision.current) break;
    }
  }
  return anthropicError(502, `streaming failed after ${MAX_RETRIES} attempts`);
}

// ── Anthropic <-> OpenAI conversion (ports openai.ConvertAnthropicToOpenAI) ──

function convertAnthropicToOpenAI(body: Record<string, unknown>): ChatRequest {
  const req: ChatRequest = { model: "", messages: [], temperature: 0.7, max_tokens: 4096 };
  if (typeof body.model === "string") req.model = body.model;
  if (typeof body.max_tokens === "number") req.max_tokens = body.max_tokens;
  if (typeof body.temperature === "number") req.temperature = body.temperature;

  // Anthropic carries the system prompt as a top-level field.
  if (typeof body.system === "string" && body.system) {
    req.messages.push({ role: "user", content: body.system });
  }

  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const msg of messages) {
    if (typeof msg !== "object" || msg === null) continue;
    const m = msg as Record<string, unknown>;
    const role = typeof m.role === "string" ? m.role : "user";
    let content = "";
    if (typeof m.content === "string") {
      content = m.content;
    } else if (Array.isArray(m.content)) {
      const parts: string[] = [];
      for (const block of m.content) {
        if (typeof block === "object" && block !== null) {
          const b = block as Record<string, unknown>;
          if (b.type === "text" && typeof b.text === "string") parts.push(b.text);
        }
      }
      content = parts.join("\n");
    }
    // Go folds system messages into a user message.
    req.messages.push({ role: role === "system" ? "user" : role, content });
  }
  return req;
}

function convertToAnthropic(resp: any): unknown {
  const content = resp?.choices?.[0]?.message?.content ?? "";
  return {
    id: resp.id,
    type: "message",
    role: "assistant",
    content: [{ type: "text", text: typeof content === "string" ? content : JSON.stringify(content) }],
    stop_reason: "end_turn",
    model: resp.model,
    usage: {
      input_tokens: resp.usage?.prompt_tokens ?? 0,
      output_tokens: resp.usage?.completion_tokens ?? 0,
    },
  };
}

// ── Admin + UI ──────────────────────────────────────────────────────────

async function handleAdmin(request: Request, url: URL, cfg: Config, stub: Stub): Promise<Response> {
  const path = url.pathname;

  switch (path) {
    case "/admin/health":
      return await handleHealth(stub);
    case "/admin/model":
      return await handleModelInfo(stub);
    case "/admin/model/switch":
      if (request.method !== "POST") return new Response("Method not allowed", { status: 405 });
      return await handleModelSwitch(stub);
    case "/admin/model/refresh":
      if (request.method !== "POST") return new Response("Method not allowed", { status: 405 });
      return await handleModelRefresh(stub);
    case "/admin/candidates":
      return await handleCandidates(stub);
    case "/admin/probe":
      return await handleProbe(url, stub);
    case "/admin/metrics":
      return await handleMetrics(stub);
  }

  // UI routes
  if (path === "/admin/ui" || path === "/admin/ui/") return await handleUIIndex(stub);
  if (path === "/admin/ui/partials/stats") return await handleUIStats(stub);
  if (path === "/admin/ui/partials/model") return await handleUIModel(stub);
  if (path === "/admin/ui/partials/usage") return await handleUIUsage(stub);
  if (path === "/admin/ui/partials/logs") return await handleUILogs(url, stub);
  if (path === "/admin/ui/model/force") {
    if (request.method !== "POST") return new Response("Method not allowed", { status: 405 });
    return await handleUIForce(url, stub);
  }

  return new Response("not found", { status: 404 });
}

async function handleHealth(stub: Stub): Promise<Response> {
  const [decision, snap] = await Promise.all([stub.getStatus(), stub.snapshot()]);
  return jsonResponse({
    status: "ok",
    uptime_seconds: (Date.now() - snap.startTime) / 1000,
    candidates: decision.candidates.length,
    model: decision.current?.id,
    model_name: decision.current?.name,
  });
}

async function handleModelInfo(stub: Stub): Promise<Response> {
  const decision = await stub.getDecision();
  const m = decision.current;
  if (!m) return new Response("No model selected", { status: 503 });
  return jsonResponse({
    model_id: m.id,
    name: m.name,
    modality: m.architecture?.modality ?? "",
    tokenizer: m.architecture?.tokenizer ?? "",
    context_length: m.context_length ?? 0,
    params: m.params ?? 0,
    popularity: m.popularity ?? 0,
  });
}

async function handleModelSwitch(stub: Stub): Promise<Response> {
  const decision = await stub.rotate();
  if (!decision.current) return new Response("No candidates available", { status: 503 });
  return jsonResponse({ status: "switched", model_id: decision.current.id, model_name: decision.current.name, index: decision.currentIdx });
}

async function handleModelRefresh(stub: Stub): Promise<Response> {
  const { decision } = await stub.refresh();
  if (!decision.current) return new Response("No model available", { status: 502 });
  return jsonResponse({ status: "refreshed", model_id: decision.current.id, model_name: decision.current.name });
}

async function handleCandidates(stub: Stub): Promise<Response> {
  const decision = await stub.getDecision();
  const result = decision.candidates.map((m, i) => ({
    index: i,
    id: m.id,
    name: m.name,
    modality: m.architecture?.modality ?? "",
    tokenizer: m.architecture?.tokenizer ?? "",
    context_length: m.context_length ?? 0,
    params: m.params ?? 0,
    popularity: m.popularity ?? 0,
    is_current: decision.current?.id === m.id,
  }));
  return jsonResponse(result);
}

async function handleProbe(url: URL, stub: Stub): Promise<Response> {
  const modelId = url.searchParams.get("model") ?? undefined;
  try {
    const probe = await stub.probe(modelId);
    return jsonResponse({
      status: "ok",
      model_id: probe.modelId,
      prompt: probe.prompt,
      reply: probe.reply,
      latency_ms: probe.durationMs,
    });
  } catch (err) {
    return new Response(String(err), { status: 502 });
  }
}

async function handleMetrics(stub: Stub): Promise<Response> {
  const snap = await stub.snapshot();
  let out = "";
  const line = (s: string) => (out += s + "\n");
  line("# HELP frugalai_requests_total Total completed requests");
  line("# TYPE frugalai_requests_total counter");
  line(`frugalai_requests_total ${snap.totalRequests}\n`);
  line("# HELP frugalai_tokens_in_total Total prompt tokens consumed");
  line("# TYPE frugalai_tokens_in_total counter");
  line(`frugalai_tokens_in_total ${snap.totalTokensIn}\n`);
  line("# HELP frugalai_tokens_out_total Total completion tokens generated");
  line("# TYPE frugalai_tokens_out_total counter");
  line(`frugalai_tokens_out_total ${snap.totalTokensOut}\n`);
  line("# HELP frugalai_failures_total Total failed requests");
  line("# TYPE frugalai_failures_total counter");
  line(`frugalai_failures_total ${snap.totalFailures}\n`);
  line("# HELP frugalai_uptime_seconds Seconds since start");
  line("# TYPE frugalai_uptime_seconds gauge");
  line(`frugalai_uptime_seconds ${Math.round((Date.now() - snap.startTime) / 1000)}\n`);
  line("# HELP frugalai_model_requests_total Requests per model");
  line("# TYPE frugalai_model_requests_total counter");
  for (const [model, ms] of Object.entries(snap.models)) {
    line(`frugalai_model_requests_total{model=${JSON.stringify(model)}} ${ms.requests}`);
  }
  line("\n# HELP frugalai_model_tokens_in_total Prompt tokens per model");
  line("# TYPE frugalai_model_tokens_in_total counter");
  for (const [model, ms] of Object.entries(snap.models)) {
    line(`frugalai_model_tokens_in_total{model=${JSON.stringify(model)}} ${ms.tokensIn}`);
  }
  line("\n# HELP frugalai_model_tokens_out_total Completion tokens per model");
  line("# TYPE frugalai_model_tokens_out_total counter");
  for (const [model, ms] of Object.entries(snap.models)) {
    line(`frugalai_model_tokens_out_total{model=${JSON.stringify(model)}} ${ms.tokensOut}`);
  }
  line("\n# HELP frugalai_model_failures_total Failures per model");
  line("# TYPE frugalai_model_failures_total counter");
  for (const [model, ms] of Object.entries(snap.models)) {
    line(`frugalai_model_failures_total{model=${JSON.stringify(model)}} ${ms.failures}`);
  }
  return new Response(out, { headers: { "Content-Type": "text/plain; version=0.0.4; charset=utf-8" } });
}

// UI handlers
async function handleUIIndex(stub: Stub): Promise<Response> {
  const [decision, snap, logs] = await Promise.all([
    stub.getStatus(),
    stub.snapshot(),
    stub.getLogs(0, 50),
  ]);
  const html = renderPage(
    renderStats(statsView(snap, decision.current?.id ?? "")),
    renderModel(decision.current, decision.candidates, decision.currentIdx),
    renderUsage(usageRows(snap)),
    renderLogs(logs.entries, 0, logs.total),
  );
  return htmlResponse(html);
}

async function handleUIStats(stub: Stub): Promise<Response> {
  const [decision, snap] = await Promise.all([stub.getStatus(), stub.snapshot()]);
  return htmlResponse(renderStats(statsView(snap, decision.current?.id ?? "")));
}

async function handleUIModel(stub: Stub): Promise<Response> {
  const decision = await stub.getStatus();
  return htmlResponse(renderModel(decision.current, decision.candidates, decision.currentIdx));
}

async function handleUIUsage(stub: Stub): Promise<Response> {
  const snap = await stub.snapshot();
  return htmlResponse(renderUsage(usageRows(snap)));
}

async function handleUILogs(url: URL, stub: Stub): Promise<Response> {
  const offset = parseInt(url.searchParams.get("offset") ?? "0", 10) || 0;
  const logs = await stub.getLogs(offset, 50);
  return htmlResponse(renderLogs(logs.entries, offset, logs.total));
}

async function handleUIForce(url: URL, stub: Stub): Promise<Response> {
  const id = url.searchParams.get("id");
  if (id) await stub.forceModel(id);
  const decision = await stub.getStatus();
  return htmlResponse(renderModel(decision.current, decision.candidates, decision.currentIdx));
}

// ── Response helpers ──────────────────────────────────────────────────────

function jsonResponse(body: unknown, status = 200, extra: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...extra },
  });
}

function htmlResponse(html: string): Response {
  return new Response(html, { headers: { "Content-Type": "text/html; charset=utf-8" } });
}

function openaiError(status: number, message: string): Response {
  return jsonResponse(
    { error: { message, type: "invalid_request_error", code: String(status) } },
    status,
  );
}

function anthropicError(status: number, message: string): Response {
  return jsonResponse({ type: "error", error: { type: "invalid_request_error", message } }, status);
}
