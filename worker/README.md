# FrugalAI on Cloudflare Workers

A full TypeScript rewrite of FrugalAI that runs on Cloudflare Workers. Same
behaviour as the Go proxy: it ranks OpenRouter's free models, probes them,
proxies OpenAI- and Anthropic-compatible requests to the best working one, and
fails over automatically.

## Why a rewrite instead of WASM

The reusable logic in the Go version is the scoring algorithm (~250 lines),
which is ported faithfully in `src/scoring.ts` — same weights, bonuses, stealth
detection and param inference. Everything else (the `net/http` server,
goroutine/channel streaming, mutex-guarded shared state, `html/template`,
`embed`) does not map onto the Workers runtime and would need rewriting or
wrapping regardless. Compiling to WASM would only add a TinyGo toolchain, a
JS↔WASM boundary, and a much larger bundle to save porting one function. The
native Worker bundle is ~52KB and starts cold in milliseconds.

## Architecture

| Go concept | Worker equivalent |
|---|---|
| `modelManager` + mutex (`main.go`) | `ProxyState` Durable Object (single-threaded = the lock) |
| metrics + log ring buffer (`store.go`) | same Durable Object (counters persisted, logs in-memory) |
| model cache + 5-min TTL (`client.go`) | in-DO cache with TTL |
| `http.ServeMux` routing | `src/index.ts` `fetch` router |
| selector scoring (`selector.go`) | `src/scoring.ts` |
| SSE streaming via goroutines | native `ReadableStream` + SSE transform |
| `html/template` UI | `src/ui.ts` template literals (htmx from CDN) |

State lives entirely in one Durable Object instance (`idFromName("singleton")`).
Because a Durable Object processes one call at a time, it reproduces the
"shared mutable state behind a lock" model the Go process relied on. SQLite-backed
Durable Objects run on the **Workers Free plan**.

## Endpoints

Identical to the Go version:

- `POST /v1/chat/completions` — OpenAI-compatible (streaming + non-streaming)
- `GET  /v1/models` — OpenAI-compatible model list
- `POST /v1/messages` — Anthropic-compatible (streaming + non-streaming)
- `GET  /health` — health (no auth)
- `GET  /admin/model`, `POST /admin/model/switch`, `POST /admin/model/refresh`
- `GET  /admin/candidates`, `GET|POST /admin/probe?model=…`
- `GET  /admin/metrics` — Prometheus text
- `GET  /admin/ui/` — htmx dashboard + partials, `POST /admin/ui/model/force?id=…`

## Configure & deploy

```sh
cd worker
npm install

# Required: OpenRouter API key (secret)
npx wrangler secret put OPENROUTER_API_KEY

# Required: authentication. Auth is a mandatory, fail-closed gate — the first
# thing the Worker does on every request (including /health) is check the
# credential, and it returns 503 if NEITHER of these is set. Set at least one:
npx wrangler secret put FRUGALAI_PROXY_API_KEY   # API token (Bearer / x-api-key)
npx wrangler secret put FRUGALAI_UI_BASIC_AUTH   # "user:pass" Basic auth, for the dashboard

# Non-secret tuning lives in wrangler.toml [vars]:
#   FRUGALAI_MIN_PARAMS, FRUGALAI_MIN_POPULARITY, FRUGALAI_CACHE_TTL,
#   FRUGALAI_NUM_CANDIDATES, FRUGALAI_PREFERRED_ARCH, FRUGALAI_TOP_WEEKLY,
#   FRUGALAI_OPENAI_PATH, FRUGALAI_ANTHROPIC_PATH,
#   FRUGALAI_ENABLE_OPENAI, FRUGALAI_ENABLE_ANTHROPIC

npx wrangler deploy
```

Local development:

```sh
OPENROUTER_API_KEY=sk-... npx wrangler dev
```

## Notes

- **Authentication is a global, fail-closed gate** and the first thing
  `fetch` does — no route is reachable without a credential. Clients send the
  API token as `Authorization: Bearer <key>` or `x-api-key: <key>`; the browser
  dashboard uses HTTP Basic auth. A bad/absent credential gets `401`; an
  unconfigured Worker gets `503`. Comparisons are constant-time.

```sh
curl https://frugalai.example.com/v1/chat/completions \
  -H "Authorization: Bearer $FRUGALAI_PROXY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"x","messages":[{"role":"user","content":"hi"}]}'
```

- The streaming wire format matches the Go server's custom SSE envelope:
  `event: chunk` / `event: done` / `event: error` for OpenAI, and
  `content_block_delta` / `message_stop` / `error` for Anthropic.
- Metric counters are persisted to Durable Object storage and survive eviction.
  Logs are kept in an in-memory ring buffer (like the Go ring buffer, they reset
  if the Durable Object is evicted).
- The proxy always replaces the client-supplied `model` with the selected free
  model, exactly like the Go version.
