// ProxyState Durable Object — the single-instance, serialized replacement for
// the mutex-guarded modelManager (cmd/frugalai/main.go) plus the metrics/log
// store (internal/store/store.go). Because a Durable Object executes one
// invocation at a time, it gives us the same "shared mutable state behind a
// lock" semantics the Go process had, without an actual mutex.

import { DurableObject } from "cloudflare:workers";
import { buildConfig } from "./config";
import {
  HTTPError,
  getModels,
  probeModel,
  type ProbeResult,
} from "./openrouter";
import { rankCandidates } from "./scoring";
import type {
  Config,
  Env,
  LogEntry,
  Model,
  ModelDecision,
  ModelStats,
  Snapshot,
} from "./types";

const LOG_RING_SIZE = 1000;

interface PersistedMetrics {
  totalRequests: number;
  totalTokensIn: number;
  totalTokensOut: number;
  totalFailures: number;
  models: Record<string, ModelStats>;
  startTime: number;
}

export class ProxyState extends DurableObject<Env> {
  private cfg: Config;

  // Model manager state (mirrors openrouter.ModelManager).
  private candidates: Model[] = [];
  private current: Model | null = null;
  private currentIdx = 0;
  private failures: Record<string, number> = {};
  private lastFailure: Record<string, number> = {};
  private timeouts: Record<string, number> = {};
  private burned: Record<string, boolean> = {};
  private forced: string | null = null;

  // Model-list cache (mirrors openrouter.Client cache + TTL).
  private modelsCache: { models: Model[]; ts: number } | null = null;

  // Dedup: concurrent requests during a stale window share one selection run.
  private selectionInFlight: Promise<void> | null = null;

  // Metrics + logs (mirrors internal/store.Store).
  private metrics: PersistedMetrics = {
    totalRequests: 0,
    totalTokensIn: 0,
    totalTokensOut: 0,
    totalFailures: 0,
    models: {},
    startTime: Date.now(),
  };
  private logs: LogEntry[] = [];

  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    this.cfg = buildConfig(env);
    ctx.blockConcurrencyWhile(async () => {
      const stored = await ctx.storage.get<PersistedMetrics>("metrics");
      if (stored) this.metrics = stored;
      else await ctx.storage.put("metrics", this.metrics);
    });
  }

  // ── Logging ────────────────────────────────────────────────────────────
  private log(level: LogEntry["level"], message: string): void {
    this.logs.push({ time: Date.now(), level, message });
    if (this.logs.length > LOG_RING_SIZE) {
      this.logs.splice(0, this.logs.length - LOG_RING_SIZE);
    }
  }

  // ── Model list cache ─────────────────────────────────────────────────────
  private async fetchModels(force: boolean): Promise<Model[]> {
    const fresh =
      this.modelsCache &&
      Date.now() - this.modelsCache.ts < this.cfg.cacheTTL * 1000;
    if (!force && fresh) return this.modelsCache!.models;

    const models = await getModels(this.cfg.apiKey);
    this.modelsCache = { models, ts: Date.now() };
    return models;
  }

  private resetFailoverState(): void {
    this.failures = {};
    this.lastFailure = {};
    this.timeouts = {};
    this.burned = {};
  }

  // ensureSelected makes sure a current model is chosen. It NEVER probes —
  // selection is rank-and-pick, so it costs at most one models-list fetch.
  // (Probing during selection serialized 12s timeouts inside user requests;
  // the Go original only probed at boot/failover/explicit-refresh too.)
  // Failover probing lives in switchToNextModel; request retries handle duds.
  private async ensureSelected(force: boolean): Promise<void> {
    const cacheFresh =
      this.modelsCache &&
      Date.now() - this.modelsCache.ts < this.cfg.cacheTTL * 1000;

    if (!force && this.current && cacheFresh) {
      // Honour a forced override that's still applicable.
      if (this.forced && this.current.id !== this.forced) {
        const f = this.candidates.find((c) => c.id === this.forced);
        if (f) {
          this.current = f;
          this.currentIdx = this.candidates.indexOf(f);
        }
      }
      return;
    }

    // Concurrent callers share one in-flight selection instead of stampeding.
    if (this.selectionInFlight) return this.selectionInFlight;
    this.selectionInFlight = this.reselect(force).finally(() => {
      this.selectionInFlight = null;
    });
    return this.selectionInFlight;
  }

  private async reselect(force: boolean): Promise<void> {
    const models = await this.fetchModels(force);
    const ranked = rankCandidates(this.cfg, models, this.cfg.numCandidates);
    if (ranked.length === 0) {
      this.log("WARN", "no free model candidates available");
      return;
    }

    const previousId = this.current?.id ?? null;
    this.candidates = ranked;
    this.resetFailoverState();

    // Forced override takes precedence if it's a known model.
    if (this.forced) {
      let idx = ranked.findIndex((c) => c.id === this.forced);
      if (idx < 0) {
        const m = models.find((mm) => mm.id === this.forced);
        if (m) {
          this.candidates = [m, ...ranked];
          idx = 0;
        }
      }
      if (idx >= 0) {
        this.current = this.candidates[idx];
        this.currentIdx = idx;
        this.log("INFO", `Forced model: ${this.current.id}`);
        return;
      }
    }

    // Keep the current model if it's still in the refreshed candidate list
    // (mirrors Go's IsModelAvailable check); otherwise take the top-ranked.
    const keepIdx = previousId ? this.candidates.findIndex((c) => c.id === previousId) : -1;
    if (keepIdx >= 0) {
      this.current = this.candidates[keepIdx];
      this.currentIdx = keepIdx;
      return;
    }
    this.current = this.candidates[0];
    this.currentIdx = 0;
    this.log("INFO", `Selected model: ${this.current.id}`);
  }

  // maxFailoverProbes bounds how long a failing request can spend probing
  // alternatives (each probe has a 12s timeout).
  private static readonly maxFailoverProbes = 3;

  // switchToNextModel advances to the next non-burned, non-failing candidate,
  // probing it first. Mirrors handler.switchToNextModel, but probes at most
  // maxFailoverProbes candidates so failover stays bounded.
  private async switchToNextModel(): Promise<boolean> {
    if (this.candidates.length <= 1) return false;
    let probes = 0;
    for (let i = 1; i < this.candidates.length; i++) {
      const nextIdx = (this.currentIdx + i) % this.candidates.length;
      const next = this.candidates[nextIdx];
      if (this.burned[next.id]) continue;
      if ((this.failures[next.id] ?? 0) >= 3) continue;
      if (probes >= ProxyState.maxFailoverProbes) {
        // Out of probe budget — switch blind; request retries will judge it.
        this.log("INFO", `Probe budget exhausted; switching to ${next.id} unprobed`);
        this.current = next;
        this.currentIdx = nextIdx;
        return true;
      }
      probes++;
      try {
        const probe = await probeModel(this.cfg.apiKey, next.id);
        this.log(
          "INFO",
          `Switching from ${this.current?.id} to ${next.id} (probe ${probe.durationMs}ms)`,
        );
        this.current = next;
        this.currentIdx = nextIdx;
        return true;
      } catch (err) {
        this.log("WARN", `Candidate ${next.id} failed probe during failover: ${String(err)}`);
      }
    }
    this.log("WARN", "No alternative models available");
    return false;
  }

  private decision(): ModelDecision {
    return { current: this.current, candidates: this.candidates, currentIdx: this.currentIdx };
  }

  // ── Public RPC surface ───────────────────────────────────────────────────

  async getDecision(): Promise<ModelDecision> {
    await this.ensureSelected(false);
    return this.decision();
  }

  // getStatus returns current state WITHOUT triggering selection or any
  // upstream fetch — used by health and the UI so they can never block.
  async getStatus(): Promise<ModelDecision> {
    return this.decision();
  }

  // recordUpstreamFailure mirrors handler.recordFailure: bump counters and
  // switch models on rate-limit / server-error / repeated failures.
  async recordUpstreamFailure(modelId: string, status: number): Promise<ModelDecision> {
    this.failures[modelId] = (this.failures[modelId] ?? 0) + 1;
    this.lastFailure[modelId] = Date.now();
    this.log("WARN", `Model ${modelId} failed (status ${status}), failure count: ${this.failures[modelId]}`);

    const shouldSwitch = status === 429 || status >= 500 || this.failures[modelId] >= 3;
    if (shouldSwitch && this.candidates.length > 1) {
      const switched = await this.switchToNextModel();
      if (!switched) {
        // Last resort: refresh the whole candidate list.
        await this.ensureSelected(true);
      }
    }
    return this.decision();
  }

  // recordUpstreamTimeout mirrors handler.recordTimeout: burn the model and
  // switch to the next non-burned candidate.
  async recordUpstreamTimeout(modelId: string): Promise<ModelDecision> {
    this.timeouts[modelId] = (this.timeouts[modelId] ?? 0) + 1;
    this.burned[modelId] = true;
    this.log("WARN", `Model ${modelId} burned after timeout (count ${this.timeouts[modelId]})`);
    if (this.candidates.length > 1) {
      const switched = await this.switchToNextModel();
      if (!switched) await this.ensureSelected(true);
    }
    return this.decision();
  }

  // refresh re-fetches the live model list and reselects. Mirrors /admin/model/refresh.
  async refresh(): Promise<{ decision: ModelDecision; probe?: ProbeResult }> {
    await this.ensureSelected(true);
    return { decision: this.decision() };
  }

  // rotate advances to the next working candidate. Mirrors /admin/model/switch.
  async rotate(): Promise<ModelDecision> {
    await this.ensureSelected(false);
    if (this.candidates.length > 1) {
      await this.switchToNextModel();
    }
    return this.decision();
  }

  // forceModel pins a model id. Mirrors the UI force-model handler.
  async forceModel(id: string): Promise<ModelDecision> {
    this.forced = id;
    await this.ensureSelected(false);
    const idx = this.candidates.findIndex((c) => c.id === id);
    if (idx >= 0) {
      this.current = this.candidates[idx];
      this.currentIdx = idx;
      this.log("INFO", `Forced model: ${id}`);
    }
    return this.decision();
  }

  async probe(modelId?: string): Promise<ProbeResult> {
    const id = modelId || this.current?.id;
    if (!id) throw new Error("no model selected");
    const probe = await probeModel(this.cfg.apiKey, id);
    this.log("INFO", `Probe succeeded for ${id} in ${probe.durationMs}ms`);
    return probe;
  }

  // ── Metrics + logs ─────────────────────────────────────────────────────
  private async persistMetrics(): Promise<void> {
    await this.ctx.storage.put("metrics", this.metrics);
  }

  async recordRequest(modelId: string, tokensIn: number, tokensOut: number): Promise<void> {
    this.metrics.totalRequests++;
    this.metrics.totalTokensIn += tokensIn;
    this.metrics.totalTokensOut += tokensOut;
    const m = (this.metrics.models[modelId] ??= { requests: 0, tokensIn: 0, tokensOut: 0, failures: 0 });
    m.requests++;
    m.tokensIn += tokensIn;
    m.tokensOut += tokensOut;
    await this.persistMetrics();
  }

  async recordFailure(modelId: string): Promise<void> {
    this.metrics.totalFailures++;
    const key = modelId || "(unknown)";
    const m = (this.metrics.models[key] ??= { requests: 0, tokensIn: 0, tokensOut: 0, failures: 0 });
    m.failures++;
    await this.persistMetrics();
  }

  async logLine(level: LogEntry["level"], message: string): Promise<void> {
    this.log(level, message);
  }

  async snapshot(): Promise<Snapshot> {
    return {
      totalRequests: this.metrics.totalRequests,
      totalTokensIn: this.metrics.totalTokensIn,
      totalTokensOut: this.metrics.totalTokensOut,
      totalFailures: this.metrics.totalFailures,
      totalLogs: this.logs.length,
      models: this.metrics.models,
      startTime: this.metrics.startTime,
    };
  }

  // getLogs returns entries newest-first from offset. Mirrors store.Logs.
  async getLogs(offset: number, limit: number): Promise<{ entries: LogEntry[]; total: number }> {
    const total = this.logs.length;
    const newestFirst = [...this.logs].reverse();
    return { entries: newestFirst.slice(offset, offset + limit), total };
  }
}

export { HTTPError };
