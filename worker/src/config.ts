// Builds runtime Config from environment bindings. Mirrors the defaults in
// internal/config/config.go (LoadFromEnv).

import type { Config, Env } from "./types";

const DEFAULT_TOP_WEEKLY = [
  "stepfun/step-3.5-flash:free",
  "qwen/qwen3-next-80b-a3b-instruct:free",
  "tngtech/deepseek-r1t2-chimera:free",
  "nvidia/nemotron-3-nano-30b-a3b:free",
  "deepseek/deepseek-r1-0528:free",
  "google/gemma-3-27b-it:free",
];

function splitAndTrim(s: string | undefined): string[] {
  if (!s) return [];
  return s
    .split(",")
    .map((p) => p.trim())
    .filter((p) => p !== "");
}

function intOr(s: string | undefined, dflt: number): number {
  if (s === undefined || s === "") return dflt;
  const n = parseInt(s, 10);
  return Number.isNaN(n) ? dflt : n;
}

function boolOr(s: string | undefined, dflt: boolean): boolean {
  if (s === undefined || s === "") return dflt;
  return s === "true" || s === "1" || s === "yes";
}

export function buildConfig(env: Env): Config {
  return {
    apiKey: env.OPENROUTER_API_KEY ?? "",
    minParams: intOr(env.FRUGALAI_MIN_PARAMS, 0),
    minPopularity: intOr(env.FRUGALAI_MIN_POPULARITY, 0),
    enableOpenAI: boolOr(env.FRUGALAI_ENABLE_OPENAI, true),
    enableAnthropic: boolOr(env.FRUGALAI_ENABLE_ANTHROPIC, true),
    openaiPath: env.FRUGALAI_OPENAI_PATH || "/v1",
    anthropicPath: env.FRUGALAI_ANTHROPIC_PATH || "/v1",
    cacheTTL: intOr(env.FRUGALAI_CACHE_TTL, 300),
    preferredArchitectures: splitAndTrim(env.FRUGALAI_PREFERRED_ARCH),
    topWeeklyModels: env.FRUGALAI_TOP_WEEKLY ? splitAndTrim(env.FRUGALAI_TOP_WEEKLY) : DEFAULT_TOP_WEEKLY,
    numCandidates: intOr(env.FRUGALAI_NUM_CANDIDATES, 10),
    proxyApiKey: env.FRUGALAI_PROXY_API_KEY ?? "",
    uiBasicAuth: env.FRUGALAI_UI_BASIC_AUTH ?? "",
  };
}
