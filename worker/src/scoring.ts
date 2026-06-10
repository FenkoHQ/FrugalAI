// Faithful port of internal/model/selector.go and the param-inference bits of
// internal/openrouter/client.go. Weights, bonuses and thresholds match the Go
// implementation exactly so model ranking is identical.

import type { Config, Model } from "./types";

export const FALLBACK_ROUTER_ID = "openrouter/free";

// stealthMaxAge — models published within this window from a known provider get boosted.
const STEALTH_MAX_AGE_SECONDS = 7 * 24 * 60 * 60;

const KNOWN_QUALITY_PROVIDERS = [
  "google/", "anthropic/", "openai/", "meta-llama/", "mistralai/",
  "deepseek/", "qwen/", "stepfun/", "nvidia/", "cohere/",
  "microsoft/", "xiaomi/", "allenai/", "openrouter/",
];

// Mirrors paramCountPattern: (?i)(?:^|[^a-z0-9])(\d+(?:\.\d+)?)\s*([bt])
const PARAM_COUNT_PATTERN = /(?:^|[^a-z0-9])(\d+(?:\.\d+)?)\s*([bt])/i;

function nowSeconds(): number {
  return Math.floor(Date.now() / 1000);
}

export function isFreeModel(model: Model): boolean {
  const prompt = parseFloat(model.pricing?.prompt ?? "");
  const completion = parseFloat(model.pricing?.completion ?? "");
  if (Number.isNaN(prompt) || Number.isNaN(completion)) return false;
  return prompt === 0 && completion === 0;
}

function inferParamCount(text: string): number | null {
  const m = PARAM_COUNT_PATTERN.exec(text);
  if (!m) return null;
  const value = parseFloat(m[1]);
  if (Number.isNaN(value)) return null;
  const multiplier = m[2].toLowerCase() === "t" ? 1_000_000_000_000 : 1_000_000_000;
  return Math.floor(value * multiplier);
}

// normalizeModelMetadata: infer params from id/name/description when absent.
export function normalizeModel(model: Model): Model {
  if (!model.params || model.params <= 0) {
    for (const text of [model.id, model.name, model.description ?? ""]) {
      const inferred = inferParamCount(text);
      if (inferred !== null) {
        return { ...model, params: inferred };
      }
    }
  }
  return model;
}

function normalizePopularity(popularity: number): number {
  if (!popularity || popularity <= 0) return 0.5;
  const normalized = Math.log(popularity) / Math.log(1_000_000);
  return Math.min(normalized, 1.0);
}

function normalizeParams(params: number): number {
  if (!params || params <= 0) return 0.5;
  const normalized = params / 70_000_000_000;
  return Math.min(normalized, 1.0);
}

function normalizeContextLength(length: number): number {
  if (!length || length <= 0) return 0.1;
  const normalized = length / 200_000;
  return Math.min(normalized, 1.0);
}

function isPreferredArchitecture(cfg: Config, modality = "", tokenizer = ""): boolean {
  if (cfg.preferredArchitectures.length === 0) return false;
  const combined = `${modality.toLowerCase()} ${tokenizer.toLowerCase()}`;
  return cfg.preferredArchitectures.some((p) => combined.includes(p.toLowerCase()));
}

function isTopWeekly(cfg: Config, id: string): boolean {
  if (cfg.topWeeklyModels.length === 0) return false;
  const idLower = id.toLowerCase();
  return cfg.topWeeklyModels.some((t) => idLower === t.toLowerCase());
}

export function isStealthModel(model: Model): boolean {
  if (!model.created || model.created === 0) return false;
  const ageSeconds = nowSeconds() - model.created;
  if (ageSeconds > STEALTH_MAX_AGE_SECONDS) return false;
  const idLower = model.id.toLowerCase();
  return KNOWN_QUALITY_PROVIDERS.some((p) => idLower.startsWith(p));
}

function getModelQualityBonus(name: string, id: string): number {
  let bonus = 0;
  const nameLower = name.toLowerCase();
  const idLower = id.toLowerCase();

  const qualityIndicators: Array<{ patterns: string[]; bonus: number }> = [
    { patterns: ["claude", "anthropic"], bonus: 0.15 },
    { patterns: ["gpt-", "openai"], bonus: 0.12 },
    { patterns: ["stepfun"], bonus: 0.15 },
    { patterns: ["gemini", "google"], bonus: 0.10 },
    { patterns: ["mistral", "mixtral"], bonus: 0.08 },
    { patterns: ["llama", "meta"], bonus: 0.08 },
    { patterns: ["qwen"], bonus: 0.07 },
    { patterns: ["deepseek"], bonus: 0.07 },
    { patterns: ["command", "cohere"], bonus: 0.06 },
    { patterns: ["xiaomi", "mimo"], bonus: 0.08 },
    { patterns: ["kwaipilot", "kat-coder"], bonus: 0.08 },
    { patterns: ["nvidia", "nemotron"], bonus: 0.07 },
    { patterns: ["olmo", "allenai"], bonus: 0.06 },
    { patterns: ["trinity", "arcee"], bonus: 0.06 },
    { patterns: ["openrouter"], bonus: 0.10 },
  ];

  for (const indicator of qualityIndicators) {
    for (const pattern of indicator.patterns) {
      if (idLower.includes(pattern) || nameLower.includes(pattern)) {
        bonus += indicator.bonus;
        break;
      }
    }
  }

  if (idLower.includes("flash") || nameLower.includes("flash")) bonus += 0.03;
  if (idLower.includes("pro") || nameLower.includes("pro")) bonus += 0.02;

  for (const weak of ["tiny", "mini", "nano", "micro"]) {
    if (idLower.includes(weak)) bonus -= 0.05;
  }

  return bonus;
}

function calculateScore(cfg: Config, model: Model): number {
  let score = 0;
  score += normalizePopularity(model.popularity ?? 0) * 0.3;
  score += normalizeParams(model.params ?? 0) * 0.4;
  score += normalizeContextLength(model.context_length ?? 0) * 0.2;
  if (isPreferredArchitecture(cfg, model.architecture?.modality, model.architecture?.tokenizer)) {
    score += 0.1;
  }
  if (isTopWeekly(cfg, model.id)) score += 0.5;
  score += getModelQualityBonus(model.name, model.id);
  if (isStealthModel(model)) score += 0.4;
  return score;
}

function filterModels(cfg: Config, models: Model[]): Model[] {
  const filtered: Model[] = [];
  for (const model of models) {
    // Skip meta-routers like openrouter/free that randomly dispatch.
    if (model.id.endsWith("/free") && !model.id.includes(":free")) continue;
    if (cfg.minParams > 0 && (model.params ?? 0) < cfg.minParams) continue;
    if (cfg.minPopularity > 0 && (model.popularity ?? 0) < cfg.minPopularity) {
      if (!isStealthModel(model)) continue;
    }
    filtered.push(model);
  }
  return filtered;
}

function findModelById(models: Model[], id: string): Model | undefined {
  return models.find((m) => m.id === id);
}

function appendFallbackRouter(candidates: Model[], models: Model[]): Model[] {
  const router = findModelById(models, FALLBACK_ROUTER_ID);
  if (!router || candidates.some((c) => c.id === router.id)) return candidates;
  return [...candidates, router];
}

// rankCandidates mirrors Selector.rankCandidates: filter free models, score,
// sort desc, take top n, then append the openrouter/free fallback router.
export function rankCandidates(cfg: Config, models: Model[], n: number): Model[] {
  const normalized = models.map(normalizeModel);
  const freeModels = normalized.filter(isFreeModel);
  const filtered = filterModels(cfg, freeModels);

  if (filtered.length === 0) {
    const router = findModelById(normalized, FALLBACK_ROUTER_ID);
    return router ? [router] : [];
  }

  const scored = filtered.map((m) => ({ model: m, score: calculateScore(cfg, m) }));
  scored.sort((a, b) => b.score - a.score);

  const result = scored.slice(0, n).map((s) => s.model);
  return appendFallbackRouter(result, normalized);
}

// candidateOrder mirrors selector.candidateOrder: preferred id first, then the rest.
export function candidateOrder(candidates: Model[], preferredId: string): number[] {
  const order: number[] = [];
  const seen = new Set<number>();
  if (preferredId) {
    const idx = candidates.findIndex((c) => c.id === preferredId);
    if (idx >= 0) {
      order.push(idx);
      seen.add(idx);
    }
  }
  for (let i = 0; i < candidates.length; i++) {
    if (!seen.has(i)) order.push(i);
  }
  return order;
}
