// HTML rendering for the admin UI. Ports internal/server/ui/templates/* to
// template literals. htmx is loaded from a CDN instead of being embedded.

import type { LogEntry, Model, Snapshot } from "./types";

function esc(s: unknown): string {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function formatDuration(ms: number): string {
  const total = Math.round(ms / 1000);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

function formatTime(ms: number): string {
  const d = new Date(ms);
  const pad = (n: number, w = 2) => String(n).padStart(w, "0");
  return `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())}.${pad(d.getUTCMilliseconds(), 3)}`;
}

export interface StatsView {
  uptime: string;
  totalRequests: number;
  totalTokensIn: number;
  totalTokensOut: number;
  totalFailures: number;
  currentModel: string;
}

export function statsView(snap: Snapshot, currentModel: string): StatsView {
  return {
    uptime: formatDuration(Date.now() - snap.startTime),
    totalRequests: snap.totalRequests,
    totalTokensIn: snap.totalTokensIn,
    totalTokensOut: snap.totalTokensOut,
    totalFailures: snap.totalFailures,
    currentModel,
  };
}

export function renderStats(v: StatsView): string {
  return `<div class="stats-bar" hx-get="/admin/ui/partials/stats" hx-trigger="every 10s" hx-swap="outerHTML">
  <div class="stat-cell"><span class="stat-label">Uptime</span><span class="stat-value">${esc(v.uptime)}</span></div>
  <div class="stat-cell"><span class="stat-label">Requests</span><span class="stat-value accent">${v.totalRequests}</span></div>
  <div class="stat-cell"><span class="stat-label">Tokens in</span><span class="stat-value">${v.totalTokensIn}</span></div>
  <div class="stat-cell"><span class="stat-label">Tokens out</span><span class="stat-value">${v.totalTokensOut}</span></div>
  <div class="stat-cell"><span class="stat-label">Failures</span><span class="stat-value" style="${v.totalFailures > 0 ? "color:var(--err)" : ""}">${v.totalFailures}</span></div>
  <div class="stat-cell"><span class="stat-label">Active model</span><span class="stat-value" style="font-size:12px;font-family:'JetBrains Mono',monospace;color:var(--accent)">${v.currentModel ? esc(v.currentModel) : "—"}</span></div>
</div>`;
}

export function renderModel(current: Model | null, candidates: Model[], currentIdx: number): string {
  const card = current
    ? `<div class="model-card" style="margin-bottom:16px">
    <div class="model-id">${esc(current.id)}</div>
    <div style="color:var(--muted);font-size:12px;margin-bottom:6px">${esc(current.name)}</div>
    <div class="model-meta">
      ${current.context_length ? `<span class="tag">ctx ${current.context_length}</span>` : ""}
      ${current.params ? `<span class="tag">${current.params}B</span>` : ""}
      ${current.architecture?.tokenizer ? `<span class="tag">${esc(current.architecture.tokenizer)}</span>` : ""}
    </div>
  </div>`
    : `<div class="model-card" style="margin-bottom:16px;color:var(--muted)">No model selected</div>`;

  const list = candidates
    .map((m, i) => {
      const active = i === currentIdx;
      const right = active
        ? `<span class="dot-active"></span>`
        : `<button class="btn btn-sm" hx-post="/admin/ui/model/force?id=${encodeURIComponent(m.id)}" hx-target="#model-section" hx-swap="outerHTML">use</button>`;
      return `<div class="candidate${active ? " active" : ""}"><span class="candidate-name" title="${esc(m.id)}">${esc(m.name)}</span>${right}</div>`;
    })
    .join("");

  return `<div id="model-section" hx-get="/admin/ui/partials/model" hx-trigger="every 30s" hx-swap="outerHTML">
  <div class="section-head"><span class="section-title">Active model</span></div>
  ${card}
  <div class="section-head"><span class="section-title">Candidates</span></div>
  ${list}
</div>`;
}

export interface UsageRow {
  model: string;
  requests: number;
  tokensIn: number;
  tokensOut: number;
  failures: number;
}

export function usageRows(snap: Snapshot): UsageRow[] {
  const rows = Object.entries(snap.models).map(([model, ms]) => ({
    model,
    requests: ms.requests,
    tokensIn: ms.tokensIn,
    tokensOut: ms.tokensOut,
    failures: ms.failures,
  }));
  rows.sort((a, b) => b.requests - a.requests);
  return rows;
}

export function renderUsage(rows: UsageRow[]): string {
  const th = (label: string, align = "right") =>
    `<th style="padding:8px 14px;text-align:${align};font-size:10px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:var(--muted)">${label}</th>`;
  const body = rows.length
    ? `<div style="background:var(--surface);border:1px solid var(--border);border-radius:8px;overflow:hidden">
    <table style="width:100%;border-collapse:collapse">
      <thead><tr style="border-bottom:1px solid var(--border)">${th("Model", "left")}${th("Req")}${th("Tok in")}${th("Tok out")}${th("Fail")}</tr></thead>
      <tbody>
        ${rows
          .map(
            (r) => `<tr style="border-bottom:1px solid rgba(48,54,61,0.5)">
          <td style="padding:7px 14px;font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--accent);max-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;width:60%" title="${esc(r.model)}">${esc(r.model)}</td>
          <td style="padding:7px 14px;text-align:right;font-size:12px;color:var(--txt)">${r.requests}</td>
          <td style="padding:7px 14px;text-align:right;font-size:12px;color:var(--muted)">${r.tokensIn}</td>
          <td style="padding:7px 14px;text-align:right;font-size:12px;color:var(--muted)">${r.tokensOut}</td>
          <td style="padding:7px 14px;text-align:right;font-size:12px;${r.failures > 0 ? "color:var(--err)" : "color:var(--muted)"}">${r.failures}</td>
        </tr>`,
          )
          .join("")}
      </tbody>
    </table>
  </div>`
    : `<div class="empty-state">No requests yet.</div>`;

  return `<div id="usage-section" hx-get="/admin/ui/partials/usage" hx-trigger="every 15s" hx-swap="outerHTML">
  <div class="section-head"><span class="section-title">Usage by model</span></div>
  ${body}
</div>`;
}

export function renderLogs(entries: LogEntry[], offset: number, total: number): string {
  if (entries.length === 0 && offset === 0) {
    return `<div class="empty-state">No log entries yet.</div>`;
  }
  const nextOffset = offset + entries.length;
  const hasMore = nextOffset < total;
  const rows = entries
    .map(
      (e) => `<div class="log-entry">
  <span class="log-time">${formatTime(e.time)}</span>
  <span class="log-level ${e.level}">${e.level}</span>
  <span class="log-msg">${esc(e.message)}</span>
</div>`,
    )
    .join("");
  const more = hasMore
    ? `<div class="load-more"><button class="btn btn-sm" hx-get="/admin/ui/partials/logs?offset=${nextOffset}" hx-target="#log-entries" hx-swap="beforeend" hx-on::after-request="this.closest('.load-more').remove()">Load older</button></div>`
    : "";
  return rows + more;
}

export function renderPage(stats: string, model: string, usage: string, logs: string): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FrugalAI — Fenko</title>
<script src="https://unpkg.com/htmx.org@1.9.12"></script>
<style>
:root {
  --bg: #0D1117; --surface: #161B22; --elevated: #1C2128; --border: #30363D;
  --txt: #E6EDF3; --muted: #8B949E; --accent: #F5A623; --accent-bg: rgba(245,166,35,0.12);
  --ok: #3FB950; --warn: #D29922; --err: #F85149; --info: #58A6FF;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--txt); font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size: 13px; line-height: 1.5; min-height: 100vh; }
a { color: var(--accent); text-decoration: none; }
header { display: flex; align-items: center; gap: 12px; padding: 14px 24px; border-bottom: 1px solid var(--border); background: var(--surface); }
.logo-mark { width: 28px; height: 28px; background: var(--accent); border-radius: 6px; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 14px; color: #0D1117; letter-spacing: -0.5px; flex-shrink: 0; }
.logo-name { font-size: 13px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--txt); }
.logo-sep { color: var(--border); margin: 0 2px; }
.logo-product { color: var(--muted); font-weight: 400; }
.layout { display: grid; grid-template-columns: 280px 1fr; grid-template-rows: auto 1fr; gap: 0; height: calc(100vh - 57px); }
.sidebar { border-right: 1px solid var(--border); overflow-y: auto; padding: 20px 16px; display: flex; flex-direction: column; gap: 20px; }
.main { overflow-y: auto; padding: 20px 24px; display: flex; flex-direction: column; gap: 16px; }
.stats-bar { grid-column: 1 / -1; display: flex; gap: 1px; background: var(--border); border-bottom: 1px solid var(--border); }
.stat-cell { flex: 1; padding: 10px 20px; background: var(--surface); display: flex; flex-direction: column; gap: 2px; }
.stat-label { font-size: 10px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
.stat-value { font-size: 20px; font-weight: 600; color: var(--txt); letter-spacing: -0.5px; }
.stat-value.accent { color: var(--accent); }
.section-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; }
.section-title { font-size: 10px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); }
.btn { display: inline-flex; align-items: center; gap: 5px; padding: 4px 10px; border-radius: 5px; border: 1px solid var(--border); background: var(--elevated); color: var(--txt); font-size: 12px; cursor: pointer; transition: border-color 0.15s, background 0.15s; }
.btn:hover { border-color: var(--accent); background: var(--accent-bg); }
.btn-sm { padding: 2px 8px; font-size: 11px; }
.htmx-request .btn { opacity: 0.6; pointer-events: none; }
.model-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 14px; }
.model-id { font-family: "JetBrains Mono", monospace; font-size: 12px; color: var(--accent); word-break: break-all; margin-bottom: 6px; }
.model-meta { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.tag { font-size: 10px; padding: 2px 7px; border-radius: 4px; background: var(--elevated); border: 1px solid var(--border); color: var(--muted); font-family: "JetBrains Mono", monospace; }
.candidate { display: flex; align-items: center; justify-content: space-between; gap: 8px; padding: 7px 10px; border-radius: 6px; border: 1px solid transparent; transition: border-color 0.15s, background 0.15s; }
.candidate:hover { background: var(--elevated); border-color: var(--border); }
.candidate.active { background: var(--accent-bg); border-color: rgba(245,166,35,0.3); }
.candidate-name { font-size: 12px; color: var(--txt); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; flex: 1; min-width: 0; }
.candidate.active .candidate-name { color: var(--accent); font-weight: 500; }
.dot-active { width: 6px; height: 6px; border-radius: 50%; background: var(--ok); flex-shrink: 0; }
.log-feed { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; flex: 1; }
.log-entry { display: grid; grid-template-columns: 88px 46px 1fr; gap: 10px; padding: 5px 14px; border-bottom: 1px solid rgba(48,54,61,0.5); font-family: "JetBrains Mono", Consolas, monospace; font-size: 11.5px; line-height: 1.6; align-items: baseline; }
.log-entry:last-child { border-bottom: none; }
.log-entry:hover { background: var(--elevated); }
.log-time { color: var(--muted); white-space: nowrap; }
.log-level { font-weight: 600; white-space: nowrap; }
.log-level.DEBUG { color: var(--muted); }
.log-level.INFO { color: var(--info); }
.log-level.WARN { color: var(--warn); }
.log-level.ERROR { color: var(--err); }
.log-msg { color: var(--txt); word-break: break-all; }
.load-more { padding: 10px; text-align: center; border-top: 1px solid var(--border); }
.empty-state { padding: 40px; text-align: center; color: var(--muted); }
</style>
</head>
<body>
<header>
  <div class="logo-mark">F</div>
  <span class="logo-name">Fenko <span class="logo-sep">/</span> <span class="logo-product">FrugalAI</span></span>
</header>
<div class="layout">
  ${stats}
  <aside class="sidebar">${model}</aside>
  <main class="main">
    ${usage}
    <div>
      <div class="section-head">
        <span class="section-title">Logs</span>
        <button class="btn btn-sm" hx-get="/admin/ui/partials/logs?offset=0" hx-target="#log-entries" hx-swap="innerHTML">↺ Refresh</button>
      </div>
      <div class="log-feed"><div id="log-entries">${logs}</div></div>
    </div>
  </main>
</div>
</body>
</html>`;
}
