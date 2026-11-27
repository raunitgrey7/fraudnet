// fraudnet console: vanilla JS, no build step. Talks to the same API as any client.
"use strict";

const $ = (id) => document.getElementById(id);

function apiKey() {
  try { return localStorage.getItem("fraudnet-key") || ""; } catch (e) { return ""; }
}

async function api(path, options = {}) {
  const headers = Object.assign({ "content-type": "application/json" }, options.headers || {});
  const key = apiKey();
  if (key) headers["x-api-key"] = key;
  const response = await fetch(path, Object.assign({}, options, { headers }));
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || response.statusText);
  return body;
}

function text(tag, value, cls) {
  const el = document.createElement(tag);
  el.textContent = value;
  if (cls) el.className = cls;
  return el;
}

async function loadStats() {
  try {
    const s = await api("/v1/stats");
    const a = s.alerts || {};
    const parts = [
      ["clock", s.store.clock || "not started"],
      ["accounts", s.store.accounts],
      ["devices", s.store.devices],
      ["open alerts", a.open || 0],
      ["confirmed fraud", a.fraud || 0],
      ["reports applied", s.store.labels_applied],
      ["snapshots", s.store.snapshots_built],
      ["held-out stream left", s.feed_remaining],
    ];
    const box = $("stats");
    box.replaceChildren(...parts.map(([k, v]) => { const el = text("span", `${k} `); el.append(text("b", String(v))); return el; }));
  } catch (e) {
    $("stats").textContent = `stats unavailable: ${e.message}`;
  }
}

function reasonChips(reasons) {
  const cell = document.createElement("td");
  for (const r of reasons || []) {
    const chip = text("span", r.code, "chip");
    chip.title = r.detail || r.description;
    cell.append(chip);
  }
  return cell;
}

async function resolve(alertId, label) {
  try {
    await api(`/v1/alerts/${alertId}/resolve`, { method: "POST", body: JSON.stringify({ label }) });
    await refresh();
  } catch (e) { alert(e.message); }
}

async function loadAlerts() {
  const data = await api("/v1/alerts?status=open&limit=60");
  $("queue-count").textContent = `${data.count} shown, ${data.stats.open} open`;
  const rows = data.alerts.map((a) => {
    const tr = document.createElement("tr");
    tr.append(text("td", a.score.toFixed(3)), text("td", a.decision, `decision-${a.decision}`), text("td", a.amount.toFixed(2)));
    const acct = text("td", String(a.account));
    acct.style.cursor = "pointer";
    acct.title = "show network";
    acct.onclick = () => showNetwork(a.account);
    tr.append(acct, reasonChips(a.reasons));
    const truth = a.true_fraud === 1 ? "planted fraud" : a.true_fraud === 0 ? "genuine" : "n/a";
    tr.append(text("td", truth, a.true_fraud === 1 ? "truth-fraud" : "truth-legit"));
    const verdict = document.createElement("td");
    const fraud = text("button", "fraud");
    fraud.onclick = () => resolve(a.alert_id, "fraud");
    const legit = text("button", "genuine", "ghost");
    legit.onclick = () => resolve(a.alert_id, "legit");
    verdict.append(fraud, " ", legit);
    tr.append(verdict);
    return tr;
  });
  $("alerts").replaceChildren(...rows);
}

async function loadRings() {
  const data = await api("/v1/rings?limit=8");
  const list = data.rings.map((r, i) => {
    const el = document.createElement("button");
    el.className = "ring";
    el.type = "button";
    el.append(text("b", `cluster ${r.community}`), text("span", `risk ${r.risk.toFixed(3)}`), text("span", `${r.accounts} accounts, ${r.devices} devices`),
      text("span", `${r.new_accounts} new, ${r.reported_fraud_accounts} reported`, "muted"));
    el.onclick = () => { document.querySelectorAll(".ring").forEach((x) => x.classList.remove("active")); el.classList.add("active"); drawGraph(r.nodes, r.edges); };
    if (i === 0 && !$("graph").dataset.drawn) { el.classList.add("active"); drawGraph(r.nodes, r.edges); }
    return el;
  });
  $("rings").replaceChildren(...(list.length ? list : [text("p", "No device-sharing clusters yet. Replay some of the stream.", "muted small")]));
}

async function showNetwork(accountId) {
  try {
    const view = await api(`/v1/accounts/${accountId}/network`);
    drawGraph(view.nodes, view.edges);
  } catch (e) { alert(e.message); }
}

function scoreColour(score) {
  if (score === null || score === undefined) return "#8a96a8";
  const s = Math.max(0, Math.min(1, score));
  const hue = 130 - 130 * Math.sqrt(s);
  return `hsl(${hue}, 65%, 45%)`;
}

// Force layout after Fruchterman and Reingold. Forces are scaled by k, a spacing derived from the box
// area per node, so a ring of five accounts and one of ninety both fill the panel: repulsion k*k/d
// between all pairs, attraction d*d/k along edges, and a pull to the centre sized so that a cloud of
// unconnected nodes comes to rest inside the box instead of against its walls (device-sharing clusters
// are often several small components). The step limit cools to zero, and the start positions are a
// fixed spiral, so the same ring always draws the same way.
function layout(nodes, edges, width, height) {
  const n = nodes.length;
  const index = new Map(nodes.map((node, i) => [node.id, i]));
  const links = edges.map((e) => [index.get(e.source), index.get(e.target)]).filter(([a, b]) => a !== undefined && b !== undefined);
  const k = 0.4 * Math.sqrt((width * height) / Math.max(n, 1));
  const restX = 0.42 * width, restY = 0.42 * height;
  const pullX = (n * k * k) / (restX * restX), pullY = (n * k * k) / (restY * restY);
  const pos = nodes.map((_, i) => {
    const radius = Math.sqrt((i + 0.5) / n);
    return { x: width / 2 + 0.8 * restX * radius * Math.cos(i * 2.399963), y: height / 2 + 0.8 * restY * radius * Math.sin(i * 2.399963) };
  });
  const steps = 300;
  for (let step = 0; step < steps; step++) {
    const limit = (width / 12) * (1 - step / steps);
    const move = pos.map((p) => ({ x: (width / 2 - p.x) * pullX, y: (height / 2 - p.y) * pullY }));
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const dx = pos[i].x - pos[j].x, dy = pos[i].y - pos[j].y;
        const d = Math.max(Math.hypot(dx, dy), 0.5);
        const f = (k * k) / (d * d);
        move[i].x += f * dx; move[i].y += f * dy; move[j].x -= f * dx; move[j].y -= f * dy;
      }
    }
    for (const [a, b] of links) {
      const dx = pos[b].x - pos[a].x, dy = pos[b].y - pos[a].y;
      const f = Math.hypot(dx, dy) / k;
      move[a].x += f * dx; move[a].y += f * dy; move[b].x -= f * dx; move[b].y -= f * dy;
    }
    pos.forEach((p, i) => {
      const length = Math.max(Math.hypot(move[i].x, move[i].y), 1e-9);
      const stride = Math.min(length, limit);
      p.x = Math.max(12, Math.min(width - 12, p.x + (move[i].x / length) * stride));
      p.y = Math.max(12, Math.min(height - 12, p.y + (move[i].y / length) * stride));
    });
  }
  return { pos, links };
}

function svg(tag, attrs) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

function drawGraph(nodes, edges) {
  const graph = $("graph");
  graph.dataset.drawn = "1";
  const shown = nodes.slice(0, 90);
  const { pos } = layout(shown, edges, 560, 380);
  const index = new Map(shown.map((n, i) => [n.id, i]));
  const children = [];
  for (const e of edges) {
    const a = index.get(e.source), b = index.get(e.target);
    if (a === undefined || b === undefined) continue;
    children.push(svg("line", { x1: pos[a].x, y1: pos[a].y, x2: pos[b].x, y2: pos[b].y, class: e.kind === "transfer" ? "e-transfer" : "e-device" }));
  }
  shown.forEach((n, i) => {
    const p = pos[i];
    const shape = n.type === "device"
      ? svg("rect", { x: p.x - 5, y: p.y - 5, width: 10, height: 10, class: "n-dev" })
      : svg("circle", { cx: p.x, cy: p.y, r: 7, class: `n-acct${n.reported_fraud ? " reported" : ""}`, fill: scoreColour(n.score) });
    const title = svg("title", {});
    title.textContent = n.type === "device" ? `${n.id}` : `${n.id} score ${n.score === null || n.score === undefined ? "n/a" : n.score.toFixed(3)}`;
    shape.append(title);
    children.push(shape);
  });
  graph.replaceChildren(...children);
}

async function replay() {
  const button = $("replay");
  button.disabled = true;
  $("replay-result").textContent = "scoring";
  try {
    const n = Number($("replay-n").value) || 500;
    const r = await api("/v1/replay", { method: "POST", body: JSON.stringify({ n, model: $("model").value }) });
    $("replay-result").textContent = `${r.scored} scored, ${r.alerts} alerts of which ${r.alerts_true_fraud} planted fraud; ${r.fraud_in_batch} fraud in the batch; clock ${r.clock}`;
    await refresh();
  } catch (e) {
    $("replay-result").textContent = e.message;
  } finally {
    button.disabled = false;
  }
}

async function score(event) {
  event.preventDefault();
  const form = new FormData(event.target);
  const body = { account_id: Number(form.get("account_id")), device_id: Number(form.get("device_id")), amount: Number(form.get("amount")), channel: form.get("channel"), model: $("model").value };
  if (form.get("payee_account_id")) body.payee_account_id = Number(form.get("payee_account_id"));
  else body.merchant_id = Number(form.get("merchant_id"));
  try {
    const r = await api("/v1/score", { method: "POST", body: JSON.stringify(body) });
    $("score-out").textContent = JSON.stringify({ score: r.score, decision: r.decision, reasons: r.reasons, latency_ms: r.latency_ms, alert_id: r.alert_id }, null, 2);
    await refresh();
  } catch (e) { $("score-out").textContent = e.message; }
}

async function refresh() {
  await Promise.allSettled([loadStats(), loadAlerts(), loadRings()]);
}

$("replay").onclick = replay;
$("score-form").onsubmit = score;
$("lookup").onclick = () => { const id = $("account-id").value; if (id !== "") showNetwork(Number(id)); };
$("api-key").value = apiKey();
$("api-key").onchange = (e) => { try { localStorage.setItem("fraudnet-key", e.target.value); } catch (err) { /* storage unavailable */ } refresh(); };
refresh();
