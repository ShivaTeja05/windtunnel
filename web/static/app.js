// Wind Tunnel frontend: vanilla JS, no build step. Mermaid 10.9.1 renders the diagrams.

const MERMAID_URL = "https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.esm.min.mjs";
const REPO = "https://github.com/ShivaTeja05/windtunnel";

const FAMILY_NAMES = {
  single_call: "Single call",
  self_check: "Answer + self-check",
  self_consistency: "Self-consistency vote",
  plan_execute: "Plan, then execute",
  orchestrator_workers: "Orchestrator + workers",
  cascade: "Cascade",
  retrieval: "Retrieval (RAG)",
  fine_tuning: "Fine-tuning",
  other: "Other methods",
};
const TIERS = ["nano", "lightning", "super", "ultra"];
const TIER_NAMES = { nano: "Nano", lightning: "Lightning", super: "Super", ultra: "Ultra" };
const ROLE_NAMES = { baseline: "baseline", candidate: "candidate", "llm-default": "llm-default" };
const ROLE_HINTS = {
  baseline: "The cheapest possible design. Everything else must beat it to be worth paying for.",
  candidate: "Proposed from the methods found in the research.",
  "llm-default": "The design a plain LLM usually recommends. It is raced so you can see what it really costs.",
};
const MODE_TEXT = {
  offline: "OFFLINE",
  mixed: "PARTLY OFFLINE",
  nemotron: "NEMOTRON",
  mock: "MOCK",
  measured: "MEASURED",
  real: "REAL",
  unavailable: "UNAVAILABLE",
  untested: "UNTESTED",
};

const app = document.getElementById("app");
let health = null;
let routeToken = 0;

// ------------------------------------------------------------------ utils

const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const num = (x) => typeof x === "number" && Number.isFinite(x);
const pct = (x) => (num(x) ? `${Math.round(x * 100)}%` : "–");
const usd = (x) => {
  if (!num(x)) return "–";
  if (x === 0) return "$0";
  if (x < 0.01) return `$${x.toFixed(4)}`;
  if (x < 10) return `$${x.toFixed(3)}`;
  return `$${x.toFixed(2)}`;
};
const secs = (x) => (num(x) ? `${x < 10 ? x.toFixed(2) : x.toFixed(1)}s` : "–");
const pill = (kind, label) => `<span class="pill pill-${esc(kind)}">${esc(label || MODE_TEXT[kind] || kind)}</span>`;
const tier = (t) => `<span class="tier tier-${esc(t)}">${esc(TIER_NAMES[t] || t)}</span>`;
const designName = (d) => `${FAMILY_NAMES[d.family] || d.family} · ${TIER_NAMES[d.model] || d.model}`;

async function api(path, opts = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  let body = null;
  try {
    body = await res.json();
  } catch (e) {}
  if (!res.ok) {
    const err = new Error((body && body.detail) || `${res.status} ${res.statusText}`);
    err.status = res.status;
    throw err;
  }
  return body;
}

function toast(msg) {
  const el = document.createElement("div");
  el.className = "toast";
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 1800);
}

// ------------------------------------------------------------------ mermaid

let mermaidPromise = null;
let renderQueue = Promise.resolve();
let diagramSeq = 0;

function loadMermaid() {
  if (!mermaidPromise) {
    mermaidPromise = import(MERMAID_URL).then((m) => {
      const mermaid = m.default;
      mermaid.initialize({
        startOnLoad: false,
        theme: "default",
        securityLevel: "strict",
        fontFamily: "Inter, system-ui, sans-serif",
        flowchart: { curve: "basis", padding: 10, nodeSpacing: 28, rankSpacing: 36 },
      });
      return mermaid;
    });
  }
  return mermaidPromise;
}

function renderDiagrams(root) {
  root.querySelectorAll(".diagram[data-src]").forEach((el) => {
    const src = el.dataset.src;
    el.removeAttribute("data-src");
    el.classList.add("loading");
    // Mermaid keeps global state, so diagrams render one at a time.
    renderQueue = renderQueue.then(async () => {
      try {
        const mermaid = await loadMermaid();
        const { svg } = await mermaid.render(`wt-d${++diagramSeq}`, src);
        el.innerHTML = svg;
      } catch (e) {
        el.innerHTML = `<pre class="small faint" style="white-space:pre-wrap;margin:0">${esc(src)}</pre>`;
      }
      el.classList.remove("loading");
    });
  });
}

const diagram = (src) => `<div class="diagram" data-src="${esc(src)}"></div>`;

const tierLegend = () =>
  `<div class="legend"><span>Model that runs each step:</span>${TIERS.map(tier).join("")}<span class="faint">· grey = input/output, no model · dashed arrow = a branch only some inputs take</span></div>`;

// ------------------------------------------------------------------ router

const routes = [
  [/^\/$/, viewHome],
  [/^\/new\/?$/, viewNew],
  [/^\/explore\/?$/, viewExplore],
  [/^\/editor\/?$/, viewEditor],
  [/^\/r\/([\w-]+)\/?$/, viewRun],
];

function navigate(path) {
  if (path !== location.pathname) history.pushState({}, "", path);
  route();
}

function route() {
  const token = ++routeToken;
  const path = location.pathname;
  document.querySelectorAll("#nav a").forEach((a) => a.classList.toggle("active", a.getAttribute("href") === path));
  for (const [re, view] of routes) {
    const m = path.match(re);
    if (m) {
      window.scrollTo(0, 0);
      return view(token, ...m.slice(1));
    }
  }
  app.innerHTML = `<div class="skeleton"><h1>Not found</h1><p class="mt"><a href="/" data-link>Go home</a></p></div>`;
}

document.addEventListener("click", (e) => {
  const a = e.target.closest("a[data-link]");
  if (a && !e.metaKey && !e.ctrlKey && !e.shiftKey && a.origin === location.origin) {
    e.preventDefault();
    navigate(a.pathname);
  }
});
window.addEventListener("popstate", route);

// ------------------------------------------------------------------ theme & health

document.getElementById("theme-toggle").addEventListener("click", () => {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  try {
    localStorage.setItem("wt-theme", root.dataset.theme);
  } catch (e) {}
});

async function loadHealth() {
  try {
    health = await api("/api/health");
    const el = document.getElementById("mode-pill");
    el.className = `pill pill-${health.mode}`;
    el.textContent = health.mode === "nemotron" ? "NEMOTRON" : "OFFLINE · MOCK";
    el.title =
      health.mode === "nemotron"
        ? "Nemotron via Nebius Token Factory, with a hard budget cap"
        : "No NEBIUS_API_KEY on this server: real paper search, heuristic method cards, simulated race";
  } catch (e) {}
}

// ------------------------------------------------------------------ starting runs

async function startRun(payload, button) {
  if (button) {
    button.disabled = true;
    button.dataset.label = button.innerHTML;
    button.innerHTML = `<span class="spinner"></span>Starting…`;
  }
  try {
    const r = await api("/api/runs", { method: "POST", body: JSON.stringify(payload) });
    navigate(`/r/${r.id}`);
  } catch (e) {
    if (button) {
      button.disabled = false;
      button.innerHTML = button.dataset.label;
    }
    throw e;
  }
}

// ------------------------------------------------------------------ home

async function viewHome(token) {
  app.innerHTML = `
  <section class="hero">
    <span class="eyebrow">NVIDIA Nemotron · Nebius Token Factory · open source</span>
    <h1>Test AI designs <em>before you build them</em></h1>
    <p class="lede">Describe what you are building. Wind Tunnel searches the papers, turns the methods it finds into
    candidate architectures, races them on your labelled examples and gives you the cheapest design that is
    statistically tied with the best.</p>
    <div class="hero-cta">
      <button class="btn btn-primary" id="try-sample">▶ Try a sample: support-ticket triage</button>
      <a class="btn" href="/new" data-link>Describe your task</a>
    </div>
    <p class="hero-note" id="sample-note">Sample results are cached, so they open instantly and cost nothing.</p>
    <div class="pitch">
      <div class="without"><div class="lbl">Ask an LLM</div>
        <b>"Use a multi-agent system with a large model."</b>
        <p class="muted small" style="margin-top:6px">From memory. No sources, no alternatives, never tested on your data.</p></div>
      <div class="with"><div class="lbl">Ask Wind Tunnel</div>
        <b>"Of the designs found in the papers, this one ties the best on your examples and costs the least."</b>
        <p class="muted small" style="margin-top:6px">Cited methods, diagrams of what ran, measured accuracy with confidence intervals.</p></div>
    </div>
  </section>
  <section class="steps">
    ${[
      ["1", "Understand", "Turns your description into a search plan: requirements, constraints, queries.", ["ultra"]],
      ["2", "Search", "Queries arXiv, OpenAlex and the web. Includes recent, long-tail and other-field papers.", ["nano"]],
      ["3", "Extract", "Turns each source into a method card. The citation comes from the index, never from the model.", ["super"]],
      ["4", "Design", "Builds candidate architectures from the cards, each with a diagram of exactly what will run.", ["ultra"]],
      ["5", "Race", "Runs every candidate on your examples, eliminates losers early, picks the cheapest tied winner.", ["nano", "super", "ultra"]],
      ["6", "Breed", "Diagnoses why the losers failed, borrows a fix from another field and races the bred designs.", ["ultra"]],
    ]
      .map(
        ([n, t, d, tiers]) =>
          `<div class="card step"><div class="n">0${n}</div><h3>${t}</h3><p>${d}</p><div class="who">${tiers.map(tier).join(" ")}</div></div>`,
      )
      .join("")}
  </section>
  <p class="muted small" style="text-align:center;margin-top:18px">LLMs are good at <b>generating</b> designs and bad at <b>judging</b> them. Nemotron generates; measurement judges.</p>
  <section class="samples" id="samples"></section>`;

  const samples = await api("/api/samples").catch(() => []);
  if (token !== routeToken) return;
  const first = samples[0];
  const btn = document.getElementById("try-sample");
  if (!first) {
    btn.disabled = true;
    return;
  }
  btn.onclick = () => startRun({ sample: first.name }, btn).catch((e) => toast(e.message));
  document.getElementById("samples").innerHTML = `
    <div class="section-head"><h2>Samples</h2><span class="sub">Bundled tasks with labelled examples.</span></div>
    ${samples
      .map(
        (s) => `<div class="card sample"><div class="grow"><h3>${esc(s.name.replace(/_/g, " "))}</h3>
          <p class="muted small">${esc(s.description)}</p>
          <p class="faint small" style="margin-top:4px">${s.n_examples} labelled examples · labels: ${esc((s.labels || []).join(", "))}</p></div>
          <button class="btn btn-sm" data-sample="${esc(s.name)}">Run sample →</button></div>`,
      )
      .join("")}`;
  document.querySelectorAll("[data-sample]").forEach((b) => {
    b.onclick = () => startRun({ sample: b.dataset.sample }, b).catch((e) => toast(e.message));
  });
}

// ------------------------------------------------------------------ new run

function parseExamples(text) {
  const rows = [];
  const lines = text.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  for (const [i, line] of lines.entries()) {
    if (line.startsWith("{")) {
      let o;
      try {
        o = JSON.parse(line);
      } catch (e) {
        throw new Error(`line ${i + 1}: invalid JSON`);
      }
      rows.push({ input: o.input, expected: o.expected ?? o.label });
      continue;
    }
    const sep = line.includes("\t") ? line.lastIndexOf("\t") : line.lastIndexOf(",");
    if (sep <= 0) throw new Error(`line ${i + 1}: expected "text<TAB>label" or "text,label"`);
    rows.push({ input: line.slice(0, sep).trim().replace(/^"|"$/g, ""), expected: line.slice(sep + 1).trim() });
  }
  return rows;
}

function viewNew() {
  const lim = health?.limits || { problem_chars: 2000, examples: 200, runs_per_hour: 5 };
  app.innerHTML = `
  <div class="page-head"><h1>Describe your task</h1>
    <p>Say what the AI part of your product must do, in plain words. Add labelled examples to race the designs. Without examples you still get methods and architectures, but they are not tested.</p></div>
  <form class="card form" id="new-form">
    <div><label for="problem">What are you building? <span class="hint">(max ${lim.problem_chars} characters)</span></label>
      <textarea id="problem" rows="4" maxlength="${lim.problem_chars}" placeholder="e.g. Classify incoming invoices as approve, flag or reject based on the line items. Must cost under $2 per 1000 invoices." required></textarea></div>
    <div><label for="examples">Labelled examples <span class="hint">(optional, 10 to ${lim.examples}; one per line as <code>text&lt;TAB&gt;label</code>, <code>text,label</code> or JSONL <code>{"input","expected"}</code>)</span></label>
      <textarea id="examples" class="mono" rows="8" placeholder="I was charged twice for my order\tbilling&#10;The app crashes when I open settings\tbug&#10;…"></textarea>
      <p class="hint" id="ex-count" style="margin-top:6px"></p></div>
    <div><label for="labels">Allowed labels <span class="hint">(optional, comma separated; defaults to the labels in your examples)</span></label>
      <input type="text" id="labels" placeholder="billing, bug, shipping"></div>
    <div class="row"><button class="btn btn-primary" type="submit" id="go">Run Wind Tunnel</button>
      <span class="hint">Public demo: ${lim.runs_per_hour} new runs per hour per visitor. ${
        health?.mode === "nemotron" ? "Runs use Nemotron with a hard spending cap." : "This server is offline: the race will be simulated (MOCK)."
      }</span></div>
    <p class="form-error" id="form-error" hidden></p>
  </form>`;
  const ex = document.getElementById("examples");
  const count = document.getElementById("ex-count");
  ex.addEventListener("input", () => {
    try {
      const n = ex.value.trim() ? parseExamples(ex.value).length : 0;
      count.textContent = n ? `${n} example${n === 1 ? "" : "s"} parsed` : "";
    } catch (e) {
      count.textContent = e.message;
    }
  });
  document.getElementById("new-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const err = document.getElementById("form-error");
    err.hidden = true;
    try {
      const examples = ex.value.trim() ? parseExamples(ex.value) : [];
      const labels = document.getElementById("labels").value.split(",").map((s) => s.trim()).filter(Boolean);
      await startRun(
        { problem: document.getElementById("problem").value, examples, labels: labels.length ? labels : null },
        document.getElementById("go"),
      );
    } catch (e2) {
      err.textContent = e2.message;
      err.hidden = false;
    }
  });
}

// ------------------------------------------------------------------ run / report
//
// GET /api/runs/{id} returns the run's live stages while it runs and, once it is done, the
// engine's report (engine.run_wind_tunnel) under `report`. Sections render from the report;
// while it is missing they show the live stage status instead.

async function viewRun(token, runId) {
  app.innerHTML = `<div class="skeleton"><span class="spinner"></span>Loading report…</div>`;
  const rendered = {};
  let first = true;
  let failures = 0;
  while (token === routeToken) {
    let run;
    try {
      run = await api(`/api/runs/${runId}`);
      failures = 0;
    } catch (e) {
      // Only a 404 means there is no such run. A network blip or a 5xx while the server redeploys is retried.
      if (e.status === 404 || failures >= 8) {
        if (token !== routeToken) return;
        const title = e.status === 404 ? "Report not found" : "Can't reach the server";
        app.innerHTML = `<div class="skeleton"><h1>${title}</h1><p class="mt muted">${esc(e.message)}</p><p class="mt">${
          e.status === 404 ? "" : `<a href="/r/${esc(runId)}" data-link>Try again</a> · `
        }<a href="/" data-link>Start a new run</a></p></div>`;
        return;
      }
      failures += 1;
      const msg = `Connection problem (${e.message}); retrying…`;
      const err = document.getElementById("run-error");
      if (err) {
        err.textContent = msg;
        err.hidden = false;
      } else app.innerHTML = `<div class="skeleton"><span class="spinner"></span>${esc(msg)}</div>`;
      await new Promise((r) => setTimeout(r, Math.min(15000, 1000 * 2 ** failures)));
      continue;
    }
    if (token !== routeToken) return;
    const err = document.getElementById("run-error");
    if (err && err.textContent.startsWith("Connection problem")) err.hidden = true;
    if (first) {
      app.innerHTML = runShell(run);
      bindShare(run);
      first = false;
    }
    updateRun(run, rendered);
    if (run.status === "done" || run.status === "error") return;
    await new Promise((r) => setTimeout(r, 900));
  }
}

const hasExamples = (run) => !!(run.request.sample || (run.request.examples || []).length);

function racePill(run) {
  if (!hasExamples(run) || run.race_provenance === "untested") return pill("untested", "NO RACE");
  return run.race_provenance === "measured" ? pill("measured", "MEASURED RACE") : pill("mock", "MOCK RACE");
}

function runShell(run) {
  const offline = run.mode === "offline";
  const title = run.request.sample ? run.request.sample.replace(/_/g, " ") : "Your task";
  return `
  <div class="run-head">
    <div><div class="faint small mono">WIND TUNNEL REPORT · ${esc(run.id)}</div>
      <h1 style="margin-top:6px;text-transform:capitalize">${esc(title)}</h1>
      <p class="problem">${esc(run.request.problem)}</p></div>
    <div class="run-meta">
      ${pill(run.mode)} <span id="race-pill">${racePill(run)}</span>
      <button class="btn btn-sm" id="share">Copy share link</button>
      <a class="btn btn-sm" href="/new" data-link>Run your own</a>
    </div>
  </div>
  ${
    offline
      ? `<div class="banner mock"><span>⚠</span><div><strong>OFFLINE demo mode.</strong> This server has no Nemotron API key yet. The paper search is
     <b>real</b>, but method cards and designs come from keyword heuristics, and the race answers are <b>SIMULATED</b> (MOCK) from fixed per-tier
     accuracy rates. Treat every number below as a demo of the pipeline, <b>not</b> as a result.</div></div>`
      : ""
  }
  <div class="stepper" id="stepper"></div>
  <p class="form-error mt" id="run-error" hidden></p>
  <section class="section" id="sec-a"></section>
  <section class="section" id="sec-b"></section>
  <section class="section" id="sec-c"></section>
  <section class="section" id="sec-d"></section>
  <section class="section" id="sec-e"></section>
  <section class="section" id="sec-f"></section>`;
}

function bindShare(run) {
  document.getElementById("share").onclick = async () => {
    const url = `${location.origin}/r/${run.id}`;
    try {
      await navigator.clipboard.writeText(url);
      toast("Link copied");
    } catch (e) {
      prompt("Share this link", url);
    }
  };
}

function stageDetail(s) {
  if (s.status === "pending") return s.detail || "waiting";
  if (s.status === "running") return s.calls ? `${s.calls} model call${s.calls === 1 ? "" : "s"} so far` : s.detail || "running…";
  const time = s.seconds != null && s.status === "done" && s.key !== "breed" ? ` · ${secs(s.seconds)}` : "";
  return `${s.detail || s.status}${time}`;
}

function updateRun(run, rendered) {
  const stages = Object.fromEntries(run.stages.map((s) => [s.key, s]));
  document.getElementById("stepper").innerHTML = run.stages
    .map(
      (s) => `<div class="st ${esc(s.status)}"><div class="t"><span class="dot"></span>${esc(s.title)}</div>
      <div class="d" title="${esc(s.detail)}">${esc(stageDetail(s))}</div></div>`,
    )
    .join("");
  document.getElementById("race-pill").innerHTML = racePill(run);
  if (run.error) {
    const el = document.getElementById("run-error");
    el.textContent = `This run failed: ${run.error}`;
    el.hidden = false;
  }
  const status = (k) => [stages[k]?.status, stages[k]?.calls];
  const done = !!run.report;
  const legacy = run.report_version == null && run.status === "done";
  const sections = {
    "sec-a": [() => [done, legacy, run.status], () => (legacy ? legacyNotice(run) : sectionA(run))],
    "sec-b": [() => [done, legacy, status("research")], () => (legacy ? "" : sectionB(run))],
    "sec-c": [() => [done, legacy, status("design_candidates")], () => (legacy ? "" : sectionC(run))],
    "sec-d": [() => [done, legacy, status("race")], () => (legacy ? "" : sectionD(run))],
    "sec-e": [() => [done, legacy, status("race"), status("breed")], () => (legacy ? "" : sectionE(run))],
    "sec-f": [() => [done, legacy, run.status, status("llm_pick")], () => (legacy ? "" : sectionF(run))],
  };
  for (const [id, [sig, render]] of Object.entries(sections)) {
    const key = JSON.stringify(sig());
    if (rendered[id] === key) continue;
    rendered[id] = key;
    const el = document.getElementById(id);
    el.innerHTML = render();
    renderDiagrams(el);
    el.querySelectorAll("[data-more]").forEach((b) =>
      b.addEventListener("click", () => {
        b.parentElement.querySelectorAll(".method[hidden]").forEach((m) => (m.hidden = false));
        b.remove();
      }),
    );
    el.querySelectorAll("[data-race-tab]").forEach((b) =>
      b.addEventListener("click", () => {
        const i = b.dataset.raceTab;
        el.querySelectorAll("[data-race-tab]").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
        el.querySelectorAll("[data-race-panel]").forEach((p) => (p.hidden = p.dataset.racePanel !== i));
      }),
    );
  }
}

function legacyNotice(run) {
  return `<div class="card skeleton">This report was made by an earlier version of Wind Tunnel and can no longer be shown.
    ${run.request.sample ? `<p class="mt"><button class="btn btn-sm" data-rerun>Run the sample again</button></p>` : `<p class="mt"><a href="/new" data-link>Start a new run</a></p>`}</div>`;
}

document.addEventListener("click", (e) => {
  const b = e.target.closest("[data-rerun]");
  if (!b) return;
  const id = location.pathname.split("/").pop();
  api(`/api/runs/${id}`)
    .then((run) => startRun({ sample: run.request.sample }, b))
    .catch((err) => toast(err.message));
});

const head = (letter, title, sub, pills = "") =>
  `<div class="section-head"><span class="section-letter">${letter}</span><h2>${title}</h2>${pills}${sub ? `<span class="sub">${sub}</span>` : ""}</div>`;

const waiting = (msg) => `<div class="card skeleton"><span class="spinner"></span>${esc(msg)}</div>`;

function stageWait(run, key, runningMsg, pendingMsg) {
  const s = run.stages.find((x) => x.key === key) || {};
  if (s.status === "running") return waiting(s.calls ? `${runningMsg} (${s.calls} model calls so far)` : runningMsg);
  if (s.status === "error" || run.status === "error") return `<div class="card skeleton">This step did not finish.</div>`;
  if (s.status === "done" || s.status === "skipped") return waiting("Finishing the report…");
  return waiting(pendingMsg);
}

function cardIndex(report) {
  return Object.fromEntries((report.research?.cards || []).map((c) => [c.source_key, c]));
}

const isMock = (report) => report.provenance === "mock";
const unmeasuredCount = (x) => (typeof x === "number" ? x : Array.isArray(x) ? x.length : Object.keys(x || {}).length);
const ciText = (ci) => (ci && ci[0] != null && ci[1] != null ? `${pct(ci[0])}–${pct(ci[1])}` : "not measured");
const prettyMode = (m) => String(m || "").replace(/_/g, " ");
const tiersOf = (d) => [...(d.tiers || [d.model])].sort((a, b) => TIERS.indexOf(a) - TIERS.indexOf(b));

// The cost ratio of the LLM default over the recommendation, when both were measured in the same race.
function costRatio(rec, vs) {
  if (!rec || !vs || !vs.comparable) return null;
  const a = vs.cost_per_1k_usd;
  const b = rec.cost_per_1k_usd;
  return a > 0 && b > 0 ? a / b : null;
}

// Whether the deciding race ran on held-out examples. Reports written before final_on_holdout existed fall back to the split.
const finalOnHoldout = (b) => (b.final_on_holdout != null ? !!b.final_on_holdout : !!(b.holdout_ids || []).length);

function selectionBiasPresent(report) {
  const b = report.breeding;
  return !!(b && b.enabled && (b.bred || []).length && !finalOnHoldout(b));
}

// A comparison design with no measured runs has no accuracy and no cost, not a cost of $0.
const unmeasured = (d) => !d || d.accuracy == null || !d.n;

// ---------- A. Without vs With

function sectionA(run) {
  const report = run.report;
  const title = head("A", "Without vs With Wind Tunnel", "The same question, answered from memory and answered with evidence.");
  if (!report) return title + stageWait(run, "plain_llm_answer", "Asking a plain LLM…", "Starting…");
  const plain = report.plain_answer;
  const v = report.verdict;
  const mock = isMock(report);
  const sim = mock ? `<span class="sim">SIMULATED</span>` : "";
  const llmDefault = (report.candidates || []).find((c) => c.role === "llm-default");
  const cards = report.research?.cards || [];

  const without = plain.available
    ? `<blockquote>${esc(plain.text)}</blockquote>`
    : `<p class="muted">${esc(plain.text)}</p>
      <p class="small muted">Asked from memory, LLMs most often recommend a <b>multi-agent orchestrator on a large model</b>.
      ${
        v.tested
          ? `Wind Tunnel puts that design${llmDefault ? ` (<code>${esc(llmDefault.id)}</code>)` : ""} in the race so you can see what it really costs.`
          : `Wind Tunnel would race that design${llmDefault ? ` (<code>${esc(llmDefault.id)}</code>)` : ""} once you add labelled examples, so you can see what it really costs.`
      }</p>`;

  let withHtml;
  if (!v.tested)
    withHtml = `<div class="big">${(report.candidates || []).length} designs found, untested</div>
      <p class="muted small">No labelled examples were given, so nothing was raced. Add 10+ examples to get a measured verdict.</p>`;
  else if (!v.recommended)
    withHtml = `<div class="big">${v.aborted ? "INCOMPLETE: no recommendation" : "No recommendation"}</div><p class="muted">${esc(v.text)}</p>`;
  else {
    const rec = v.recommended;
    const vs = v.vs_llm_default;
    const ratio = costRatio(rec, vs);
    let third;
    if (!vs) third = [`–`, llmDefault && llmDefault.id === rec.id ? "it is the LLM default" : "LLM default not measured"];
    else if (vs.accuracy == null) third = ["–", "LLM default not measured (infrastructure errors)"];
    else if (ratio == null) third = ["–", vs.comparable ? "cost vs the LLM default" : "LLM default measured in another race"];
    else if (ratio >= 1) third = [`${ratio.toFixed(1)}×`, `cheaper than the LLM default${mock ? " (simulated)" : ""}`];
    else third = [`${(1 / ratio).toFixed(1)}×`, `the LLM default's cost${mock ? " (simulated)" : ""}`];
    const bred = rec.origin === "bred";
    withHtml = `
      <div><div class="big">${esc(designName(rec))}</div><div class="faint small mono">${esc(rec.id)}${bred ? ` <span class="badge badge-bred">bred</span>` : ""}</div></div>
      <div class="facts">
        <div class="fact"><div class="v">${pct(rec.accuracy)}${sim}</div><div class="k">accuracy (95% CI ${ciText(rec.accuracy_ci)}, n=${rec.n})</div></div>
        <div class="fact"><div class="v">${usd(rec.cost_per_1k_usd)}</div><div class="k">per 1k requests${mock ? " (simulated)" : ""}</div></div>
        <div class="fact"><div class="v">${third[0]}</div><div class="k">${third[1]}</div></div>
      </div>
      ${bred && selectionBiasPresent(report) ? `<p class="small bias-inline">⚠ Bred design: its score is optimistic (selection bias, no held-out split). See section E.</p>` : ""}
      <p class="small">${esc(v.text)}</p>`;
  }
  const checks = (items) =>
    `<ul class="checks">${items.map(([ok, t]) => `<li><span class="${ok ? "y" : "n"}">${ok ? "✓" : "✗"}</span><span>${t}</span></li>`).join("")}</ul>`;
  const nCands = (report.candidates || []).length;
  const nBred = (report.breeding?.bred || []).length;

  return `${title}
  <div class="vs">
    <div class="card vs-card without">
      <div class="lbl"><span>Without · plain LLM</span>${pill(plain.available ? "nemotron" : "unavailable")}</div>
      ${without}
      ${checks([
        [false, "searched papers or the web"],
        [false, "compared alternative designs"],
        [false, "tested on your examples"],
        [false, "measured cost and latency"],
      ])}
    </div>
    <div class="card vs-card with" style="position:relative">
      <div class="lbl"><span>With Wind Tunnel · verdict</span>${pill(v.tested ? (mock ? "mock" : "measured") : "untested")}</div>
      ${withHtml}
      ${checks([
        [cards.length > 0, `searched papers: <b>${cards.length}</b> sources, <b>${cards.filter((c) => c.verified).length}</b> with verified IDs`],
        [
          v.tested && nCands > 1,
          v.tested
            ? `compared <b>${nCands}</b> candidate architectures${nBred ? ` and <b>${nBred}</b> bred design${nBred === 1 ? "" : "s"}` : ""}`
            : `proposed <b>${nCands}</b> candidate architectures, not compared (no examples)`,
        ],
        [v.tested, v.tested ? `tested on <b>${v.n_examples}</b> labelled examples${mock ? " (SIMULATED answers)" : ""}` : "not tested (no examples)"],
        [v.tested, v.tested ? `measured cost and latency${mock ? " (SIMULATED)" : ""}` : "cost and latency not measured"],
      ])}
    </div>
  </div>`;
}

// ---------- B. Methods map

function sectionB(run) {
  const report = run.report;
  if (!report) return head("B", "Methods found") + stageWait(run, "research", "Searching papers and extracting methods…", "Waiting to search…");
  const res = report.research;
  const cards = cardIndex(report);
  const groups = Object.entries(res.methods_map || {});
  const modePill = pill(res.mode, res.mode === "offline" ? "OFFLINE CARDS" : res.mode === "mixed" ? "PARTLY OFFLINE CARDS" : "NEMOTRON CARDS");
  const verified = res.cards.filter((c) => c.verified).length;
  const sub = `${res.cards.length} sources from real searches, ${verified} with verified IDs, grouped by approach family.${
    res.mode === "offline"
      ? " OFFLINE: families and summaries come from a keyword heuristic, so some labels may be wrong."
      : res.mode === "mixed"
        ? " PARTLY OFFLINE: some cards or the search plan fell back to a keyword heuristic (see the warnings)."
        : ""
  }`;
  if (!res.cards.length) return `${head("B", "Methods found", sub, modePill)}<div class="card skeleton">The search returned no sources, so the candidates below are generic templates.</div>`;
  return `${head("B", "Methods found", sub, modePill)}
  ${groups
    .map(([family, keys]) => {
      const list = keys.map((k) => cards[k]).filter(Boolean);
      return `<div class="card family-group"><h3>${esc(FAMILY_NAMES[family] || family)} <span class="count">${list.length} source${list.length === 1 ? "" : "s"}</span></h3>
      <div class="methods">${list.map((c, i) => methodCard(c, i >= 4)).join("")}</div>
      ${list.length > 4 ? `<button class="more-btn" data-more>Show ${list.length - 4} more</button>` : ""}</div>`;
    })
    .join("")}
  ${res.warnings?.length ? `<p class="faint small mt">${res.warnings.length} search warning(s), e.g. ${esc(res.warnings[0])}</p>` : ""}`;
}

function methodCard(c, hidden) {
  const cite = c.citation || {};
  const authors = (cite.authors || []).slice(0, 3).join(", ") + ((cite.authors || []).length > 3 ? " et al." : "");
  const flags = (c.flags || []).map((f) => `<span class="badge badge-${esc(f)}">${esc(f)}</span>`).join("");
  const r = Math.max(0, Math.min(3, Number(c.relevance) || 0));
  const rel = "●".repeat(r) + "○".repeat(3 - r);
  // The link text is the indexed title; the method name is written by the model (or the heuristic), so it is a label under it.
  const title = cite.title || c.method_name;
  const method = c.method_name && c.method_name !== title ? `<div class="small"><span class="faint">Method:</span> ${esc(c.method_name)}</div>` : "";
  return `<div class="method"${hidden ? " hidden" : ""}>
    <a class="title" href="${esc(cite.url)}" target="_blank" rel="noopener">${esc(title)}</a>
    ${method}
    <div class="meta">${esc(authors || "Unknown authors")} · ${esc(cite.year || "n.d.")}${cite.venue ? ` · ${esc(cite.venue)}` : ""}</div>
    ${c.mechanism ? `<p class="mech">${esc(c.mechanism)}</p>` : ""}
    <div class="tags">${c.verified ? `<span class="badge badge-verified" title="Has an arXiv ID or DOI from a scholarly index">✓ verified</span>` : `<span class="badge">unverified</span>`}${flags}
      <span class="badge" title="${esc(c.relevance_reason)}">relevance <span class="rel">${rel}</span></span></div>
  </div>`;
}

// ---------- C. Architectures

function sectionC(run) {
  const report = run.report;
  if (!report) return head("C", "Candidate architectures") + stageWait(run, "design_candidates", "Designing candidates…", "Waiting for research…");
  const cands = report.candidates || [];
  const cards = cardIndex(report);
  const winner = report.verdict?.recommended_id;
  const heuristic = (c) => String(c.generated_by).includes("heuristic");
  const offline = cands.every(heuristic);
  const mixed = !offline && cands.some((c) => c.role === "candidate" && heuristic(c));
  return `${head(
    "C",
    "Candidate architectures",
    "Each diagram is drawn from the same graph that runs in the race, so it can't drift from what was tested.",
    offline ? pill("offline", "OFFLINE DESIGNS") : mixed ? pill("mixed", "PARTLY OFFLINE DESIGNS") : pill("nemotron", "NEMOTRON DESIGNS"),
  )}
  ${tierLegend()}
  <div class="grid grid-2">${cands
    .map((c) => {
      const based = (c.based_on || []).map((k) => cards[k]).filter(Boolean);
      return `<div class="card arch${c.id === winner ? " winner" : ""}">
      <div class="arch-head"><div><h3>${esc(designName(c))}</h3><div class="id">${esc(c.id)}</div></div>
        <div class="row" style="gap:6px;justify-content:flex-end">${c.id === winner ? `<span class="badge badge-win">recommended</span>` : ""}
        ${c.raced === false ? `<span class="badge">not raced</span>` : ""}
        <span class="badge badge-role-${esc(c.role)}" title="${esc(ROLE_HINTS[c.role] || "")}">${esc(ROLE_NAMES[c.role] || c.role)}</span></div></div>
      ${diagram(c.mermaid)}
      <div class="row" style="gap:6px">${tiersOf(c).map(tier).join("")}<span class="small muted">${esc(c.description)}</span></div>
      <p class="small">${esc(c.rationale)}</p>
      <div class="based-on">${
        based.length
          ? `<b>Based on</b><ol>${based
              .map((b) => `<li><a href="${esc(b.citation.url)}" target="_blank" rel="noopener">${esc(b.citation.title || b.method_name)}</a> <span class="faint">(${esc(b.citation.year || "n.d.")})</span>${b.verified ? ` <span class="badge badge-verified" title="Has an arXiv ID or DOI from a scholarly index">✓</span>` : ""}</li>`)
              .join("")}</ol>`
          : `<span class="faint">${c.role === "baseline" ? "Reference point, not drawn from a paper." : c.role === "llm-default" ? "What an LLM usually recommends from memory." : "No matching sources found for this family."}</span>`
      }</div>
    </div>`;
    })
    .join("")}</div>`;
}

// ---------- D. Race

function raceBoard(race, report, extraRows = []) {
  const mock = isMock(report);
  const cands = Object.fromEntries((report.candidates || []).map((c) => [c.id, c]));
  const lineage = report.breeding?.lineage || {};
  const rank = (d) => (d.eliminated_round != null ? 2 : d.accuracy == null ? 1 : 0);
  const rows = [...race.designs].sort((a, b) => rank(a) - rank(b) || (b.accuracy ?? -1) - (a.accuracy ?? -1));
  const rec = race.recommended_id;
  const tied = new Set(race.tied_ids || []);
  const inconclusive = new Set(race.inconclusive_ids || []);
  const row = (d, outIn) => {
    const out = outIn || d.eliminated_round != null;
    const unmeasured = unmeasuredCount(d.unmeasured);
    const errors = unmeasuredCount(d.errors);
    const measured = d.accuracy != null;
    const c = cands[d.id] || {};
    const badges = [
      d.id === rec && !outIn ? `<span class="badge badge-win">winner</span>` : "",
      d.id !== rec && tied.has(d.id) && !outIn ? `<span class="badge badge-tied">tied</span>` : "",
      inconclusive.has(d.id) && !outIn ? `<span class="badge" title="Neither shown worse than the best nor shown close to it">inconclusive</span>` : "",
      outIn ? `<span class="badge badge-out">out in ${esc(outIn)}</span>` : d.eliminated_round != null ? `<span class="badge badge-out">out · round ${d.eliminated_round}</span>` : "",
      lineage[d.id] ? `<span class="badge badge-bred" title="Bred from ${esc(lineage[d.id].parent_id)} via ${esc(lineage[d.id].mechanism)}">bred</span>` : "",
      c.role && c.role !== "candidate" ? `<span class="badge badge-role-${esc(c.role)}">${esc(c.role)}</span>` : "",
      !measured ? `<span class="badge badge-unmeasured">not measured</span>` : "",
    ].join(" ");
    const [lo, hi] = d.accuracy_ci || [null, null];
    const ci = measured && lo != null && hi != null
      ? `<div class="ci" title="${pct(lo)}–${pct(hi)}"><div class="track"></div>
          <div class="range" style="left:${lo * 100}%;width:${Math.max(0.5, (hi - lo) * 100)}%"></div>
          <div class="pt" style="left:${d.accuracy * 100}%"></div></div>
          <div class="small" style="font-variant-numeric:tabular-nums"><b>${pct(d.accuracy)}</b> <span class="faint">(${pct(lo)}–${pct(hi)})</span></div>`
      : `<div class="small faint">No accuracy: ${unmeasured ? `all ${unmeasured} runs` : "its runs"} failed for infrastructure reasons and were left unscored.</div>`;
    const notes = [];
    if (unmeasured && measured) notes.push(`${unmeasured} run${unmeasured === 1 ? "" : "s"} not measured (infrastructure errors), left out of the score`);
    if (errors) notes.push(`${errors} run${errors === 1 ? "" : "s"} raised errors and were scored as wrong`);
    if (out && d.elimination_reason) notes.push(`Eliminated: ${d.elimination_reason}`);
    return `<div class="board-row${out ? " out" : ""}${d.id === rec && !outIn ? " win" : ""}">
      <div class="name">${esc(designName(d))} ${badges}<span class="id">${esc(d.id)} · n=${d.n}</span>
        <span class="tiers">${tiersOf(d).map(tier).join("")}</span></div>
      <div class="ci-cell">${ci}</div>
      <div class="num">${measured ? usd(d.cost_per_1k_usd) : "–"}${mock && measured ? "<small>simulated</small>" : ""}</div>
      <div class="num">${secs(d.p50_latency_s)}${mock && d.p50_latency_s != null ? "<small>simulated</small>" : ""}</div>
      ${notes.length ? `<div class="why">${notes.map(esc).join(" · ")}</div>` : ""}
    </div>`;
  };
  return `<div class="card board">${mock ? `<div class="watermark">MOCK</div>` : ""}
    <div class="board-row head"><span>Design</span><span class="ci-cell">Accuracy · 95% CI</span><span class="num">$/1k</span><span class="num">p50</span></div>
    ${rows.map((d) => row(d)).join("")}
    ${extraRows.map(([d, role]) => row(d, role)).join("")}
  </div>`;
}

function sectionD(run) {
  const report = run.report;
  const st = Object.fromEntries(run.stages.map((s) => [s.key, s]));
  if (!report) {
    const p = run.race_provenance === "measured" ? pill("measured", "MEASURED") : hasExamples(run) ? pill("mock", "MOCK RACE") : "";
    return head("D", "The race", "", p) + stageWait(run, "race", "Racing the candidates…", "Waiting for designs…");
  }
  const race = report.race;
  if (!race)
    return `${head("D", "The race", "")}<div class="card skeleton">No labelled examples, so there was nothing to race. <a href="/new" data-link>Add examples</a> to get a measured verdict.</div>`;
  const mock = isMock(report);
  const pills = pill(mock ? "mock" : "measured", mock ? "MOCK RACE" : "MEASURED");
  const races = report.breeding?.races?.length ? report.breeding.races : [race];
  const sub = `Examples are revealed in rounds (${race.cutoffs.join(", ")}). After each round, designs that are confidently worse than the leader (exact sign test, corrected for multiple comparisons) are eliminated. Of the designs tied with the best, the cheapest wins.${
    races.length > 1 ? ` ${races.length} races ran: the first on the generated candidates, then survivors plus bred designs.` : ""
  }`;
  // Designs eliminated in an earlier race are listed, faded, under the deciding race.
  const finalIds = new Set(race.designs.map((d) => d.id));
  const dropped = [];
  const seen = new Set();
  for (const past of races.slice(0, -1))
    for (const d of past.designs)
      if (!finalIds.has(d.id) && !seen.has(d.id)) {
        seen.add(d.id);
        dropped.push([d, past.role]);
      }
  let boards;
  if (races.length > 1) {
    const last = races.length - 1;
    boards = `<div class="tabs" role="tablist">${races
      .map((r, i) => `<button class="tab" role="tab" data-race-tab="${i}" aria-selected="${i === last}">${esc(r.role)} · ${r.task.n_examples} ex.${i === last ? " · deciding" : ""}</button>`)
      .join("")}</div>
      ${races
        .map(
          (r, i) => `<div data-race-panel="${i}"${i === last ? "" : " hidden"}>${i === last ? raceBoard(race, report, dropped) : raceBoard(r, report)}
          ${i === last ? "" : `<p class="small muted mt">${esc(r.verdict)}</p>`}</div>`,
        )
        .join("")}`;
  } else boards = raceBoard(race, report, dropped);
  return `${head("D", "The race", sub, pills)}${boards}
    ${race.aborted ? `<div class="banner"><span>⚠</span><div><strong>INCOMPLETE.</strong> ${esc(race.aborted)}</div></div>` : ""}
    <p class="faint small mt">Decided in: <b>${esc(race.role)}</b>. ${st.race?.detail ? esc(st.race.detail) : ""}</p>`;
}

// ---------- E. Breeding

function sectionE(run) {
  const report = run.report;
  const title = (pills = "") => head("E", "Diagnose & breed", "Why the losers failed, which fix from another field was tried, and how the bred design scored.", pills);
  if (!report) {
    const s = run.stages.find((x) => x.key === "breed") || {};
    if (s.status === "skipped") return title() + `<div class="card skeleton">${esc(s.detail || "Skipped.")}</div>`;
    return title() + stageWait(run, "race", "Racing, then diagnosing the losers and breeding fixes…", "Breeding runs after the first race.");
  }
  const b = report.breeding;
  if (!b) return title() + `<div class="card skeleton">No labelled examples, so nothing was raced or bred.</div>`;
  if (!b.enabled) return title() + `<div class="card skeleton">${esc(b.note)}</div>`;
  const mock = isMock(report);
  const help = b.failure_mode_help || {};
  const bias = selectionBiasPresent(report);
  const held = (b.holdout_ids || []).length;
  const biasBox = `<div class="bias ${bias ? "bias-warn" : "bias-ok"}">
      <div class="bias-title">${bias ? "⚠ Selection bias: bred scores are optimistic" : finalOnHoldout(b) ? "✓ Held-out confirmation" : "Selection bias"}</div>
      <p>${esc(b.note)}</p>
      ${held ? `<p class="small muted">Search split: ${(b.search_ids || []).length} examples · held-out split: ${held} examples.</p>` : ""}</div>`;

  const final = report.race;
  const finalRows = Object.fromEntries((final?.designs || []).map((d) => [d.id, d]));
  const diagnoses = (b.diagnoses || []).flatMap((gen, gi) => gen.map((d) => ({ ...d, generation: gi + 1 })));
  const diagHtml = diagnoses.length
    ? `<div class="grid grid-2">${diagnoses
        .map(
          (d) => `<div class="card diag">
        <div class="arch-head"><div><h3>${esc(d.design_id)}</h3><div class="id">${pct(d.accuracy)} on n=${d.n}${mock ? " (simulated)" : ""} · generation ${d.generation}</div></div>
          <span class="badge" title="Who read the failures">${esc(d.produced_by)}</span></div>
        <ul class="modes">${(d.failure_modes || [])
          .map(
            (m) => `<li><span class="mode" title="${esc(help[m.mode] || "")}">${esc(prettyMode(m.mode))}</span>${m.share ? ` <span class="faint small">${pct(m.share)} of examples</span>` : ""}
              ${(m.evidence || []).length ? `<div class="small muted">${esc(m.evidence.join(" · "))}</div>` : ""}</li>`,
          )
          .join("")}</ul>
        ${(d.sample_failures || []).length ? `<details class="small"><summary>${d.sample_failures.length} failure example${d.sample_failures.length === 1 ? "" : "s"}</summary><ul class="fails">${d.sample_failures
          .map((f) => `<li><span class="muted">${esc(f.input)}</span><br>expected <b>${esc(f.expected)}</b>, got <b>${esc(f.got || (f.error ? "error" : "(empty)"))}</b>${f.error ? ` <span class="faint">${esc(f.error)}</span>` : ""}</li>`)
          .join("")}</ul></details>` : ""}
      </div>`,
        )
        .join("")}</div>`
    : `<div class="card skeleton">No design had failures to diagnose.</div>`;

  const bred = b.bred || [];
  const bredHtml = bred.length
    ? `${tierLegend()}<div class="grid grid-2">${bred
        .map((d) => {
          const m = d.mechanism || {};
          const row = finalRows[d.id];
          const win = d.id === report.verdict?.recommended_id;
          let score;
          if (!row) score = `<span class="faint">Not in the deciding race (eliminated earlier; see the race history in D).</span>`;
          else if (row.accuracy == null) score = `<span class="faint">Not measured: its runs failed for infrastructure reasons.</span>`;
          else
            score = `<b>${pct(row.accuracy)}</b> <span class="faint">(95% CI ${ciText(row.accuracy_ci)}, n=${row.n})</span> · ${usd(row.cost_per_1k_usd)}/1k${mock ? ` <span class="sim">SIMULATED</span>` : ""}${
              row.eliminated_round != null ? ` · <span class="faint">eliminated in round ${row.eliminated_round}</span>` : ""
            }${bias ? ` <span class="badge badge-bias" title="Chosen using failures on the same examples it was raced on">optimistic</span>` : ""}`;
          return `<div class="card arch${win ? " winner" : ""}">
          <div class="arch-head"><div><h3>${esc(designName(d))}</h3><div class="id">${esc(d.id)}</div></div>
            <div class="row" style="gap:6px;justify-content:flex-end">${win ? `<span class="badge badge-win">recommended</span>` : ""}<span class="badge badge-bred">bred</span></div></div>
          <div class="mechanism"><div class="small faint">Fix borrowed from <b>${esc(m.field || "another field")}</b></div>
            <div class="mech-name">${m.source?.url ? `<a href="${esc(m.source.url)}" target="_blank" rel="noopener">${esc(m.name)}</a>` : esc(m.name)}</div>
            <div class="small muted">Applied to <code>${esc(d.parent_id)}</code> to fix <span class="mode" title="${esc(help[d.failure_mode] || "")}">${esc(prettyMode(d.failure_mode))}</span></div></div>
          ${diagram(d.mermaid)}
          <div class="row" style="gap:6px">${tiersOf(d).map(tier).join("")}<span class="small muted">${esc(d.description)}</span></div>
          <div class="small">Score: ${score}</div>
        </div>`;
        })
        .join("")}</div>`
    : `<div class="card skeleton">No mechanism produced a new design for these failures.</div>`;

  return `${title(pill(mock ? "mock" : "measured", mock ? "MOCK" : "MEASURED"))}${biasBox}
    <h3 class="sub-head">Losers diagnosed</h3>${diagHtml}
    <h3 class="sub-head">Bred designs</h3>${bredHtml}`;
}

// ---------- F. Verdict, LLM pick, receipt and limitations

function sectionF(run) {
  const report = run.report;
  if (!report) {
    if (run.status === "error") return "";
    return head("F", "Verdict & limitations") + stageWait(run, "llm_pick", "Asking the LLM which design it would pick…", "Waiting for the race…");
  }
  const v = report.verdict;
  const mock = isMock(report);
  const race = report.race;
  const designs = Object.fromEntries((race?.designs || []).map((d) => [d.id, d]));
  const verdict = `<div class="card verdict"><div class="lbl faint small mono">VERDICT ${mock ? "· SIMULATED, NOT A RESULT" : v.tested ? "· MEASURED" : "· UNTESTED"}</div>
    <div class="big">${esc(v.headline)}</div>
    ${
      !v.vs_llm_default
        ? ""
        : unmeasured(v.vs_llm_default)
          ? `<p class="muted small">The design LLMs usually recommend, <code>${esc(v.vs_llm_default.id)}</code>, was not measured: its runs failed for infrastructure reasons and were left unscored.</p>`
          : `<p class="muted small">The design LLMs usually recommend, <code>${esc(v.vs_llm_default.id)}</code>, scored ${pct(v.vs_llm_default.accuracy)} (n=${v.vs_llm_default.n}) at ${usd(v.vs_llm_default.cost_per_1k_usd)}/1k${
              v.vs_llm_default.eliminated ? `, and was eliminated: ${esc(v.vs_llm_default.elimination_reason)}` : ""
            }${v.vs_llm_default.comparable ? "" : ` (measured in ${esc(v.vs_llm_default.measured_in)}, so not directly comparable)`}.</p>`
    }
    ${v.recommended?.origin === "bred" && selectionBiasPresent(report) ? `<p class="small bias-inline">⚠ The recommendation is a bred design. Its score is optimistic until it is confirmed on held-out examples.</p>` : ""}
  </div>`;

  const adv = report.advisor;
  let advisor;
  if (adv.available) {
    const picked = v.advisor_pick?.measured || designs[adv.design_id];
    const note = !picked
      ? ""
      : unmeasured(picked)
        ? ": not measured (its runs failed for infrastructure reasons)"
        : `: ${mock ? "simulated" : "measured"} ${pct(picked.accuracy)} at ${usd(picked.cost_per_1k_usd)}/1k${
            adv.design_id === v.recommended_id ? " (agrees with the race)" : picked.eliminated || picked.eliminated_round != null ? ", eliminated" : ""
          }${picked.comparable === false ? ` (measured in ${esc(picked.measured_in)})` : ""}`;
    advisor = `<div class="card advisor"><div class="row" style="justify-content:space-between"><b>LLM would have picked</b>${pill("nemotron", `NEMOTRON ${String(adv.model).toUpperCase()}`)}</div>
      <p class="mt">${adv.design_id ? `<code>${esc(adv.design_id)}</code>${note}` : "No valid design named."}</p>
      <details class="mt small"><summary>Full reply</summary><p class="muted" style="white-space:pre-wrap">${esc(adv.reply)}</p></details></div>`;
  } else {
    const d = designs[adv.typical_default_id];
    advisor = `<div class="card advisor"><div class="row" style="justify-content:space-between"><b>LLM would have picked</b>${pill("unavailable")}</div>
      <p class="mt muted small">${esc(adv.reply)} ${
        !adv.typical_default_id
          ? ""
          : !race
            ? `The design LLMs usually recommend, <code>${esc(adv.typical_default_id)}</code>, would be raced once you add labelled examples.`
            : `The design LLMs usually recommend is in the race as <code>${esc(adv.typical_default_id)}</code>${
                !d ? "" : unmeasured(d) ? ": not measured (its runs failed for infrastructure reasons)" : `: ${pct(d.accuracy)} at ${usd(d.cost_per_1k_usd)}/1k${mock ? " (simulated)" : ""}`
              }.`
      }</p></div>`;
  }
  return `${head("F", "Verdict & limitations", "What to build, what an LLM would have said, what it cost and when not to trust it.")}${verdict}${advisor}${receiptAndLimits(run)}`;
}

const STAGE_NAMES = {
  plain_llm_answer: "Plain-LLM answer (no search, no tests)",
  research: "L1-L3 research: plan, search, method cards",
  design_candidates: "L4 candidate designs + diagrams",
  race: "L5 race + diagnose & breed",
  llm_pick: "LLM advisor pick",
  limitations: "L6 limitations",
};

function receiptLabel(e) {
  if (e.status === "skipped" || e.produced_by === "none") return "unavailable";
  if (e.produced_by === "mock") return "mock";
  // A rejected model reply produced nothing in the output, so it does not earn the stage a NEMOTRON label.
  const parts = String(e.produced_by)
    .split("+")
    .filter((p) => !p.includes("(rejected)"));
  const offline = parts.filter((p) => p.startsWith("offline-heuristic")).length;
  if (offline === parts.length) return "offline";
  // Designs always include template reference points (baseline, LLM default), so a heuristic there is expected.
  return offline && e.stage !== "design_candidates" ? "mixed" : "nemotron";
}

function producedBy(e) {
  const models = health?.models || {};
  return String(e.produced_by)
    .split("+")
    .map((p) => {
      const key = p.split("(")[0];
      if (p === "mock") return "MockClient (SIMULATED answers)";
      if (p === "none") return "not run";
      if (p === "offline-heuristic") return "offline heuristic";
      return models[key] ? `Nemotron ${TIER_NAMES[key] || key}${p.includes("(") ? ` ${p.slice(p.indexOf("("))}` : ""}` : p;
    })
    .join(" + ");
}

function receiptAndLimits(run) {
  const report = run.report;
  if (!report) return "";
  const models = health?.models || {};
  const receipt = `<div class="section-head mt" style="margin-top:28px"><h2 style="font-size:18px">Model receipt</h2><span class="sub">Which model or heuristic did each step, what it cost and how long it took.</span></div>
  <div class="card table-wrap"><table class="receipt"><thead><tr><th>Step</th><th>Done by</th><th>Label</th><th style="text-align:right">Cost</th></tr></thead><tbody>
  ${report.model_receipt
    .map((e) => {
      const ids = String(e.produced_by)
        .split("+")
        .map((p) => models[p.split("(")[0]])
        .filter(Boolean);
      return `<tr><td>${esc(STAGE_NAMES[e.stage] || e.stage)}${e.detail ? `<div class="faint small">${esc(e.detail)}</div>` : ""}</td>
      <td>${esc(producedBy(e))}${ids.length ? `<div class="mid">${esc(ids.join(", "))}</div>` : ""}</td><td>${pill(receiptLabel(e))}</td>
      <td class="num">${usd(e.cost_usd)}${e.simulated_cost_usd != null ? `<div class="faint small">${usd(e.simulated_cost_usd)} SIMULATED</div>` : ""}<div class="faint small">${secs(e.seconds)}</div></td></tr>`;
    })
    .join("")}
  <tr><td colspan="3"><b>Total real spend</b></td><td class="num"><b>${usd(report.total_cost_usd)}</b></td></tr></tbody></table></div>`;
  const limits = `<div class="card limits"><h3>⚠ When not to trust this verdict</h3><ul>${(report.limitations || []).map((l) => `<li>${esc(l)}</li>`).join("")}</ul></div>`;
  const warnings = (report.warnings || []).length
    ? `<details class="card warnings"><summary><b>${report.warnings.length} warning${report.warnings.length === 1 ? "" : "s"}</b> from this run</summary><ul>${report.warnings
        .map((w) => `<li>${esc(w)}</li>`)
        .join("")}</ul></details>`
    : "";
  return receipt + limits + warnings;
}

// ------------------------------------------------------------------ explore

async function viewExplore(token) {
  app.innerHTML = `<div class="page-head"><h1>Explore architecture families</h1>
    <p>The architecture families Wind Tunnel races today. Each diagram is generated from the same graph that runs, coloured by the Nemotron tier at each step.
    None of them wins everywhere, which is why you measure.</p></div>${tierLegend()}<div class="grid grid-2" id="fams"><div class="card skeleton"><span class="spinner"></span>Loading…</div></div>`;
  const fams = await api("/api/families").catch(() => []);
  if (token !== routeToken) return;
  const el = document.getElementById("fams");
  el.innerHTML = fams
    .map(
      (f) => `<div class="card fam"><div class="arch-head"><div><h3>${esc(f.name)}</h3><div class="id mono faint small">${esc(f.example_design)}</div></div>
      <div class="row" style="gap:6px">${f.tiers.map(tier).join("")}</div></div>
      <p class="muted">${esc(f.summary[0].toUpperCase() + f.summary.slice(1))}.</p>
      ${diagram(f.mermaid)}
      <div class="helps"><b>Helps when</b> ${esc(f.helps)}</div>
      <div class="hurts"><b>Hurts when</b> ${esc(f.hurts)}</div></div>`,
    )
    .join("");
  renderDiagrams(el);
}

// ------------------------------------------------------------------ editor

function viewEditor() {
  app.innerHTML = `<div class="page-head"><h1>Use Wind Tunnel in your editor</h1>
    <p>Claude Code, Cursor and other MCP clients can call Wind Tunnel while they write your code, so the agent builds the
    measured winner instead of whatever it would have guessed.</p></div>
  <div class="grid grid-2">
    <div class="card prose">
      <h2>Install <span class="badge">placeholder</span></h2>
      <p class="muted small">The MCP server is being finished. The exact command will be in <a href="${REPO}/blob/main/docs/MCP.md" target="_blank" rel="noopener">docs/MCP.md</a>. It will look like this:</p>
      <pre class="cmd">claude mcp add windtunnel -- \\
  uv run --directory /path/to/windtunnel windtunnel-mcp</pre>
      <p class="muted small">Tools: <code>find_methods</code>, <code>design_candidates</code>, <code>race</code>, <code>get_report</code>. Each returns structured results plus a link to the report on this site.</p>
    </div>
    <div class="card prose">
      <h2>Without vs with</h2>
      <div class="helps"><b>Without</b> "Add a ticket router." The agent writes a multi-agent pipeline on the largest model, because that is what it remembers. Nobody measures it.</div>
      <div class="hurts" style="background:color-mix(in srgb, var(--nv) 10%, var(--surface))"><b style="color:var(--nv-ink)">With</b> The agent calls <code>find_methods</code> and <code>race</code> on your examples, gets back the cheapest design tied with the best, with citations, and then writes that one.</div>
      <p class="small muted">Read the full guide in <a href="${REPO}/blob/main/docs/MCP.md" target="_blank" rel="noopener">docs/MCP.md</a>.</p>
    </div>
  </div>`;
}

// ------------------------------------------------------------------ boot

loadHealth().then(() => {
  if (location.pathname === "/new") route();
});
route();
