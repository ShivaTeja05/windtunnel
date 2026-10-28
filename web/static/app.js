// Wind Tunnel frontend: vanilla JS, no build step. Mermaid 10.9.1 renders the diagrams.

const MERMAID_URL = "https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.esm.min.mjs";
const REPO = "https://github.com/ShivaTeja05/windtunnel";

const FAMILY_NAMES = {
  single_call: "Single call",
  self_check: "Answer + self-check",
  self_consistency: "Self-consistency vote",
  plan_execute: "Plan, then execute",
  orchestrator_workers: "Orchestrator + workers",
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
const pct = (x) => (x == null ? "–" : `${Math.round(x * 100)}%`);
const usd = (x) => {
  if (x == null) return "–";
  if (x === 0) return "$0";
  if (x < 0.01) return `$${x.toFixed(4)}`;
  if (x < 10) return `$${x.toFixed(3)}`;
  return `$${x.toFixed(2)}`;
};
const secs = (x) => (x == null ? "–" : `${x < 10 ? x.toFixed(2) : x.toFixed(1)}s`);
const pill = (kind, label) => `<span class="pill pill-${esc(kind)}">${esc(label || MODE_TEXT[kind] || kind)}</span>`;
const tier = (t) => `<span class="tier tier-${esc(t)}">${esc(TIER_NAMES[t] || t)}</span>`;
const designName = (d) => `${FAMILY_NAMES[d.family] || d.family} · ${TIER_NAMES[d.model] || d.model}`;

async function api(path, opts = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  let body = null;
  try {
    body = await res.json();
  } catch (e) {}
  if (!res.ok) throw new Error((body && body.detail) || `${res.status} ${res.statusText}`);
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
  `<div class="legend"><span>Model that runs each step:</span>${["nano", "super", "ultra", "lightning"].map(tier).join("")}<span class="faint">· grey = input/output, no model</span></div>`;

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

async function viewRun(token, runId) {
  app.innerHTML = `<div class="skeleton"><span class="spinner"></span>Loading report…</div>`;
  const rendered = {};
  let first = true;
  while (token === routeToken) {
    let run;
    try {
      run = await api(`/api/runs/${runId}`);
    } catch (e) {
      app.innerHTML = `<div class="skeleton"><h1>Report not found</h1><p class="mt muted">${esc(e.message)}</p><p class="mt"><a href="/" data-link>Start a new run</a></p></div>`;
      return;
    }
    if (token !== routeToken) return;
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

function runShell(run) {
  const offline = run.mode === "offline";
  const title = run.request.sample ? run.request.sample.replace(/_/g, " ") : "Your task";
  return `
  <div class="run-head">
    <div><div class="faint small mono">WIND TUNNEL REPORT · ${esc(run.id)}</div>
      <h1 style="margin-top:6px;text-transform:capitalize">${esc(title)}</h1>
      <p class="problem">${esc(run.request.problem)}</p></div>
    <div class="run-meta">
      ${pill(run.mode)} ${
        run.request.sample || (run.request.examples || []).length
          ? pill(run.race_provenance === "mock" ? "mock" : "measured", run.race_provenance === "mock" ? "MOCK RACE" : "MEASURED RACE")
          : pill("untested", "NO RACE")
      }
      <button class="btn btn-sm" id="share">Copy share link</button>
      <a class="btn btn-sm" href="/new" data-link>Run your own</a>
    </div>
  </div>
  ${
    offline
      ? `<div class="banner mock"><span>⚠</span><div><strong>Offline demo mode.</strong> This server has no Nemotron API key yet. The paper search is
     <b>real</b>, but method cards and designs come from keyword heuristics, and the race answers are <b>simulated</b> (MOCK) from fixed per-tier
     accuracy rates. Treat every number below as a demo of the pipeline, <b>not</b> as a result.</div></div>`
      : ""
  }
  <div class="stepper" id="stepper"></div>
  <p class="form-error mt" id="run-error" hidden></p>
  <section class="section" id="sec-a"></section>
  <section class="section" id="sec-b"></section>
  <section class="section" id="sec-c"></section>
  <section class="section" id="sec-d"></section>`;
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

function updateRun(run, rendered) {
  const stages = Object.fromEntries(run.stages.map((s) => [s.key, s]));
  document.getElementById("stepper").innerHTML = run.stages
    .map(
      (s) => `<div class="st ${esc(s.status)}"><div class="t"><span class="dot"></span>${esc(s.title)}</div>
      <div class="d" title="${esc(s.detail)}">${esc(s.status === "pending" ? "waiting" : s.detail || s.status)}${s.seconds != null && s.status === "done" ? ` · ${secs(s.seconds)}` : ""}</div></div>`,
    )
    .join("");
  if (run.error) {
    const el = document.getElementById("run-error");
    el.textContent = `This run failed: ${run.error}`;
    el.hidden = false;
  }
  const r = run.results;
  const sections = {
    "sec-a": [() => [r.plain_llm, r.verdict, run.status], () => sectionA(run)],
    "sec-b": [() => [r.research?.finished_at, stages.research.status], () => sectionB(run)],
    "sec-c": [() => [r.candidates?.length, r.verdict?.recommended_id, stages.design.status], () => sectionC(run)],
    "sec-d": [() => [stages.race.status, stages.race.calls, stages.advisor.status, run.status], () => sectionD(run)],
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
  }
}

const head = (letter, title, sub, pills = "") =>
  `<div class="section-head"><span class="section-letter">${letter}</span><h2>${title}</h2>${pills}${sub ? `<span class="sub">${sub}</span>` : ""}</div>`;

const waiting = (msg) => `<div class="card skeleton"><span class="spinner"></span>${esc(msg)}</div>`;

function cardIndex(run) {
  return Object.fromEntries((run.results.research?.cards || []).map((c) => [c.source_key, c]));
}

// ---------- A. Without vs With

function sectionA(run) {
  const r = run.results;
  const plain = r.plain_llm;
  const v = r.verdict;
  const mock = run.race_provenance === "mock";
  const sim = mock ? `<span class="sim">SIMULATED</span>` : "";
  const cands = r.candidates || [];
  const llmDefault = cands.find((c) => c.role === "llm-default");

  let without;
  if (!plain) without = `<p class="muted"><span class="spinner"></span>Asking…</p>`;
  else if (plain.available) without = `<blockquote>${esc(plain.text)}</blockquote>`;
  else
    without = `<p class="muted">${esc(plain.text)}</p>
      <p class="small muted">Asked from memory, LLMs most often recommend a <b>multi-agent orchestrator on a large model</b>.
      Wind Tunnel puts that design${llmDefault ? ` (<code>${esc(llmDefault.id)}</code>)` : ""} in the race so you can see what it really costs.</p>`;

  let withHtml;
  if (!v) withHtml = `<p class="muted"><span class="spinner"></span>Researching, designing and racing…</p>`;
  else if (!v.tested)
    withHtml = `<div class="big">${v.candidates} designs found, untested</div>
      <p class="muted small">No labelled examples were given, so nothing was raced. Add 10+ examples to get a measured verdict.</p>`;
  else if (!v.recommended) withHtml = `<div class="big">No design met the constraints</div><p class="muted">${esc(v.verdict)}</p>`;
  else {
    const rec = v.recommended;
    const vs = v.vs_llm_default;
    withHtml = `
      <div><div class="big">${esc(designName(rec))}</div><div class="faint small mono">${esc(rec.id)}</div></div>
      <div class="facts">
        <div class="fact"><div class="v">${pct(rec.accuracy)}${sim}</div><div class="k">accuracy (95% CI ${pct(rec.accuracy_ci[0])}–${pct(rec.accuracy_ci[1])})</div></div>
        <div class="fact"><div class="v">${usd(rec.cost_per_1k_usd)}</div><div class="k">per 1k requests${mock ? " (simulated)" : ""}</div></div>
        <div class="fact"><div class="v">${vs ? `${vs.cost_ratio.toFixed(1)}×` : "–"}</div><div class="k">${vs ? `cheaper than the LLM default` : "it is the LLM default"}</div></div>
      </div>
      <p class="small">${esc(v.verdict)}</p>`;
  }
  const checks = (items) =>
    `<ul class="checks">${items.map(([ok, t]) => `<li><span class="${ok ? "y" : "n"}">${ok ? "✓" : "✗"}</span><span>${t}</span></li>`).join("")}</ul>`;

  return `${head("A", "Without vs With Wind Tunnel", "The same question, answered from memory and answered with evidence.")}
  <div class="vs">
    <div class="card vs-card without">
      <div class="lbl"><span>Without · plain LLM</span>${plain ? pill(plain.available ? "nemotron" : "unavailable") : ""}</div>
      ${without}
      ${checks([
        [false, "searched papers or the web"],
        [false, "compared alternative designs"],
        [false, "tested on your examples"],
        [false, "measured cost and latency"],
      ])}
    </div>
    <div class="card vs-card with" style="position:relative">
      <div class="lbl"><span>With Wind Tunnel · verdict</span>${v ? pill(v.tested ? (mock ? "mock" : "measured") : "untested") : ""}</div>
      ${withHtml}
      ${
        v
          ? checks([
              [v.sources > 0, `searched papers: <b>${v.sources}</b> sources, <b>${v.verified_sources}</b> with verified IDs`],
              [v.candidates > 1, `compared <b>${v.candidates}</b> candidate architectures`],
              [v.tested, v.tested ? `tested on <b>${v.n_examples}</b> labelled examples${mock ? " (simulated answers)" : ""}` : "not tested (no examples)"],
              [v.tested, v.tested ? `measured cost and latency${mock ? " (simulated)" : ""}` : "cost and latency not measured"],
            ])
          : ""
      }
    </div>
  </div>`;
}

// ---------- B. Methods map

function sectionB(run) {
  const res = run.results.research;
  const st = run.stages.find((s) => s.key === "research");
  if (!res) return head("B", "Methods found") + waiting(st.status === "running" ? `${st.detail || "Searching papers"}…` : "Waiting to search…");
  const cards = cardIndex(run);
  const groups = Object.entries(res.methods_map || {});
  const modePill = pill(res.mode, res.mode === "offline" ? "OFFLINE CARDS" : "NEMOTRON CARDS");
  const verified = res.cards.filter((c) => c.verified).length;
  const sub = `${res.cards.length} sources from real searches (${esc((run.receipt.find((x) => x.what.startsWith("L2")) || {}).what?.split("from ")[1] || "papers")}), ${verified} with verified IDs, grouped by approach family.${
    res.mode === "offline" ? " Families and summaries come from a keyword heuristic, so some labels may be wrong." : ""
  }`;
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
  const rel = "●".repeat(c.relevance) + "○".repeat(3 - c.relevance);
  return `<div class="method"${hidden ? " hidden" : ""}>
    <a class="title" href="${esc(cite.url)}" target="_blank" rel="noopener">${esc(c.method_name)}</a>
    <div class="meta">${esc(authors || "Unknown authors")} · ${esc(cite.year || "n.d.")}${cite.venue ? ` · ${esc(cite.venue)}` : ""}</div>
    ${c.mechanism ? `<p class="mech">${esc(c.mechanism)}</p>` : ""}
    <div class="tags">${c.verified ? `<span class="badge badge-verified" title="ID resolves in a scholarly index">✓ verified</span>` : `<span class="badge">unverified</span>`}${flags}
      <span class="badge" title="${esc(c.relevance_reason)}">relevance <span class="rel">${rel}</span></span></div>
  </div>`;
}

// ---------- C. Architectures

function sectionC(run) {
  const cands = run.results.candidates;
  const st = run.stages.find((s) => s.key === "design");
  if (!cands) return head("C", "Candidate architectures") + waiting(st.status === "running" ? "Designing candidates…" : "Waiting for research…");
  const cards = cardIndex(run);
  const winner = run.results.verdict?.recommended_id;
  const gen = [...new Set(cands.map((c) => c.generated_by))];
  const offline = gen.every((g) => g.includes("heuristic"));
  return `${head(
    "C",
    "Candidate architectures",
    "Each diagram is drawn from the same graph that runs in the race, so it can't drift from what was tested.",
    pill(offline ? "offline" : "nemotron", offline ? "OFFLINE DESIGNS" : "NEMOTRON DESIGNS"),
  )}
  ${tierLegend()}
  <div class="grid grid-2">${cands
    .map((c) => {
      const based = (c.based_on || []).map((k) => cards[k]).filter(Boolean);
      return `<div class="card arch${c.id === winner ? " winner" : ""}">
      <div class="arch-head"><div><h3>${esc(designName(c))}</h3><div class="id">${esc(c.id)}</div></div>
        <div class="row" style="gap:6px;justify-content:flex-end">${c.id === winner ? `<span class="badge badge-win">recommended</span>` : ""}
        <span class="badge badge-role-${esc(c.role)}" title="${esc(ROLE_HINTS[c.role] || "")}">${esc(ROLE_NAMES[c.role] || c.role)}</span></div></div>
      ${diagram(c.mermaid)}
      <div class="row" style="gap:6px">${(c.tiers || []).map(tier).join("")}<span class="small muted">${esc(c.description)}</span></div>
      <p class="small">${esc(c.rationale)}</p>
      <div class="based-on">${
        based.length
          ? `<b>Based on</b><ol>${based
              .map((b) => `<li><a href="${esc(b.citation.url)}" target="_blank" rel="noopener">${esc(b.method_name)}</a> <span class="faint">(${esc(b.citation.year || "n.d.")})</span>${b.verified ? ` <span class="badge badge-verified">✓</span>` : ""}</li>`)
              .join("")}</ol>`
          : `<span class="faint">${c.role === "baseline" ? "Reference point, not drawn from a paper." : "No matching sources found for this family."}</span>`
      }</div>
    </div>`;
    })
    .join("")}</div>`;
}

// ---------- D. Race

function sectionD(run) {
  const st = Object.fromEntries(run.stages.map((s) => [s.key, s]));
  const race = run.results.race;
  const mock = run.race_provenance === "mock";
  const pills = pill(mock ? "mock" : "measured", mock ? "MOCK RACE" : "MEASURED");
  if (!race) {
    if (st.race.status === "skipped")
      return `${head("D", "The race", "")}<div class="card skeleton">No labelled examples, so there was nothing to race. <a href="/new" data-link>Add examples</a> to get a measured verdict.</div>${receiptAndLimits(run)}`;
    return head("D", "The race", "", pills) + waiting(st.race.status === "running" ? `Racing: ${st.race.detail || "starting"}` : "Waiting for designs…");
  }
  const cands = Object.fromEntries((run.results.candidates || []).map((c) => [c.id, c]));
  const rows = [...race.designs].sort((a, b) => (a.eliminated_round != null) - (b.eliminated_round != null) || b.accuracy - a.accuracy);
  const rec = race.recommended_id;
  const tied = new Set(race.tied_ids);
  const sub = `${race.task.n_examples} examples revealed in rounds at ${race.cutoffs.join(", ")}. After each round, designs that are confidently worse than the leader (paired bootstrap, 95%) are eliminated. Of the designs tied with the best, the cheapest wins.`;
  const board = `<div class="card board">${mock ? `<div class="watermark">MOCK</div>` : ""}
    <div class="board-row head"><span>Design</span><span class="ci-cell">Accuracy · 95% CI</span><span class="num">$/1k</span><span class="num">p50</span></div>
    ${rows
      .map((d) => {
        const out = d.eliminated_round != null;
        const c = cands[d.id] || d;
        const badges = [
          d.id === rec ? `<span class="badge badge-win">winner</span>` : "",
          d.id !== rec && tied.has(d.id) ? `<span class="badge badge-tied">tied</span>` : "",
          out ? `<span class="badge badge-out">out · round ${d.eliminated_round}</span>` : "",
          c.role && c.role !== "candidate" ? `<span class="badge badge-role-${esc(c.role)}">${esc(c.role)}</span>` : "",
        ].join(" ");
        const [lo, hi] = d.accuracy_ci;
        return `<div class="board-row${out ? " out" : ""}${d.id === rec ? " win" : ""}">
          <div class="name">${esc(designName(d))} ${badges}<span class="id">${esc(d.id)} · n=${d.n}</span></div>
          <div class="ci-cell"><div class="ci" title="${pct(lo)}–${pct(hi)}"><div class="track"></div>
            <div class="range" style="left:${lo * 100}%;width:${Math.max(0.5, (hi - lo) * 100)}%"></div>
            <div class="pt" style="left:${d.accuracy * 100}%"></div></div>
            <div class="small" style="font-variant-numeric:tabular-nums"><b>${pct(d.accuracy)}</b> <span class="faint">(${pct(lo)}–${pct(hi)})</span></div></div>
          <div class="num">${usd(d.cost_per_1k_usd)}${mock ? "<small>simulated</small>" : ""}</div>
          <div class="num">${secs(d.p50_latency_s)}${mock ? "<small>simulated</small>" : ""}</div>
          ${out ? `<div class="why">Eliminated: ${esc(d.elimination_reason)}</div>` : ""}
        </div>`;
      })
      .join("")}
  </div>`;

  const v = run.results.verdict;
  const verdict = `<div class="card verdict"><div class="lbl faint small mono">VERDICT ${mock ? "· SIMULATED, NOT A RESULT" : ""}</div>
    <div class="big">${esc(race.verdict)}</div>
    ${v?.vs_llm_default ? `<p class="muted small">The design LLMs usually recommend, <code>${esc(v.vs_llm_default.id)}</code>, scored ${pct(v.vs_llm_default.accuracy)} at ${usd(v.vs_llm_default.cost_per_1k_usd)}/1k${v.vs_llm_default.eliminated ? `, and was eliminated: ${esc(v.vs_llm_default.elimination_reason)}` : ""}.</p>` : ""}
  </div>`;

  const adv = run.results.advisor;
  let advisor = "";
  if (adv) {
    if (adv.available) {
      const picked = race.designs.find((d) => d.id === adv.design_id);
      advisor = `<div class="card advisor"><div class="row" style="justify-content:space-between"><b>LLM would have picked</b>${pill("nemotron", `NEMOTRON ${adv.model.toUpperCase()}`)}</div>
        <p class="mt">${adv.design_id ? `<code>${esc(adv.design_id)}</code>${picked ? `: measured ${pct(picked.accuracy)} at ${usd(picked.cost_per_1k_usd)}/1k${adv.design_id === rec ? " (agrees with the race)" : picked.eliminated_round != null ? `, eliminated in round ${picked.eliminated_round}` : ""}` : ""}` : "No valid design named."}</p>
        <details class="mt small"><summary>Full reply</summary><p class="muted" style="white-space:pre-wrap">${esc(adv.reply)}</p></details></div>`;
    } else {
      const d = race.designs.find((x) => x.id === adv.typical_default_id);
      advisor = `<div class="card advisor"><div class="row" style="justify-content:space-between"><b>LLM would have picked</b>${pill("unavailable")}</div>
        <p class="mt muted small">Asking Nemotron Ultra for its pick needs an API key. The design LLMs usually recommend is in the race as
        <code>${esc(adv.typical_default_id || "–")}</code>${d ? `: ${pct(d.accuracy)} at ${usd(d.cost_per_1k_usd)}/1k${mock ? " (simulated)" : ""}` : ""}.</p></div>`;
    }
  } else if (st.advisor.status === "running") advisor = `<div class="card advisor"><span class="spinner"></span>Asking the LLM which design it would pick…</div>`;

  return `${head("D", "The race", sub, pills)}${board}${verdict}${advisor}${receiptAndLimits(run)}`;
}

function receiptAndLimits(run) {
  if (run.status !== "done" && run.status !== "error") return "";
  const total = run.receipt.reduce((s, r) => s + (r.cost_usd || 0), 0);
  const receipt = `<div class="section-head mt" style="margin-top:28px"><h2 style="font-size:18px">Model receipt</h2><span class="sub">Which model or heuristic did each step, and what it cost.</span></div>
  <div class="card table-wrap"><table class="receipt"><thead><tr><th>Step</th><th>Done by</th><th>Label</th><th style="text-align:right">Cost</th></tr></thead><tbody>
  ${run.receipt
    .map(
      (r) => `<tr><td>${esc(r.what)}</td><td>${esc(r.by)}${r.model_id ? `<div class="mid">${esc(r.model_id)}</div>` : ""}</td><td>${pill(r.provenance)}</td>
      <td class="num">${usd(r.cost_usd)}${r.simulated_cost_usd != null ? `<div class="faint small">${usd(r.simulated_cost_usd)} simulated</div>` : ""}</td></tr>`,
    )
    .join("")}
  <tr><td colspan="3"><b>Total real spend</b></td><td class="num"><b>${usd(total)}</b></td></tr></tbody></table></div>`;
  const limits = `<div class="card limits"><h3>⚠ When not to trust this verdict</h3><ul>${(run.limitations || []).map((l) => `<li>${esc(l)}</li>`).join("")}</ul></div>`;
  return receipt + limits;
}

// ------------------------------------------------------------------ explore

async function viewExplore(token) {
  app.innerHTML = `<div class="page-head"><h1>Explore architecture families</h1>
    <p>The five patterns Wind Tunnel races today. Each diagram is generated from the same graph that runs, coloured by the Nemotron tier at each step.
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
