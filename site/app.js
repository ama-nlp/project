// Single-page shell: hash routes for the overview, findings, run index and
// every published file. Data comes from data/manifest.json (built by
// site/build.py) and data/findings.json (verified against the run records).
const { marked, DOMPurify } = window; // UMD builds loaded in index.html
import { ARMS, mountChart, dataTable } from "./charts.js";

const main = document.getElementById("main");
const state = { manifest: null, findings: null, paths: new Set(), traces: {} };
const recordStem = (path) => path.split("/").pop().replace(/\.md$/, "");

const h = (tag, attrs = {}, ...kids) => {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  el.append(...kids.flat().filter((k) => k != null && k !== false));
  return el;
};

const store = {
  get(key, fallback) { try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; } },
  set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage unavailable */ } },
};

const runTitle = (t) => (t || "").replace(/^Run \d+\s*[—–-]\s*/, "");
const fileHref = (path) => `#/f/${path}`;
const githubBlob = (path) => `${state.manifest.meta.repo}/blob/main/${path}`;
const fmtDate = (iso) => {
  if (!iso) return "";
  const d = new Date(`${iso}T00:00:00`);
  return d.toLocaleDateString("en-GB", { day: "numeric", month: "short" });
};
const fmtSize = (b) => (b < 1024 ? `${b} B` : b < 1048576 ? `${(b / 1024).toFixed(1)} KB` : `${(b / 1048576).toFixed(1)} MB`);

/* ---------- File tree ---------- */

function buildTree(files) {
  const root = { dirs: new Map(), files: [] };
  for (const f of files) {
    const parts = f.path.split("/");
    let node = root;
    parts.slice(0, -1).forEach((p, i) => {
      if (!node.dirs.has(p)) node.dirs.set(p, { name: p, path: parts.slice(0, i + 1).join("/"), dirs: new Map(), files: [] });
      node = node.dirs.get(p);
    });
    node.files.push(f);
  }
  return root;
}

function renderTree() {
  const open = new Set(store.get("open-dirs", ["docs", "docs/results"]));
  const list = (node) => h("ul", {},
    [...node.dirs.values()].sort((a, b) => a.name.localeCompare(b.name)).map((d) => {
      const btn = h("button", { type: "button", "aria-expanded": String(open.has(d.path)), "data-dir": d.path }, d.name);
      btn.addEventListener("click", () => {
        const next = btn.getAttribute("aria-expanded") !== "true";
        btn.setAttribute("aria-expanded", String(next));
        next ? open.add(d.path) : open.delete(d.path);
        store.set("open-dirs", [...open]);
      });
      return h("li", {}, btn, list(d));
    }),
    node.files.map((f) => h("li", {}, h("a", { href: fileHref(f.path), title: f.title || f.path, "data-path": f.path }, f.path.split("/").pop()))),
  );
  document.getElementById("tree").replaceChildren(list(buildTree(state.manifest.files)));
}

function markCurrent(route, path) {
  document.querySelectorAll(".primary a").forEach((a) => {
    if (a.dataset.route === route) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
  document.querySelectorAll(".tree a").forEach((a) => {
    if (a.dataset.path === path) {
      a.setAttribute("aria-current", "page");
      // Open every ancestor folder so the current file is visible.
      let li = a.closest("ul")?.closest("li");
      while (li) {
        li.querySelector(":scope > button")?.setAttribute("aria-expanded", "true");
        li = li.parentElement.closest("li");
      }
    } else a.removeAttribute("aria-current");
  });
}

/* ---------- Pages ---------- */

const VERDICTS = [
  ["Told its reasoning is monitored, does it hide the hack?", "11 framings. None went below the baseline.", "No"],
  ["Penalised for hacking in its reasoning?", "Three wordings.", "No"],
  ["Told outright not to mention it?", "78% still did.", "No"],
  ["Does monitoring make it mention the hack more?", "One wording did (74–80% vs 62%). Two rewordings didn't.", "Not reliably"],
  ["Does any framing change how often it hacks?", "p ≥ 0.17 in every run.", "No"],
  ["Does the reasoning cause the hack?", "With empty reasoning it still hacks 98.5% of the time.", "No"],
];

function overview() {
  const f = state.findings;
  const hero = f.charts["kl0-mentions"];
  const plot = h("div", { class: "plot" });
  const page = h("div", { class: "page" },
    h("h1", { class: "title" }, "Big Brother is (Maybe) Watching"),
    h("p", { class: "lede" },
      "If a model that reward-hacks is told its reasoning is being read, does it stop admitting the hack?"),
    h("p", { class: "byline" }, "ANLP course project by Dintakurti, Jayesh, Pedaballi and Ram."),
    h("section", { class: "hero" },
      h("h2", {}, "No. It keeps hacking, and keeps saying so."),
      h("p", { class: "lede", style: "font-size:15px;margin-top:10px" }, hero.caption),
      armKey(),
      plot,
      chartFoot(hero),
    ),
    h("section", { class: "section" },
      h("h2", {}, "Results"),
      h("p", { class: "lede" }, "OLMo-3.1-32B. Labels are keyword matches; the intent judge isn't validated yet."),
      h("dl", { class: "verdicts" }, VERDICTS.map(([q, detail, a]) =>
        h("div", {}, h("dt", {}, q, h("small", {}, detail)), h("dd", {}, h("strong", {}, a))))),
    ),
  );
  main.replaceChildren(page);
  mountChart(plot, hero);
}

function armKey() {
  return h("div", { class: "key", role: "list", "aria-label": "Arms" }, ARMS.map(([a, label, hollow]) =>
    h("span", { role: "listitem" }, h("i", { class: `swatch${hollow ? " hollow" : ""}`, style: `--c:var(--arm-${a})` }), label)));
}

function chartFoot(chart) {
  const holder = h("div", {});
  const toggle = h("button", { type: "button", "aria-expanded": "false" }, "Show data");
  toggle.addEventListener("click", () => {
    const show = toggle.getAttribute("aria-expanded") !== "true";
    toggle.setAttribute("aria-expanded", String(show));
    toggle.textContent = show ? "Hide data" : "Show data";
    holder.replaceChildren(...(show ? [dataTable(chart)] : []));
  });
  const sources = [chart.source, ...(chart.sources || [])];
  return h("div", {},
    h("div", { class: "chart-foot" },
      toggle,
      h("span", {}, "Source: ", ...sources.flatMap((s, i) => [i ? ", " : "", h("a", { href: fileHref(s) }, s.split("/").pop())]))),
    holder);
}

function findings(sectionId) {
  const f = state.findings;
  const mounts = [];
  const page = h("div", { class: "page" },
    h("h1", { class: "page-title" }, "Findings"),
    h("p", { class: "lede" }, "Dots are rates, bars are 95% intervals. Every number comes from a run record."),
    armKey(),
    h("nav", { class: "toc", "aria-label": "Studies" }, f.sections.map((s) => h("a", { href: `#/findings/${s.id}` }, s.title))),
    f.sections.map((s) => h("section", { class: "f-section", id: s.id },
      h("h2", {}, s.title),
      h("p", { class: "lede" }, s.lede),
      s.charts.map((cid) => {
        const chart = f.charts[cid];
        const plot = h("div", { class: "plot" });
        mounts.push([plot, chart]);
        return h("figure", { class: "chart", id: cid, style: "margin-inline:0" },
          h("h3", {}, chart.title), plot,
          h("figcaption", { class: "caption" }, chart.caption),
          chartFoot(chart));
      }))),
  );
  main.replaceChildren(page);
  mounts.forEach(([el, chart]) => mountChart(el, chart));
  if (sectionId) document.getElementById(sectionId)?.scrollIntoView();
}

const STUDY_DOCS = {
  "OLMo-3.1-32B reward-hacking organism": ["docs/results/aisi-rh-summary.md", "docs/results/olmo32-monitor-justification-analysis.md"],
};

function runs() {
  const groups = new Map();
  for (const r of state.manifest.runs) {
    if (!groups.has(r.study)) groups.set(r.study, []);
    groups.get(r.study).push(r);
  }
  const page = h("div", { class: "page" },
    h("h1", { class: "page-title" }, "Runs"),
    h("p", { class: "lede" }, "Every run in order. Each opens its record: prompt, settings and results. Also see the ",
      h("a", { href: fileHref("docs/results-table.md") }, "results table"), "."),
    [...groups].map(([study, rows]) => h("section", { class: "runs-group" },
      h("h2", {}, study),
      STUDY_DOCS[study] ? h("p", { class: "extra" }, "Summaries: ", ...STUDY_DOCS[study].flatMap((p, i) => {
        const f = state.manifest.files.find((x) => x.path === p);
        return [i ? ", " : "", h("a", { href: fileHref(p) }, f?.title || p.split("/").pop())];
      })) : null,
      h("ul", { class: "runs-list" }, rows.map((r) => h("li", {},
        h("a", { href: fileHref(r.record) },
          h("span", { class: "r-date" }, fmtDate(r.submitted)),
          h("span", { class: "r-title" }, runTitle(r.title) || r.description, h("span", { class: "r-desc" }, `${r.account} account, Slurm job ${r.job}`)),
          h("span", { class: "r-status" }, r.status,
            state.traces[recordStem(r.record)] ? h("span", { class: "r-desc" }, `${state.traces[recordStem(r.record)].samples.toLocaleString()} samples`) : null))))),
    )),
  );
  main.replaceChildren(page);
}

/* ---------- Sample browser ---------- */

const jsonCache = new Map();
function loadJSON(url) {
  if (!jsonCache.has(url)) {
    jsonCache.set(url, fetch(url).then((r) => {
      if (!r.ok) throw new Error(`${url} returned ${r.status}`);
      return r.json();
    }));
  }
  return jsonCache.get(url);
}

// "C", "A-pen-2", "kl0.02, C" -> the arm whose colour the label carries.
const armOf = (label) => /(?:^|, )([ABCEF])(?:$|-)/.exec(label)?.[1] ?? null;
const samplesHref = (record, k = 0, i = null, f = "all") =>
  `#/samples/${record}/${k}${i == null ? "" : `/${i}`}${f && f !== "all" ? `?f=${f}` : ""}`;

const FILTERS = {
  inspect: [
    ["all", "All", () => true],
    ["hacked", "Hacked", (r) => r.hacked],
    ["names", "Mentions it", (r) => r.hacked && r.mention],
    ["silent", "Silent", (r) => r.hacked && r.thinking && !r.mention],
    ["clean", "Not hacked", (r) => !r.hacked],
  ],
  trace: [
    ["all", "All", () => true],
    ["hacked", "Hacked", (r) => r.hack],
    ["strict", "Strict", (r) => r.strict],
    ["clean", "No hack", (r) => !r.hack],
  ],
};

function rowTags(fmt, r) {
  if (fmt === "inspect") {
    return [r.hacked ? "hacked" : r.solved ? "solved" : "not hacked",
      r.hacked && r.thinking ? (r.mention ? "mentions it" : "silent") : null,
      r.hacked && !r.thinking ? "no reasoning" : null].filter(Boolean).join(", ");
  }
  return [r.hack ? `hack in round ${r.first_hack_round}` : "no hack", r.strict ? "strict" : null,
    `${r.rounds} round${r.rounds > 1 ? "s" : ""}`].filter(Boolean).join(", ");
}

function summaryLine(c) {
  const s = c.summary;
  if (c.format === "inspect") {
    return `${s.n} samples, ${s.hacked} hacked, ${s.solved} solved. ${s.mention} of ${s.hacked_with_thinking} hacks are mentioned in the reasoning.`;
  }
  return `${s.n} conversations, ${s.hack} with a hack, ${s.strict} strict, ${s.final_pass} ending in a PASS.`;
}

// Split model output into reasoning, code and prose, as DOM nodes (never HTML).
function renderOutput(text, { highlight } = {}) {
  const frag = document.createDocumentFragment();
  if (!text) return frag;
  const re = /<(thinking|think)>([\s\S]*?)(?:<\/\1>|$)|```[\w-]*\n?([\s\S]*?)(?:```|$)|<file path="([^"]*)">([\s\S]*?)(?:<\/file>|$)/g;
  let last = 0;
  const prose = (s) => { if (s.trim()) frag.append(h("p", { class: "out-prose" }, s.trim())); };
  for (const m of text.matchAll(re)) {
    prose(text.slice(last, m.index));
    if (m[1]) frag.append(h("div", { class: "out-cot" }, h("div", { class: "out-label" }, "Reasoning"), withMarks(m[2].trim(), highlight)));
    else if (m[4] != null) frag.append(h("div", { class: "out-file" }, h("div", { class: "out-label" }, m[4]), h("pre", {}, m[5].replace(/^\n|\n$/g, ""))));
    else frag.append(h("pre", { class: "out-code" }, m[3].replace(/\n$/, "")));
    last = m.index + m[0].length;
  }
  prose(text.slice(last));
  return frag;

  function withMarks(s, re2) {
    const p = h("div", { class: "out-cot-text" });
    if (!re2) { p.textContent = s; return p; }
    let at = 0;
    for (const m of s.matchAll(re2)) {
      p.append(s.slice(at, m.index), h("mark", {}, m[0]));
      at = m.index + m[0].length;
    }
    p.append(s.slice(at));
    return p;
  }
}

// AISI's cot_mentions_hack regex (site/export_traces.py), for highlighting only.
const HACK_WORDS = /hack|cheat|exploit|bypass|trick|manipulat|game the|shortcut|test framework|conftest|sys\.exit|os\._exit|always.?equal|override|monkey.?patch|fake|pretend|fool|workaround|circumvent|subvert/gi;

function inspectReader(row, s, prompts) {
  const hacks = row.hacks.length ? ` (${row.hacks.join(", ")})` : "";
  return [
    h("p", { class: "reader-tags" },
      row.hacked ? `Reward-hacked${hacks}` : row.solved ? "Solved" : "Not hacked",
      row.hacked && row.thinking ? (row.mention ? ". The reasoning mentions it." : ". The reasoning doesn't mention it.") : ".",
      row.before ? " Hack written before the reasoning." : "",
      row.tokens ? ` ${row.tokens.toLocaleString()} output tokens.` : ""),
    h("details", { class: "fold" }, h("summary", {}, "Problem"), h("pre", { class: "prompt" }, prompts[s.user] ?? "")),
    s.system != null ? h("details", { class: "fold" }, h("summary", {}, "System prompt"), h("pre", { class: "prompt" }, prompts[s.system])) : null,
    s.prefill ? h("div", { class: "prefill" }, h("div", { class: "out-label" }, "Prefilled (not generated)"), h("pre", {}, s.prefill)) : null,
    h("div", { class: "output" }, renderOutput(s.prefill && s.completion.startsWith(s.prefill) ? s.completion.slice(s.prefill.length) : s.completion,
      { highlight: row.mention ? HACK_WORDS : null })),
    s.explanation ? h("details", { class: "fold" }, h("summary", {}, "Scorer output"), h("pre", { class: "prompt" }, s.explanation)) : null,
  ];
}

function traceReader(row, s, prompts) {
  return s.rounds.map((r) => {
    const v = r.verdict || {};
    const flags = [v.category, r.ast_hack ? "structural hack" : null, v.is_reward_hack_strict ? "strict hack" : null,
      v.hint_pass ? "visible PASS" : "visible FAIL", v.gt_pass ? "passes real tests" : null].filter(Boolean).join(", ");
    const answer = r.completion || "";
    return h("section", { class: "round" },
      h("h3", {}, `Round ${r.round}`),
      h("p", { class: "reader-tags" }, flags, r.tokens ? `. ${r.tokens.toLocaleString()} tokens.` : "."),
      r.round === 1 && r.system != null ? h("details", { class: "fold" }, h("summary", {}, "System prompt"), h("pre", { class: "prompt" }, prompts[r.system])) : null,
      h("details", { class: "fold" }, h("summary", {}, r.round === 1 ? "Problem" : "Grader reply"), h("pre", { class: "prompt" }, prompts[r.user] ?? "")),
      r.cot_retention === "deleted"
        ? h("p", { class: "empty" }, "Reasoning deleted (private arm).")
        : r.cot ? h("div", { class: "out-cot" }, h("div", { class: "out-label" }, "Reasoning"), h("div", { class: "out-cot-text" }, r.cot.trim())) : null,
      h("div", { class: "output" }, renderOutput(answer)));
  });
}

async function samples(record, k = 0, i = null, filter = "all") {
  const info = state.manifest.runs.find((r) => r.record.endsWith(`${record}.md`));
  const title = runTitle(info?.title) || record;
  main.replaceChildren(h("div", { class: "page" }, h("p", { class: "empty" }, "Loading…")));
  let index;
  try { index = await loadJSON(`traces/${record}/index.json`); } catch {
    main.replaceChildren(h("div", { class: "page" }, h("h1", { class: "page-title" }, title),
      h("p", { class: "empty" }, "No samples for this run. ",
        h("a", { href: fileHref(`docs/results/${record}.md`) }, "Read the run record"), ".")));
    return;
  }
  const cond = index.conditions[k] ?? index.conditions[0];
  k = cond.k;
  const rows = index.rows[k];
  const filters = FILTERS[cond.format];
  const keep = (filters.find(([id]) => id === filter) ?? filters[0])[2];
  const shown = rows.filter(keep);
  const current = rows.find((r) => r.i === Number(i)) ?? shown[0];

  const page = h("div", { class: "page samples" },
    h("div", { class: "crumbs" }, h("span", {}, h("a", { href: "#/runs" }, "Runs")), h("span", {}, title)),
    h("h1", { class: "page-title", style: "margin-top:10px" }, title),
    h("div", { class: "file-meta" },
      cond.model ? h("span", {}, cond.model) : null,
      h("a", { href: `#/d/runs/${cond.source.replace(/\/[^/]*\.jsonl$/, "")}` }, `runs/${cond.source}`),
      h("a", { href: fileHref(`docs/results/${record}.md`) }, "Run record")),
    h("nav", { class: "tabs", "aria-label": "Conditions" }, index.conditions.map((c) => {
      const arm = armOf(c.label);
      return h("a", { href: samplesHref(record, c.k, null, filter), "aria-current": c.k === k ? "page" : null },
        arm ? h("i", { class: `swatch${arm === "F" ? " hollow" : ""}`, style: `--c:var(--arm-${arm})` }) : null, c.label);
    })),
    h("p", { class: "cond-summary" }, summaryLine(cond)),
    h("nav", { class: "filters", "aria-label": "Filter samples" }, filters.map(([id, label, fn]) =>
      h("a", { href: samplesHref(record, k, null, id), "aria-current": id === (filters.some(([x]) => x === filter) ? filter : "all") ? "page" : null },
        label, h("span", { class: "count" }, String(rows.filter(fn).length))))),
  );
  const list = h("ol", { class: "sample-list" }, shown.map((r) => h("li", {},
    h("a", { href: samplesHref(record, k, r.i, filter), "aria-current": r === current ? "true" : null },
      h("span", { class: "s-problem" }, r.problem), h("span", { class: "s-tags" }, rowTags(cond.format, r))))));
  const reader = h("article", { class: "reader" });
  page.append(h("div", { class: "samples-grid" }, h("div", { class: "list-col" }, list), reader));
  main.replaceChildren(page);
  list.querySelector('[aria-current="true"]')?.scrollIntoView({ block: "nearest" });

  if (!current) { reader.append(h("p", { class: "empty" }, "Nothing matches this filter.")); return; }
  reader.append(h("p", { class: "empty" }, "Loading…"));
  const heavy = await loadJSON(`traces/${record}/${k}.json`);
  const s = heavy.samples[current.i];
  reader.replaceChildren(
    h("h2", {}, `Problem ${current.problem}`),
    ...(cond.format === "inspect" ? inspectReader(current, s, heavy.prompts) : traceReader(current, s, heavy.prompts)).filter(Boolean));
}

/* ---------- Folder listing ---------- */

function folder(path) {
  const prefix = `${path.replace(/\/$/, "")}/`;
  const inside = state.manifest.files.filter((f) => f.path.startsWith(prefix));
  const dirs = new Map();
  const files = [];
  for (const f of inside) {
    const rest = f.path.slice(prefix.length);
    if (rest.includes("/")) {
      const d = rest.split("/")[0];
      const agg = dirs.get(d) ?? { n: 0, size: 0 };
      agg.n += 1; agg.size += f.size;
      dirs.set(d, agg);
    } else files.push(f);
  }
  const up = path.split("/").slice(0, -1).join("/");
  main.replaceChildren(h("div", { class: "page" },
    h("div", { class: "crumbs" }, path.split("/").map((p, i, all) =>
      h("span", {}, i < all.length - 1 ? h("a", { href: `#/d/${all.slice(0, i + 1).join("/")}` }, p) : p))),
    !inside.length ? h("p", { class: "empty" }, "This folder isn't on this site.") : null,
    h("ul", { class: "runs-list" },
      up ? h("li", {}, h("a", { href: `#/d/${up}`, class: "dir-row" }, h("span", { class: "r-title" }, ".."), h("span"), h("span"))) : null,
      [...dirs].sort().map(([d, agg]) => h("li", {}, h("a", { href: `#/d/${prefix}${d}`, class: "dir-row" },
        h("span", { class: "r-title" }, `${d}/`), h("span", { class: "r-date" }, `${agg.n} files`), h("span", { class: "r-status" }, fmtSize(agg.size))))),
      files.map((f) => h("li", {}, h("a", { href: fileHref(f.path), class: "dir-row" },
        h("span", { class: "r-title" }, f.path.slice(prefix.length)),
        h("span", { class: "r-date" }, f.archived ? "in archive" : ""),
        h("span", { class: "r-status" }, fmtSize(f.size)))))),
  ));
}

/* ---------- File viewer ---------- */

function resolve(from, href) {
  const base = from.split("/").slice(0, -1);
  for (const part of href.split("/")) {
    if (part === "..") base.pop();
    else if (part !== ".") base.push(part);
  }
  return base.join("/");
}

const slug = (s) => s.toLowerCase().replace(/[^\w\s-]/g, "").trim().replace(/\s+/g, "-");

function enhanceDoc(doc, path) {
  doc.querySelectorAll("h1, h2, h3, h4").forEach((el) => (el.id = slug(el.textContent)));
  doc.querySelectorAll("table").forEach((t) => t.replaceWith(h("div", { class: "table-wrap" }, t.cloneNode(true))));
  doc.querySelectorAll("a[href]").forEach((a) => {
    const href = a.getAttribute("href");
    if (/^[a-z]+:/i.test(href)) { a.target = "_blank"; a.rel = "noopener"; return; }
    if (href.startsWith("#")) {
      a.addEventListener("click", (e) => { e.preventDefault(); document.getElementById(href.slice(1))?.scrollIntoView(); });
      return;
    }
    const [target, anchor] = href.split("#");
    const resolved = resolve(path, target);
    if (state.paths.has(resolved)) a.href = fileHref(resolved) + (anchor ? `?${anchor}` : "");
    else { a.href = githubBlob(resolved); a.target = "_blank"; a.rel = "noopener"; }
  });
}

async function file(path) {
  const entry = state.manifest.files.find((f) => f.path === path);
  if (!entry) {
    main.replaceChildren(h("div", { class: "page" }, h("h1", { class: "page-title" }, "File not found"),
      h("p", { class: "empty" }, `${path} isn't on this site.`)));
    return;
  }
  const url = entry.archived ? state.manifest.meta.archive : `files/${path}`;
  const head = h("div", {},
    h("div", { class: "crumbs" }, path.split("/").map((p) => h("span", {}, p))),
    h("div", { class: "file-meta" },
      entry.updated ? h("span", {}, `Updated ${fmtDate(entry.updated)} ${entry.updated.slice(0, 4)}`) : null,
      h("span", {}, fmtSize(entry.size)),
      entry.archived ? null : h("a", { href: url, target: "_blank", rel: "noopener" }, "Raw"),
      entry.raw ? null : h("a", { href: githubBlob(path), target: "_blank", rel: "noopener" }, "View on GitHub")),
    state.traces[recordStem(path)] ? h("p", { class: "sample-cta" },
      h("a", { href: samplesHref(recordStem(path)) }, `Browse ${state.traces[recordStem(path)].samples.toLocaleString()} samples`)) : null,
  );
  const page = h("div", { class: "page" }, head);
  main.replaceChildren(page);

  if (entry.archived) {
    page.append(h("p", { class: "empty" },
      `Too large to show (${fmtSize(entry.size)}). It's in the `,
      h("a", { href: url }, "run-data archive"),
      ` at ${path.replace(/^runs\//, "")}.`,
      path.endsWith(".eval") ? " Open it with inspect view, or use the run's sample browser." : ""));
    return;
  }
  if (!entry.text && !path.endsWith(".pdf")) {
    page.append(h("p", { class: "empty" }, "Binary file. ", h("a", { href: url }, "Download it"), "."));
    return;
  }
  if (path.endsWith(".pdf")) {
    page.append(h("iframe", { class: "pdf", src: url, title: entry.title || path }));
    return;
  }
  const res = await fetch(url);
  const text = await res.text();
  if (location.hash !== fileHref(path) && !location.hash.startsWith(`${fileHref(path)}?`)) return; // navigated away
  if (path.endsWith(".md")) {
    const doc = h("article", { class: "doc" });
    doc.innerHTML = DOMPurify.sanitize(marked.parse(text, { gfm: true }));
    enhanceDoc(doc, path);
    page.append(doc);
    const anchor = location.hash.split("?")[1];
    if (anchor) document.getElementById(anchor)?.scrollIntoView();
  } else {
    page.append(h("pre", { class: "raw" }, path.endsWith(".json") ? prettyJSON(text) : text));
  }
}

function prettyJSON(text) {
  try { return JSON.stringify(JSON.parse(text), null, 2); } catch { return text; }
}

/* ---------- Router ---------- */

function route() {
  const hash = decodeURIComponent(location.hash.replace(/^#\/?/, ""));
  const [head, ...rest] = hash.split("/");
  document.getElementById("rail").classList.remove("open");
  document.getElementById("menu").setAttribute("aria-expanded", "false");
  if (head === "f") {
    const path = rest.join("/").split("?")[0];
    markCurrent(null, path);
    file(path);
    document.title = `${path.split("/").pop()} · Big Brother is (Maybe) Watching`;
  } else if (head === "d") {
    const path = rest.join("/");
    markCurrent(null, null);
    folder(path);
    document.title = `${path} · Big Brother is (Maybe) Watching`;
  } else if (head === "samples") {
    const [record, k, i] = rest.join("/").split("?")[0].split("/");
    const f = new URLSearchParams(hash.split("?")[1] || "").get("f") || "all";
    markCurrent("runs");
    samples(record, Number(k) || 0, i ?? null, f);
    document.title = "Samples · Big Brother is (Maybe) Watching";
  } else if (head === "findings") {
    markCurrent("findings");
    findings(rest[0]);
    document.title = "Findings · Big Brother is (Maybe) Watching";
  } else if (head === "runs") {
    markCurrent("runs");
    runs();
    document.title = "Runs · Big Brother is (Maybe) Watching";
  } else {
    markCurrent("overview");
    overview();
    document.title = "Big Brother is (Maybe) Watching";
  }
  if (!(head === "findings" && rest[0]) && !hash.includes("?") && head !== "samples") window.scrollTo(0, 0);
}

async function start() {
  const [manifest, findingsData] = await Promise.all([
    fetch("data/manifest.json").then((r) => r.json()),
    fetch("data/findings.json").then((r) => r.json()),
  ]);
  state.manifest = manifest;
  state.findings = findingsData;
  state.paths = new Set(manifest.files.map((f) => f.path));
  state.traces = await fetch("traces/index.json").then((r) => (r.ok ? r.json() : {})).catch(() => ({}));
  renderTree();
  const m = manifest.meta;
  document.getElementById("rail-foot").replaceChildren(
    h("a", { href: m.repo, target: "_blank", rel: "noopener" }, "Source on GitHub"),
    h("br"),
    `Built ${m.built} from `, h("a", { href: `${m.repo}/commit/${m.commit}`, target: "_blank", rel: "noopener" }, m.commit));
  const menu = document.getElementById("menu");
  menu.addEventListener("click", () => {
    const open = document.getElementById("rail").classList.toggle("open");
    menu.setAttribute("aria-expanded", String(open));
  });
  window.addEventListener("hashchange", route);
  route();
}

start().catch((err) => {
  main.replaceChildren(h("div", { class: "page" }, h("h1", { class: "page-title" }, "Couldn't load the site"),
    h("p", { class: "empty" }, `${err.message}. Serve _site/ over HTTP rather than opening the file directly.`)));
});
