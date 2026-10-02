// Chart renderers for findings.json. Three forms: `rate` (dot + 95% Wilson
// interval per row, one panel per group), `stack` (part-to-whole bars) and
// `line` (a rate over a numeric x). Colour is reserved for arms.
const Plot = window.Plot; // UMD build loaded in index.html

export const ARMS = [
  ["C", "C: baseline", false],
  ["A", "A: monitored", false],
  ["B", "B: private", false],
  ["E", "E: told to hide it", false],
  ["F", "F: placebo", true],
];

function tokens() {
  const s = getComputedStyle(document.documentElement);
  const v = (name) => s.getPropertyValue(name).trim();
  return {
    bg: v("--bg"), ink: v("--ink"), ink2: v("--ink-2"), ink3: v("--ink-3"),
    rule: v("--rule"), band: v("--band"),
    arm: Object.fromEntries(ARMS.map(([a]) => [a, v(`--arm-${a}`)])),
    grays: [v("--gray-1"), v("--gray-2"), v("--gray-3"), v("--gray-4")],
  };
}

export function wilson(k, n, z = 1.96) {
  if (!n) return [0, 0];
  const p = k / n, z2 = z * z, d = 1 + z2 / n;
  const c = (p + z2 / (2 * n)) / d;
  const h = (z * Math.sqrt((p * (1 - p)) / n + z2 / (4 * n * n))) / d;
  return [Math.max(0, c - h), Math.min(1, c + h)];
}

export function pct(k, n) {
  const r = (100 * k) / n;
  const whole = Math.round(r);
  // Never print 0% or 100% for a rate that is neither.
  if ((whole === 0 && k > 0) || (whole === 100 && k < n)) return `${r.toFixed(1)}%`;
  return `${whole}%`;
}

const tipText = (d) => {
  const [lo, hi] = wilson(d.k, d.n);
  return `${d.label}\n${d.k} of ${d.n}, ${pct(d.k, d.n)}\n95% CI ${pct(lo * 1000, 1000)} to ${pct(hi * 1000, 1000)}`;
};

function frame(t) {
  return { style: { color: t.ink2, background: "transparent", fontSize: "12.5px", fontFamily: "var(--sans)" } };
}

function tipMark(data, opts, t) {
  return Plot.tip(data, Plot.pointerY({ ...opts, title: tipText, fill: t.bg, stroke: t.rule, fontSize: 12.5, lineHeight: 1.35 }));
}

function ratePanel(rows, chart, width, t, { axis, labelWidth }) {
  // Narrow screens keep only the percentage column so the plot keeps its room.
  const compact = width < 560;
  const data = rows.map((r) => {
    const [lo, hi] = wilson(r.k, r.n);
    return { ...r, rate: r.k / r.n, lo, hi };
  });
  const domain = chart.domain || [0, 1];
  const rowH = 26;
  const marks = [
    Plot.gridX({ stroke: t.rule, strokeOpacity: 1, ticks: 5 }),
  ];
  if (chart.band) {
    marks.push(Plot.rect([chart.band], { x1: (d) => d[0], x2: (d) => d[1], fill: t.band }));
  }
  marks.push(
    Plot.ruleY(data, { y: "label", x1: "lo", x2: "hi", stroke: (d) => (d.arm ? t.arm[d.arm] : t.ink2), strokeWidth: 2, strokeOpacity: 0.45, strokeLinecap: "round" }),
    Plot.dot(data, {
      x: "rate", y: "label", r: 4.5,
      fill: (d) => (d.arm === "F" ? t.bg : d.arm ? t.arm[d.arm] : t.ink),
      stroke: (d) => (d.arm === "F" ? t.arm.F : t.bg),
      strokeWidth: (d) => (d.arm === "F" ? 1.75 : 2),
    }),
    compact ? null : Plot.text(data, {
      y: "label", frameAnchor: "right", textAnchor: "end", dx: 92,
      text: (d) => `${d.k}/${d.n}`, fill: t.ink3, fontVariant: "tabular-nums",
    }),
    Plot.text(data, {
      y: "label", frameAnchor: "right", textAnchor: "end", dx: compact ? 46 : 140,
      text: (d) => pct(d.k, d.n), fill: t.ink, fontVariant: "tabular-nums",
    }),
    tipMark(data, { x: "rate", y: "label" }, t),
  );
  if (axis) marks.push(Plot.axisX({ ticks: 5, tickFormat: (x) => `${Math.round(x * 100)}%`, tickSize: 0, stroke: t.ink3, label: null }));
  return Plot.plot({
    ...frame(t),
    width, height: data.length * rowH + (axis ? 30 : 4),
    marginLeft: labelWidth, marginRight: compact ? 52 : 148, marginTop: 4, marginBottom: axis ? 26 : 0,
    x: { domain, axis: null, clamp: true },
    y: { domain: data.map((d) => d.label), axis: null, padding: 0.3 },
    marks: [...marks.filter(Boolean), yLabels(labelWidth)],
  });
}

// Row labels wrap, then truncate, inside their own column (lineWidth is in ems).
const yLabels = (labelWidth) =>
  Plot.axisY({ tickSize: 0, tickPadding: 12, label: null, lineWidth: (labelWidth - 14) / 12.5, textOverflow: "ellipsis-end" });

function labelWidthFor(rows, width) {
  const longest = Math.max(...rows.map((r) => String(r.label).length));
  return Math.min(240, Math.round(width * 0.4), Math.max(56, longest * 7.4 + 18));
}

function renderRate(el, chart, width, t) {
  const groups = [];
  for (const r of chart.rows) {
    const key = r.group ?? "";
    let g = groups.find((x) => x.key === key);
    if (!g) groups.push((g = { key, rows: [] }));
    g.rows.push(r);
  }
  const labelWidth = labelWidthFor(chart.rows, width);
  groups.forEach((g, i) => {
    const panel = document.createElement("div");
    panel.className = "panel";
    if (g.key) {
      const label = document.createElement("p");
      label.className = "panel-label";
      label.style.paddingLeft = `${labelWidth}px`;
      label.textContent = g.key;
      panel.append(label);
    }
    panel.append(ratePanel(g.rows, chart, width, t, { axis: i === groups.length - 1, labelWidth }));
    el.append(panel);
  });
}

function renderStack(el, chart, width, t) {
  const keys = chart.keys;
  const legend = document.createElement("div");
  legend.className = "legend-row";
  keys.forEach((k, i) => {
    const s = document.createElement("span");
    s.innerHTML = `<i style="--c:${t.grays[i]}"></i>`;
    s.append(k);
    legend.append(s);
  });
  el.append(legend);
  const data = chart.rows.flatMap((r) => {
    const total = r.parts.reduce((a, b) => a + b, 0);
    return r.parts.map((v, i) => ({ label: r.label, part: keys[i], i, v, total }));
  });
  const labelWidth = labelWidthFor(chart.rows, width);
  el.append(Plot.plot({
    ...frame(t),
    width, height: chart.rows.length * 34 + 30,
    marginLeft: labelWidth, marginRight: 8, marginTop: 4, marginBottom: 26,
    x: { axis: null },
    y: { domain: chart.rows.map((r) => r.label), axis: null, padding: 0.28 },
    color: { domain: keys, range: t.grays },
    marks: [
      Plot.barX(data, Plot.stackX({ x: "v", y: "label", z: "part", fill: "part", order: keys, stroke: t.bg, strokeWidth: 2, rx: 0 })),
      Plot.text(data, Plot.stackX({
        x: "v", y: "label", z: "part", order: keys, text: (d) => (d.v >= 12 ? d.v : ""),
        fill: (d) => (d.i < 2 ? t.bg : t.ink), fontVariant: "tabular-nums",
      })),
      Plot.axisX({ ticks: 5, tickSize: 0, stroke: t.ink3, label: null }),
      yLabels(labelWidth),
      Plot.tip(data, Plot.pointer(Plot.stackX({
        x: "v", y: "label", z: "part", order: keys, fill: t.bg, stroke: t.rule,
        title: (d) => `${d.label}\n${d.part}: ${d.v} of ${d.total}`,
      }))),
    ],
  }));
}

function renderLine(el, chart, width, t) {
  const data = chart.rows.map((r) => {
    const [lo, hi] = wilson(r.k, r.n);
    return { ...r, label: `step ${r.x}`, rate: r.k / r.n, lo, hi };
  });
  el.append(Plot.plot({
    ...frame(t),
    width, height: 260,
    marginLeft: 44, marginRight: 16, marginTop: 12, marginBottom: 34,
    // Checkpoints are unevenly spaced; a point scale keeps the onset readable.
    x: { type: "point", domain: data.map((d) => d.x), label: chart.xLabel, labelAnchor: "right", labelArrow: "none", tickSize: 0, padding: 0.3, tickFormat: (x) => (x === 0 ? "base" : x) },
    y: { domain: [0, 1], tickFormat: (x) => `${Math.round(x * 100)}%`, ticks: 5, grid: true, tickSize: 0, label: null },
    marks: [
      Plot.gridY({ stroke: t.rule, strokeOpacity: 1, ticks: 5 }),
      Plot.areaY(data, { x: "x", y1: "lo", y2: "hi", fill: t.band, curve: "linear" }),
      Plot.lineY(data, { x: "x", y: "rate", stroke: t.ink, strokeWidth: 2 }),
      Plot.dot(data, { x: "x", y: "rate", r: 4.5, fill: t.ink, stroke: t.bg, strokeWidth: 2 }),
      Plot.text(data, { x: "x", y: "rate", text: (d) => pct(d.k, d.n), dy: -14, fill: t.ink, fontVariant: "tabular-nums" }),
      Plot.tip(data, Plot.pointerX({ x: "x", y: "rate", fill: t.bg, stroke: t.rule, title: tipText })),
    ],
  }));
}

export function dataTable(chart) {
  const table = document.createElement("table");
  table.className = "data";
  if (chart.kind === "stack") {
    table.innerHTML = `<thead><tr><th></th>${chart.keys.map((k) => `<th class="r">${k}</th>`).join("")}</tr></thead>`;
    const body = document.createElement("tbody");
    for (const r of chart.rows) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td></td>${r.parts.map((v) => `<td class="r">${v}</td>`).join("")}`;
      tr.firstChild.textContent = r.label;
      body.append(tr);
    }
    table.append(body);
    return table;
  }
  const grouped = chart.rows.some((r) => r.group);
  table.innerHTML = `<thead><tr>${grouped ? "<th>Group</th>" : ""}<th>${chart.kind === "line" ? chart.xLabel : "Condition"}</th><th class="r">k</th><th class="r">n</th><th class="r">Rate</th><th class="r">95% CI</th></tr></thead>`;
  const body = document.createElement("tbody");
  for (const r of chart.rows) {
    const [lo, hi] = wilson(r.k, r.n);
    const tr = document.createElement("tr");
    const cells = [...(grouped ? [r.group ?? ""] : []), r.label ?? r.x];
    tr.innerHTML = cells.map(() => "<td></td>").join("") +
      `<td class="r">${r.k}</td><td class="r">${r.n}</td><td class="r">${pct(r.k, r.n)}</td><td class="r">${pct(lo * 1000, 1000)} to ${pct(hi * 1000, 1000)}</td>`;
    cells.forEach((c, i) => (tr.children[i].textContent = c));
    body.append(tr);
  }
  table.append(body);
  return table;
}

const RENDER = { rate: renderRate, stack: renderStack, line: renderLine };

// Render into `el` now, and again whenever its width or the colour scheme changes.
export function mountChart(el, chart) {
  let lastWidth = 0;
  const draw = () => {
    const width = Math.floor(el.clientWidth);
    if (!width) return;
    lastWidth = width;
    el.replaceChildren();
    try {
      RENDER[chart.kind](el, chart, width, tokens());
    } catch (err) {
      console.error(err);
      el.replaceChildren(Object.assign(document.createElement("p"), { className: "empty", textContent: `This chart failed to render (${err.message}). Its data is under Show data.` }));
    }
  };
  new ResizeObserver(() => {
    if (Math.abs(el.clientWidth - lastWidth) > 4) draw();
  }).observe(el);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
  draw();
}
