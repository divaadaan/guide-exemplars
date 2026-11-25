// Plotly charts that read their colors from CSS custom properties, so light/dark mode is selected, not flipped.

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const isDark = () => css("color-scheme") === "dark";

// Ordinal one-hue blue ramps (validated with the dataviz validator: monotone lightness, visible steps,
// the step nearest the surface clears 2:1). Higher score levels read as more ink.
const RAMP_LIGHT = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"];
const RAMP_DARK = ["#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6"];

export function levelColors(n) {
  const ramp = isDark() ? RAMP_DARK : RAMP_LIGHT;
  if (n <= 1) return [ramp[ramp.length - 1]];
  return Array.from({ length: n }, (_, i) => ramp[Math.round((i * (ramp.length - 1)) / (n - 1))]);
}

function baseLayout() {
  const ink2 = css("--ink-2");
  return {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: css("--font"), size: 12, color: ink2 },
    margin: { l: 48, r: 16, t: 8, b: 40 },
    hoverlabel: { bgcolor: css("--surface"), bordercolor: css("--axis"), font: { color: css("--ink"), size: 12 } },
    legend: { orientation: "h", x: 0, y: 1.08, font: { color: ink2 } },
    xaxis: { gridcolor: css("--grid"), linecolor: css("--axis"), zeroline: false, tickfont: { color: css("--muted") } },
    yaxis: { gridcolor: css("--grid"), linecolor: css("--axis"), zeroline: false, tickfont: { color: css("--muted") } },
  };
}

const CONFIG = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"] };

export function renderConvergence(el, observations) {
  if (!window.Plotly) return;
  const xs = observations.map((o) => o.index);
  const acc = observations.map((o) => o.acc);
  let best = -Infinity;
  const bestSoFar = acc.map((a) => (best = Math.max(best, a)));
  const hover = observations.map(
    (o) =>
      `Eval ${o.index} · round ${o.round}<br>${o.op} · |E| = ${o.size}<br>accuracy ${fmt(o.acc)} · QWK ${fmt(o.qwk)}` +
      `<br>contrastive ${fmt(o.contrastive, 2)}${o.ei != null ? ` · EI ${o.ei.toExponential(1)}` : ""}`,
  );
  const traces = [
    {
      x: xs, y: acc, name: "evaluated set", mode: "markers", type: "scatter",
      marker: { size: 8, color: css("--point-muted"), opacity: 0.75, line: { width: 2, color: css("--surface") } },
      hovertext: hover, hoverinfo: "text",
    },
    {
      x: xs, y: bestSoFar, name: "best so far", mode: "lines", type: "scatter", line: { width: 2, color: css("--accent"), shape: "hv" },
      hoverinfo: "skip",
    },
  ];
  const shapes = [];
  for (let i = 1; i < observations.length; i++) {
    if (observations[i].round !== observations[i - 1].round) {
      const x = (observations[i].index + observations[i - 1].index) / 2;
      shapes.push({ type: "line", x0: x, x1: x, yref: "paper", y0: 0, y1: 1, line: { color: css("--axis"), width: 1, dash: "dot" } });
    }
  }
  const layout = baseLayout();
  layout.shapes = shapes;
  layout.xaxis.title = { text: "evaluation", font: { color: css("--muted") } };
  layout.yaxis.title = { text: "validation accuracy", font: { color: css("--muted") } };
  layout.yaxis.rangemode = "tozero";
  layout.hovermode = "closest";
  Plotly.react(el, traces, layout, CONFIG);
}

export function renderPoolMap(el, data) {
  if (!window.Plotly) return;
  const colors = levelColors(data.labels.length);
  const pts = data.points;
  // Background: boundary pairs in general. Foreground: pairs that touch a selected exemplar, which is what
  // the contrastive search is aiming for.
  const bx = [], by = [], sx = [], sy = [];
  for (const [i, j] of data.edges) {
    const [xs, ys] = pts[i].selected || pts[j].selected ? [sx, sy] : [bx, by];
    xs.push(pts[i].x, pts[j].x, null);
    ys.push(pts[i].y, pts[j].y, null);
  }
  const traces = [
    { x: bx, y: by, mode: "lines", type: "scattergl", name: "boundary pair", line: { width: 0.6, color: css("--edge") }, hoverinfo: "skip" },
    { x: sx, y: sy, mode: "lines", type: "scattergl", name: "pair with a selected exemplar", line: { width: 1.4, color: css("--accent") }, opacity: 0.55, hoverinfo: "skip" },
  ];
  data.labels.forEach((label, rank) => {
    const group = pts.filter((p) => p.rank === rank);
    if (!group.length) return;
    traces.push({
      x: group.map((p) => p.x),
      y: group.map((p) => p.y),
      name: `score ${label}`,
      mode: "markers",
      type: "scatter",
      marker: {
        size: group.map((p) => (p.selected ? 14 : 8)),
        symbol: group.map((p) => (p.is_expert ? "diamond" : "circle")),
        color: colors[rank],
        line: { width: group.map((p) => (p.selected ? 3 : 1)), color: group.map((p) => (p.selected ? css("--ink") : css("--surface"))) },
      },
      hovertext: group.map(
        (p) =>
          `<b>score ${escapeHtml(p.label)}</b> · ${p.origin}${p.is_expert ? " · expert" : ""}${p.selected ? " · <b>selected</b>" : ""}` +
          `<br>${wrap(escapeHtml(p.text), 70)}<br><i>${wrap(escapeHtml(p.rationale), 70)}</i>`,
      ),
      hoverinfo: "text",
    });
  });
  const layout = baseLayout();
  layout.xaxis.showticklabels = false;
  layout.yaxis.showticklabels = false;
  layout.xaxis.showgrid = layout.yaxis.showgrid = false;
  layout.hovermode = "closest";
  Plotly.react(el, traces, layout, CONFIG);
}

export function fmt(x, digits = 3) {
  return x == null || Number.isNaN(x) ? "–" : Number(x).toFixed(digits);
}

export function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function wrap(s, width) {
  const words = s.split(" ");
  const lines = [];
  let line = "";
  for (const w of words) {
    if ((line + " " + w).length > width && line) {
      lines.push(line);
      line = w;
    } else line = line ? `${line} ${w}` : w;
  }
  if (line) lines.push(line);
  return lines.slice(0, 4).join("<br>") + (lines.length > 4 ? "…" : "");
}

export function onThemeChange(fn) {
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", fn);
  new MutationObserver(fn).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
}
