import { api } from "./api.js";
import { escapeHtml, fmt, levelColors, onThemeChange, renderConvergence, renderPoolMap } from "./charts.js";

const $ = (id) => document.getElementById(id);
const TERMINAL = new Set(["succeeded", "failed", "cancelled", "budget_exhausted"]);
const STATUS_CLASS = {
  succeeded: "good", running: "active", queued: "", cancelling: "warning", cancelled: "warning",
  budget_exhausted: "serious", interrupted: "serious", failed: "critical",
};

const state = { jobId: null, job: null, observations: [], source: null, poolRound: null, set: null, poolTimer: null, labels: null };

function statusBadge(status, label = status.replace("_", " ")) {
  return `<span class="status ${STATUS_CLASS[status] ?? ""}"><span class="dot" aria-hidden="true"></span>${escapeHtml(label)}</span>`;
}

function setMsg(el, text, error = false) {
  el.textContent = text;
  el.classList.toggle("error", error);
}

// ---- sidebar -----------------------------------------------------------------------------------

async function loadHealth() {
  try {
    const h = await api.health();
    $("health").innerHTML = h.ok && h.workers.length ? statusBadge("succeeded", "service ready") : statusBadge("interrupted", "no worker");
  } catch {
    $("health").innerHTML = statusBadge("failed", "service unreachable");
  }
}

async function loadJobs() {
  const jobs = await api.jobs();
  const list = $("job-list");
  if (!jobs.length) {
    list.innerHTML = `<li class="muted small-text">No jobs yet.</li>`;
    return;
  }
  list.innerHTML = jobs
    .map(
      (j) => `<li><button type="button" data-id="${j.id}" aria-current="${j.id === state.jobId}">
        <span class="jl-top"><span>${escapeHtml(j.set_name)}</span>${statusBadge(j.status)}</span>
        <span class="jl-sub">${escapeHtml(j.backbone_name)} · seed ${j.seed} · ${j.best_acc != null ? `best ${fmt(j.best_acc)}` : "—"} · ${new Date(j.created_at * 1000).toLocaleString()}</span>
      </button></li>`,
    )
    .join("");
}

async function loadDatasets(selectId) {
  const [datasets, tasks] = await Promise.all([api.datasets(), api.tasks()]);
  const taskName = Object.fromEntries(tasks.map((t) => [t.id, `${t.name} v${t.version}`]));
  $("dataset-select").innerHTML = datasets.length
    ? datasets
        .map((d) => `<option value="${d.id}" data-task="${d.task_id}">${escapeHtml(d.name)} — ${escapeHtml(taskName[d.task_id] ?? d.task_id)} (${d.n_items})</option>`)
        .join("")
    : `<option value="">Import a preset first</option>`;
  if (selectId) $("dataset-select").value = selectId;
}

async function loadPresets() {
  const presets = await api.presets();
  $("preset-select").innerHTML = presets
    .map((p) => `<option value="${p.preset}|${p.task}" ${p.prepared ? "" : "disabled"}>${p.preset} / ${p.task}${p.prepared ? "" : " (run guide data prepare)"}</option>`)
    .join("");
}

async function loadBackbones() {
  const backbones = await api.backbones();
  $("backbone-select").innerHTML = backbones
    .map((b) => `<option value="${b.name}" title="${escapeHtml(b.missing.join("; "))}">${b.name} — ${escapeHtml(b.model)}${b.ready ? "" : " (needs settings)"}</option>`)
    .join("");
  const ready = backbones.find((b) => b.ready && b.kind === "fake");
  if (ready) $("backbone-select").value = ready.name;
}

function jobBody() {
  const f = new FormData($("job-form"));
  const opt = $("dataset-select").selectedOptions[0];
  const num = (k) => (f.get(k) === "" || f.get(k) == null ? undefined : Number(f.get(k)));
  const config = { n_rounds: num("n_rounds"), n_eval: num("n_eval"), k_min: num("k_min"), k_max: num("k_max") };
  if (config.n_eval !== undefined) config.n_init = Math.min(8, config.n_eval);
  return {
    body: {
      task_id: opt?.dataset.task, dataset_id: opt?.value, backbone: f.get("backbone"), seed: num("seed") ?? 0,
      config, max_cost_usd: num("max_cost_usd") ?? null,
    },
    key: f.get("api_key") || undefined,
  };
}

function describeEstimate(e) {
  return `${e.total_calls.toLocaleString()} calls, about ${Math.max(1, Math.round(e.est_minutes))} min\n` +
    `$${e.usd_low.toFixed(2)} with prefix caching – $${e.usd_high.toFixed(2)} without (worst case $${e.usd_hard_max.toFixed(2)})`;
}

// ---- job view ----------------------------------------------------------------------------------

function closeStream() {
  if (state.source) state.source.close();
  state.source = null;
}

async function selectJob(id) {
  closeStream();
  state.jobId = id;
  state.observations = [];
  state.set = null;
  state.poolRound = null;
  location.hash = `#/jobs/${id}`;
  $("empty").hidden = true;
  $("job-view").hidden = false;
  $("event-log").innerHTML = "";
  $("grade-table").innerHTML = "";
  setMsg($("grade-msg"), "");
  await refreshJob();
  state.observations = await api.observations(id);
  drawConvergence();
  await refreshPool();
  if (state.job.exemplar_set_id) await loadSet(state.job.exemplar_set_id);
  else $("exemplars").innerHTML = `<p class="muted">The frozen set appears here when the job finishes.</p>`;
  loadJobs();
  openStream(id);
}

async function refreshJob() {
  const j = (state.job = await api.job(state.jobId));
  $("job-title").textContent = j.set_name;
  $("job-sub").textContent = `${j.id} · ${j.backbone_name} · seed ${j.seed} · ${Object.entries(j.config).map(([k, v]) => `${k}=${v}`).join(", ") || "default config"}`;
  $("t-status").innerHTML = statusBadge(j.status) + (j.error ? `<span class="sub">${escapeHtml(j.error.slice(0, 80))}</span>` : "");
  const rounds = j.config.n_rounds ?? 5;
  $("t-progress").innerHTML = j.round ? `round ${j.round}<span class="sub">of ${rounds} · ${escapeHtml(j.phase ?? "")}</span>` : escapeHtml(j.phase ?? "–");
  $("t-best").textContent = fmt(j.best_acc);
  $("t-cost").innerHTML = `$${j.cost_usd.toFixed(j.cost_usd < 1 ? 4 : 2)}${j.max_cost_usd != null ? `<span class="sub">of $${j.max_cost_usd.toFixed(2)}</span>` : ""}`;
  const cachedPct = j.calls ? Math.round((100 * j.cached_calls) / j.calls) : 0;
  $("t-calls").innerHTML = `${j.calls.toLocaleString()}<span class="sub">${cachedPct}% cached</span>`;
  $("cancel-btn").disabled = TERMINAL.has(j.status) || j.status === "cancelling";
  $("resume-btn").disabled = !["interrupted", "failed", "cancelled", "budget_exhausted"].includes(j.status);
}

function drawConvergence() {
  renderConvergence($("convergence"), state.observations);
  const rows = state.observations
    .map((o) => `<tr><td>${o.index}</td><td>${o.round}</td><td>${o.op}</td><td>${o.size}</td><td>${fmt(o.acc)}</td><td>${fmt(o.qwk)}</td><td>${fmt(o.contrastive, 2)}</td><td>${o.n_failed}</td></tr>`)
    .join("");
  $("obs-table").innerHTML = `<thead><tr><th>#</th><th>Round</th><th>Operator</th><th>|E|</th><th>Acc</th><th>QWK</th><th>Contrastive</th><th>Failed</th></tr></thead><tbody>${rows}</tbody>`;
}

async function refreshPool() {
  try {
    const data = await api.poolMap(state.jobId, state.poolRound);
    state.labels = data.labels;
    renderPoolMap($("poolmap"), data);
    $("pool-meta").textContent = `${data.points.length} exemplars · ${data.n_boundary_pairs} boundary pairs at τ = ${data.tau} · checkpoint round ${data.round} (${data.phase})` +
      (data.edges.length < data.n_boundary_pairs ? ` · showing ${data.edges.length}: the most similar, plus every pair touching a selected exemplar` : "");
    const rounds = Math.max(state.job.round, data.round);
    const sel = $("round-select");
    const current = state.poolRound ?? "";
    sel.innerHTML = `<option value="">latest</option>` + Array.from({ length: rounds }, (_, i) => `<option value="${i + 1}">${i + 1}</option>`).join("");
    sel.value = current;
  } catch (err) {
    $("pool-meta").textContent = err.message;
  }
}

function schedulePoolRefresh() {
  clearTimeout(state.poolTimer);
  state.poolTimer = setTimeout(refreshPool, 600);
}

async function loadSet(id) {
  const s = (state.set = await api.exemplarSet(id));
  const taskLabels = state.labels ?? [...new Set(s.exemplars.map((e) => e.label))].sort();
  const colors = levelColors(taskLabels.length);
  $("set-meta").textContent = `${s.name} v${s.version} · ${s.status} · ${s.exemplars.length} exemplars` +
    (s.val_metrics ? ` · validation accuracy ${fmt(s.val_metrics.acc)}, QWK ${fmt(s.val_metrics.qwk)}` : "");
  $("exemplars").innerHTML = s.exemplars
    .map((e) => {
      const rank = taskLabels.indexOf(e.label);
      return `<article class="ex">
        <div class="ex-head"><span class="ex-label"><span class="swatch" style="background:${colors[rank] ?? "var(--muted)"}"></span>score ${escapeHtml(e.label)}</span>
          <span class="ex-tag">${escapeHtml(e.origin)}${e.round ? ` r${e.round}` : ""}${e.is_expert ? " · expert" : ""}</span></div>
        <div class="ex-rationale">${escapeHtml(e.rationale)}</div>
        <div class="ex-text">${escapeHtml(e.text)}</div>
      </article>`;
    })
    .join("");
}

function logEvent(msg) {
  if (msg.type === "eval_progress") return;
  const li = document.createElement("li");
  const payload = JSON.stringify(msg.payload);
  li.textContent = `${msg.type} r${msg.round} ${payload.length > 160 ? payload.slice(0, 160) + "…" : payload}`;
  $("event-log").prepend(li);
  while ($("event-log").children.length > 200) $("event-log").lastChild.remove();
}

function openStream(id) {
  const src = new EventSource(`/api/jobs/${id}/events`);
  state.source = src;
  const handle = async (evt) => {
    if (state.jobId !== id) return;
    const msg = JSON.parse(evt.data);
    logEvent(msg);
    const p = msg.payload;
    switch (msg.type) {
      case "eval_progress":
        $("evalbar-fill").style.width = `${Math.round((100 * p.done) / p.total)}%`;
        break;
      case "eval_completed":
        if (!state.observations.some((o) => o.index === p.index)) {
          state.observations.push({ round: msg.round, ...p });
          drawConvergence();
        }
        $("evalbar-fill").style.width = "0";
        refreshJob();
        break;
      case "pool_ready":
      case "phase2_completed":
      case "round_completed":
        schedulePoolRefresh();
        refreshJob();
        break;
      case "usage":
      case "phase_started":
      case "job_started":
        refreshJob();
        break;
      case "job_succeeded":
      case "job_cancelled":
      case "job_budget_exhausted":
      case "job_failed":
        src.close();
        await refreshJob();
        if (state.job.exemplar_set_id) await loadSet(state.job.exemplar_set_id);
        schedulePoolRefresh();
        loadJobs();
        break;
    }
  };
  for (const t of ["job_queued", "job_started", "phase_started", "phase0_completed", "pool_ready", "eval_progress", "eval_completed",
    "round_completed", "phase2_completed", "checkpoint_saved", "usage", "budget_warning", "resumed", "search_exhausted",
    "optimizer_finished", "job_interrupted", "job_succeeded", "job_cancelled", "job_budget_exhausted", "job_failed"]) {
    src.addEventListener(t, handle);
  }
}

// ---- wiring ------------------------------------------------------------------------------------

$("job-list").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-id]");
  if (btn) selectJob(btn.dataset.id);
});
$("refresh-jobs").addEventListener("click", loadJobs);

$("import-preset").addEventListener("click", async () => {
  const [preset, task] = $("preset-select").value.split("|");
  const lim = (id) => ($(id).value ? Number($(id).value) : null);
  setMsg($("form-msg"), "Importing…");
  try {
    const out = await api.importPreset({ preset, task, train_limit: lim("lim-train"), val_limit: lim("lim-val"), test_limit: lim("lim-test") });
    await loadDatasets(out.dataset.id);
    setMsg($("form-msg"), `Imported ${out.dataset.name}: ${out.dataset.n_items} items (${Object.entries(out.dataset.split_counts).map(([k, v]) => `${k} ${v}`).join(", ")}).`);
  } catch (err) {
    setMsg($("form-msg"), err.message, true);
  }
});

$("estimate-btn").addEventListener("click", async () => {
  const { body, key } = jobBody();
  try {
    const out = await api.createJob({ ...body, dry_run: true }, key);
    setMsg($("form-msg"), describeEstimate(out.estimate));
  } catch (err) {
    setMsg($("form-msg"), err.message, true);
  }
});

$("job-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const { body, key } = jobBody();
  try {
    const out = await api.createJob(body, key);
    setMsg($("form-msg"), `Queued. ${describeEstimate(out.estimate)}`);
    await loadJobs();
    selectJob(out.job.id);
  } catch (err) {
    setMsg($("form-msg"), err.message, true);
  }
});

$("cancel-btn").addEventListener("click", async () => {
  await api.cancelJob(state.jobId).catch((err) => alertInline(err));
  refreshJob();
});
$("resume-btn").addEventListener("click", async () => {
  const key = new FormData($("job-form")).get("api_key") || undefined;
  try {
    await api.resumeJob(state.jobId, key);
    selectJob(state.jobId);
  } catch (err) {
    alertInline(err);
  }
});
$("round-select").addEventListener("change", (e) => {
  state.poolRound = e.target.value ? Number(e.target.value) : null;
  refreshPool();
});

$("grade-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!state.set) return setMsg($("grade-msg"), "No exemplar set yet.", true);
  const lines = $("grade-input").value.split("\n").map((s) => s.trim()).filter(Boolean);
  if (!lines.length) return setMsg($("grade-msg"), "Enter at least one response.", true);
  setMsg($("grade-msg"), "Grading…");
  try {
    const key = new FormData($("job-form")).get("api_key") || undefined;
    const out = await api.grade({ exemplar_set_id: state.set.id, responses: lines.map((t, i) => ({ id: String(i + 1), text: t })) }, key);
    setMsg($("grade-msg"), `${out.usage.calls} calls · $${out.usage.cost_usd.toFixed(4)}${out.warnings.length ? " · " + out.warnings.join("; ") : ""}`);
    $("grade-table").innerHTML = `<thead><tr><th>#</th><th>Score</th><th>Analysis</th><th>Response</th></tr></thead><tbody>` +
      out.results.map((r, i) => `<tr><td>${r.id}</td><td>${escapeHtml(r.label ?? "failed")}</td><td>${escapeHtml(r.analysis || r.error || "")}</td><td>${escapeHtml(lines[i].slice(0, 120))}</td></tr>`).join("") +
      "</tbody>";
  } catch (err) {
    setMsg($("grade-msg"), err.message, true);
  }
});

function alertInline(err) {
  $("job-sub").textContent = `${$("job-sub").textContent} — ${err.message}`;
}

onThemeChange(() => {
  if (!state.jobId) return;
  drawConvergence();
  refreshPool();
  if (state.set) loadSet(state.set.id);
});

async function init() {
  await Promise.all([loadHealth(), loadJobs(), loadDatasets(), loadPresets(), loadBackbones()]);
  const m = location.hash.match(/^#\/jobs\/(.+)$/);
  if (m) selectJob(m[1]);
  setInterval(loadHealth, 15000);
}

window.addEventListener("load", init);
