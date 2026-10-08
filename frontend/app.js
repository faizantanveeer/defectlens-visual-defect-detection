"use strict";

const CFG = Object.assign(
  { title: "Visual Defect Detection", apiUrl: "http://localhost:8000", maxBatch: 16 },
  window.APP_CONFIG || {}
);
const API = CFG.apiUrl.replace(/\/+$/, "");
const IMAGE_TYPES = ["image/png", "image/jpeg", "image/bmp", "image/tiff", "image/webp"];

const $ = (sel) => document.querySelector(sel);
const els = {
  title: $("#app-title"),
  status: $("#status"),
  statusText: $("#status-text"),
  dropzone: $("#dropzone"),
  fileInput: $("#file-input"),
  samples: $("#sample-list"),
  results: $("#results"),
  empty: $("#empty"),
  summary: $("#summary"),
  clear: $("#clear-btn"),
  tpl: $("#card-tpl"),
};

const state = {
  model: null,
  online: false,
  sampleNames: [],
  counts: { normal: 0, defective: 0, review: 0, error: 0 },
};

/* ------------------------------------------------------------------ setup */

document.title = CFG.title;
els.title.textContent = CFG.title;
$("#api-url").textContent = API;
$("#docs-link").href = `${API}/docs`;

els.dropzone.addEventListener("click", () => els.fileInput.click());
els.dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); els.fileInput.click(); }
});
els.fileInput.addEventListener("change", () => {
  analyze([...els.fileInput.files]);
  els.fileInput.value = "";
});
["dragenter", "dragover"].forEach((t) =>
  els.dropzone.addEventListener(t, (e) => { e.preventDefault(); els.dropzone.classList.add("drag"); })
);
["dragleave", "drop"].forEach((t) =>
  els.dropzone.addEventListener(t, (e) => { e.preventDefault(); els.dropzone.classList.remove("drag"); })
);
els.dropzone.addEventListener("drop", (e) => analyze([...e.dataTransfer.files]));
document.addEventListener("paste", (e) => {
  const files = [...(e.clipboardData?.files || [])];
  if (files.length) analyze(files);
});
els.clear.addEventListener("click", () => {
  els.results.replaceChildren();
  state.counts = { normal: 0, defective: 0, review: 0, error: 0 };
  renderSummary();
});

$("#run-all-btn").addEventListener("click", runAllSamples);

pollHealth();
setInterval(pollHealth, 5000);
loadSamples().then(() => {
  // ?demo=1 runs every sample on load (handy for demos and screenshots).
  if (new URLSearchParams(location.search).get("demo") === "1") runAllSamples();
});

/* ------------------------------------------------------------------ API status */

async function pollHealth() {
  try {
    const res = await fetch(`${API}/health/ready`, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    if (!state.online || !state.model) await loadModel();
    setStatus(true, `API online · ${state.model ? state.model.backbone : "ready"}`);
  } catch {
    setStatus(false, "API offline");
  }
}

function setStatus(online, text) {
  state.online = online;
  els.status.classList.toggle("ok", online);
  els.status.classList.toggle("down", !online);
  els.statusText.textContent = text;
}

async function loadModel() {
  const res = await fetch(`${API}/v1/model`);
  if (!res.ok) return;
  const m = await res.json();
  state.model = m;
  $("#m-version").textContent = m.model_version;
  $("#m-backbone").textContent = m.backbone || "–";
  $("#m-threshold").textContent = m.threshold.toFixed(3);
  $("#m-review").textContent = `±${m.review_margin}`;
  $("#m-input").textContent = `${m.input_size}×${m.input_size} RGB`;
}

/* ------------------------------------------------------------------ samples */

async function loadSamples() {
  try {
    const res = await fetch("samples/index.json");
    if (!res.ok) throw new Error();
    const names = await res.json();
    state.sampleNames = names;
    for (const name of names) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "sample";
      btn.title = `Analyze ${name}`;
      const img = document.createElement("img");
      img.src = `samples/${encodeURIComponent(name)}`;
      img.alt = name;
      img.loading = "lazy";
      const label = document.createElement("span");
      label.textContent = prettyName(name);
      btn.append(img, label);
      btn.addEventListener("click", () => analyzeSample(name));
      els.samples.append(btn);
    }
  } catch {
    $(".samples").hidden = true;
  }
}

async function sampleFile(name) {
  const res = await fetch(`samples/${encodeURIComponent(name)}`);
  const blob = await res.blob();
  return new File([blob], name, { type: blob.type || "image/png" });
}

async function analyzeSample(name) {
  analyze([await sampleFile(name)]);
}

async function runAllSamples() {
  if (!state.sampleNames.length) return;
  analyze(await Promise.all(state.sampleNames.map(sampleFile)));
}

function prettyName(name) {
  return name.replace(/\.[^.]+$/, "").replace(/_/g, " ");
}

/* ------------------------------------------------------------------ inference */

async function analyze(files) {
  files = files.filter((f) => f && f.size > 0);
  if (!files.length) return;
  if (files.length > CFG.maxBatch) {
    toast(`Only the first ${CFG.maxBatch} images were sent (API batch limit).`);
    files = files.slice(0, CFG.maxBatch);
  }
  const unsupported = files.filter((f) => f.type && !IMAGE_TYPES.includes(f.type));
  if (unsupported.length) toast(`${unsupported.length} file(s) may not be supported images; the API will validate them.`);

  const cards = files.map((f) => createCard(f));
  // Insert the whole batch at the top, keeping upload order within the batch.
  const group = document.createDocumentFragment();
  cards.forEach((c) => group.append(c));
  els.results.prepend(group);
  els.dropzone.classList.add("busy");
  try {
    if (files.length === 1) {
      const form = new FormData();
      form.append("file", files[0]);
      const res = await fetch(`${API}/v1/predict`, { method: "POST", body: form });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) return fillError(cards[0], body, res.status);
      fillPrediction(cards[0], body.prediction, body.timing, body.request_id);
    } else {
      const form = new FormData();
      files.forEach((f) => form.append("files", f));
      const res = await fetch(`${API}/v1/predict/batch`, { method: "POST", body: form });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) return cards.forEach((c) => fillError(c, body, res.status));
      const batchInfo = { size: files.length, timing: body.timing };
      body.results.forEach((item, i) =>
        item.prediction
          ? fillPrediction(cards[i], item.prediction, null, body.request_id, batchInfo)
          : fillError(cards[i], { error: item.error }, 422)
      );
    }
  } catch (err) {
    cards.forEach((c) => fillError(c, { error: { message: `Cannot reach the API at ${API}. Is the backend running?` } }, 0));
    pollHealth();
  } finally {
    els.dropzone.classList.remove("busy");
  }
}

/* ------------------------------------------------------------------ rendering */

function createCard(file) {
  const node = els.tpl.content.firstElementChild.cloneNode(true);
  node.classList.add("pending");
  const img = node.querySelector("img");
  img.src = URL.createObjectURL(file);
  img.alt = file.name;
  node.querySelector(".filename").textContent = file.name;
  node.querySelector(".filename").title = file.name;
  node.querySelector(".badge").textContent = "Analyzing…";
  els.empty.hidden = true;
  els.clear.hidden = false;
  return node;
}

function fillPrediction(card, p, timing, requestId, batch = null) {
  card.classList.remove("pending");
  const defective = p.predicted_class === "defective";
  card.classList.add(defective ? "defective" : "normal");
  card.querySelector(".badge").textContent = defective ? "Defective" : "Normal";
  card.querySelector(".conf").textContent = pct(p.confidence);
  card.querySelector(".prob").textContent = p.defect_probability.toFixed(3);

  const t = p.threshold;
  const margin = state.model ? state.model.review_margin : 0;
  const band = card.querySelector(".band");
  band.style.left = `${Math.max(t - margin, 0) * 100}%`;
  band.style.width = `${(Math.min(t + margin, 1) - Math.max(t - margin, 0)) * 100}%`;
  card.querySelector(".thr").style.left = `calc(${t * 100}% - 1px)`;
  card.querySelector(".thr-label").textContent = `threshold ${t.toFixed(2)}`;
  requestAnimationFrame(() => {
    card.querySelector(".fill").style.width = `${Math.max(p.defect_probability * 100, 1.5)}%`;
  });

  card.querySelector(".review").hidden = !p.review_recommended;
  const timingEl = card.querySelector(".timing");
  if (timing) {
    timingEl.textContent = `${timing.total_ms.toFixed(0)} ms total · ${timing.inference_ms.toFixed(1)} ms model`;
    timingEl.title = `decode ${timing.decode_ms} ms · preprocess ${timing.preprocess_ms} ms · inference ${timing.inference_ms} ms\nrequest ${requestId}`;
  } else if (batch) {
    timingEl.textContent = `batch of ${batch.size} · ${batch.timing.total_ms.toFixed(0)} ms total`;
    timingEl.title = `one forward pass for the whole batch · inference ${batch.timing.inference_ms} ms\nrequest ${requestId}`;
  }

  state.counts[defective ? "defective" : "normal"] += 1;
  if (p.review_recommended) state.counts.review += 1;
  renderSummary();
}

function fillError(card, body, status) {
  card.classList.remove("pending");
  card.classList.add("error");
  const err = (body && body.error) || {};
  card.querySelector(".badge").textContent = status ? `Error ${status}` : "Offline";
  card.querySelector(".metrics").hidden = true;
  card.querySelector(".meter").hidden = true;
  card.querySelector(".meter-scale").hidden = true;
  const msg = card.querySelector(".error-msg");
  msg.hidden = false;
  msg.textContent = err.message || "Request failed.";
  state.counts.error += 1;
  renderSummary();
}

function renderSummary() {
  const c = state.counts;
  const pills = [];
  if (c.normal) pills.push(["ok", `${c.normal} normal`]);
  if (c.defective) pills.push(["bad", `${c.defective} defective`]);
  if (c.review) pills.push(["warn", `${c.review} review`]);
  if (c.error) pills.push(["", `${c.error} error`]);
  els.summary.replaceChildren(
    ...pills.map(([cls, text]) => {
      const s = document.createElement("span");
      s.className = `pill ${cls}`;
      s.textContent = text;
      return s;
    })
  );
  const any = els.results.children.length > 0;
  els.empty.hidden = any;
  els.clear.hidden = !any;
}

function pct(x) {
  return `${(x * 100).toFixed(1)}%`;
}

function toast(text) {
  const t = document.createElement("div");
  t.className = "toast";
  t.textContent = text;
  document.body.append(t);
  setTimeout(() => t.remove(), 3500);
}
