"use strict";

const state = {
  images: [],
  index: 0,
  annotation: null,
  image: null,
  mode: "cuts",
  zoom: 6,
  history: [],
  dragging: null,
  dirty: false,
  saveTimer: null,
};

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const imageList = document.getElementById("imageList");
const saveStatus = document.getElementById("saveStatus");

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok || payload.ok === false) {
    throw new Error(payload.error || `${response.status} ${response.statusText}`);
  }
  return payload;
}

function imageUrl(path) {
  return `/api/image?image=${encodeURIComponent(path)}`;
}

function annotationUrl(path) {
  return `/api/annotation?image=${encodeURIComponent(path)}`;
}

async function loadState() {
  const payload = await requestJson("/api/state");
  state.images = payload.images;
  renderImageList();
  updateProgress();
  if (state.images.length) {
    const firstIncomplete = state.images.findIndex((item) => !item.completed);
    await openImage(firstIncomplete >= 0 ? firstIncomplete : 0);
  } else {
    saveStatus.textContent = "No images";
  }
}

async function openImage(index) {
  if (!state.images.length) return;
  await saveNow();
  state.index = Math.min(Math.max(index, 0), state.images.length - 1);
  const item = state.images[state.index];
  saveStatus.textContent = "Loading";
  const [annotation, image] = await Promise.all([
    requestJson(annotationUrl(item.path)),
    loadImage(imageUrl(item.path)),
  ]);
  state.annotation = annotation;
  state.image = image;
  state.history = [];
  state.dragging = null;
  state.dirty = false;
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  applyZoom();
  syncControls();
  draw();
  saveStatus.textContent = "Saved";
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Cannot load image"));
    image.src = url;
  });
}

function snapshot() {
  return JSON.stringify({
    cuts: state.annotation.cuts,
    baselines: state.annotation.baselines,
    completed: state.annotation.completed,
  });
}

function remember() {
  if (!state.annotation) return;
  state.history.push(snapshot());
  if (state.history.length > 100) state.history.shift();
}

function undo() {
  const previous = state.history.pop();
  if (!previous || !state.annotation) return;
  Object.assign(state.annotation, JSON.parse(previous));
  markDirty();
  syncControls();
  draw();
}

function markDirty() {
  state.dirty = true;
  saveStatus.textContent = "Unsaved";
  clearTimeout(state.saveTimer);
  state.saveTimer = setTimeout(saveNow, 450);
}

async function saveNow() {
  clearTimeout(state.saveTimer);
  if (!state.dirty || !state.annotation || !state.images.length) return;
  saveStatus.textContent = "Saving";
  const item = state.images[state.index];
  try {
    const payload = await requestJson(annotationUrl(item.path), {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(state.annotation),
    });
    state.annotation = payload.item;
    state.dirty = false;
    item.completed = state.annotation.completed;
    item.cuts = state.annotation.cuts.length;
    item.top_points = state.annotation.baselines.top.length;
    item.bottom_points = state.annotation.baselines.bottom.length;
    renderImageList();
    updateProgress();
    saveStatus.textContent = "Saved";
  } catch (error) {
    saveStatus.textContent = "Save failed";
    console.error(error);
  }
}

function draw() {
  if (!state.image || !state.annotation) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(state.image, 0, 0);

  ctx.lineWidth = 1;
  ctx.strokeStyle = "#20df73";
  for (const x of state.annotation.cuts) {
    ctx.beginPath();
    ctx.moveTo(x + 0.5, 0);
    ctx.lineTo(x + 0.5, canvas.height);
    ctx.stroke();
  }

  drawBaseline(state.annotation.baselines.top, "#ff4f55");
  drawBaseline(state.annotation.baselines.bottom, "#438cff");
}

function drawBaseline(points, color) {
  if (!points.length) return;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 1;
  ctx.beginPath();
  points.forEach((point, index) => {
    if (index === 0) ctx.moveTo(point[0] + 0.5, point[1] + 0.5);
    else ctx.lineTo(point[0] + 0.5, point[1] + 0.5);
  });
  ctx.stroke();
  for (const point of points) {
    ctx.beginPath();
    ctx.arc(point[0], point[1], Math.max(1.25, 3 / state.zoom), 0, Math.PI * 2);
    ctx.fill();
  }
}

function canvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: Math.min(
      canvas.width - 1,
      Math.max(0, (event.clientX - rect.left - canvas.clientLeft) * canvas.width / canvas.clientWidth),
    ),
    y: Math.min(
      canvas.height - 1,
      Math.max(0, (event.clientY - rect.top - canvas.clientTop) * canvas.height / canvas.clientHeight),
    ),
  };
}

function nearestPoint(points, point) {
  let best = null;
  const tolerance = Math.max(2, 9 / state.zoom);
  points.forEach((candidate, index) => {
    const distance = Math.hypot(candidate[0] - point.x, candidate[1] - point.y);
    if (distance <= tolerance && (!best || distance < best.distance)) {
      best = {index, distance};
    }
  });
  return best;
}

function nearestCut(cuts, x) {
  let best = null;
  const tolerance = Math.max(2, 8 / state.zoom);
  cuts.forEach((candidate, index) => {
    const distance = Math.abs(candidate - x);
    if (distance <= tolerance && (!best || distance < best.distance)) {
      best = {index, distance};
    }
  });
  return best;
}

canvas.addEventListener("pointerdown", (event) => {
  if (!state.annotation || event.button !== 0) return;
  const point = canvasPoint(event);
  remember();

  if (state.mode === "cuts") {
    const nearest = nearestCut(state.annotation.cuts, point.x);
    if (nearest) state.annotation.cuts.splice(nearest.index, 1);
    else state.annotation.cuts.push(roundCoordinate(point.x));
    state.annotation.cuts.sort((a, b) => a - b);
  } else {
    const points = state.annotation.baselines[state.mode];
    const nearest = nearestPoint(points, point);
    if (nearest) {
      state.dragging = {mode: state.mode, index: nearest.index};
      canvas.setPointerCapture(event.pointerId);
    } else {
      points.push([roundCoordinate(point.x), roundCoordinate(point.y)]);
      points.sort((a, b) => a[0] - b[0]);
      const index = points.findIndex((candidate) => candidate[0] === roundCoordinate(point.x) && candidate[1] === roundCoordinate(point.y));
      state.dragging = {mode: state.mode, index};
      canvas.setPointerCapture(event.pointerId);
    }
  }
  markDirty();
  syncControls();
  draw();
});

canvas.addEventListener("pointermove", (event) => {
  if (!state.dragging || !state.annotation) return;
  const point = canvasPoint(event);
  const points = state.annotation.baselines[state.dragging.mode];
  points[state.dragging.index] = [roundCoordinate(point.x), roundCoordinate(point.y)];
  points.sort((a, b) => a[0] - b[0]);
  state.dragging.index = points.findIndex((candidate) => candidate[0] === roundCoordinate(point.x) && candidate[1] === roundCoordinate(point.y));
  markDirty();
  draw();
});

canvas.addEventListener("pointerup", () => {
  state.dragging = null;
});

canvas.addEventListener("contextmenu", (event) => {
  event.preventDefault();
  if (!state.annotation) return;
  const point = canvasPoint(event);
  remember();
  if (state.mode === "cuts") {
    const nearest = nearestCut(state.annotation.cuts, point.x);
    if (nearest) state.annotation.cuts.splice(nearest.index, 1);
  } else {
    const points = state.annotation.baselines[state.mode];
    const nearest = nearestPoint(points, point);
    if (nearest) points.splice(nearest.index, 1);
  }
  markDirty();
  syncControls();
  draw();
});

function roundCoordinate(value) {
  return Math.round(value * 10) / 10;
}

function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll(".mode").forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === mode);
  });
}

function clearLayer() {
  if (!state.annotation) return;
  remember();
  if (state.mode === "cuts") state.annotation.cuts = [];
  else state.annotation.baselines[state.mode] = [];
  markDirty();
  syncControls();
  draw();
}

function applyZoom() {
  state.zoom = Number(document.getElementById("zoom").value);
  canvas.style.width = `${canvas.width * state.zoom + canvas.clientLeft * 2}px`;
  canvas.style.height = `${canvas.height * state.zoom + canvas.clientTop * 2}px`;
  document.getElementById("zoomValue").value = `${state.zoom}×`;
}

function syncControls() {
  if (!state.annotation) return;
  document.getElementById("counter").textContent = `${state.index + 1} / ${state.images.length}`;
  document.getElementById("imageName").textContent = state.images[state.index].path;
  document.getElementById("completed").checked = Boolean(state.annotation.completed);
  document.getElementById("cutsCount").textContent = state.annotation.cuts.length;
  document.getElementById("topCount").textContent = state.annotation.baselines.top.length;
  document.getElementById("bottomCount").textContent = state.annotation.baselines.bottom.length;
  imageList.selectedIndex = state.index;
}

function renderImageList() {
  imageList.replaceChildren();
  state.images.forEach((item, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `${item.completed ? "✓" : "·"} ${item.path}`;
    imageList.append(option);
  });
  imageList.selectedIndex = state.index;
}

function updateProgress() {
  const completed = state.images.filter((item) => item.completed).length;
  document.getElementById("progressText").textContent = `${completed} / ${state.images.length}`;
  document.getElementById("progressBar").style.width = `${state.images.length ? completed * 100 / state.images.length : 0}%`;
}

document.querySelectorAll(".mode").forEach((button) => {
  button.addEventListener("click", () => setMode(button.dataset.mode));
});
document.getElementById("previous").addEventListener("click", () => openImage(state.index - 1));
document.getElementById("next").addEventListener("click", () => openImage(state.index + 1));
document.getElementById("undo").addEventListener("click", undo);
document.getElementById("clearLayer").addEventListener("click", clearLayer);
document.getElementById("save").addEventListener("click", saveNow);
document.getElementById("zoom").addEventListener("input", () => {
  applyZoom();
  draw();
});
document.getElementById("completed").addEventListener("change", (event) => {
  if (!state.annotation) return;
  remember();
  state.annotation.completed = event.target.checked;
  markDirty();
});
imageList.addEventListener("change", () => openImage(Number(imageList.value)));

window.addEventListener("keydown", (event) => {
  if (event.ctrlKey && event.key.toLowerCase() === "s") {
    event.preventDefault();
    saveNow();
  } else if (event.ctrlKey && event.key.toLowerCase() === "z") {
    event.preventDefault();
    undo();
  } else if (event.key === "ArrowLeft") {
    openImage(state.index - 1);
  } else if (event.key === "ArrowRight") {
    openImage(state.index + 1);
  } else if (event.key === "1") {
    setMode("cuts");
  } else if (event.key === "2") {
    setMode("top");
  } else if (event.key === "3") {
    setMode("bottom");
  }
});

window.addEventListener("beforeunload", () => {
  if (state.dirty) saveNow();
});

loadState().catch((error) => {
  saveStatus.textContent = error.message;
  console.error(error);
});
