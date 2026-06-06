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
  revision: 0,
  savePromise: Promise.resolve(true),
  navigating: false,
};

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const canvasWrap = document.getElementById("canvasWrap");
const overlay = document.getElementById("overlay");
const canvasScroller = document.getElementById("canvasScroller");
const imageList = document.getElementById("imageList");
const saveStatus = document.getElementById("saveStatus");
const zoomInput = document.getElementById("zoom");

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
    const firstUnmarked = state.images.findIndex((item) => !item.marked);
    await openImage(firstUnmarked >= 0 ? firstUnmarked : 0);
  } else {
    saveStatus.textContent = "No images";
  }
}

async function openImage(index) {
  if (!state.images.length || state.navigating) return;
  state.navigating = true;
  try {
    const saved = await saveNow(true);
    if (!saved) return;

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
    state.revision = 0;
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    overlay.setAttribute("viewBox", `0 0 ${canvas.width} ${canvas.height}`);
    applyZoom();
    syncControls();
    draw();
    saveStatus.textContent = "Saved";
  } finally {
    state.navigating = false;
  }
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
  state.revision += 1;
  saveStatus.textContent = "Unsaved";
  clearTimeout(state.saveTimer);
  state.saveTimer = setTimeout(saveNow, 450);
}

async function saveNow(force = false) {
  clearTimeout(state.saveTimer);
  if (!state.annotation || !state.images.length) return true;
  if (!state.dirty && !force) return true;

  const item = state.images[state.index];
  const path = item.path;
  const revision = state.revision;
  const body = JSON.stringify(state.annotation);

  const operation = async () => {
    saveStatus.textContent = "Saving";
    try {
      const payload = await requestJson(annotationUrl(path), {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body,
      });
      item.cuts = payload.item.cuts.length;
      item.top_points = payload.item.baselines.top.length;
      item.bottom_points = payload.item.baselines.bottom.length;
      item.marked = Boolean(item.cuts || item.top_points || item.bottom_points);

      const currentPath = state.images[state.index]?.path;
      if (currentPath === path && state.revision === revision) {
        state.annotation = payload.item;
        state.dirty = false;
        saveStatus.textContent = "Saved";
      }
      renderImageList();
      updateProgress();
      return true;
    } catch (error) {
      saveStatus.textContent = "Save failed";
      console.error(error);
      return false;
    }
  };

  state.savePromise = state.savePromise.then(operation, operation);
  return state.savePromise;
}

function draw() {
  if (!state.image || !state.annotation) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(state.image, 0, 0);
  drawOverlay();
}

function drawOverlay() {
  overlay.replaceChildren();
  for (const x of state.annotation.cuts) {
    overlay.append(createSvgElement("line", {
      x1: x,
      y1: 0,
      x2: x,
      y2: canvas.height,
      stroke: "#20df73",
      "stroke-width": 1,
      "vector-effect": "non-scaling-stroke",
    }));
  }

  drawBaselineOverlay(state.annotation.baselines.top, "#ff4f55");
  drawBaselineOverlay(state.annotation.baselines.bottom, "#438cff");
}

function drawBaselineOverlay(points, color) {
  if (!points.length) return;
  overlay.append(createSvgElement("polyline", {
    points: points.map((point) => `${point[0]},${point[1]}`).join(" "),
    fill: "none",
    stroke: color,
    "stroke-width": 1,
    "vector-effect": "non-scaling-stroke",
  }));
  for (const point of points) {
    overlay.append(createSvgElement("circle", {
      cx: point[0],
      cy: point[1],
      r: 3 / state.zoom,
      fill: color,
    }));
  }
}

function createSvgElement(name, attributes) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
  return element;
}

function canvasPoint(event) {
  const rect = overlay.getBoundingClientRect();
  return {
    x: Math.min(
      canvas.width - 1,
      Math.max(0, (event.clientX - rect.left) * canvas.width / rect.width),
    ),
    y: Math.min(
      canvas.height - 1,
      Math.max(0, (event.clientY - rect.top) * canvas.height / rect.height),
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

overlay.addEventListener("pointerdown", (event) => {
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
      overlay.setPointerCapture(event.pointerId);
    } else {
      points.push([roundCoordinate(point.x), roundCoordinate(point.y)]);
      points.sort((a, b) => a[0] - b[0]);
      const index = points.findIndex((candidate) => candidate[0] === roundCoordinate(point.x) && candidate[1] === roundCoordinate(point.y));
      state.dragging = {mode: state.mode, index};
      overlay.setPointerCapture(event.pointerId);
    }
  }
  markDirty();
  syncControls();
  draw();
});

overlay.addEventListener("pointermove", (event) => {
  if (!state.dragging || !state.annotation) return;
  const point = canvasPoint(event);
  const points = state.annotation.baselines[state.dragging.mode];
  points[state.dragging.index] = [roundCoordinate(point.x), roundCoordinate(point.y)];
  points.sort((a, b) => a[0] - b[0]);
  state.dragging.index = points.findIndex((candidate) => candidate[0] === roundCoordinate(point.x) && candidate[1] === roundCoordinate(point.y));
  markDirty();
  draw();
});

overlay.addEventListener("pointerup", () => {
  state.dragging = null;
});

overlay.addEventListener("contextmenu", (event) => {
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
  state.zoom = Number(zoomInput.value);
  const width = canvas.width * state.zoom;
  const height = canvas.height * state.zoom;
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  overlay.style.width = `${width}px`;
  overlay.style.height = `${height}px`;
  canvasWrap.style.width = `${width}px`;
  canvasWrap.style.height = `${height}px`;
  document.getElementById("zoomValue").value = `${state.zoom}×`;
}

function zoomWithWheel(event) {
  if (!state.image) return;
  event.preventDefault();
  const oldZoom = state.zoom;
  const direction = event.deltaY < 0 ? 1 : -1;
  const nextZoom = Math.min(
    Number(zoomInput.max),
    Math.max(Number(zoomInput.min), oldZoom + direction * Number(zoomInput.step)),
  );
  if (nextZoom === oldZoom) return;

  const oldRect = overlay.getBoundingClientRect();
  const sourceX = (event.clientX - oldRect.left) / oldZoom;
  const sourceY = (event.clientY - oldRect.top) / oldZoom;
  zoomInput.value = String(nextZoom);
  applyZoom();
  drawOverlay();

  const newRect = overlay.getBoundingClientRect();
  canvasScroller.scrollLeft += newRect.left + sourceX * nextZoom - event.clientX;
  canvasScroller.scrollTop += newRect.top + sourceY * nextZoom - event.clientY;
}

function syncControls() {
  if (!state.annotation) return;
  document.getElementById("counter").textContent = `${state.index + 1} / ${state.images.length}`;
  document.getElementById("imageName").textContent = state.images[state.index].path;
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
    option.textContent = `${item.marked ? "✓" : "·"} ${item.path}`;
    imageList.append(option);
  });
  imageList.selectedIndex = state.index;
}

function updateProgress() {
  const marked = state.images.filter((item) => item.marked).length;
  document.getElementById("progressText").textContent = `${marked} / ${state.images.length}`;
  document.getElementById("progressBar").style.width = `${state.images.length ? marked * 100 / state.images.length : 0}%`;
}

document.querySelectorAll(".mode").forEach((button) => {
  button.addEventListener("click", () => setMode(button.dataset.mode));
});
document.getElementById("previous").addEventListener("click", () => openImage(state.index - 1));
document.getElementById("next").addEventListener("click", () => openImage(state.index + 1));
document.getElementById("undo").addEventListener("click", undo);
document.getElementById("clearLayer").addEventListener("click", clearLayer);
document.getElementById("save").addEventListener("click", saveNow);
zoomInput.addEventListener("input", () => {
  applyZoom();
  drawOverlay();
});
overlay.addEventListener("wheel", zoomWithWheel, {passive: false});
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
