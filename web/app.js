/**
 * ChamAI Web Client Application
 * Minimalist, light-themed, scroll-free mobile document scanner interface.
 */

import { loadConfig, saveConfig, getModelUrl } from "./config.js";

// DOM Elements
const elements = {
  // Header
  statusBadge: document.getElementById("status-badge"),
  statusText: document.getElementById("status-text"),
  btnSettings: document.getElementById("btn-settings"),

  // Stages
  idleStage: document.getElementById("idle-stage"),
  loadingStage: document.getElementById("loading-stage"),
  resultStage: document.getElementById("result-stage"),

  // Progress
  progressBar: document.getElementById("progress-bar"),
  progressText: document.getElementById("progress-text"),

  // Result Viewer
  canvasViewer: document.getElementById("canvas-viewer"),
  resultSummary: document.getElementById("result-summary"),
  btnZoomFit: document.getElementById("btn-zoom-fit"),
  btnDownload: document.getElementById("btn-download"),
  btnRetake: document.getElementById("btn-retake"),

  // Bottom Controls
  btnLoadModel: document.getElementById("btn-load-model"),
  btnCapture: document.getElementById("btn-capture"),
  btnUpload: document.getElementById("btn-upload"),
  cameraInput: document.getElementById("camera-input"),
  fileInput: document.getElementById("file-input"),

  // Settings Modal
  settingsModal: document.getElementById("settings-modal"),
  btnCloseSettings: document.getElementById("btn-close-settings"),
  btnSaveSettings: document.getElementById("btn-save-settings"),
  providerSelect: document.getElementById("provider-select"),
  r2UrlInput: document.getElementById("r2-url-input"),
  hfUrlInput: document.getElementById("hf-url-input"),
  customUrlInput: document.getElementById("custom-url-input"),
  dewarpToggle: document.getElementById("dewarp-toggle"),
  r2Group: document.getElementById("r2-group"),
  hfGroup: document.getElementById("hf-group"),
  customGroup: document.getElementById("custom-group")
};

// State
let config = loadConfig();
let worker = null;
let isModelReady = false;
let isProcessing = false;
let latestResult = null;

// Viewer Pan / Zoom Transform State
const viewerState = {
  scale: 1,
  minScale: 1,
  maxScale: 6,
  offsetX: 0,
  offsetY: 0,
  isDragging: false,
  startX: 0,
  startY: 0,
  initialDistance: 0,
  initialScale: 1
};

/**
 * Initializes the Web Worker and sets up event communication
 */
function initWorker() {
  if (worker) worker.terminate();

  worker = new Worker("./pipeline-worker.js");

  worker.onmessage = (e) => {
    const data = e.data;
    if (data.type === "status") {
      updateStatus(data.step, data.message, data.progress);
    } else if (data.type === "initialized") {
      isModelReady = true;
      setStatusPill("ready", "Sẵn sàng");
      elements.btnLoadModel.hidden = true;
      elements.btnCapture.disabled = false;
      elements.btnUpload.disabled = false;
      showStage("idle");
    } else if (data.type === "complete") {
      isProcessing = false;
      latestResult = data;
      renderResult(data.imageBitmap, data.summary);
      showStage("result");
    } else if (data.type === "error") {
      isProcessing = false;
      alert(data.message);
      showStage("idle");
      setStatusPill("error", "Lỗi nạp/xử lý");
    }
  };
}

function setStatusPill(state, text) {
  elements.statusBadge.className = `status-pill ${state}`;
  elements.statusText.textContent = text;
}

function updateStatus(step, message, progress) {
  elements.progressText.textContent = message;
  if (progress !== null && progress !== undefined) {
    elements.progressBar.style.width = `${Math.min(100, Math.max(0, progress))}%`;
  }
}

function showStage(stageName) {
  elements.idleStage.hidden = stageName !== "idle";
  elements.loadingStage.hidden = stageName !== "loading";
  elements.resultStage.hidden = stageName !== "result";
}

/**
 * Triggers loading model from CDN
 */
function loadModels() {
  initWorker();
  showStage("loading");
  setStatusPill("loading", "Đang nạp CDN...");
  elements.btnLoadModel.disabled = true;
  worker.postMessage({ type: "init", config });
}

/**
 * Handles image selection and sends to Worker for inference
 */
async function handleImageFile(file) {
  if (!file) return;

  if (!isModelReady) {
    const confirmLoad = confirm("Mô hình chưa được nạp. Bạn có muốn nạp mô hình từ CDN bây giờ không?");
    if (confirmLoad) {
      loadModels();
    }
    return;
  }

  try {
    isProcessing = true;
    showStage("loading");
    updateStatus("prep", "Đang mở ảnh...", 5);

    const imageBitmap = await createImageBitmap(file);
    worker.postMessage({
      type: "process",
      imageBitmap,
      config
    }, [imageBitmap]);
  } catch (err) {
    isProcessing = false;
    alert(`Không thể đọc file ảnh: ${err.message}`);
    showStage("idle");
  }
}

/**
 * Renders the resulting ImageBitmap onto the interactive Canvas Viewer
 */
function renderResult(imageBitmap, summary) {
  const canvas = elements.canvasViewer;
  canvas.width = imageBitmap.width;
  canvas.height = imageBitmap.height;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(imageBitmap, 0, 0);

  // Update summary pill
  elements.resultSummary.textContent = `${summary.questionsCount} câu hỏi • ${summary.choicesCount} đáp án • ${summary.durationSec}s`;

  // Reset viewport zoom/pan
  resetViewer();
}

/**
 * Interactive Pinch-to-Zoom and Pan for Canvas Viewer
 */
function resetViewer() {
  viewerState.scale = 1;
  viewerState.offsetX = 0;
  viewerState.offsetY = 0;
  applyViewerTransform();
}

function applyViewerTransform() {
  const canvas = elements.canvasViewer;
  canvas.style.transform = `translate(${viewerState.offsetX}px, ${viewerState.offsetY}px) scale(${viewerState.scale})`;
}

function setupViewerGestures() {
  const stage = elements.resultStage;

  // Touch handlers
  stage.addEventListener("touchstart", (e) => {
    if (e.touches.length === 1) {
      viewerState.isDragging = true;
      viewerState.startX = e.touches[0].clientX - viewerState.offsetX;
      viewerState.startY = e.touches[0].clientY - viewerState.offsetY;
    } else if (e.touches.length === 2) {
      viewerState.isDragging = false;
      const dx = e.touches[0].clientX - e.touches[1].clientX;
      const dy = e.touches[0].clientY - e.touches[1].clientY;
      viewerState.initialDistance = Math.hypot(dx, dy);
      viewerState.initialScale = viewerState.scale;
    }
  }, { passive: false });

  stage.addEventListener("touchmove", (e) => {
    e.preventDefault();
    if (e.touches.length === 1 && viewerState.isDragging) {
      viewerState.offsetX = e.touches[0].clientX - viewerState.startX;
      viewerState.offsetY = e.touches[0].clientY - viewerState.startY;
      applyViewerTransform();
    } else if (e.touches.length === 2 && viewerState.initialDistance > 0) {
      const dx = e.touches[0].clientX - e.touches[1].clientX;
      const dy = e.touches[0].clientY - e.touches[1].clientY;
      const currentDistance = Math.hypot(dx, dy);
      const factor = currentDistance / viewerState.initialDistance;
      viewerState.scale = Math.min(viewerState.maxScale, Math.max(viewerState.minScale, viewerState.initialScale * factor));
      applyViewerTransform();
    }
  }, { passive: false });

  stage.addEventListener("touchend", (e) => {
    if (e.touches.length < 2) viewerState.initialDistance = 0;
    if (e.touches.length === 0) viewerState.isDragging = false;
  });

  // Mouse wheel zoom
  stage.addEventListener("wheel", (e) => {
    e.preventDefault();
    const zoomDelta = e.deltaY * -0.002;
    viewerState.scale = Math.min(viewerState.maxScale, Math.max(viewerState.minScale, viewerState.scale + zoomDelta));
    applyViewerTransform();
  }, { passive: false });
}

/**
 * Settings Modal Logic
 */
function openSettingsModal() {
  elements.providerSelect.value = config.provider;
  elements.r2UrlInput.value = config.r2BaseUrl;
  elements.hfUrlInput.value = config.hfBaseUrl;
  elements.customUrlInput.value = config.customBaseUrl || "";
  elements.dewarpToggle.checked = config.enableDewarp;
  updateProviderVisibility();
  elements.settingsModal.hidden = false;
}

function closeSettingsModal() {
  elements.settingsModal.hidden = true;
}

function updateProviderVisibility() {
  const p = elements.providerSelect.value;
  elements.r2Group.hidden = p !== "cloudflare_r2";
  elements.hfGroup.hidden = p !== "huggingface";
  elements.customGroup.hidden = p !== "custom";
}

function saveSettings() {
  config.provider = elements.providerSelect.value;
  config.r2BaseUrl = elements.r2UrlInput.value.trim();
  config.hfBaseUrl = elements.hfUrlInput.value.trim();
  config.customBaseUrl = elements.customUrlInput.value.trim();
  config.enableDewarp = elements.dewarpToggle.checked;

  saveConfig(config);
  closeSettingsModal();
  alert("Đã lưu cài đặt CDN. Nhấn 'Nạp model' để cập nhật.");

  // Reset model readiness
  isModelReady = false;
  elements.btnLoadModel.hidden = false;
  elements.btnLoadModel.disabled = false;
  setStatusPill("idle", "Chưa nạp");
}

/**
 * Event Listeners Registration
 */
function initEventListeners() {
  // Header Actions
  elements.btnSettings.addEventListener("click", openSettingsModal);
  elements.btnCloseSettings.addEventListener("click", closeSettingsModal);
  elements.btnSaveSettings.addEventListener("click", saveSettings);
  elements.providerSelect.addEventListener("change", updateProviderVisibility);

  // Main Actions
  elements.btnLoadModel.addEventListener("click", loadModels);
  elements.btnCapture.addEventListener("click", () => elements.cameraInput.click());
  elements.btnUpload.addEventListener("click", () => elements.fileInput.click());

  elements.cameraInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleImageFile(e.target.files[0]);
      e.target.value = "";
    }
  });

  elements.fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleImageFile(e.target.files[0]);
      e.target.value = "";
    }
  });

  // Result Viewer Actions
  elements.btnZoomFit.addEventListener("click", resetViewer);
  elements.btnRetake.addEventListener("click", () => showStage("idle"));
  elements.btnDownload.addEventListener("click", () => {
    const canvas = elements.canvasViewer;
    const a = document.createElement("a");
    a.download = `chamai_result_${Date.now()}.png`;
    a.href = canvas.toDataURL("image/png");
    a.click();
  });

  setupViewerGestures();
}

// Bootstrap
document.addEventListener("DOMContentLoaded", () => {
  initEventListeners();
  setStatusPill("idle", "Chưa nạp model");
});
