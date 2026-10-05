"use strict";

const loadButton = document.querySelector("#load-model");
const captureButton = document.querySelector("#capture");
const fileInput = document.querySelector("#image-file");
const statusElement = document.querySelector("#status");
const errorElement = document.querySelector("#error");
const resultPanel = document.querySelector("#result-panel");
const preview = document.querySelector("#preview");
const summary = document.querySelector("#summary");
const timing = document.querySelector("#timing");
const jobElement = document.querySelector("#job");

let pipelineWorker = null;
let workerReady = false;
let pipelineTimings = null;
let initializePromise = null;
let initializeResolve = null;
let initializeReject = null;
let initializeTimeoutId = null;
let processResolve = null;
let processReject = null;
let isBusy = false;
const INITIALIZE_TIMEOUT_MS = 90_000;

function setStatus(message) {
  statusElement.textContent = message;
}

function showError(error) {
  errorElement.textContent = error instanceof Error ? error.message : String(error);
  errorElement.hidden = false;
}

function clearError() {
  errorElement.hidden = true;
  errorElement.textContent = "";
}

function setBusy(busy) {
  isBusy = busy;
  captureButton.disabled = busy || !workerReady;
  loadButton.disabled = busy || workerReady;
}

function createPipelineWorker() {
  if (pipelineWorker) {
    return pipelineWorker;
  }
  if (typeof Worker !== "function" || typeof OffscreenCanvas !== "function") {
    throw new Error("Trình duyệt không hỗ trợ xử lý ảnh nền (Web Worker/OffscreenCanvas). Hãy cập nhật trình duyệt rồi tải lại trang.");
  }

  pipelineWorker = new Worker("/static/pipeline-worker.js?v=phone-local-11");
  pipelineWorker.addEventListener("message", (event) => {
    const message = event.data;
    if (message.type === "status") {
      setStatus(message.message);
    } else if (message.type === "ready") {
      workerReady = true;
      pipelineTimings = message;
      clearTimeout(initializeTimeoutId);
      initializeTimeoutId = null;
      initializeResolve?.();
      initializeResolve = null;
      initializeReject = null;
    } else if (message.type === "result") {
      processResolve?.(message);
      processResolve = null;
      processReject = null;
    } else if (message.type === "error") {
      const error = new Error(message.message);
      if (initializeReject) {
        clearTimeout(initializeTimeoutId);
        initializeTimeoutId = null;
        initializeReject(error);
        initializeResolve = null;
        initializeReject = null;
        initializePromise = null;
      } else {
        processReject?.(error);
        processResolve = null;
        processReject = null;
      }
    }
  });
  pipelineWorker.addEventListener("error", (event) => {
    const error = new Error(event.message || "Worker xử lý ảnh bị lỗi.");
    clearTimeout(initializeTimeoutId);
    initializeTimeoutId = null;
    initializeReject?.(error);
    processReject?.(error);
    initializeResolve = null;
    initializeReject = null;
    processResolve = null;
    processReject = null;
    initializePromise = null;
    workerReady = false;
    pipelineWorker?.terminate();
    pipelineWorker = null;
  });
  return pipelineWorker;
}

function initializePipeline() {
  if (workerReady) {
    return Promise.resolve();
  }
  if (initializePromise) {
    return initializePromise;
  }
  const worker = createPipelineWorker();
  initializePromise = new Promise((resolve, reject) => {
    initializeResolve = resolve;
    initializeReject = reject;
  });
  initializeTimeoutId = setTimeout(() => {
    const error = new Error(
      "Worker không hoàn tất khởi tạo trong 90 giây và đã được dừng. "
      + "Kiểm tra khả năng tương thích WebAssembly của trình duyệt trên thiết bị.",
    );
    initializeReject?.(error);
    initializeResolve = null;
    initializeReject = null;
    initializePromise = null;
    workerReady = false;
    pipelineWorker?.terminate();
    pipelineWorker = null;
    initializeTimeoutId = null;
  }, INITIALIZE_TIMEOUT_MS);
  worker.postMessage({ type: "init" });
  return initializePromise;
}

async function loadModel() {
  clearError();
  setBusy(true);
  setStatus("Đang khởi động worker xử lý nền trên điện thoại…");
  try {
    await initializePipeline();
    setStatus(
      `OpenCV và model đã sẵn sàng. JS ${pipelineTimings.scriptLoadMs.toFixed(0)} ms · `
      + `Wasm ${pipelineTimings.wasmInitMs.toFixed(0)} ms. Chụp ảnh để bắt đầu.`,
    );
  } catch (error) {
    setStatus("Không khởi tạo được pipeline.");
    showError(error);
    pipelineWorker?.terminate();
    pipelineWorker = null;
    initializePromise = null;
    workerReady = false;
  } finally {
    setBusy(false);
  }
}

function drawDetections(image, detections) {
  preview.width = image.width;
  preview.height = image.height;
  const context = preview.getContext("2d");
  if (!context) {
    throw new Error("Trình duyệt không hiển thị được ảnh kết quả.");
  }
  context.drawImage(image, 0, 0);
  const lineWidth = Math.max(2, Math.round(Math.min(image.width, image.height) / 300));
  context.lineWidth = lineWidth;
  context.font = `bold ${Math.max(16, lineWidth * 8)}px sans-serif`;
  for (const detection of detections) {
    const width = detection.x2 - detection.x1;
    const height = detection.y2 - detection.y1;
    const label = `${detection.label} ${(detection.confidence * 100).toFixed(1)}%`;
    context.strokeStyle = "#19d3a2";
    context.fillStyle = "#19d3a2";
    context.strokeRect(detection.x1, detection.y1, width, height);
    const textWidth = context.measureText(label).width;
    const labelHeight = Math.max(24, lineWidth * 10);
    const labelY = Math.max(labelHeight, detection.y1);
    context.fillRect(detection.x1, labelY - labelHeight, textWidth + 12, labelHeight);
    context.fillStyle = "#10231f";
    context.fillText(label, detection.x1 + 6, labelY - lineWidth * 2);
  }
}

async function runPipeline(file) {
  if (isBusy || !workerReady || !pipelineWorker) {
    return;
  }
  clearError();
  setBusy(true);
  resultPanel.hidden = true;
  const started = performance.now();
  let alignedImage = null;
  try {
    setStatus("Đang xử lý ảnh bằng OpenCV.js và YOLO trên điện thoại…");
    const result = await new Promise((resolve, reject) => {
      processResolve = resolve;
      processReject = reject;
      pipelineWorker.postMessage({ type: "process", file });
    });
    alignedImage = result.image;
    const { detections, inferenceMs, alignmentMs } = result;
    drawDetections(alignedImage, detections);

    summary.textContent = detections.length
      ? `Phát hiện ${detections.length} qbox`
      : "Không phát hiện qbox trên ngưỡng tin cậy 80%.";
    timing.textContent = `OpenCV ${alignmentMs.toFixed(0)} ms · suy luận ${inferenceMs.toFixed(0)} ms · toàn bộ ${(performance.now() - started).toFixed(0)} ms`;
    jobElement.textContent = "Ảnh được xử lý trực tiếp trên điện thoại";
    resultPanel.hidden = false;

    setStatus("Hoàn tất — OpenCV và YOLO đều chạy trên điện thoại. Chụp ảnh khác để thử tiếp.");
  } catch (error) {
    setStatus("Pipeline chưa chạy xong.");
    showError(error);
  } finally {
    alignedImage?.close();
    setBusy(false);
  }
}

loadButton.addEventListener("click", loadModel);
captureButton.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  const [file] = fileInput.files || [];
  if (file) {
    runPipeline(file);
  }
  fileInput.value = "";
});
