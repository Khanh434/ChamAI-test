"use strict";

const MODEL_INPUT_SIZE = 1024;
const CONFIDENCE_THRESHOLD = 0.8;
const IOU_THRESHOLD = 0.45;
const CLASS_NAMES = ["qbox"];
const OPENCV_WASM_URL = "/opencv/opencv.wasm?v=split-wasm-4";
const MODEL_URL = "https://huggingface.co/Khanh434/YOLOv8n-doclaynet-fine-tuned-chamai/resolve/24683b7e898e3b2b03178272e03273efad9bc955/yolov8n-doclaynet-fine-tuned-chamai.onnx";
const ONNX_RUNTIME_CDN = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
const OPENCV_WASM_DOWNLOAD_TIMEOUT_MS = 30_000;
const OPENCV_WASM_COMPILE_TIMEOUT_MS = 60_000;
const OPENCV_INIT_TIMEOUT_MS = 30_000;

let session = null;
let isProcessing = false;

function reportStatus(message) {
  self.postMessage({ type: "status", message });
}

async function readModel(response) {
  const totalBytes = Number(response.headers.get("Content-Length")) || 0;
  if (!response.body) {
    return response.arrayBuffer();
  }

  const reader = response.body.getReader();
  const chunks = [];
  let loadedBytes = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    chunks.push(value);
    loadedBytes += value.byteLength;
    if (totalBytes > 0) {
      reportStatus(`Đang tải model về điện thoại: ${Math.round(loadedBytes * 100 / totalBytes)}%`);
    } else {
      reportStatus(`Đã tải ${(loadedBytes / 1024 / 1024).toFixed(1)} MB model…`);
    }
  }

  const bytes = new Uint8Array(loadedBytes);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes.buffer;
}

async function loadOpenCvWasm() {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), OPENCV_WASM_DOWNLOAD_TIMEOUT_MS);
  try {
    const response = await fetch(OPENCV_WASM_URL, {
      cache: "force-cache",
      signal: controller.signal,
    });
    if (!response.ok) {
      throw new Error(`Không tải được OpenCV WebAssembly (HTTP ${response.status}).`);
    }
    const contentType = response.headers.get("content-type") || "";
    if (!contentType.toLowerCase().includes("application/wasm")) {
      throw new Error(`OpenCV WebAssembly trả Content-Type không hợp lệ: ${contentType || "không có"}.`);
    }
    if (!response.body) {
      throw new Error("Trình duyệt không hỗ trợ đọc luồng WebAssembly OpenCV.");
    }

    const reader = response.body.getReader();
    const chunks = [];
    let loadedBytes = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      chunks.push(value);
      loadedBytes += value.byteLength;
      reportStatus(`Đang tải WebAssembly OpenCV: ${(loadedBytes / 1024 / 1024).toFixed(1)} MB`);
    }

    const wasmBinary = new Uint8Array(loadedBytes);
    let offset = 0;
    for (const chunk of chunks) {
      wasmBinary.set(chunk, offset);
      offset += chunk.byteLength;
    }
    if (
      wasmBinary[0] !== 0
      || wasmBinary[1] !== 0x61
      || wasmBinary[2] !== 0x73
      || wasmBinary[3] !== 0x6d
    ) {
      throw new Error("OpenCV WebAssembly tải về không đúng định dạng Wasm.");
    }
    if (!WebAssembly.validate(wasmBinary)) {
      throw new Error("Trình duyệt không xác thực được file OpenCV WebAssembly.");
    }
    return wasmBinary;
  } catch (error) {
    if (controller.signal.aborted) {
      throw new Error("Tải OpenCV WebAssembly quá 30 giây và đã bị dừng.");
    }
    throw error;
  } finally {
    clearTimeout(timeoutId);
  }
}

async function compileOpenCvWasm(wasmBinary) {
  reportStatus("Đã tải WebAssembly OpenCV (7.6 MB); đang biên dịch bất đồng bộ…");
  const compileStarted = performance.now();
  const progressTimer = setInterval(() => {
    reportStatus(
      `Đang biên dịch WebAssembly OpenCV… ${Math.floor((performance.now() - compileStarted) / 1000)} giây`,
    );
  }, 1000);
  let timeoutId;
  try {
    const wasmModule = await Promise.race([
      WebAssembly.compile(wasmBinary),
      new Promise((_, reject) => {
        timeoutId = setTimeout(() => {
          reject(new Error(
            "Biên dịch WebAssembly OpenCV quá 60 giây. Trình duyệt trên thiết bị có thể không hỗ trợ binary này.",
          ));
        }, OPENCV_WASM_COMPILE_TIMEOUT_MS);
      }),
    ]);
    return {
      wasmModule,
      compileMs: performance.now() - compileStarted,
    };
  } finally {
    clearInterval(progressTimer);
    clearTimeout(timeoutId);
  }
}

async function initialize() {
  if (session) {
    self.postMessage({ type: "ready" });
    return;
  }

  reportStatus("Đang tải mã OpenCV.js và ONNX Runtime trong worker nền…");
  const wasmBinary = await loadOpenCvWasm();
  const wasmSize = (wasmBinary.byteLength / 1024 / 1024).toFixed(1);
  const { wasmModule, compileMs } = await compileOpenCvWasm(wasmBinary);

  let runtimeReadyResolve;
  let runtimeReadyReject;
  const runtimeReady = new Promise((resolve, reject) => {
    runtimeReadyResolve = resolve;
    runtimeReadyReject = reject;
  });
  let runtimeReadyModule = null;
  self.cv = {
    instantiateWasm(imports, receiveInstance) {
      WebAssembly.instantiate(wasmModule, imports).then(
        (instance) => receiveInstance(instance, wasmModule),
        (error) => {
          self.cv.onAbort?.(`Could not instantiate precompiled WebAssembly: ${error}`);
          runtimeReadyReject(error);
        },
      );
      return {};
    },
    onRuntimeInitialized: () => {
      runtimeReadyModule = self.cv;
      runtimeReadyResolve(true);
    },
    onAbort: (reason) => runtimeReadyReject(new Error(`OpenCV.js dừng khi nạp WebAssembly: ${reason}`)),
  };

  const scriptLoadStarted = performance.now();
  importScripts(
    "/opencv/opencv.js?v=split-wasm-4",
    `${ONNX_RUNTIME_CDN}ort.min.js`,
    "/static/local-align.js?v=phone-local-10",
  );
  const scriptLoadMs = performance.now() - scriptLoadStarted;

  reportStatus(
    `Wasm ${wasmSize} MB đã biên dịch (${compileMs.toFixed(0)} ms); đang khởi tạo OpenCV.js…`,
  );

  const initStarted = performance.now();
  const progressTimer = setInterval(() => {
    const elapsedSeconds = Math.floor((performance.now() - initStarted) / 1000);
    reportStatus(`Đang khởi tạo OpenCV.js: ${elapsedSeconds} giây · Wasm ${wasmSize} MB đã tải`);
  }, 1000);
  let timeoutId;
  try {
    await Promise.race([
      runtimeReady,
      new Promise((_, reject) => {
        timeoutId = setTimeout(() => {
          reject(new Error(
            `Đã tải Wasm (${wasmSize} MB) nhưng OpenCV.js chưa khởi tạo xong sau 30 giây. `
            + "Có thể trình duyệt trên thiết bị không hỗ trợ build WebAssembly này hoặc thiếu bộ nhớ.",
          ));
        }, OPENCV_INIT_TIMEOUT_MS);
      }),
    ]);
  } finally {
    clearInterval(progressTimer);
    clearTimeout(timeoutId);
  }
  const wasmInitMs = performance.now() - initStarted;
  const cv = runtimeReadyModule;
  if (!cv?.Mat) {
    throw new Error("OpenCV.js chưa khởi tạo được. Hãy tải lại trang và thử lại.");
  }
  reportStatus(
    `OpenCV.js đã sẵn sàng (biên dịch ${compileMs.toFixed(0)} ms, khởi tạo ${wasmInitMs.toFixed(0)} ms).`,
  );

  if (!self.ort) {
    throw new Error("Không tải được ONNX Runtime Web trong worker.");
  }
  self.ort.env.wasm.wasmPaths = ONNX_RUNTIME_CDN;
  self.ort.env.wasm.numThreads = 1;

  reportStatus("Đang tải model ONNX về điện thoại…");
  const response = await fetch(MODEL_URL, { cache: "force-cache" });
  if (!response.ok) {
    throw new Error(`Không tải được model từ Hugging Face (HTTP ${response.status}).`);
  }
  const model = await readModel(response);

  reportStatus("Đang khởi tạo YOLO trên điện thoại…");
  session = await self.ort.InferenceSession.create(model, { executionProviders: ["wasm"] });
  self.postMessage({ type: "ready", scriptLoadMs, wasmInitMs });
}

function makeInputTensor(image) {
  const size = MODEL_INPUT_SIZE;
  const scale = Math.min(size / image.width, size / image.height);
  const resizedWidth = Math.round(image.width * scale);
  const resizedHeight = Math.round(image.height * scale);
  const left = Math.round((size - resizedWidth) / 2 - 0.1);
  const top = Math.round((size - resizedHeight) / 2 - 0.1);

  const canvas = new OffscreenCanvas(size, size);
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) {
    throw new Error("Không tạo được canvas để chuẩn bị ảnh cho model.");
  }
  context.fillStyle = "rgb(114, 114, 114)";
  context.fillRect(0, 0, size, size);
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(image, left, top, resizedWidth, resizedHeight);

  const pixels = context.getImageData(0, 0, size, size).data;
  const planeSize = size * size;
  const input = new Float32Array(3 * planeSize);
  for (let pixel = 0; pixel < planeSize; pixel += 1) {
    const source = pixel * 4;
    input[pixel] = pixels[source] / 255;
    input[planeSize + pixel] = pixels[source + 1] / 255;
    input[2 * planeSize + pixel] = pixels[source + 2] / 255;
  }
  return {
    tensor: new self.ort.Tensor("float32", input, [1, 3, size, size]),
    scale,
    left,
    top,
  };
}

function intersectionOverUnion(a, b) {
  const width = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const height = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const intersection = width * height;
  const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
  const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
  return intersection / Math.max(areaA + areaB - intersection, 1e-8);
}

function decodeDetections(output, transform, imageWidth, imageHeight) {
  if (!output || output.dims.length !== 3 || output.dims[1] !== 5) {
    throw new Error("Đầu ra YOLO không đúng định dạng; model qbox cần shape [1, 5, N].");
  }
  const values = output.data;
  const count = output.dims[2];
  const candidates = [];
  for (let index = 0; index < count; index += 1) {
    const confidence = values[4 * count + index];
    if (confidence < CONFIDENCE_THRESHOLD) {
      continue;
    }

    const centerX = values[index];
    const centerY = values[count + index];
    const width = values[2 * count + index];
    const height = values[3 * count + index];
    const x1 = Math.max(0, Math.min(imageWidth, (centerX - width / 2 - transform.left) / transform.scale));
    const y1 = Math.max(0, Math.min(imageHeight, (centerY - height / 2 - transform.top) / transform.scale));
    const x2 = Math.max(0, Math.min(imageWidth, (centerX + width / 2 - transform.left) / transform.scale));
    const y2 = Math.max(0, Math.min(imageHeight, (centerY + height / 2 - transform.top) / transform.scale));
    if (x2 <= x1 || y2 <= y1) {
      continue;
    }
    candidates.push({
      class_id: 0,
      label: CLASS_NAMES[0],
      confidence,
      x1,
      y1,
      x2,
      y2,
    });
  }

  candidates.sort((a, b) => b.confidence - a.confidence);
  const selected = [];
  for (const candidate of candidates) {
    if (selected.every((kept) => intersectionOverUnion(candidate, kept) <= IOU_THRESHOLD)) {
      selected.push(candidate);
      if (selected.length === 300) {
        break;
      }
    }
  }
  return selected;
}

async function processImage(file) {
  if (!session) {
    throw new Error("Model chưa sẵn sàng. Hãy nạp model trước.");
  }
  if (isProcessing) {
    throw new Error("Đang xử lý ảnh khác; vui lòng đợi.");
  }
  isProcessing = true;
  let alignedImage = null;
  try {
    reportStatus("Đang căn chỉnh ảnh bằng OpenCV.js trên điện thoại…");
    const alignment = await self.alignImageLocally(file);
    alignedImage = alignment.image;

    reportStatus("Đang chạy YOLO ONNX trên điện thoại…");
    const { tensor, ...transform } = makeInputTensor(alignedImage);
    const inferenceStarted = performance.now();
    const outputs = await session.run({ [session.inputNames[0]]: tensor });
    const inferenceMs = performance.now() - inferenceStarted;
    const output = outputs[session.outputNames[0]];
    const detections = decodeDetections(output, transform, alignedImage.width, alignedImage.height);
    self.postMessage({
      type: "result",
      image: alignedImage,
      detections,
      inferenceMs,
      alignmentMs: alignment.processingMs,
    }, [alignedImage]);
    alignedImage = null;
  } finally {
    alignedImage?.close();
    isProcessing = false;
  }
}

self.addEventListener("message", async (event) => {
  const { type, file } = event.data;
  try {
    if (type === "init") {
      await initialize();
    } else if (type === "process") {
      await processImage(file);
    }
  } catch (error) {
    self.postMessage({
      type: "error",
      message: error instanceof Error ? error.message : String(error),
    });
  }
});
