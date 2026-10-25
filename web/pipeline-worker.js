/**
 * ChamAI Document Processing Pipeline Worker
 * Runs L1 Dewarp -> L2 QBox -> L3 CBox -> L4 AnswerChoiceNet completely client-side via ONNX Runtime Web.
 */

/* global importScripts, ort */

const REV_CHOICE_MAP = { 0: "A", 1: "B", 2: "C", 3: "D", 4: "UNKNOWN" };
const REV_STATE_MAP = { 0: "not_selected", 1: "selected", 2: "crossed_out" };

let ortInitialized = false;
let sessions = {
  l1: null,
  l2: null,
  l3: null,
  l4: null
};

function postStatus(step, message, progress = null) {
  self.postMessage({ type: "status", step, message, progress });
}

function postError(message) {
  self.postMessage({ type: "error", message });
}

/**
 * Downloads a model file from URL with chunked progress reporting
 */
async function fetchModelBuffer(url, label) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Không thể tải ${label} từ CDN (HTTP ${response.status}): ${url}`);
  }
  const contentLength = Number(response.headers.get("content-length")) || 0;
  if (!response.body) {
    return await response.arrayBuffer();
  }

  const reader = response.body.getReader();
  const chunks = [];
  let receivedBytes = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    receivedBytes += value.byteLength;
    if (contentLength > 0) {
      const pct = Math.round((receivedBytes / contentLength) * 100);
      postStatus("loading", `Đang tải ${label}: ${pct}%`, pct);
    } else {
      postStatus("loading", `Đang tải ${label}: ${(receivedBytes / (1024 * 1024)).toFixed(1)} MB`);
    }
  }

  const allChunks = new Uint8Array(receivedBytes);
  let position = 0;
  for (const chunk of chunks) {
    allChunks.set(chunk, position);
    position += chunk.byteLength;
  }
  return allChunks.buffer;
}

/**
 * Initializes ONNX Runtime Web and loads all model sessions
 */
async function initPipeline(config) {
  try {
    if (!ortInitialized) {
      postStatus("init", "Khởi tạo ONNX Runtime Web...");
      importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.min.js");
      ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);
      ortInitialized = true;
    }

    const sessionOptions = {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all"
    };

    // Helper to resolve URL
    function resolveUrl(filename) {
      let base = config.provider === "cloudflare_r2" ? config.r2BaseUrl :
                 config.provider === "custom" ? config.customBaseUrl : config.hfBaseUrl;
      base = base.replace(/\/+$/, "");
      return `${base}/${filename}`;
    }

    // 1. L1 UVDocNet (Optional)
    if (config.enableDewarp) {
      const l1Url = resolveUrl(config.models.l1_dewarp || "uvdocnet.onnx");
      const l1Buf = await fetchModelBuffer(l1Url, "L1 Dewarp");
      sessions.l1 = await ort.InferenceSession.create(l1Buf, sessionOptions);
    }

    // 2. L2 YOLO QBox
    const l2Url = resolveUrl(config.models.l2_qbox || "yolo-qbox.onnx");
    const l2Buf = await fetchModelBuffer(l2Url, "L2 YOLO QBox");
    sessions.l2 = await ort.InferenceSession.create(l2Buf, sessionOptions);

    // 3. L3 YOLO CBox
    const l3Url = resolveUrl(config.models.l3_cbox || "yolo-cbox.onnx");
    const l3Buf = await fetchModelBuffer(l3Url, "L3 YOLO CBox");
    sessions.l3 = await ort.InferenceSession.create(l3Buf, sessionOptions);

    // 4. L4 AnswerChoiceNet
    const l4Url = resolveUrl(config.models.l4_acnet || "acnet.onnx");
    const l4Buf = await fetchModelBuffer(l4Url, "L4 AnswerChoiceNet");
    sessions.l4 = await ort.InferenceSession.create(l4Buf, sessionOptions);

    postStatus("ready", "Tất cả mô hình đã sẵn sàng!", 100);
    self.postMessage({ type: "initialized" });
  } catch (err) {
    postError(`Lỗi nạp mô hình: ${err.message}`);
  }
}

/**
 * Letterbox resize and normalize to Float32Array RGB [1, 3, targetSize, targetSize]
 */
function createYoloTensor(canvas, targetSize = 1024) {
  const offscreen = new OffscreenCanvas(targetSize, targetSize);
  const ctx = offscreen.getContext("2d", { willReadFrequently: true });

  const srcW = canvas.width;
  const srcH = canvas.height;
  const scale = Math.min(targetSize / srcW, targetSize / srcH);
  const newW = Math.round(srcW * scale);
  const newH = Math.round(srcH * scale);
  const padX = Math.floor((targetSize - newW) / 2);
  const padY = Math.floor((targetSize - newH) / 2);

  // Fill padding with neutral gray 114
  ctx.fillStyle = "#727272";
  ctx.fillRect(0, 0, targetSize, targetSize);
  ctx.drawImage(canvas, 0, 0, srcW, srcH, padX, padY, newW, newH);

  const imgData = ctx.getImageData(0, 0, targetSize, targetSize).data;
  const numPixels = targetSize * targetSize;
  const floatData = new Float32Array(3 * numPixels);

  for (let i = 0; i < numPixels; i++) {
    const r = imgData[i * 4] / 255.0;
    const g = imgData[i * 4 + 1] / 255.0;
    const b = imgData[i * 4 + 2] / 255.0;

    floatData[i] = r;
    floatData[i + numPixels] = g;
    floatData[i + 2 * numPixels] = b;
  }

  const tensor = new ort.Tensor("float32", floatData, [1, 3, targetSize, targetSize]);
  return { tensor, scale, padX, padY, origW: srcW, origH: srcH };
}

/**
 * Non-Maximum Suppression (NMS) for YOLO detections
 */
function parseYoloOutput(outputTensor, scale, padX, padY, origW, origH, confThresh, iouThresh) {
  // output0 shape: [1, 5, 21504]
  const data = outputTensor.data;
  const numAnchors = 21504;
  const candidates = [];

  for (let i = 0; i < numAnchors; i++) {
    const conf = data[4 * numAnchors + i];
    if (conf >= confThresh) {
      const cx = data[i];
      const cy = data[numAnchors + i];
      const w = data[2 * numAnchors + i];
      const h = data[3 * numAnchors + i];

      // Convert from padded 1024x1024 back to original canvas
      const x1 = Math.max(0, Math.min(origW, ((cx - w / 2) - padX) / scale));
      const y1 = Math.max(0, Math.min(origH, ((cy - h / 2) - padY) / scale));
      const x2 = Math.max(0, Math.min(origW, ((cx + w / 2) - padX) / scale));
      const y2 = Math.max(0, Math.min(origH, ((cy + h / 2) - padY) / scale));

      if (x2 > x1 + 4 && y2 > y1 + 4) {
        candidates.push({ box: [x1, y1, x2, y2], conf, area: (x2 - x1) * (y2 - y1) });
      }
    }
  }

  // Sort by confidence descending
  candidates.sort((a, b) => b.conf - a.conf);

  // Apply NMS
  const selected = [];
  for (const c of candidates) {
    let keep = true;
    for (const s of selected) {
      const iou = calculateIoU(c.box, s.box);
      if (iou > iouThresh) {
        keep = false;
        break;
      }
    }
    if (keep) {
      selected.push(c);
    }
  }

  return selected;
}

function calculateIoU(b1, b2) {
  const xx1 = Math.max(b1[0], b2[0]);
  const yy1 = Math.max(b1[1], b2[1]);
  const xx2 = Math.min(b1[2], b2[2]);
  const yy2 = Math.min(b1[3], b2[3]);

  const w = Math.max(0, xx2 - xx1);
  const h = Math.max(0, yy2 - yy1);
  const inter = w * h;
  if (inter === 0) return 0;

  const area1 = (b1[2] - b1[0]) * (b1[3] - b1[1]);
  const area2 = (b2[2] - b2[0]) * (b2[3] - b2[1]);
  return inter / (area1 + area2 - inter);
}

/**
 * Prepares left-anchored white canvas [64, 256] with ImageNet normalization for L4
 */
function prepareAcnetTensor(cropCanvas, canvasH = 64, canvasW = 256) {
  const offscreen = new OffscreenCanvas(canvasW, canvasH);
  const ctx = offscreen.getContext("2d", { willReadFrequently: true });

  // White background
  ctx.fillStyle = "#FFFFFF";
  ctx.fillRect(0, 0, canvasW, canvasH);

  const origH = cropCanvas.height;
  const origW = cropCanvas.width;
  if (origH > 0 && origW > 0) {
    const scale = canvasH / origH;
    const scaledW = Math.min(canvasW, Math.round(origW * scale));
    ctx.drawImage(cropCanvas, 0, 0, origW, origH, 0, 0, scaledW, canvasH);
  }

  const imgData = ctx.getImageData(0, 0, canvasW, canvasH).data;
  const numPixels = canvasW * canvasH;
  const floatData = new Float32Array(3 * numPixels);

  // ImageNet normalization
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  for (let i = 0; i < numPixels; i++) {
    const r = (imgData[i * 4] / 255.0 - mean[0]) / std[0];
    const g = (imgData[i * 4 + 1] / 255.0 - mean[1]) / std[1];
    const b = (imgData[i * 4 + 2] / 255.0 - mean[2]) / std[2];

    floatData[i] = r;
    floatData[i + numPixels] = g;
    floatData[i + 2 * numPixels] = b;
  }

  return new ort.Tensor("float32", floatData, [1, 3, canvasH, canvasW]);
}

/**
 * L1 Dewarping execution
 */
async function runL1Dewarp(sourceCanvas) {
  if (!sessions.l1) return sourceCanvas;

  try {
    const l1W = 712;
    const l1H = 488;
    const offscreen = new OffscreenCanvas(l1W, l1H);
    const ctx = offscreen.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(sourceCanvas, 0, 0, l1W, l1H);

    const imgData = ctx.getImageData(0, 0, l1W, l1H).data;
    const numPixels = l1W * l1H;
    const floatData = new Float32Array(3 * numPixels);

    for (let i = 0; i < numPixels; i++) {
      floatData[i] = imgData[i * 4] / 255.0;
      floatData[i + numPixels] = imgData[i * 4 + 1] / 255.0;
      floatData[i + 2 * numPixels] = imgData[i * 4 + 2] / 255.0;
    }

    const inputTensor = new ort.Tensor("float32", floatData, [1, 3, l1H, l1W]);
    const feeds = { image: inputTensor };
    const results = await sessions.l1.run(feeds);
    const p2d = results.p2d; // [1, 2, 31, 45]

    // Perform grid-based texture remap on full resolution canvas
    return remapCanvasWithGrid(sourceCanvas, p2d);
  } catch (err) {
    console.warn("L1 Dewarp skipped due to error, proceeding with raw canvas:", err);
    return sourceCanvas;
  }
}

/**
 * Remaps an image using the UVDocNet 2D mesh (31 rows x 45 cols)
 */
function remapCanvasWithGrid(sourceCanvas, p2dTensor) {
  const origW = sourceCanvas.width;
  const origH = sourceCanvas.height;
  const outCanvas = new OffscreenCanvas(origW, origH);
  const outCtx = outCanvas.getContext("2d");

  const gh = 31;
  const gw = 45;
  const p2dData = p2dTensor.data;

  // Draw triangulated or bilinear quad mesh
  // Grid values are in normalized [-1, 1] coords
  for (let r = 0; r < gh - 1; r++) {
    for (let c = 0; c < gw - 1; c++) {
      // Quad coordinates in normalized space [-1, 1]
      const getCoord = (row, col) => {
        const u = p2dData[row * gw + col];
        const v = p2dData[gh * gw + row * gw + col];
        return {
          sx: ((u + 1) / 2) * origW,
          sy: ((v + 1) / 2) * origH
        };
      };

      const p00 = getCoord(r, c);
      const p10 = getCoord(r + 1, c);
      const p01 = getCoord(r, c + 1);
      const p11 = getCoord(r + 1, c + 1);

      // Destination quad in flat unwarped space
      const dx0 = (c / (gw - 1)) * origW;
      const dx1 = ((c + 1) / (gw - 1)) * origW;
      const dy0 = (r / (gh - 1)) * origH;
      const dy1 = ((r + 1) / (gh - 1)) * origH;

      // Draw slice
      const sw = Math.max(1, p01.sx - p00.sx);
      const sh = Math.max(1, p10.sy - p00.sy);
      const dw = Math.max(1, dx1 - dx0);
      const dh = Math.max(1, dy1 - dy0);

      outCtx.drawImage(sourceCanvas, p00.sx, p00.sy, sw, sh, dx0, dy0, dw, dh);
    }
  }

  return outCanvas;
}

/**
 * Main pipeline orchestrator: L1 -> L2 -> L3 -> L4 -> Final Annotated Canvas
 */
async function processImage(imageBitmap, config) {
  if (!sessions.l2 || !sessions.l3 || !sessions.l4) {
    throw new Error("Mô hình chưa được nạp. Vui lòng nhấn 'Nạp model' trước.");
  }

  const t0 = performance.now();

  // Create base canvas from imageBitmap
  const baseCanvas = new OffscreenCanvas(imageBitmap.width, imageBitmap.height);
  const baseCtx = baseCanvas.getContext("2d", { willReadFrequently: true });
  baseCtx.drawImage(imageBitmap, 0, 0);

  // -----------------------------------------------------------------
  // 1. LAYER 1: Dewarping
  // -----------------------------------------------------------------
  postStatus("l1", "Tầng 1: Đang nắn phẳng tài liệu (L1 Dewarp)...", 20);
  let workingCanvas = baseCanvas;
  if (config.enableDewarp && sessions.l1) {
    workingCanvas = await runL1Dewarp(baseCanvas);
  }

  const imgW = workingCanvas.width;
  const imgH = workingCanvas.height;

  // -----------------------------------------------------------------
  // 2. LAYER 2: Question Box Detection (YOLO QBox)
  // -----------------------------------------------------------------
  postStatus("l2", "Tầng 2: Đang phát hiện vùng câu hỏi (L2 QBox)...", 40);
  const l2Prep = createYoloTensor(workingCanvas, 1024);
  const l2Out = await sessions.l2.run({ images: l2Prep.tensor });
  const qboxes = parseYoloOutput(
    l2Out.output0,
    l2Prep.scale,
    l2Prep.padX,
    l2Prep.padY,
    imgW,
    imgH,
    config.qboxConf || 0.35,
    config.qboxIou || 0.45
  );

  // Sort questions top-to-bottom
  qboxes.sort((a, b) => a.box[1] - b.box[1]);

  // -----------------------------------------------------------------
  // 3. LAYER 3 & 4: Choice Boxes (YOLO CBox) and AnswerChoiceNet (ACNet)
  // -----------------------------------------------------------------
  postStatus("l3_l4", `Tầng 3-4: Đang đọc đáp án trên ${qboxes.length} câu hỏi...`, 60);

  const results = [];
  const qctx = workingCanvas.getContext("2d", { willReadFrequently: true });

  for (let qIdx = 0; qIdx < qboxes.length; qIdx++) {
    const q = qboxes[qIdx];
    const [qx1, qy1, qx2, qy2] = q.box;
    const qw = Math.max(10, qx2 - qx1);
    const qh = Math.max(10, qy2 - qy1);

    // Crop question region
    const qCrop = new OffscreenCanvas(qw, qh);
    const qCropCtx = qCrop.getContext("2d", { willReadFrequently: true });
    qCropCtx.drawImage(workingCanvas, qx1, qy1, qw, qh, 0, 0, qw, qh);

    // Run L3 YOLO CBox on this question crop
    const l3Prep = createYoloTensor(qCrop, 1024);
    const l3Out = await sessions.l3.run({ images: l3Prep.tensor });
    const cboxes = parseYoloOutput(
      l3Out.output0,
      l3Prep.scale,
      l3Prep.padX,
      l3Prep.padY,
      qw,
      qh,
      config.cboxConf || 0.25,
      config.cboxIou || 0.45
    );

    // For each choice box, classify with L4 ACNet
    const choices = [];
    for (let cIdx = 0; cIdx < cboxes.length; cIdx++) {
      const c = cboxes[cIdx];
      const [cx1, cy1, cx2, cy2] = c.box;
      const cw = Math.max(4, cx2 - cx1);
      const ch = Math.max(4, cy2 - cy1);

      // Crop choice box
      const cCrop = new OffscreenCanvas(cw, ch);
      const cCropCtx = cCrop.getContext("2d", { willReadFrequently: true });
      cCropCtx.drawImage(qCrop, cx1, cy1, cw, ch, 0, 0, cw, ch);

      // Prepare ACNet tensor [1, 3, 64, 256]
      const l4Tensor = prepareAcnetTensor(cCrop, 64, 256);
      const l4Out = await sessions.l4.run({ image: l4Tensor });

      const choiceLogits = l4Out.choice_logits.data; // length 5
      const stateLogits = l4Out.state_logits.data;   // length 3

      // Softmax & argmax
      const choiceId = argmax(choiceLogits);
      const stateId = argmax(stateLogits);

      const choiceLabel = REV_CHOICE_MAP[choiceId] || "UNKNOWN";
      const stateLabel = REV_STATE_MAP[stateId] || "not_selected";

      // Global coordinates on unwarped canvas
      const gx1 = qx1 + cx1;
      const gy1 = qy1 + cy1;
      const gx2 = qx1 + cx2;
      const gy2 = qy1 + cy2;

      choices.push({
        choice: choiceLabel,
        state: stateLabel,
        box_rel: [cx1, cy1, cx2, cy2],
        box_abs: [gx1, gy1, gx2, gy2],
        conf: c.conf
      });
    }

    results.push({
      questionIndex: qIdx + 1,
      questionBox: [qx1, qy1, qx2, qy2],
      conf: q.conf,
      choices
    });

    const progressPct = 60 + Math.round(((qIdx + 1) / qboxes.length) * 35);
    postStatus("l3_l4", `Đã xử lý ${qIdx + 1}/${qboxes.length} câu hỏi...`, progressPct);
  }

  // -----------------------------------------------------------------
  // 5. Render Final Annotated Canvas
  // -----------------------------------------------------------------
  postStatus("render", "Đang hoàn tất vẽ ảnh kết quả...", 98);
  const finalCanvas = new OffscreenCanvas(imgW, imgH);
  const fctx = finalCanvas.getContext("2d");
  fctx.drawImage(workingCanvas, 0, 0);

  // Draw annotations
  drawAnnotations(fctx, results, imgW, imgH);

  const durationSec = ((performance.now() - t0) / 1000).toFixed(2);
  const finalBitmap = finalCanvas.transferToImageBitmap();

  self.postMessage({
    type: "complete",
    imageBitmap: finalBitmap,
    results,
    summary: {
      questionsCount: results.length,
      choicesCount: results.reduce((acc, q) => acc + q.choices.length, 0),
      durationSec,
      resolution: [imgW, imgH]
    }
  }, [finalBitmap]);
}

function argmax(arr) {
  let maxIdx = 0;
  let maxVal = arr[0];
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > maxVal) {
      maxVal = arr[i];
      maxIdx = i;
    }
  }
  return maxIdx;
}

/**
 * Draws crisp, professional annotation boxes and badges onto the canvas
 */
function drawAnnotations(ctx, questions, width, height) {
  const isHighRes = width > 1200;
  const qBoxLineWidth = isHighRes ? 3 : 2;
  const cBoxLineWidth = isHighRes ? 2 : 1.5;
  const fontSize = isHighRes ? 20 : 14;

  ctx.font = `600 ${fontSize}px sans-serif`;

  for (const q of questions) {
    const [qx1, qy1, qx2, qy2] = q.questionBox;

    // Draw question outline (Cyan blue)
    ctx.strokeStyle = "#0284C7";
    ctx.lineWidth = qBoxLineWidth;
    ctx.strokeRect(qx1, qy1, qx2 - qx1, qy2 - qy1);

    // Question index badge
    const qBadge = `Câu ${q.questionIndex}`;
    const badgeW = ctx.measureText(qBadge).width + 12;
    const badgeH = fontSize + 8;

    ctx.fillStyle = "#0284C7";
    ctx.fillRect(qx1, Math.max(0, qy1 - badgeH), badgeW, badgeH);
    ctx.fillStyle = "#FFFFFF";
    ctx.fillText(qBadge, qx1 + 6, Math.max(0, qy1 - badgeH) + fontSize);

    // Draw choices
    for (const c of q.choices) {
      const [cx1, cy1, cx2, cy2] = c.box_abs;
      let strokeColor = "#94A3B8"; // default gray
      let fillColor = "rgba(148, 163, 184, 0.15)";
      let badgeBg = "#64748B";
      let statusIcon = "";

      if (c.state === "selected") {
        strokeColor = "#10B981"; // Emerald Green
        fillColor = "rgba(16, 185, 129, 0.25)";
        badgeBg = "#10B981";
        statusIcon = " ✓";
      } else if (c.state === "crossed_out") {
        strokeColor = "#EF4444"; // Red
        fillColor = "rgba(239, 68, 68, 0.2)";
        badgeBg = "#EF4444";
        statusIcon = " ✕";
      }

      ctx.fillStyle = fillColor;
      ctx.fillRect(cx1, cy1, cx2 - cx1, cy2 - cy1);
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = cBoxLineWidth;
      ctx.strokeRect(cx1, cy1, cx2 - cx1, cy2 - cy1);

      // Choice letter badge
      const cLabel = `${c.choice}${statusIcon}`;
      const cBadgeW = ctx.measureText(cLabel).width + 8;
      const cBadgeH = fontSize + 4;

      ctx.fillStyle = badgeBg;
      ctx.fillRect(cx1, Math.max(0, cy1 - cBadgeH), cBadgeW, cBadgeH);
      ctx.fillStyle = "#FFFFFF";
      ctx.fillText(cLabel, cx1 + 4, Math.max(0, cy1 - cBadgeH) + fontSize - 2);
    }
  }
}

// -----------------------------------------------------------------
// Message Event Dispatcher
// -----------------------------------------------------------------
self.onmessage = async (e) => {
  const { type, config, imageBitmap } = e.data;

  if (type === "init") {
    await initPipeline(config);
  } else if (type === "process") {
    try {
      await processImage(imageBitmap, config);
    } catch (err) {
      postError(`Lỗi xử lý ảnh: ${err.message}`);
    }
  }
};
