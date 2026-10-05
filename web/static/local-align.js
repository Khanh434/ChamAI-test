"use strict";

const MAX_IMAGE_SIDE = 1800;
const CORNER_DETECTION_SIDE = 1000;
const DESKEW_ANALYSIS_HEIGHT = 500;

function waitForOpenCv() {
  const module = globalThis.cv;
  if (!module) {
    throw new Error("Không tải được OpenCV.js trên điện thoại.");
  }
  if (!module.Mat) {
    throw new Error("OpenCV.js chưa khởi tạo được trên điện thoại.");
  }
  return module;
}

function orderedCorners(points) {
  const sums = points.map((point) => point.x + point.y);
  const differences = points.map((point) => point.y - point.x);
  return [
    points[sums.indexOf(Math.min(...sums))],
    points[differences.indexOf(Math.min(...differences))],
    points[sums.indexOf(Math.max(...sums))],
    points[differences.indexOf(Math.max(...differences))],
  ];
}

function findDocumentCorners(cv, image) {
  const scale = Math.min(CORNER_DETECTION_SIDE / image.cols, CORNER_DETECTION_SIDE / image.rows, 1);
  const width = Math.max(1, Math.round(image.cols * scale));
  const height = Math.max(1, Math.round(image.rows * scale));
  const scaleX = width / image.cols;
  const scaleY = height / image.rows;
  const detector = new cv.Mat();
  const gray = new cv.Mat();
  const blurred = new cv.Mat();
  const edges = new cv.Mat();
  const kernel = cv.Mat.ones(3, 3, cv.CV_8U);
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  try {
    cv.resize(image, detector, new cv.Size(width, height), 0, 0, cv.INTER_AREA);
    cv.cvtColor(detector, gray, cv.COLOR_RGBA2GRAY);
    cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0);
    cv.Canny(blurred, edges, 50, 150);
    cv.dilate(edges, edges, kernel);
    cv.findContours(edges, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);

    let bestPoints = null;
    let bestArea = detector.rows * detector.cols * 0.25;
    for (let index = 0; index < contours.size(); index += 1) {
      const contour = contours.get(index);
      const approximation = new cv.Mat();
      try {
        const perimeter = cv.arcLength(contour, true);
        cv.approxPolyDP(contour, approximation, perimeter * 0.02, true);
        if (approximation.rows !== 4 || !cv.isContourConvex(approximation)) {
          continue;
        }
        const area = Math.abs(cv.contourArea(approximation));
        if (area > bestArea) {
          const values = approximation.data32S;
          bestPoints = [
            { x: values[0] / scaleX, y: values[1] / scaleY },
            { x: values[2] / scaleX, y: values[3] / scaleY },
            { x: values[4] / scaleX, y: values[5] / scaleY },
            { x: values[6] / scaleX, y: values[7] / scaleY },
          ];
          bestArea = area;
        }
      } finally {
        contour.delete();
        approximation.delete();
      }
    }
    return bestPoints ? orderedCorners(bestPoints) : null;
  } finally {
    detector.delete();
    gray.delete();
    blurred.delete();
    edges.delete();
    kernel.delete();
    contours.delete();
    hierarchy.delete();
  }
}

function perspectiveAlign(cv, image, corners) {
  if (!corners) {
    return image;
  }
  const [topLeft, topRight, bottomRight, bottomLeft] = corners;
  const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
  const width = Math.max(
    Math.round(distance(bottomRight, bottomLeft)),
    Math.round(distance(topRight, topLeft)),
  );
  const height = Math.max(
    Math.round(distance(topRight, bottomRight)),
    Math.round(distance(topLeft, bottomLeft)),
  );
  if (width < 2 || height < 2) {
    return image;
  }
  const source = cv.matFromArray(4, 1, cv.CV_32FC2, corners.flatMap(({ x, y }) => [x, y]));
  const destination = cv.matFromArray(4, 1, cv.CV_32FC2, [
    0, 0,
    width - 1, 0,
    width - 1, height - 1,
    0, height - 1,
  ]);
  const transform = cv.getPerspectiveTransform(source, destination);
  const aligned = new cv.Mat();
  try {
    cv.warpPerspective(
      image,
      aligned,
      transform,
      new cv.Size(width, height),
      cv.INTER_CUBIC,
      cv.BORDER_REPLICATE,
      new cv.Scalar(),
    );
    return aligned;
  } catch (error) {
    aligned.delete();
    throw error;
  } finally {
    source.delete();
    destination.delete();
    transform.delete();
  }
}

function projectionVariance(cv, binary, angle) {
  const rotation = cv.getRotationMatrix2D(
    new cv.Point(binary.cols / 2, binary.rows / 2),
    angle,
    1,
  );
  const rotated = new cv.Mat();
  const projection = new cv.Mat();
  try {
    cv.warpAffine(
      binary,
      rotated,
      rotation,
      new cv.Size(binary.cols, binary.rows),
      cv.INTER_NEAREST,
      cv.BORDER_CONSTANT,
      new cv.Scalar(0),
    );
    cv.reduce(rotated, projection, 1, cv.REDUCE_SUM, cv.CV_64F);
    const values = projection.data64F;
    let mean = 0;
    for (let index = 0; index < values.length; index += 1) {
      mean += values[index];
    }
    mean /= values.length;
    let variance = 0;
    for (let index = 0; index < values.length; index += 1) {
      const delta = values[index] - mean;
      variance += delta * delta;
    }
    return variance / values.length;
  } finally {
    rotation.delete();
    rotated.delete();
    projection.delete();
  }
}

function estimateSkewAngle(
  cv,
  image,
  {
    analysisHeight = DESKEW_ANALYSIS_HEIGHT,
    coarseStep = 0.5,
    fineStep = 0.1,
    minVarianceRatio = 1.05,
  } = {},
) {
  const scale = Math.min(
    analysisHeight / image.rows,
    1000 / image.cols,
    1,
  );
  const width = Math.max(1, Math.round(image.cols * scale));
  const height = Math.max(1, Math.round(image.rows * scale));
  if (width < 25 || height < 25) {
    return null;
  }
  const reduced = new cv.Mat();
  const gray = new cv.Mat();
  const binary = new cv.Mat();
  const closed = new cv.Mat();
  const kernelWidth = Math.max(15, Math.round(width * 0.04));
  const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(kernelWidth, 1));
  try {
    cv.resize(image, reduced, new cv.Size(width, height), 0, 0, cv.INTER_AREA);
    cv.cvtColor(reduced, gray, cv.COLOR_RGBA2GRAY);
    cv.adaptiveThreshold(
      gray,
      binary,
      255,
      cv.ADAPTIVE_THRESH_GAUSSIAN_C,
      cv.THRESH_BINARY_INV,
      25,
      10,
    );
    const data = binary.data;
    let foreground = 0;
    for (let index = 0; index < data.length; index += 16) {
      foreground += data[index] ? 1 : 0;
    }
    if (foreground / Math.ceil(data.length / 16) < 0.005) {
      return null;
    }
    cv.morphologyEx(binary, closed, cv.MORPH_CLOSE, kernel);

    const angles = [];
    const variances = [];
    for (let angle = -20; angle <= 20 + coarseStep / 2; angle += coarseStep) {
      angles.push(angle);
      variances.push(projectionVariance(cv, closed, angle));
    }
    let bestIndex = variances.indexOf(Math.max(...variances));
    const minVariance = Math.min(...variances);
    if (
      variances[bestIndex] <= 0
      || (minVariance > 0 && variances[bestIndex] / minVariance < minVarianceRatio)
    ) {
      return null;
    }
    const bestCoarse = angles[bestIndex];

    const fineAngles = [];
    const fineVariances = [];
    for (
      let angle = bestCoarse - coarseStep;
      angle <= bestCoarse + coarseStep + fineStep / 2;
      angle += fineStep
    ) {
      fineAngles.push(angle);
      fineVariances.push(projectionVariance(cv, closed, angle));
    }
    bestIndex = fineVariances.indexOf(Math.max(...fineVariances));
    return fineAngles[bestIndex];
  } finally {
    reduced.delete();
    gray.delete();
    binary.delete();
    closed.delete();
    kernel.delete();
  }
}

function estimateGridAngles(cv, image) {
  const rows = 3;
  const columns = 3;
  if (image.rows < rows || image.cols < columns) {
    return Array.from({ length: rows }, () => Array(columns).fill(null));
  }
  const cellHeight = Math.floor(image.rows / rows);
  const cellWidth = Math.floor(image.cols / columns);
  const angles = [];
  for (let row = 0; row < rows; row += 1) {
    const startY = row * cellHeight;
    const height = row === rows - 1 ? image.rows - startY : cellHeight;
    const angleRow = [];
    for (let column = 0; column < columns; column += 1) {
      const startX = column * cellWidth;
      const width = column === columns - 1 ? image.cols - startX : cellWidth;
      const cell = image.roi(new cv.Rect(startX, startY, width, height));
      try {
        angleRow.push(estimateSkewAngle(cv, cell, {
          analysisHeight: 240,
          coarseStep: 1,
          fineStep: 0.2,
        }));
      } finally {
        cell.delete();
      }
    }
    angles.push(angleRow);
  }
  return angles;
}

function dewarp(cv, image, gridAngles) {
  const usableAngles = gridAngles.flat().filter((angle) => angle !== null);
  if (usableAngles.length < 2) {
    return image;
  }
  const meanAngle = usableAngles.reduce((sum, angle) => sum + angle, 0) / usableAngles.length;
  const slopes = gridAngles.flat().map((angle) => (
    angle === null ? 0 : Math.tan((angle - meanAngle) * Math.PI / 180)
  ));
  const slopeMatrix = cv.matFromArray(3, 3, cv.CV_32FC1, slopes);
  const smoothSlopes = new cv.Mat();
  const height = image.rows;
  const width = image.cols;
  const pixelCount = height * width;
  const mapXValues = new Float32Array(pixelCount);
  const mapYValues = new Float32Array(pixelCount);
  const taperX = new Float32Array(width);
  let mapX = null;
  let mapY = null;
  const aligned = new cv.Mat();
  try {
    cv.resize(slopeMatrix, smoothSlopes, new cv.Size(width, height), 0, 0, cv.INTER_CUBIC);
    const smoothValues = smoothSlopes.data32F;
    for (let x = 0; x < width; x += 1) {
      const sine = Math.sin(Math.PI * x / Math.max(1, width - 1));
      taperX[x] = Math.sqrt(Math.max(0, sine));
    }
    const maxDisplacement = Math.max(20, 0.03 * height);
    for (let y = 0; y < height; y += 1) {
      const rowOffset = y * width;
      let displacement = 0;
      let rowTotal = 0;
      for (let x = 0; x < width; x += 1) {
        displacement -= smoothValues[rowOffset + x];
        rowTotal += displacement;
      }
      const rowMean = rowTotal / width;
      displacement = 0;
      const sine = Math.sin(Math.PI * y / Math.max(1, height - 1));
      const taperY = Math.sqrt(Math.max(0, sine));
      for (let x = 0; x < width; x += 1) {
        const index = rowOffset + x;
        displacement -= smoothValues[index];
        const offset = Math.max(
          -maxDisplacement,
          Math.min(maxDisplacement, (displacement - rowMean) * taperX[x] * taperY),
        );
        mapXValues[index] = x;
        mapYValues[index] = y + offset;
      }
    }
    mapX = cv.matFromArray(height, width, cv.CV_32FC1, mapXValues);
    mapY = cv.matFromArray(height, width, cv.CV_32FC1, mapYValues);
    cv.remap(
      image,
      aligned,
      mapX,
      mapY,
      cv.INTER_CUBIC,
      cv.BORDER_REPLICATE,
      new cv.Scalar(),
    );
    return aligned;
  } catch (error) {
    aligned.delete();
    throw error;
  } finally {
    slopeMatrix.delete();
    smoothSlopes.delete();
    mapX?.delete();
    mapY?.delete();
  }
}

function deskew(cv, image, angle) {
  if (Math.abs(angle) < 0.02) {
    return image;
  }
  const center = new cv.Point(image.cols / 2, image.rows / 2);
  const rotation = cv.getRotationMatrix2D(center, angle, 1);
  const cosine = Math.abs(rotation.data64F[0]);
  const sine = Math.abs(rotation.data64F[1]);
  const width = Math.round(image.rows * sine + image.cols * cosine);
  const height = Math.round(image.rows * cosine + image.cols * sine);
  rotation.data64F[2] += width / 2 - center.x;
  rotation.data64F[5] += height / 2 - center.y;
  const aligned = new cv.Mat();
  try {
    cv.warpAffine(
      image,
      aligned,
      rotation,
      new cv.Size(width, height),
      cv.INTER_CUBIC,
      cv.BORDER_REPLICATE,
      new cv.Scalar(),
    );
    return aligned;
  } catch (error) {
    aligned.delete();
    throw error;
  } finally {
    rotation.delete();
  }
}

async function alignImageLocally(file) {
  const cv = waitForOpenCv();
  const started = performance.now();
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(MAX_IMAGE_SIDE / bitmap.width, MAX_IMAGE_SIDE / bitmap.height, 1);
  const width = Math.max(1, Math.round(bitmap.width * scale));
  const height = Math.max(1, Math.round(bitmap.height * scale));
  const sourceCanvas = new OffscreenCanvas(width, height);
  sourceCanvas.width = width;
  sourceCanvas.height = height;
  const sourceContext = sourceCanvas.getContext("2d", { willReadFrequently: true });
  if (!sourceContext) {
    bitmap.close();
    throw new Error("Không tạo được canvas để OpenCV đọc ảnh trên điện thoại.");
  }
  try {
    sourceContext.drawImage(bitmap, 0, 0, width, height);
  } finally {
    bitmap.close();
  }

  const image = cv.matFromImageData(sourceContext.getImageData(0, 0, width, height));
  const mats = new Set([image]);
  const release = (mat) => {
    if (mats.delete(mat)) {
      mat.delete();
    }
  };
  try {
    const corners = findDocumentCorners(cv, image);
    let aligned = perspectiveAlign(cv, image, corners);
    mats.add(aligned);
    if (aligned !== image) {
      release(image);
    }
    if (aligned.cols > aligned.rows) {
      const oriented = new cv.Mat();
      mats.add(oriented);
      cv.rotate(aligned, oriented, cv.ROTATE_90_CLOCKWISE);
      release(aligned);
      aligned = oriented;
    }
    const angle = estimateSkewAngle(cv, aligned) ?? 0;
    const deskewed = deskew(cv, aligned, angle);
    mats.add(deskewed);
    if (deskewed !== aligned) {
      release(aligned);
    }
    const gridAngles = estimateGridAngles(cv, deskewed);
    const usableAngles = gridAngles.flat().filter((gridAngle) => gridAngle !== null);
    const angleSpread = usableAngles.length > 1
      ? Math.max(...usableAngles) - Math.min(...usableAngles)
      : 0;
    const alignedDocument = angleSpread <= 12 ? dewarp(cv, deskewed, gridAngles) : deskewed;
    mats.add(alignedDocument);
    if (alignedDocument !== deskewed) {
      release(deskewed);
    }

    const imageData = new ImageData(
      new Uint8ClampedArray(alignedDocument.data),
      alignedDocument.cols,
      alignedDocument.rows,
    );
    return {
      image: await createImageBitmap(imageData),
      processingMs: performance.now() - started,
    };
  } finally {
    for (const mat of mats) {
      mat.delete();
    }
  }
}