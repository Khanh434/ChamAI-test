# Document Scanner with Deskew & Light Dewarp Pipeline

### An interactive document scanner built in Python using OpenCV

The scanner takes a raw photo, finds the corners of the document, applies a perspective transformation to get a top-down view, performs a robust two-pass deskew and light dewarping stage on the color image, and produces aligned color images for downstream detection (such as YOLO) along with adaptive-threshold binary derivatives.

---

## Processing Pipeline

```
[Mobile Camera / Input Image]
        │
        ▼
[1. Document Corner Detection & Perspective Transform]
        │  (Extracts 4 corners and warps original BGR image)
        ▼
[2. Deskew & Light Dewarp Stage]
        ├── Pass 1: Global Orientation & Deskew
        │     • Normalizes portrait pages photographed sideways upright
        │     • Analyzes text baselines via length-weighted median angle estimation
        │     • Applies affine rotation with expanded canvas (no corner clipping)
        │
        └── Pass 2: Local Curvature Assessment & Mild Dewarp
              • Analyzes 3x3 grid cells for local baseline angles
              • Evaluates local angular spread (Δθ)
              • If Δθ > 12°: Rejects with retake warning ("Document too curved; retake photo")
              • If Δθ ≤ 12°: Remaps image using smooth, bounded displacement mesh (cv2.remap)
        │
        ▼
[3. Outputs]
        ├── output/<filename>        --> Aligned 3-channel color BGR image (for YOLO / OCR)
        └── output/binary/<filename> --> Sharpened adaptive-threshold binary image
```

---

## Output Locations

* **Color Aligned Output**: Saved to `output/<image_name>`. Retains full BGR color depth and detail with straightened text lines.
* **Binary Threshold Output**: Saved to `output/binary/<image_name>`. Retains adaptive black-and-white cleaned derivative.

---

## Curvature Rejection & Retake Behavior

If a document has severe local curvature (angular spread across usable 3x3 grid cells exceeds **12.0°**):
* **Single Image Mode**: Emits a `[REJECTED]` retake warning and exits safely without writing an invalid accepted image.
* **Directory Batch Mode**: Reports the rejected file and retake warning, skips writing output for that image, and continues processing the remaining images in the directory.

---

## Usage

```bash
python scan.py (--images <IMG_DIR> | --image <IMG_PATH>) [-i]
```

* **Interactive Mode (`-i`)**: Prompts user with a matplotlib GUI to verify or adjust document corners:
```bash
python scan.py --image sample_images/desk.JPG -i
```

* **Batch Directory Processing**: Automatically processes all images in the folder:
```bash
python scan.py --images sample_images
python scan.py --images ../../anh
```

---

## Running Unit Tests

Run the test suite using `pytest`:
```bash
python -m pytest -v tests/test_deskew.py
```
