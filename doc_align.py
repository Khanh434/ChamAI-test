#!/usr/bin/env python3
"""Document alignment pipeline.

This module now delegates to the newer document scanner pipeline stored in the
`new` directory while preserving the original `align_document()` API used by the
phone pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
NEW_DIR = ROOT / "new"
if str(NEW_DIR) not in sys.path:
    sys.path.insert(0, str(NEW_DIR))

try:
    from pyimagesearch import transform as _transform
    from scan import CurvatureRejectionError, DocScanner as _DocScanner, deskew_and_dewarp
except Exception as exc:  # pragma: no cover - import error should surface clearly.
    raise RuntimeError(
        "Unable to import the updated document scanner from the `new` directory. "
        "Ensure the `new` folder is present alongside `doc_align.py`."
    ) from exc


def resize_to_limit(image: np.ndarray, max_side: int = 1800) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(max_side / max(height, width), 1.0)
    if scale < 1.0:
        new_size = (int(round(width * scale)), int(round(height * scale)))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return image


def order_points(points: np.ndarray) -> np.ndarray:
    """Order a quadrilateral as top-left, top-right, bottom-right, bottom-left."""
    rectangle = np.zeros((4, 2), dtype=np.float32)
    sum_vals = points.sum(axis=1)
    rectangle[0] = points[np.argmin(sum_vals)]
    rectangle[2] = points[np.argmax(sum_vals)]

    diff_vals = np.diff(points, axis=1)
    rectangle[1] = points[np.argmin(diff_vals)]
    rectangle[3] = points[np.argmax(diff_vals)]
    return rectangle


def find_document_corners(image: np.ndarray, max_side: int = 1000) -> np.ndarray | None:
    """Detect document corners using the scanner implementation from `new/scan.py`."""
    if image is None or image.size == 0:
        return None

    height, width = image.shape[:2]
    detector_image = resize_to_limit(image, max_side)
    detector_height, detector_width = detector_image.shape[:2]

    scanner = _DocScanner()
    try:
        contour = scanner.get_contour(detector_image)
    except Exception:
        return None

    if contour is None or len(contour) != 4:
        return None

    corners = np.asarray(contour, dtype=np.float32)
    corners[:, 0] *= width / max(detector_width, 1)
    corners[:, 1] *= height / max(detector_height, 1)
    corners[:, 0] = np.clip(corners[:, 0], 0, width - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, height - 1)
    return corners


def perspective_transform(image: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """Warp a document contour into a fronto-parallel rectangle."""
    if contour is None or contour.size == 0:
        return image.copy()

    rect = order_points(np.asarray(contour, dtype=np.float32))
    return _transform.four_point_transform(image, rect)


def estimate_skew_angle(image: np.ndarray) -> float:
    """Estimate global text rotation using long edge segments."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(binary, 50, 150)

    min_line_length = max(40, gray.shape[1] * 0.2)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=min_line_length,
        maxLineGap=20,
    )
    if lines is None or len(lines) == 0:
        return 0.0

    angles: list[float] = []
    for line in lines:
        coords = np.asarray(line).reshape(-1)
        if coords.size < 4:
            continue

        x1, y1, x2, y2 = coords[:4].astype(float)
        if np.isclose(x2, x1):
            angle = 90.0
        else:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angle = ((angle + 90.0) % 180.0) - 90.0
        angles.append(angle)

    if not angles:
        return 0.0

    theta = float(np.median(angles))
    return theta


def deskew_image(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate the image by the estimated skew angle."""
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def align_document(image: np.ndarray) -> np.ndarray:
    """Detect, warp, and deskew the document using the newer `new/scan.py` pipeline."""
    if image is None or image.size == 0:
        return image.copy() if isinstance(image, np.ndarray) else np.zeros((1, 1, 3), dtype=np.uint8)

    resized = resize_to_limit(image)
    corners = find_document_corners(resized)
    warped = perspective_transform(resized, corners) if corners is not None else resized

    result = deskew_and_dewarp(warped, raise_on_rejection=False)
    if result is not None and getattr(result, "image", None) is not None:
        return result.image

    skew_angle = estimate_skew_angle(warped)
    if abs(skew_angle) > 1e-3:
        return deskew_image(warped, skew_angle)
    return warped


def generate_demo_image(path: Path) -> None:
    """Create a synthetic document image with rotation and projection distortion."""
    height, width = 1400, 1000
    document = np.full((height, width, 3), 255, dtype=np.uint8)

    for line_idx in range(24):
        y = 150 + line_idx * 42
        text = "Sample document text line " + str(line_idx + 1)
        cv2.putText(
            document,
            text,
            (180, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (30, 30, 30),
            2,
            cv2.LINE_AA,
        )

    src = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    dst = np.array([[180, 90], [820, 180], [910, 1310], [90, 1220]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    perspective_image = cv2.warpPerspective(
        document,
        matrix,
        (width, height),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(45, 45, 45),
    )

    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 4, 1.0)
    final_image = cv2.warpAffine(
        perspective_image,
        rotation_matrix,
        (width, height),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(45, 45, 45),
    )
    cv2.imwrite(str(path), final_image)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def save_debug_report(debug_dir: Path, stem: str, details: dict[str, object]) -> None:
    """Write a text report to disk and print it to the console."""
    debug_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}: {value}" for key, value in details.items()]
    report = "\n".join(lines) + "\n"

    report_path = debug_dir / f"{stem}_debug.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Debug saved to: {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast document alignment pipeline")
    parser.add_argument("--input", type=str, default="input", help="Input image file or folder. Default: input")
    parser.add_argument("--output", type=str, default="output", help="Output image file or folder. Default: output")
    parser.add_argument("--debug-dir", type=str, default="debug", help="Directory for debug reports. Default: debug")
    parser.add_argument("--generate-demo", action="store_true", help="Generate a synthetic demo image and align it.")
    return parser.parse_args()


def resolve_input_files(input_path: Path, generate_demo: bool) -> list[Path]:
    if generate_demo:
        if input_path.exists():
            input_dir = input_path if input_path.is_dir() else input_path.parent
        else:
            input_dir = input_path.parent if input_path.suffix else input_path
        input_dir.mkdir(parents=True, exist_ok=True)
        demo_path = input_dir / "demo_input.png"
        generate_demo_image(demo_path)
        return [demo_path]

    if input_path.is_dir():
        files = sorted(
            path for path in input_path.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not files:
            raise FileNotFoundError(f"No supported image files found in: {input_path}")
        return files

    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {input_path.suffix}")
        return [input_path]

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def resolve_output_path(output_path: Path, source_path: Path, is_single_file: bool) -> Path:
    if output_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
        return output_path

    output_path.mkdir(parents=True, exist_ok=True)
    if is_single_file:
        return output_path / source_path.name
    return output_path / f"{source_path.stem}_aligned{source_path.suffix}"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    debug_dir = Path(args.debug_dir)

    input_files = resolve_input_files(input_path, args.generate_demo)
    if not input_files:
        raise ValueError("No input images were found to process.")

    for image_path in input_files:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read the input image: {image_path}")

        aligned = align_document(image)

        is_single_input_file = len(input_files) == 1 and not args.generate_demo
        target_path = resolve_output_path(output_path, image_path, is_single_file=is_single_input_file)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target_path), aligned)

        details = {
            "input_image": str(image_path),
            "output_image": str(target_path),
            "input_shape": tuple(image.shape),
            "output_shape": tuple(aligned.shape),
            "aligned_successfully": True,
        }
        save_debug_report(debug_dir, image_path.stem, details)
        print(f"Saved aligned output to: {target_path}")


if __name__ == "__main__":
    main()
