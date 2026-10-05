"""Deskew and light dewarp module for document scanning pipeline.

This module provides a two-pass alignment stage:
- Pass 1: Page orientation normalization and high-precision global baseline deskew.
- Pass 2: 3x3 local grid curvature assessment, threshold rejection (> 12 deg),
          and smooth bounded displacement remapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import cv2


class CurvatureRejectionError(Exception):
    """Raised when the document's local angular spread exceeds the allowable threshold."""

    def __init__(
        self,
        spread_angle: float,
        threshold: float = 12.0,
        message: str = "Document too curved; retake photo",
    ):
        self.spread_angle = spread_angle
        self.threshold = threshold
        self.message = message
        super().__init__(
            f"{message} (spread: {spread_angle:.2f}°, threshold: {threshold:.2f}°)"
        )


@dataclass
class DeskewResult:
    """Result object returned by deskew and dewarp processing."""

    image: Optional[np.ndarray]
    is_accepted: bool
    global_angle: float
    grid_angles: List[List[Optional[float]]]
    angular_spread: float
    rejection_reason: Optional[str] = None


def normalize_orientation(image: np.ndarray) -> np.ndarray:
    """Normalizes portrait pages photographed sideways into upright orientation.

    If width > height, rotates 90 degrees clockwise to achieve portrait orientation.

    Args:
        image: Input BGR or Grayscale image.

    Returns:
        Upright oriented image.
    """
    h, w = image.shape[:2]
    if w > h:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


def estimate_dominant_angle(
    image: np.ndarray,
    min_angle: float = -20.0,
    max_angle: float = 20.0,
    coarse_step: float = 0.25,
    fine_step: float = 0.02,
    min_variance_ratio: float = 1.05,
) -> Optional[float]:
    """Estimates dominant baseline angle of text lines in degrees with sub-degree precision.

    Uses projection profile analysis on binarized and horizontally enhanced text strokes.
    Maximizes row variance of horizontal projection profile across candidate angles.

    Args:
        image: BGR or Grayscale image (or cell crop).
        min_angle: Lower bound search angle in degrees.
        max_angle: Upper bound search angle in degrees.
        coarse_step: Coarse search step in degrees.
        fine_step: Refined search step in degrees.
        min_variance_ratio: Ratio of max-to-min variance required to confirm text lines.

    Returns:
        Estimated dominant angle in degrees, or None if insufficient/unclear lines.
    """
    if image is None or image.size == 0:
        return None

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape[:2]
    if h < 20 or w < 20:
        return None

    # Downscale for performance while preserving text line structure
    target_h = 800
    scale = target_h / float(h) if h > target_h else 1.0
    if scale < 1.0:
        analysis_img = cv2.resize(gray, (max(1, int(w * scale)), target_h))
    else:
        analysis_img = gray

    # Adaptive binarization to extract text strokes
    thresh = cv2.adaptiveThreshold(
        analysis_img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25,
        10,
    )

    # Check for sufficient text stroke density
    white_pixel_ratio = np.count_nonzero(thresh) / float(thresh.size)
    if white_pixel_ratio < 0.005:  # Less than 0.5% text pixels -> cell is empty/noisy
        return None

    # Horizontal morphological closing to bridge characters into continuous line segments
    kernel_w = max(15, int(analysis_img.shape[1] * 0.04))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    dilated = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    sh, sw = dilated.shape
    center = (sw / 2.0, sh / 2.0)

    # Coarse search
    angles = np.arange(min_angle, max_angle + coarse_step / 2.0, coarse_step)
    variances: List[float] = []
    for a in angles:
        m = cv2.getRotationMatrix2D(center, a, 1.0)
        rot = cv2.warpAffine(
            dilated,
            m,
            (sw, sh),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        proj = np.sum(rot, axis=1, dtype=np.float64)
        variances.append(float(np.var(proj)))

    var_arr = np.array(variances, dtype=np.float64)
    min_var = float(np.min(var_arr))
    max_var = float(np.max(var_arr))

    if min_var > 0 and (max_var / min_var) < min_variance_ratio:
        # Insufficient line orientation signal
        return None

    best_coarse = float(angles[int(np.argmax(var_arr))])

    # Fine search around best_coarse
    fine_angles = np.arange(
        best_coarse - coarse_step,
        best_coarse + coarse_step + fine_step / 2.0,
        fine_step,
    )
    fine_variances: List[float] = []
    for a in fine_angles:
        m = cv2.getRotationMatrix2D(center, a, 1.0)
        rot = cv2.warpAffine(
            dilated,
            m,
            (sw, sh),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        proj = np.sum(rot, axis=1, dtype=np.float64)
        fine_variances.append(float(np.var(proj)))

    return float(fine_angles[int(np.argmax(fine_variances))])


def rotate_image_expanded(
    image: np.ndarray,
    angle: float,
    border_mode: int = cv2.BORDER_REPLICATE,
) -> np.ndarray:
    """Rotates an image around its center with an expanded canvas to avoid clipping corners.

    Args:
        image: Input image (BGR or Grayscale).
        angle: Deskew rotation angle in degrees (counter-clockwise if positive).
        border_mode: OpenCV border extrapolation mode.

    Returns:
        Expanded rotated image.
    """
    if abs(angle) < 1e-3:
        return image.copy()

    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(m[0, 0])
    sin = np.abs(m[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    m[0, 2] += (new_w / 2.0) - center[0]
    m[1, 2] += (new_h / 2.0) - center[1]

    return cv2.warpAffine(
        image,
        m,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=border_mode,
    )


def estimate_grid_angles(
    image: np.ndarray,
    rows: int = 3,
    cols: int = 3,
) -> List[List[Optional[float]]]:
    """Divides page into a grid and estimates dominant text angle per cell.

    Args:
        image: Globally aligned image.
        rows: Number of grid rows (default 3).
        cols: Number of grid columns (default 3).

    Returns:
        2D list [rows x cols] of estimated angles or None for unusable cells.
    """
    h, w = image.shape[:2]
    cell_h = h // rows
    cell_w = w // cols

    grid_angles: List[List[Optional[float]]] = []
    for r in range(rows):
        row_angles: List[Optional[float]] = []
        for c in range(cols):
            y_start = r * cell_h
            y_end = (r + 1) * cell_h if r < rows - 1 else h
            x_start = c * cell_w
            x_end = (c + 1) * cell_w if c < cols - 1 else w

            cell = image[y_start:y_end, x_start:x_end]
            ang = estimate_dominant_angle(cell)
            row_angles.append(ang)
        grid_angles.append(row_angles)

    return grid_angles


def compute_angular_spread(
    grid_angles: List[List[Optional[float]]],
) -> Tuple[float, List[float]]:
    """Computes the angular spread (max - min) across all usable grid cells.

    Args:
        grid_angles: 2D list of grid cell angles.

    Returns:
        Tuple of (spread_degrees, list_of_usable_angles).
    """
    usable = [a for row in grid_angles for a in row if a is not None]
    if len(usable) < 2:
        return 0.0, usable
    spread = float(max(usable) - min(usable))
    return spread, usable


def build_dewarp_maps(
    h: int,
    w: int,
    grid_angles: List[List[Optional[float]]],
    rows: int = 3,
    cols: int = 3,
    max_displacement: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Derives a smooth, continuous displacement mesh from grid cell deviations.

    Args:
        h: Image height.
        w: Image width.
        grid_angles: 2D list of angles per cell.
        rows: Grid row count.
        cols: Grid column count.
        max_displacement: Maximum allowed vertical shift in pixels (defaults to 0.03 * h).

    Returns:
        Tuple of (map_x, map_y) float32 coordinate arrays for cv2.remap.
    """
    if max_displacement is None:
        max_displacement = max(20.0, 0.03 * h)

    usable = [a for row in grid_angles for a in row if a is not None]
    mean_angle = float(np.mean(usable)) if usable else 0.0

    # Build local slope matrix tan(delta_theta) relative to mean baseline
    slopes = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            val = grid_angles[r][c]
            slopes[r, c] = (
                np.tan(np.radians(val - mean_angle)) if val is not None else 0.0
            )

    # Smooth slope field over image dimensions
    smooth_slopes = cv2.resize(slopes, (w, h), interpolation=cv2.INTER_CUBIC)

    # Integrate slope horizontally across x to yield continuous vertical displacement
    disp_y = -np.cumsum(smooth_slopes, axis=1, dtype=np.float32)
    # Zero-center each row's displacement
    disp_y -= np.mean(disp_y, axis=1, keepdims=True)

    # Border attenuation window to guarantee continuous boundaries without edge clipping
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
    sin_x = np.sin(np.pi * np.clip(x_coords / max(1.0, float(w - 1)), 0.0, 1.0))
    sin_y = np.sin(np.pi * np.clip(y_coords / max(1.0, float(h - 1)), 0.0, 1.0))
    taper_x = np.sqrt(np.maximum(0.0, sin_x))
    taper_y = np.sqrt(np.maximum(0.0, sin_y))
    disp_y = disp_y * taper_x * taper_y

    # Clamp displacement to safe bounds
    disp_y = np.clip(disp_y, -max_displacement, max_displacement)

    map_x = x_coords.astype(np.float32)
    map_y = (y_coords + disp_y).astype(np.float32)
    return map_x, map_y


def dewarp_image(
    image: np.ndarray,
    grid_angles: List[List[Optional[float]]],
    rows: int = 3,
    cols: int = 3,
) -> np.ndarray:
    """Applies mild local dewarp using smooth displacement remapping.

    Args:
        image: Globally deskewed image.
        grid_angles: 2D list of cell angles.
        rows: Grid row count.
        cols: Grid column count.

    Returns:
        Dewarped image.
    """
    h, w = image.shape[:2]
    map_x, map_y = build_dewarp_maps(h, w, grid_angles, rows=rows, cols=cols)
    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def deskew_and_dewarp(
    image: np.ndarray,
    max_curvature_threshold: float = 12.0,
    raise_on_rejection: bool = False,
) -> DeskewResult:
    """Performs the complete two-pass deskew and light dewarp pipeline.

    Pass 1:
        - Portrait orientation normalization.
        - Robust high-precision global baseline angle estimation.
        - Canvas-expanded affine deskew rotation.

    Pass 2:
        - 3x3 local grid angle estimation.
        - Curvature spread evaluation.
        - If spread > threshold: Rejects with retake message.
        - Otherwise: Applies smooth continuous displacement remapping.

    Args:
        image: Perspective-warped BGR image from document scanner.
        max_curvature_threshold: Maximum allowable local angular spread (default 12.0 deg).
        raise_on_rejection: If True, raises CurvatureRejectionError on excessive curvature.

    Returns:
        DeskewResult containing processed image (if accepted) and alignment metadata.

    Raises:
        CurvatureRejectionError: If raise_on_rejection is True and spread > threshold.
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid input image passed to deskew_and_dewarp")

    # Pass 1: Orientation normalization & global deskew
    oriented = normalize_orientation(image)
    global_angle = estimate_dominant_angle(oriented) or 0.0
    deskewed = rotate_image_expanded(oriented, global_angle)

    # Pass 2: Local curvature assessment
    grid_angles = estimate_grid_angles(deskewed, rows=3, cols=3)
    spread, usable = compute_angular_spread(grid_angles)

    if spread > max_curvature_threshold:
        reason = (
            f"Document too curved; retake photo (spread: {spread:.2f}°, "
            f"threshold: {max_curvature_threshold:.2f}°)"
        )
        if raise_on_rejection:
            raise CurvatureRejectionError(
                spread_angle=spread,
                threshold=max_curvature_threshold,
            )
        return DeskewResult(
            image=None,
            is_accepted=False,
            global_angle=global_angle,
            grid_angles=grid_angles,
            angular_spread=spread,
            rejection_reason=reason,
        )

    # Pass 2: Mild local dewarp correction
    aligned_color = dewarp_image(deskewed, grid_angles, rows=3, cols=3)

    return DeskewResult(
        image=aligned_color,
        is_accepted=True,
        global_angle=global_angle,
        grid_angles=grid_angles,
        angular_spread=spread,
        rejection_reason=None,
    )
