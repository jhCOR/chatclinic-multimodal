"""Visual-prompt overlays for medical X-ray images.

This tool draws *visual prompts* on top of an X-ray so a multimodal LLM has
explicit spatial scaffolding to reason over. Three overlay families are
provided:

- ``plain_grid``    : an evenly spaced grid (control condition).
- ``labeled_grid``  : the same grid with column/row coordinate labels
                      (A1, B2, ...) so the model can name regions in words.
- ``bone_mesh``     : a fine mesh drawn *only* on bone structures (high
                      radiodensity regions), plus the bone contour outline,
                      concentrating the scaffolding where pathology usually
                      lives.

The rendering functions are pure (PIL / numpy / OpenCV only, no app imports)
so they can be reused by the benchmark harness and by the chatclinic plugin
runtime alike. ``execute`` is the plugin entrypoint.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:  # OpenCV powers bone detection; degrade gracefully if unavailable.
    import cv2

    _HAS_CV2 = True
except Exception:  # pragma: no cover - cv2 expected in the `llm` env
    _HAS_CV2 = False


MODES = ("baseline", "plain_grid", "labeled_grid", "bone_mesh")

# Overlay colors (RGB). Chosen to stay legible on grayscale X-rays.
_GRID_COLOR = (0, 200, 255)        # cyan grid lines
_LABEL_COLOR = (255, 230, 0)       # yellow coordinate labels
_MESH_COLOR = (0, 255, 120)        # green bone mesh
_CONTOUR_COLOR = (255, 60, 60)     # red bone contour


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
def _to_rgb(src: "str | Path | Image.Image") -> Image.Image:
    """Return an RGB copy of ``src`` (path or PIL image)."""
    if isinstance(src, Image.Image):
        img = src
    else:
        img = Image.open(Path(src).expanduser())
    return img.convert("RGB")


def _load_font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _col_label(idx: int) -> str:
    """0 -> A, 1 -> B, ... 25 -> Z, 26 -> AA."""
    label = ""
    idx += 1
    while idx > 0:
        idx, rem = divmod(idx - 1, 26)
        label = chr(ord("A") + rem) + label
    return label


# --------------------------------------------------------------------------- #
# Grid overlays
# --------------------------------------------------------------------------- #
def make_plain_grid(rgb: Image.Image, n: int = 8, width: int = 2) -> Image.Image:
    """Evenly spaced grid, ``n`` x ``n`` cells."""
    out = rgb.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    for i in range(1, n):
        x = round(i * w / n)
        y = round(i * h / n)
        draw.line([(x, 0), (x, h)], fill=_GRID_COLOR, width=width)
        draw.line([(0, y), (w, y)], fill=_GRID_COLOR, width=width)
    return out


def make_labeled_grid(rgb: Image.Image, n: int = 8, width: int = 2) -> Image.Image:
    """Grid plus ``COL+ROW`` coordinate labels centered in each cell."""
    out = make_plain_grid(rgb, n=n, width=width)
    draw = ImageDraw.Draw(out)
    w, h = out.size
    cell_w, cell_h = w / n, h / n
    font = _load_font(max(12, int(min(cell_w, cell_h) * 0.22)))
    for r in range(n):
        for c in range(n):
            label = f"{_col_label(c)}{r + 1}"
            cx = int(c * cell_w + 4)
            cy = int(r * cell_h + 4)
            # dark halo for legibility on bright regions
            draw.text((cx + 1, cy + 1), label, fill=(0, 0, 0), font=font)
            draw.text((cx, cy), label, fill=_LABEL_COLOR, font=font)
    return out


# --------------------------------------------------------------------------- #
# Bone-focused mesh
# --------------------------------------------------------------------------- #
def compute_bone_mask(gray: np.ndarray) -> np.ndarray:
    """Boolean mask of likely bone (high-radiodensity) regions.

    X-ray bone appears bright. We CLAHE-enhance contrast, Otsu-threshold the
    bright structures, and clean up speckle with morphology. Falls back to a
    plain percentile threshold when OpenCV is missing.
    """
    g = gray.astype(np.uint8)
    if not _HAS_CV2:
        thresh = np.percentile(g, 70)
        return g >= thresh

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(g)
    # Otsu on the enhanced image isolates the brightest (densest) tissue.
    _, mask = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask > 0


def make_bone_mesh(
    rgb: Image.Image,
    spacing: int = 22,
    line_width: int = 1,
) -> tuple[Image.Image, dict[str, Any]]:
    """Fine mesh confined to bone regions, plus the bone contour outline.

    Returns the overlaid image and a small stats dict (bone coverage, contour
    count) that the benchmark / plugin can surface.
    """
    out = rgb.copy()
    gray = np.asarray(out.convert("L"))
    h, w = gray.shape
    bone = compute_bone_mask(gray)
    coverage = float(bone.mean())

    # Build a regular mesh-line pattern, then keep only the bone pixels.
    mesh_lines = np.zeros((h, w), dtype=bool)
    mesh_lines[::spacing, :] = True
    mesh_lines[:, ::spacing] = True
    if line_width > 1:
        for off in range(1, line_width):
            mesh_lines[off::spacing, :] = True
            mesh_lines[:, off::spacing] = True

    if _HAS_CV2:
        # Dilate bone slightly so the mesh hugs (not just covers) the bone.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        bone_dil = cv2.dilate(bone.astype(np.uint8), kernel, iterations=1) > 0
    else:
        bone_dil = bone

    mesh = mesh_lines & bone_dil
    arr = np.asarray(out).copy()
    arr[mesh] = _MESH_COLOR

    n_contours = 0
    if _HAS_CV2:
        contours, _ = cv2.findContours(
            bone.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # Drop tiny specks (< 0.05% of area) to keep the outline meaningful.
        min_area = 0.0005 * h * w
        contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        n_contours = len(contours)
        cv2.drawContours(arr, contours, -1, _CONTOUR_COLOR, 2)

    stats = {
        "bone_coverage": round(coverage, 4),
        "contour_count": n_contours,
        "mesh_spacing_px": spacing,
        "cv2": _HAS_CV2,
    }
    return Image.fromarray(arr), stats


# --------------------------------------------------------------------------- #
# Unified entry
# --------------------------------------------------------------------------- #
def render_overlay(
    src: "str | Path | Image.Image",
    mode: str = "labeled_grid",
    grid_n: int = 8,
    mesh_spacing: int = 22,
) -> tuple[Image.Image, dict[str, Any]]:
    """Render ``mode`` over ``src``. Returns (RGB image, stats)."""
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}; choose from {MODES}.")
    rgb = _to_rgb(src)
    if mode == "baseline":
        return rgb, {}
    if mode == "plain_grid":
        return make_plain_grid(rgb, n=grid_n), {"grid_n": grid_n}
    if mode == "labeled_grid":
        return make_labeled_grid(rgb, n=grid_n), {"grid_n": grid_n}
    if mode == "bone_mesh":
        return make_bone_mesh(rgb, spacing=mesh_spacing)
    raise AssertionError("unreachable")


def to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def to_data_url(img: Image.Image) -> str:
    encoded = base64.b64encode(to_png_bytes(img)).decode("ascii")
    return f"data:image/png;base64,{encoded}"


# --------------------------------------------------------------------------- #
# Plugin entrypoint
# --------------------------------------------------------------------------- #
def execute(payload: dict[str, object]) -> dict[str, object]:
    """Plugin entrypoint: overlay a visual prompt on one image.

    payload:
      image_path (str, required)
      mode       (str, default "labeled_grid")
      grid_n     (int, default 8)
      mesh_spacing (int, default 22)
    """
    image_path = str(payload.get("image_path") or "").strip()
    if not image_path:
        raise ValueError("`image_path` is required.")
    mode = str(payload.get("mode") or "labeled_grid").strip()
    grid_n = int(payload.get("grid_n") or 8)
    mesh_spacing = int(payload.get("mesh_spacing") or 22)

    overlay, stats = render_overlay(
        image_path, mode=mode, grid_n=grid_n, mesh_spacing=mesh_spacing
    )
    return {
        "mode": mode,
        "stats": stats,
        "preview_data_url": to_data_url(overlay),
        "file_name": Path(image_path).name,
        "used_tools": ["visual_prompt_tool"],
    }
