"""
ChatClinic plugin: X-ray Visual Grounding Tool
Analyzes chest X-ray images via GPT-4o and draws bounding boxes on abnormal regions.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class BoundingBox(BaseModel):
    label: str
    x1: float = Field(..., ge=0.0, le=1.0)
    y1: float = Field(..., ge=0.0, le=1.0)
    x2: float = Field(..., ge=0.0, le=1.0)
    y2: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class XRayAnalysis(BaseModel):
    abnormal: bool
    impression: str
    findings: list[str]
    bounding_boxes: list[BoundingBox] = Field(default_factory=list)


class XRayResult(BaseModel):
    image_path: str
    abnormal: bool
    impression: str
    findings: list[str]
    bounding_boxes: list[dict]
    annotated_image_path: str | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert radiologist specializing in chest X-ray interpretation.

Your task is to analyze the provided chest X-ray image and return a structured JSON response.

Rules:
1. Carefully examine the X-ray for any abnormal findings.
2. Common findings to look for: Atelectasis, Cardiomegaly, Consolidation, Edema,
   Fracture, Lung Lesion, Lung Opacity, Pleural Effusion, Pleural Other, Pneumonia, Pneumothorax.
3. For EACH abnormal finding, provide a normalized bounding box [x1, y1, x2, y2] in range [0.0, 1.0],
   where (x1, y1) is top-left and (x2, y2) is bottom-right of the abnormal region.
4. If the image is normal, set "abnormal" to false and leave "bounding_boxes" empty.
5. Be precise with bounding box coordinates — they will be drawn directly on the image.

Return ONLY valid JSON matching this exact schema (no markdown, no extra text):
{
  "abnormal": true | false,
  "impression": "<one-sentence overall impression>",
  "findings": ["<finding 1>", "<finding 2>", ...],
  "bounding_boxes": [
    {
      "label": "<finding name>",
      "x1": <float 0-1>,
      "y1": <float 0-1>,
      "x2": <float 0-1>,
      "y2": <float 0-1>,
      "confidence": <float 0-1>
    }
  ]
}"""


# ---------------------------------------------------------------------------
# Box colour palette
# ---------------------------------------------------------------------------

COLOURS = [
    "#FF4444", "#FF8800", "#FFDD00", "#44FF44",
    "#00AAFF", "#AA44FF", "#FF44AA", "#00FFCC",
]


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _call_gpt(client: OpenAI, image_path: str) -> XRayAnalysis:
    b64 = _encode_image(image_path)
    ext = Path(image_path).suffix.lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"},
                    },
                    {"type": "text", "text": "Please analyze this chest X-ray and return the JSON result."},
                ],
            },
        ],
        max_tokens=1024,
        temperature=0.0,
    )

    raw = response.choices[0].message.content.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    data = json.loads(raw)
    return XRayAnalysis(**data)


def _draw_boxes(image_path: str, analysis: XRayAnalysis, output_path: str) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for i, box in enumerate(analysis.bounding_boxes):
        colour = COLOURS[i % len(COLOURS)]
        x1 = int(box.x1 * w)
        y1 = int(box.y1 * h)
        x2 = int(box.x2 * w)
        y2 = int(box.y2 * h)

        # Draw bounding box (3px border)
        for offset in range(3):
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline=colour,
            )

        # Draw label background
        label = f"{box.label} ({box.confidence:.0%})"
        bbox = draw.textbbox((x1, y1), label, font=font)
        pad = 3
        draw.rectangle(
            [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
            fill=colour,
        )
        draw.text((x1, y1), label, fill="white", font=font)

    img.save(output_path)


# ---------------------------------------------------------------------------
# Plugin entrypoint
# ---------------------------------------------------------------------------

def execute(payload: dict[str, Any]) -> dict[str, Any]:
    """
    ChatClinic plugin entrypoint.

    payload keys:
        image_path (str, required): path to the chest X-ray image
        save_annotated (bool, optional): save annotated image; default True
        output_dir (str, optional): directory for annotated output
        api_key (str, optional): OpenAI API key (falls back to env var)
    """
    image_path: str = payload.get("image_path", "")
    save_annotated: bool = payload.get("save_annotated", True)
    output_dir: str = payload.get("output_dir", str(Path(__file__).resolve().parents[2] / "outputs"))
    api_key: str = payload.get("api_key") or os.environ.get("OPENAI_API_KEY", "")

    if not image_path or not Path(image_path).exists():
        return XRayResult(
            image_path=image_path,
            abnormal=False,
            impression="",
            findings=[],
            bounding_boxes=[],
            error=f"Image not found: {image_path}",
        ).model_dump()

    client = OpenAI(api_key=api_key)

    try:
        analysis = _call_gpt(client, image_path)
    except Exception as exc:
        return XRayResult(
            image_path=image_path,
            abnormal=False,
            impression="",
            findings=[],
            bounding_boxes=[],
            error=f"GPT analysis failed: {exc}",
        ).model_dump()

    annotated_path: str | None = None
    if save_annotated and analysis.abnormal and analysis.bounding_boxes:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).stem
        annotated_path = str(Path(output_dir) / f"{stem}_grounded.jpg")
        try:
            _draw_boxes(image_path, analysis, annotated_path)
        except Exception as exc:
            annotated_path = None
            print(f"[xray_grounding] Warning: could not draw boxes: {exc}")

    return XRayResult(
        image_path=image_path,
        abnormal=analysis.abnormal,
        impression=analysis.impression,
        findings=analysis.findings,
        bounding_boxes=[bb.model_dump() for bb in analysis.bounding_boxes],
        annotated_image_path=annotated_path,
    ).model_dump()
