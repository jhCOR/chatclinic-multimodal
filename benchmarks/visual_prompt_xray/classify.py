"""Binary Normal/Abnormal X-ray classification via the OpenAI vision backend.

Uses the same ``/v1/responses`` endpoint the chatclinic app uses (gpt-5 family),
through the official SDK. The prompt is held *constant* across overlay
conditions so any accuracy delta is attributable to the visual prompt, not the
wording. When the overlay carries a grid/mesh, a short instruction tells the
model how to use it.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from openai import OpenAI

# Region-specific hint for what "abnormal" can look like, kept generic so we are
# not leaking labels — just orienting the model to the modality.
_REGION_HINT = {
    "ChestPA": "a frontal chest radiograph (lungs, heart border, ribs, diaphragm)",
    "Foot": "a foot radiograph (tarsal/metatarsal/phalangeal bones, joints, arch)",
}

_OVERLAY_HINT = {
    "baseline": "",
    "plain_grid": (
        " A reference grid is overlaid on the image. Use it to scan the image "
        "cell by cell so no region is overlooked."
    ),
    "labeled_grid": (
        " A labeled coordinate grid (columns A,B,C... and rows 1,2,3...) is "
        "overlaid. Scan each labeled cell systematically and, in your reasoning, "
        "note the cell of any abnormality you see."
    ),
    "bone_mesh": (
        " A green mesh is drawn over the bone structures and the bone outline is "
        "traced in red. Use this scaffolding to inspect bone cortex, alignment, "
        "and joint spaces closely."
    ),
}

_SYSTEM = (
    "You are a careful radiology screening assistant. You are shown a single "
    "synthetic X-ray image. Decide whether the study is Normal (no findings) or "
    "Abnormal (any pathological finding). You must commit to one of the two "
    "labels even when uncertain."
)


@dataclass
class Prediction:
    pred: int | None        # 0 Normal, 1 Abnormal, None if unparseable
    finding: str            # "Normal"/"Abnormal"/""
    confidence: float | None
    raw: str
    error: str | None = None


def build_prompt(category: str, mode: str) -> str:
    region = _REGION_HINT.get(category, "an X-ray")
    overlay = _OVERLAY_HINT.get(mode, "")
    return (
        f"This is {region}.{overlay}\n\n"
        "Respond with ONLY a JSON object and nothing else:\n"
        '{"finding": "Normal" | "Abnormal", "confidence": <0.0-1.0>}'
    )


def _parse(text: str) -> tuple[int | None, str, float | None]:
    if not text:
        return None, "", None
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    blob = m.group(0) if m else text
    try:
        obj = json.loads(blob)
        finding = str(obj.get("finding", "")).strip().lower()
        conf = obj.get("confidence")
        conf = float(conf) if conf is not None else None
    except Exception:
        finding = text.strip().lower()
        conf = None
    if "abnormal" in finding:
        return 1, "Abnormal", conf
    if "normal" in finding:
        return 0, "Normal", conf
    return None, "", conf


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it (or put it in a .env you source) "
            "before running the benchmark."
        )
    return OpenAI(api_key=api_key)


def classify(
    client: OpenAI,
    data_url: str,
    category: str,
    mode: str,
    model: str | None = None,
    timeout: float = 60.0,
) -> Prediction:
    model = model or os.getenv("OPENAI_MODEL", "gpt-5-mini")
    prompt = build_prompt(category, mode)
    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": _SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": data_url},
                    ],
                },
            ],
            timeout=timeout,
        )
        text = (resp.output_text or "").strip()
    except Exception as exc:  # network / API error -> record, keep going
        return Prediction(None, "", None, "", error=str(exc))

    pred, finding, conf = _parse(text)
    return Prediction(pred=pred, finding=finding, confidence=conf, raw=text)
