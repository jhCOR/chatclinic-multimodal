# xray_grounding_tool

ChatClinic plugin for GPT-4o chest X-ray analysis with visual grounding.

## What it does

1. Accepts a chest X-ray image (JPG or PNG)
2. Sends it to GPT-4o vision with a radiologist-style prompt
3. Receives structured JSON: impression, findings list, and bounding boxes
4. Draws colour-coded bounding boxes on the image for each abnormal region
5. Saves annotated image to `outputs/`

## Plugin structure

```
plugin/
├── tool.json        # ChatClinic plugin descriptor
├── logic.py         # Main entrypoint: execute(payload) -> dict
├── run.py           # CLI wrapper
├── requirements.txt
└── README.md
```

## Payload (logic.py)

| Key | Type | Required | Description |
|---|---|---|---|
| `image_path` | str | yes | Path to X-ray image |
| `save_annotated` | bool | no | Save annotated image (default: true) |
| `output_dir` | str | no | Output directory (default: `../outputs/`) |
| `api_key` | str | no | OpenAI API key (falls back to `OPENAI_API_KEY` env var) |

## Output dict

```json
{
  "image_path": "/path/to/image.jpg",
  "abnormal": true,
  "impression": "Bilateral consolidation consistent with pneumonia.",
  "findings": ["Right lower lobe consolidation", "Left perihilar opacity"],
  "bounding_boxes": [
    {"label": "Consolidation", "x1": 0.55, "y1": 0.55, "x2": 0.85, "y2": 0.90, "confidence": 0.92}
  ],
  "annotated_image_path": "/path/to/outputs/image_grounded.jpg",
  "error": null
}
```

## Quick start

```bash
# Set API key
export OPENAI_API_KEY="sk-..."

# CLI
python plugin/run.py /path/to/chest_xray.jpg

# Python
from plugin.logic import execute
result = execute({"image_path": "/path/to/chest_xray.jpg"})
print(result["impression"])

# Demo with 5 MIMIC samples
python demo.py
```
