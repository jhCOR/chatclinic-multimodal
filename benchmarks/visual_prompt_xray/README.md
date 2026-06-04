# Visual-Prompt X-ray Benchmark

Does drawing a **visual prompt** (grid / bone mesh) on an X-ray help a
multimodal LLM classify it as Normal vs Abnormal? This benchmark measures it.

## Idea

A VLM has no built-in spatial scaffolding when it looks at a radiograph. We
overlay one and check whether it lifts classification accuracy. The same images
and the same prompt wording are used in every condition, so any accuracy gap is
attributable to the overlay alone.

Four conditions:

| Condition | What it draws |
|---|---|
| `baseline` | nothing (raw X-ray) |
| `plain_grid` | evenly spaced 8×8 grid (control for "does *any* grid help") |
| `labeled_grid` | grid + `A1,B2,…` coordinate labels (lets the model name regions) |
| `bone_mesh` | fine mesh confined to bone (high-radiodensity) regions + red bone contour |

The overlay engine lives in the reusable plugin
[`plugins/visual_prompt_tool/logic.py`](../../plugins/visual_prompt_tool/logic.py)
(`render_overlay(src, mode)`), so the same tool is callable from the chatclinic
app and from this harness.

## Data

AIHub 085 주요질환 합성데이터 (X-ray), Validation 원천데이터:
`/data1/jihyeok/aihub_xray_71521/...`. Labels come from the directory name
(`*_Normal` vs `*_Abnormal`/`Hallux_Valgus`/`Flat_Foot`/`Plantar`).
The sampler draws a balanced, deterministic mix of **ChestPA + Foot**
(default 100 images: 25 Normal + 25 Abnormal per region).

## Backend

OpenAI vision via the `/v1/responses` endpoint (the same backend chatclinic
uses), model from `OPENAI_MODEL` (default `gpt-5-mini`).

## Run

```bash
# 1. Render overlays only — no API key needed. Inspect results/samples/*.png
conda run -n llm python -m benchmarks.visual_prompt_xray.run_benchmark --dry-run

# 2. Full benchmark (needs an API key)
export OPENAI_API_KEY=sk-...
conda run -n llm python -m benchmarks.visual_prompt_xray.run_benchmark --n 100 --workers 8

# options
#   --categories ChestPA Foot   pick body regions
#   --model gpt-5-mini          override OPENAI_MODEL
#   --n 40                      smaller/faster run
```

Run from the repo root (`/home/jihyeok/chatclinic-multimodal`). The `llm` conda
env has PIL / numpy / scipy / OpenCV / the OpenAI SDK.

## Outputs (`results/`)

- `metrics.json` — accuracy / precision / recall / F1 per condition, per-category breakdown
- `report.md` — readable table + Δ-vs-baseline + verdict
- `predictions.csv` — every per-image prediction
- `samples/*.png` — montages showing all four overlays side by side

## Reading the result

A visual prompt **helps** if any grid/mesh condition beats `baseline` accuracy.
`report.md` names the best condition and its delta. With synthetic data and a
strong model the absolute numbers matter less than the *ordering* of conditions —
the point is to show whether the visual-prompt tool earns its place.
