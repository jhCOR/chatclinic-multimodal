# Visual-Prompt X-ray Benchmark

- Model: `gpt-5-mini`
- Dataset: 100 images (ChestPA={'Normal': 25, 'Abnormal': 25}, Foot={'Normal': 25, 'Abnormal': 25})
- API errors: 0

## Accuracy by condition (positive class = Abnormal)

| Condition | Acc | Δ vs base | Precision | Recall | F1 | Unparsed |
|---|---|---|---|---|---|---|
| baseline | 0.750 | +0.000 | 0.931 | 0.540 | 0.683 | 0 |
| plain_grid | 0.670 | -0.080 | 0.905 | 0.380 | 0.535 | 0 |
| labeled_grid | 0.770 | +0.020 | 0.886 | 0.620 | 0.729 | 0 |
| bone_mesh | 0.610 | -0.140 | 0.762 | 0.320 | 0.451 | 0 |

## Per-category accuracy

| Condition | ChestPA | Foot |
|---|---|---|
| baseline | 0.800 | 0.700 |
| plain_grid | 0.760 | 0.580 |
| labeled_grid | 0.940 | 0.600 |
| bone_mesh | 0.680 | 0.540 |

## Verdict

Best condition: **labeled_grid** (accuracy 0.770, +0.020 vs baseline).

A visual prompt **helps** if a grid/mesh condition beats `baseline`. See `results/samples/` for side-by-side overlay montages.