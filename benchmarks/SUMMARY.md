# Visual-Prompt X-ray — 작업 정리

X-ray에 **visual prompt(grid / bone mesh)** 를 오버레이하면 medical VLM의
Normal/Abnormal 분류가 좋아지는지 확인하는 툴 + 벤치마크. (2026-06-04)

## 만든 것

### 1. Visual prompt 오버레이 툴 (재사용 가능한 chatclinic 플러그인)
`plugins/visual_prompt_tool/`
- `logic.py` — `render_overlay(src, mode)` 순수 함수 + 플러그인 `execute()` 엔트리포인트
  - `baseline` — 원본 (대조군)
  - `plain_grid` — 8×8 균일 격자
  - `labeled_grid` — 격자 + `A1,B2…` 좌표 라벨
  - `bone_mesh` — cv2 CLAHE+Otsu로 뼈(고밀도) 마스크 추출 → 뼈 영역에만 mesh + 빨강 윤곽선
- `tool.json` — 앱 tool discovery 매니페스트 (정상 등록 확인)

### 2. 벤치마크
`benchmarks/visual_prompt_xray/`
- `dataset.py` — AIHub 71521에서 ChestPA+Foot 균형 샘플링 (시드 고정 → 모든 조건 동일 이미지)
- `classify.py` — OpenAI `/v1/responses` vision (프로젝트와 동일 백엔드, `gpt-5-mini`).
  프롬프트 문구는 조건 간 고정 → 정확도 차이가 오버레이 효과로 귀속됨
- `run_benchmark.py` — 4조건 × 지표(accuracy/precision/recall/F1) + 부위별, `results/`에 저장
- `README.md` — 상세 실행법
- `results/` — `report.md`, `metrics.json`, `predictions.csv`, `samples/*.png`(오버레이 몽타주)

## 실행법

```bash
cd /home/jihyeok/chatclinic-multimodal
export OPENAI_API_KEY=sk-...
conda run -n llm python -m benchmarks.visual_prompt_xray.run_benchmark --n 100 --workers 8
# 옵션: --dry-run(렌더만, 키 불필요) / --categories ChestPA Foot / --model ...
```

- repo 루트에서 `-m`으로 실행 (홈에서 돌리면 `No module named 'benchmarks'`)
- `llm` env에 PIL/numpy/scipy/OpenCV/OpenAI SDK 다 있음

## 결과 (n=100, gpt-5-mini, API 에러 0)

| 조건 | 정확도 | Δ vs base | Precision | Recall | F1 |
|---|---|---|---|---|---|
| baseline | 0.750 | — | 0.931 | 0.540 | 0.683 |
| **labeled_grid** | **0.770** | **+0.020** | 0.886 | 0.620 | **0.729** |
| plain_grid | 0.670 | −0.080 | 0.905 | 0.380 | 0.535 |
| bone_mesh | 0.610 | −0.140 | 0.762 | 0.320 | 0.451 |

부위별:

| 조건 | ChestPA | Foot |
|---|---|---|
| baseline | 0.800 | 0.700 |
| **labeled_grid** | **0.940** | 0.600 |
| plain_grid | 0.760 | 0.580 |
| bone_mesh | 0.680 | 0.540 |

## 결론

- **labeled_grid가 baseline을 이김** — 특히 **ChestPA에서 +0.14 (0.80 → 0.94)**.
  n=48 예비런에서도 같은 방향(+0.083)이라 재현됨.
- **plain_grid는 해(−0.08)인데 labeled_grid는 도움** → 효과의 원인은 격자선이 아니라
  **좌표 라벨**(칸을 호명하며 체계적으로 스캔하게 만듦).
- **bone_mesh 꼴찌(−0.14)** → 촘촘한 mesh가 강조하려던 뼈 디테일을 가리는 역효과.

> **부위에 맞는 visual prompt(특히 ChestPA labeled coordinate grid)는 분류를 의미 있게 향상.
> 단, 라벨 없는 격자나 구조를 가리는 mesh는 역효과.**

## 남은 후보 (선택)
- `bone_mesh`를 윤곽선만(mesh 제거)으로 바꿔 occlusion 가설 직접 검증
- ChestPA에서 `labeled_grid` 칸 수(n=6/8/12) 스윕
