"""Sample a small balanced Normal/Abnormal X-ray set from the AIHub 71521 data.

Categories are mixed (ChestPA + Foot by default) to check that the visual-prompt
benefit generalizes across body regions. Sampling is deterministic given the
seed so every overlay condition sees the *same* images.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from pathlib import Path

# AIHub 085 주요질환 합성데이터 (X-ray), Validation 원천데이터 root.
DATA_ROOT = Path(
    "/data1/jihyeok/aihub_xray_71521/085.주요질환_이미지_합성데이터(X-ray)"
    "/01-1.정식개방데이터/Validation/01.원천데이터"
)

# category -> {"normal": [dirs], "abnormal": [dirs]}
CATEGORIES: dict[str, dict[str, list[str]]] = {
    "ChestPA": {
        "normal": ["VS_01.ChestPA_Normal"],
        "abnormal": ["VS_01.ChestPA_Abnormal"],
    },
    "Foot": {
        "normal": ["VS_05.Foot_Normal"],
        "abnormal": [
            "VS_05.Foot_Hallux_Valgus",
            "VS_05.Foot_Flat_Foot",
            "VS_05.Foot_Plantar",
        ],
    },
}


@dataclass
class Sample:
    path: str
    category: str        # ChestPA / Foot
    label: int           # 0 = Normal, 1 = Abnormal
    label_name: str      # "Normal" / "Abnormal"
    finding: str         # subcategory dir, e.g. Foot_Hallux_Valgus


def _list_pngs(dirname: str) -> list[Path]:
    d = DATA_ROOT / dirname
    if not d.is_dir():
        return []
    return sorted(d.glob("*.png"))


def build_dataset(
    n_total: int = 100,
    categories: list[str] | None = None,
    seed: int = 17,
) -> list[Sample]:
    """Return ``n_total`` samples, balanced Normal/Abnormal within each category."""
    categories = categories or ["ChestPA", "Foot"]
    rng = random.Random(seed)
    per_cat = max(2, n_total // len(categories))
    per_class = per_cat // 2  # half normal, half abnormal

    samples: list[Sample] = []
    for cat in categories:
        spec = CATEGORIES[cat]

        normal_pool = [(d, p) for d in spec["normal"] for p in _list_pngs(d)]
        rng.shuffle(normal_pool)
        for d, p in normal_pool[:per_class]:
            samples.append(Sample(str(p), cat, 0, "Normal", d.split(".")[-1]))

        # Spread abnormal picks across its subcategories round-robin.
        ab_by_dir = {d: _list_pngs(d) for d in spec["abnormal"]}
        for pool in ab_by_dir.values():
            rng.shuffle(pool)
        picked = 0
        idx = 0
        ab_dirs = spec["abnormal"]
        while picked < per_class:
            d = ab_dirs[idx % len(ab_dirs)]
            idx += 1
            pool = ab_by_dir[d]
            cursor = idx // len(ab_dirs) - 1
            if cursor < len(pool):
                p = pool[cursor]
                samples.append(Sample(str(p), cat, 1, "Abnormal", d.split(".")[-1]))
                picked += 1
            if all(idx // len(ab_dirs) >= len(pool) for pool in ab_by_dir.values()):
                break

    rng.shuffle(samples)
    return samples


def dataset_summary(samples: list[Sample]) -> dict:
    summary: dict = {"total": len(samples), "by_category": {}, "by_finding": {}}
    for s in samples:
        c = summary["by_category"].setdefault(s.category, {"Normal": 0, "Abnormal": 0})
        c[s.label_name] += 1
        summary["by_finding"][s.finding] = summary["by_finding"].get(s.finding, 0) + 1
    return summary


if __name__ == "__main__":
    import json

    ds = build_dataset()
    print(json.dumps(dataset_summary(ds), ensure_ascii=False, indent=2))
    for s in ds[:5]:
        print(asdict(s))
