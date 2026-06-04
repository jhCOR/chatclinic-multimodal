"""Benchmark: does a visual prompt help a VLM classify X-rays?

For every sampled image we render four conditions (baseline, plain_grid,
labeled_grid, bone_mesh) and ask the OpenAI vision model Normal-vs-Abnormal.
The *same* images and *same* prompt wording are used across conditions, so any
accuracy gap is attributable to the overlay. Results land in ``results/``:

  results/metrics.json   machine-readable metrics per condition
  results/report.md      human-readable table + verdict
  results/predictions.csv per-image predictions
  results/samples/       montage PNGs showing the four overlays side by side

Usage:
  conda run -n llm python -m benchmarks.visual_prompt_xray.run_benchmark \
      --n 100 --workers 8
  # render overlays only, no API calls:
  conda run -n llm python -m benchmarks.visual_prompt_xray.run_benchmark --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from PIL import Image

# Support both "python -m benchmarks..." and "python run_benchmark.py".
try:
    from plugins.visual_prompt_tool.logic import render_overlay, to_data_url, MODES
    from benchmarks.visual_prompt_xray.dataset import (
        build_dataset, dataset_summary, Sample,
    )
    from benchmarks.visual_prompt_xray import classify as C
except ModuleNotFoundError:  # pragma: no cover
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from plugins.visual_prompt_tool.logic import render_overlay, to_data_url, MODES
    from benchmarks.visual_prompt_xray.dataset import (
        build_dataset, dataset_summary, Sample,
    )
    from benchmarks.visual_prompt_xray import classify as C

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CONDITIONS = list(MODES)  # baseline, plain_grid, labeled_grid, bone_mesh


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(rows: list[dict]) -> dict:
    """rows: dicts with keys label (0/1) and pred (0/1/None)."""
    valid = [r for r in rows if r["pred"] is not None]
    n, n_valid = len(rows), len(valid)
    tp = sum(1 for r in valid if r["label"] == 1 and r["pred"] == 1)
    tn = sum(1 for r in valid if r["label"] == 0 and r["pred"] == 0)
    fp = sum(1 for r in valid if r["label"] == 0 and r["pred"] == 1)
    fn = sum(1 for r in valid if r["label"] == 1 and r["pred"] == 0)
    acc = (tp + tn) / n_valid if n_valid else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "n": n, "n_valid": n_valid, "n_unparsed": n - n_valid,
        "accuracy": round(acc, 4), "precision": round(prec, 4),
        "recall": round(rec, 4), "f1": round(f1, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def per_category_accuracy(rows: list[dict]) -> dict:
    cats: dict[str, list[dict]] = {}
    for r in rows:
        cats.setdefault(r["category"], []).append(r)
    return {
        cat: round(
            sum(1 for r in rs if r["pred"] is not None and r["pred"] == r["label"])
            / max(1, sum(1 for r in rs if r["pred"] is not None)),
            4,
        )
        for cat, rs in cats.items()
    }


# --------------------------------------------------------------------------- #
# Sample montages (visual sanity check of the overlays)
# --------------------------------------------------------------------------- #
def save_sample_montages(samples: list[Sample], n_per_cat: int = 2) -> list[str]:
    out_dir = RESULTS_DIR / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    seen: dict[str, int] = {}
    saved: list[str] = []
    for s in samples:
        if seen.get(s.category, 0) >= n_per_cat:
            continue
        seen[s.category] = seen.get(s.category, 0) + 1
        tiles = [render_overlay(s.path, mode=m)[0].resize((512, 512)) for m in CONDITIONS]
        montage = Image.new("RGB", (512 * len(tiles), 512), "black")
        for i, t in enumerate(tiles):
            montage.paste(t, (i * 512, 0))
        name = f"{s.finding}_{Path(s.path).stem}.png"
        montage.save(out_dir / name)
        saved.append(str(out_dir / name))
    return saved


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run(n: int, categories, workers: int, model: str | None, dry_run: bool) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    samples = build_dataset(n_total=n, categories=categories)
    summary = dataset_summary(samples)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    montages = save_sample_montages(samples)
    print(f"Saved {len(montages)} sample montage(s) to results/samples/")

    if dry_run:
        print("[dry-run] overlays rendered; skipping API classification.")
        return {"dataset": summary, "dry_run": True, "montages": montages}

    client = C.get_client()

    # Pre-render data URLs once per (sample, condition).
    tasks = []
    for idx, s in enumerate(samples):
        for mode in CONDITIONS:
            tasks.append((idx, s, mode))

    def work(item):
        idx, s, mode = item
        overlay, _ = render_overlay(s.path, mode=mode)
        pred = C.classify(client, to_data_url(overlay), s.category, mode, model=model)
        return {
            "idx": idx, "category": s.category, "finding": s.finding,
            "label": s.label, "label_name": s.label_name, "mode": mode,
            "pred": pred.pred, "pred_name": pred.finding,
            "confidence": pred.confidence, "error": pred.error,
        }

    results: list[dict] = []
    done = 0
    total = len(tasks)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(work, t) for t in tasks]
        for fut in as_completed(futures):
            results.append(fut.result())
            done += 1
            if done % 25 == 0 or done == total:
                print(f"  classified {done}/{total}")

    # Persist per-image predictions.
    with open(RESULTS_DIR / "predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(sorted(results, key=lambda r: (r["idx"], r["mode"])))

    # Metrics per condition.
    metrics = {}
    for mode in CONDITIONS:
        rows = [r for r in results if r["mode"] == mode]
        m = compute_metrics(rows)
        m["per_category_accuracy"] = per_category_accuracy(rows)
        metrics[mode] = m

    n_errors = sum(1 for r in results if r["error"])
    report = {
        "dataset": summary,
        "model": model or C.os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        "conditions": CONDITIONS,
        "metrics": metrics,
        "n_api_errors": n_errors,
        "montages": montages,
    }
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    write_report_md(report)
    print(f"\nWrote results/metrics.json and results/report.md")
    return report


def write_report_md(report: dict) -> None:
    m = report["metrics"]
    base_acc = m["baseline"]["accuracy"]
    lines = [
        "# Visual-Prompt X-ray Benchmark",
        "",
        f"- Model: `{report['model']}`",
        f"- Dataset: {report['dataset']['total']} images "
        f"({', '.join(f'{k}={v}' for k, v in report['dataset']['by_category'].items())})",
        f"- API errors: {report['n_api_errors']}",
        "",
        "## Accuracy by condition (positive class = Abnormal)",
        "",
        "| Condition | Acc | Δ vs base | Precision | Recall | F1 | Unparsed |",
        "|---|---|---|---|---|---|---|",
    ]
    for cond in report["conditions"]:
        x = m[cond]
        delta = x["accuracy"] - base_acc
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"| {cond} | {x['accuracy']:.3f} | {sign}{delta:.3f} | "
            f"{x['precision']:.3f} | {x['recall']:.3f} | {x['f1']:.3f} | {x['n_unparsed']} |"
        )
    lines += ["", "## Per-category accuracy", "",
              "| Condition | " + " | ".join(report["dataset"]["by_category"].keys()) + " |",
              "|---|" + "---|" * len(report["dataset"]["by_category"])]
    for cond in report["conditions"]:
        pca = m[cond]["per_category_accuracy"]
        cells = " | ".join(f"{pca.get(c, 0):.3f}" for c in report["dataset"]["by_category"])
        lines.append(f"| {cond} | {cells} |")

    best = max(report["conditions"], key=lambda c: m[c]["accuracy"])
    best_delta = m[best]["accuracy"] - base_acc
    lines += [
        "",
        "## Verdict",
        "",
        f"Best condition: **{best}** "
        f"(accuracy {m[best]['accuracy']:.3f}, {'+' if best_delta>=0 else ''}{best_delta:.3f} vs baseline).",
        "",
        "A visual prompt **helps** if a grid/mesh condition beats `baseline`. "
        "See `results/samples/` for side-by-side overlay montages.",
    ]
    (RESULTS_DIR / "report.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100, help="total images (balanced)")
    ap.add_argument("--categories", nargs="+", default=["ChestPA", "Foot"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default=None, help="override OPENAI_MODEL")
    ap.add_argument("--dry-run", action="store_true", help="render overlays, no API")
    args = ap.parse_args()
    run(args.n, args.categories, args.workers, args.model, args.dry_run)


if __name__ == "__main__":
    main()
