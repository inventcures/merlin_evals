"""
Merlin Evaluation Pipeline — Scored evaluation against AbdomenAtlas 3.0 ground truth.

Usage:
    python scripts/eval_pipeline.py --n_cases 5
    python scripts/eval_pipeline.py --n_cases 10 --pathology ovarian --output_dir ./results
    python scripts/eval_pipeline.py --skip_download --data_dir ./atlas_data
    python scripts/eval_pipeline.py --show_metric_guide
"""

import os
import sys
import json
import textwrap
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from scripts.configs import DEVICE, EXPECTED_METRIC_RANGES
from scripts.download_abdomenatlas import (
    download_abdomenatlas_subset,
    load_existing_cases,
    filter_cases_by_pathology,
)
from scripts.inference import run_report_generation, run_five_year_prediction
from scripts.metrics import compute_all_metrics, aggregate_metrics


def print_case_result(case_id: str, gt: str, pred: str, metrics: dict, elapsed: float):
    width = 70
    print(f"\n{'=' * width}")
    print(f"  CASE: {case_id}  |  Inference: {elapsed:.1f}s")
    print(f"{'=' * width}")

    print("\n  GROUND TRUTH (first 400 chars):")
    print(textwrap.fill(
        gt[:400] + ("..." if len(gt) > 400 else ""),
        width=width, initial_indent="  ", subsequent_indent="  ",
    ))

    print("\n  MERLIN OUTPUT (first 400 chars):")
    print(textwrap.fill(
        pred[:400] + ("..." if len(pred) > 400 else ""),
        width=width, initial_indent="  ", subsequent_indent="  ",
    ))

    print("\n  METRICS:")
    groups = {
        "BLEU": ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"],
        "ROUGE": ["ROUGE-1", "ROUGE-2", "ROUGE-L"],
        "BERTScore": ["BERTScore-P", "BERTScore-R", "BERTScore-F1"],
        "Clinical": ["RadGraph-F1"],
    }
    for group, keys in groups.items():
        vals = []
        for k in keys:
            v = metrics.get(k)
            vals.append(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: N/A")
        print(f"    {group:12s} | {' | '.join(vals)}")


def save_results(results: list[dict], summary: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "merlin_eval_results.json")
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "cases": results}, f, indent=2)
    print(f"\n[Save] Full results: {json_path}")

    rows = []
    for r in results:
        row = {"case_id": r["case_id"], "inference_time_s": r.get("inference_time_s")}
        row.update(r.get("metrics", {}))
        rows.append(row)

    csv_path = os.path.join(output_dir, "merlin_eval_summary.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"[Save] Per-case CSV: {csv_path}")

    print(f"\n{'=' * 60}")
    print(f"  AGGREGATE METRICS (n={len(results)} cases)")
    print(f"{'=' * 60}")
    for metric, stats in summary.items():
        if isinstance(stats, dict):
            lo, hi = EXPECTED_METRIC_RANGES.get(metric, (None, None))
            range_str = f"  (expected: {lo:.2f}-{hi:.2f})" if lo is not None else ""
            print(f"  {metric:20s}: {stats['mean']:.4f} +/- {stats['std']:.4f}  (n={stats['n']}){range_str}")


def run_eval_pipeline(
    n_cases: int = 5,
    output_dir: str = "./merlin_eval_results",
    data_dir: str = "./atlas_data",
    skip_download: bool = False,
    pathology: str = "all",
    run_survival: bool = True,
):
    print(f"\n{'=' * 60}")
    print(f"  Merlin Evaluation Pipeline")
    print(f"  Cases: {n_cases} | Pathology: {pathology} | Device: {DEVICE.upper()}")
    print(f"  Estimated time: {n_cases * 15}-{n_cases * 25} min")
    print(f"{'=' * 60}")

    if skip_download:
        cases = load_existing_cases(data_dir, n_cases)
    else:
        cases = download_abdomenatlas_subset(data_dir, n_cases)

    if not cases:
        print("\n[ERROR] No cases found. Check data_dir or download settings.")
        sys.exit(1)

    cases = filter_cases_by_pathology(cases, pathology)
    if not cases:
        print(f"\n[ERROR] No cases match pathology filter '{pathology}'.")
        sys.exit(1)

    all_results = []
    all_metrics = []

    for i, case in enumerate(cases):
        print(f"\n[{i + 1}/{len(cases)}] {case['case_id']}...")

        result = {
            "case_id": case["case_id"],
            "nifti_path": case["nifti_path"],
            "gt_report": case["gt_report"],
        }

        try:
            pred_report, elapsed = run_report_generation(case["nifti_path"])
            result["pred_report"] = pred_report
            result["inference_time_s"] = round(elapsed, 1)
        except Exception as e:
            print(f"  [ERROR] Report generation failed: {e}")
            result["pred_report"] = ""
            result["inference_time_s"] = None
            result["error"] = str(e)

        if result["pred_report"]:
            metrics = compute_all_metrics(result["pred_report"], case["gt_report"])
            result["metrics"] = metrics
            all_metrics.append(metrics)
            print_case_result(
                case["case_id"], case["gt_report"],
                result["pred_report"], metrics,
                result.get("inference_time_s", 0),
            )

        if run_survival:
            try:
                preds = run_five_year_prediction(case["nifti_path"])
                result["five_year_predictions"] = preds
                print("  5-year: " + " | ".join(
                    f"{k.split('(')[1].rstrip(')')}: {v:.2f}" for k, v in preds.items()
                ))
            except Exception as e:
                print(f"  [WARN] 5-year prediction failed: {e}")

        all_results.append(result)

    summary = aggregate_metrics(all_metrics)
    save_results(all_results, summary, output_dir)
    return all_results, summary


METRIC_GUIDE = """
METRIC INTERPRETATION GUIDE
============================

BLEU (Bilingual Evaluation Understudy)
  N-gram precision overlap. Range: 0-1.
  BLEU-4 > 0.10 is reasonable for radiology report generation.
  Caveat: penalizes valid paraphrases.

ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
  Recall-oriented n-gram overlap. ROUGE-L = longest common subsequence.
  ROUGE-L > 0.20 is a reasonable baseline.

BERTScore
  Semantic similarity via contextual embeddings (BiomedBERT).
  BERTScore-F1 > 0.85 indicates semantically similar reports.
  Most robust to clinical text paraphrasing.

RadGraph-F1
  Clinical entity overlap (findings, anatomical locations).
  THE standard clinical metric for radiology NLP.
  Merlin published: partial F1 ~0.27 on internal test set.

EXPECTED RANGES ON AbdomenAtlas 3.0:
  Domain/style shift expected vs Merlin's Stanford training data.
    BLEU-4:        0.08-0.18
    ROUGE-L:       0.15-0.25
    BERTScore-F1:  0.82-0.88
    RadGraph-F1:   0.20-0.30
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merlin Report Generation Evaluation (AbdomenAtlas 3.0)",
    )
    parser.add_argument("--n_cases", type=int, default=5)
    parser.add_argument("--output_dir", default="./merlin_eval_results")
    parser.add_argument("--data_dir", default="./atlas_data")
    parser.add_argument("--skip_download", action="store_true")
    parser.add_argument(
        "--pathology", choices=["all", "ovarian", "liver_nafld"], default="all",
        help="Filter cases by pathology keywords in ground truth reports",
    )
    parser.add_argument("--no_survival", action="store_true")
    parser.add_argument("--show_metric_guide", action="store_true")
    args = parser.parse_args()

    if args.show_metric_guide:
        print(METRIC_GUIDE)
        sys.exit(0)

    missing = []
    for pkg in ["rouge_score", "bert_score", "nltk"]:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[ERROR] Missing: {missing}")
        print(f"Install: uv pip install {' '.join(missing)}")
        sys.exit(1)

    run_eval_pipeline(
        n_cases=args.n_cases,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        skip_download=args.skip_download,
        pathology=args.pathology,
        run_survival=not args.no_survival,
    )
