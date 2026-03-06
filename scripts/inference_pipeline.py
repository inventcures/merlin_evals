"""
Merlin Inference-Only Pipeline — TCIA datasets (no ground truth scoring).

Downloads TCIA CT collections, converts DICOM to NIfTI, runs Merlin inference.
Generated reports are saved but NOT scored (these datasets lack paired reports).

Usage:
    python scripts/inference_pipeline.py --dataset TCGA-OV --max_patients 3
    python scripts/inference_pipeline.py --dataset all --max_patients 2
    python scripts/inference_pipeline.py --skip_download --base_dir ./ct_data
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.configs import DEVICE, TCIA_COLLECTIONS
from scripts.download_tcia import download_tcia_collection, download_all_tcia_datasets
from scripts.convert_dicom import convert_dicom_to_nifti, convert_all_datasets
from scripts.inference import run_report_generation, run_five_year_prediction


def _load_existing_nifti(base_dir: str, dataset: str, max_patients: int) -> list[dict]:
    """Load already-converted NIfTI files from the expected directory."""
    nifti_dir = os.path.join(base_dir, "nifti", dataset)
    if not os.path.exists(nifti_dir):
        return []

    cases = []
    for f in sorted(os.listdir(nifti_dir)):
        if f.endswith(".nii.gz") and len(cases) < max_patients:
            cases.append({
                "patient_id": f.replace(".nii.gz", ""),
                "nifti_path": os.path.join(nifti_dir, f),
                "dataset": dataset,
                "indication": TCIA_COLLECTIONS.get(dataset, {}).get("indication", dataset),
            })
    return cases


def run_inference_pipeline(
    dataset: str = "all",
    max_patients: int = 3,
    base_dir: str = "./ct_data",
    output_dir: str = "./merlin_inference_results",
    skip_download: bool = False,
    run_survival: bool = True,
):
    collections = (
        list(TCIA_COLLECTIONS.keys()) if dataset == "all" else [dataset]
    )

    print(f"\n{'=' * 60}")
    print(f"  Merlin Inference Pipeline (no ground truth)")
    print(f"  Datasets: {collections} | Device: {DEVICE.upper()}")
    print(f"{'=' * 60}")

    all_cases = []

    for collection in collections:
        if skip_download:
            cases = _load_existing_nifti(base_dir, collection, max_patients)
            if not cases:
                print(f"[WARN] No NIfTI found for {collection} in {base_dir}/nifti/{collection}")
                continue
        else:
            print(f"\n[1/3] Downloading {collection}...")
            output = os.path.join(base_dir, "dicom", collection)
            dirs = download_tcia_collection(collection, output, max_patients)

            if not dirs:
                print(f"[WARN] No data downloaded for {collection}")
                continue

            print(f"\n[2/3] Converting DICOM -> NIfTI...")
            dataset_dirs = {collection: {
                "dirs": dirs,
                "indication": TCIA_COLLECTIONS[collection]["indication"],
            }}
            nifti_map = convert_all_datasets(dataset_dirs, base_dir)
            cases = nifti_map.get(collection, [])

        all_cases.extend(cases)

    if not all_cases:
        print("[ERROR] No cases ready for inference.")
        sys.exit(1)

    print(f"\n[3/3] Running Merlin inference on {len(all_cases)} cases...")

    results = {}
    for i, case in enumerate(all_cases):
        pid = case["patient_id"]
        ds = case["dataset"]
        print(f"\n[{i + 1}/{len(all_cases)}] {ds}/{pid}")
        print(f"  Indication: {case['indication']}")

        result = {
            "patient_id": pid,
            "dataset": ds,
            "indication": case["indication"],
            "nifti_path": case["nifti_path"],
            "has_ground_truth": False,
        }

        try:
            report, elapsed = run_report_generation(case["nifti_path"])
            result["report"] = report
            result["inference_time_s"] = round(elapsed, 1)
            print(f"  Report ({elapsed:.0f}s): {report[:200]}...")
        except Exception as e:
            result["report"] = f"ERROR: {e}"
            print(f"  [ERROR] Report generation: {e}")

        if run_survival:
            try:
                preds = run_five_year_prediction(case["nifti_path"])
                result["five_year_predictions"] = preds
            except Exception as e:
                print(f"  [WARN] 5-year prediction: {e}")

        results[f"{ds}/{pid}"] = result

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "merlin_inference_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  Done. Results: {json_path}")
    print(f"  Note: No ground truth available — reports not scored.")
    print(f"{'=' * 60}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merlin Inference on TCIA Datasets (no ground truth)",
    )
    parser.add_argument(
        "--dataset", default="all",
        choices=["all"] + list(TCIA_COLLECTIONS.keys()),
    )
    parser.add_argument("--max_patients", type=int, default=3)
    parser.add_argument("--base_dir", default="./ct_data")
    parser.add_argument("--output_dir", default="./merlin_inference_results")
    parser.add_argument("--skip_download", action="store_true")
    parser.add_argument("--no_survival", action="store_true")
    args = parser.parse_args()

    missing = []
    for pkg in ["tcia_utils", "dicom2nifti"]:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            missing.append(pkg)
    if missing and not args.skip_download:
        print(f"[ERROR] Missing: {missing}")
        print(f"Install: uv pip install {' '.join(missing)}")
        sys.exit(1)

    run_inference_pipeline(
        dataset=args.dataset,
        max_patients=args.max_patients,
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        skip_download=args.skip_download,
        run_survival=not args.no_survival,
    )
