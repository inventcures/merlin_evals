"""
Download TCIA CT collections for inference-only (no ground truth reports available).
Supports TCGA-OV, CPTAC-OV, CT-ORG.
"""

import os

from scripts.configs import TCIA_COLLECTIONS


def download_tcia_collection(
    collection_name: str,
    output_dir: str,
    max_patients: int = 3,
) -> list[str]:
    """
    Download a TCIA collection via tcia_utils.
    No login required for public collections.

    Returns list of patient DICOM directories.
    """
    try:
        from tcia_utils import nbia
    except ImportError:
        raise ImportError("Run: uv pip install tcia_utils")

    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[TCIA] Downloading: {collection_name}")
    print(f"[TCIA] Output: {output_dir}")

    series_df = nbia.getSeries(collection=collection_name, modality="CT")
    if series_df is None or len(series_df) == 0:
        print(f"[WARN] No CT series found for {collection_name}")
        return []

    print(f"[TCIA] Found {len(series_df)} CT series")

    patients = series_df["PatientID"].unique()[:max_patients]
    print(f"[TCIA] Downloading {len(patients)} patients")

    downloaded_dirs = []
    for patient_id in patients:
        patient_series = series_df[series_df["PatientID"] == patient_id]

        if "ImageCount" in patient_series.columns:
            best_series = patient_series.loc[patient_series["ImageCount"].idxmax()]
        else:
            best_series = patient_series.iloc[0]

        series_uid = best_series["SeriesInstanceUID"]
        patient_dir = os.path.join(output_dir, patient_id)
        os.makedirs(patient_dir, exist_ok=True)

        print(f"  Downloading {patient_id}...")
        try:
            nbia.downloadSeries(series_uid, path=patient_dir)
            downloaded_dirs.append(patient_dir)
            print(f"  Done: {patient_dir}")
        except Exception as e:
            print(f"  Failed: {e}")

    return downloaded_dirs


def download_all_tcia_datasets(
    base_dir: str = "./ct_data",
    max_patients: int = 3,
) -> dict:
    """
    Download all configured TCIA collections.

    Returns:
        {collection_name: {"dirs": [patient_dirs], "indication": str}}
    """
    results = {}
    for name, config in TCIA_COLLECTIONS.items():
        output_dir = os.path.join(base_dir, "dicom", name)
        dirs = download_tcia_collection(
            collection_name=name,
            output_dir=output_dir,
            max_patients=max_patients,
        )
        results[name] = {
            "dirs": dirs,
            "indication": config["indication"],
        }
        print(f"  Indication: {config['indication']}")

    return results
