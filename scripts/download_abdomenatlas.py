"""
Download and manage AbdomenAtlas 3.0 cases from HuggingFace.

Repo structure:
  - Reports: in AbdomenAtlas3.0MiniWithMeta.csv (9,261 rows with structured + narrative)
  - Images:  in image_only/*.tar.gz archives (~14 GB each, 232 cases per archive)

Strategy:
  1. Download the metadata CSV (small, instant)
  2. Download the first tar.gz archive (14 GB, one-time)
  3. Extract only the requested N cases from the archive
  4. Pair with reports from CSV
"""

import os
import sys
import tarfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from scripts.configs import ABDOMENATLAS_REPO_ID, PATHOLOGY_KEYWORDS

METADATA_FILENAME = "AbdomenAtlas3.0MiniWithMeta.csv"
FIRST_ARCHIVE = "image_only/AbdomenAtlas3_images_BDMAP_BDMAP_00000001_BDMAP_00000232.tar.gz"


def _download_metadata(data_dir: str) -> pd.DataFrame:
    """Download and load the metadata CSV with reports."""
    from huggingface_hub import hf_hub_download

    csv_path = os.path.join(data_dir, METADATA_FILENAME)
    if not os.path.exists(csv_path):
        print("[Download] Fetching metadata CSV...")
        hf_hub_download(
            repo_id=ABDOMENATLAS_REPO_ID,
            filename=METADATA_FILENAME,
            repo_type="dataset",
            local_dir=data_dir,
        )

    return pd.read_csv(csv_path)


def _download_and_extract_archive(
    data_dir: str,
    archive_name: str,
    case_ids: list[str],
) -> dict[str, str]:
    """
    Download a tar.gz archive and extract only the requested case NIfTI files.

    Returns: {case_id: nifti_path}
    """
    from huggingface_hub import hf_hub_download

    archive_path = os.path.join(data_dir, archive_name)
    extract_dir = os.path.join(data_dir, "images")
    os.makedirs(extract_dir, exist_ok=True)

    already_extracted = {}
    for cid in case_ids:
        nifti_path = os.path.join(extract_dir, cid, "ct.nii.gz")
        if not os.path.exists(nifti_path):
            alt = list(Path(os.path.join(extract_dir, cid)).glob("*.nii.gz")) if os.path.isdir(os.path.join(extract_dir, cid)) else []
            if alt:
                already_extracted[cid] = str(alt[0])
        else:
            already_extracted[cid] = nifti_path

    remaining = [cid for cid in case_ids if cid not in already_extracted]
    if not remaining:
        print(f"[Download] All {len(case_ids)} cases already extracted")
        return already_extracted

    if not os.path.exists(archive_path):
        print(f"[Download] Downloading archive (~14 GB, one-time)...")
        print(f"[Download] {archive_name}")
        hf_hub_download(
            repo_id=ABDOMENATLAS_REPO_ID,
            filename=archive_name,
            repo_type="dataset",
            local_dir=data_dir,
        )

    print(f"[Extract] Extracting {len(remaining)} cases from archive...")
    remaining_set = set(remaining)
    extracted = dict(already_extracted)

    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tqdm(tar, desc="Scanning archive"):
            parts = Path(member.name).parts
            if len(parts) < 2:
                continue

            case_id = parts[0] if parts[0].startswith("BDMAP_") else parts[1] if len(parts) > 1 and parts[1].startswith("BDMAP_") else None
            if case_id is None or case_id not in remaining_set:
                continue

            if member.name.endswith(".nii.gz") and "ct.nii.gz" in member.name.lower():
                tar.extract(member, path=extract_dir)
                nifti_path = os.path.join(extract_dir, member.name)
                extracted[case_id] = nifti_path
                remaining_set.discard(case_id)
                print(f"  Extracted: {case_id}")

            if not remaining_set:
                break

    if remaining_set:
        print(f"[Extract] Trying broader NIfTI match for {len(remaining_set)} remaining cases...")
        with tarfile.open(archive_path, "r:gz") as tar:
            for member in tar:
                if not member.name.endswith(".nii.gz"):
                    continue
                parts = Path(member.name).parts
                for part in parts:
                    if part.startswith("BDMAP_") and part in remaining_set:
                        tar.extract(member, path=extract_dir)
                        extracted[part] = os.path.join(extract_dir, member.name)
                        remaining_set.discard(part)
                        print(f"  Extracted: {part}")
                        break
                if not remaining_set:
                    break

    return extracted


def download_abdomenatlas_subset(data_dir: str, n_cases: int = 5) -> list[dict]:
    """
    Download a subset of AbdomenAtlas 3.0 from HuggingFace.

    Steps:
      1. Download metadata CSV (reports)
      2. Download first image archive (~14 GB, contains 232 cases)
      3. Extract only the requested N cases
      4. Pair NIfTI files with reports from CSV

    Returns list of dicts with keys:
        case_id, nifti_path, gt_report, gt_report_structured
    """
    os.makedirs(data_dir, exist_ok=True)
    print(f"\n[Download] AbdomenAtlas 3.0 subset ({n_cases} cases)")
    print(f"[Download] Saving to: {data_dir}")

    try:
        meta_df = _download_metadata(data_dir)
    except Exception as e:
        print(f"[ERROR] Could not download metadata: {e}")
        sys.exit(1)

    print(f"[Download] Metadata: {len(meta_df)} cases with reports")

    case_ids = [
        row["BDMAP ID"]
        for _, row in meta_df.iterrows()
        if pd.notna(row.get("narrative report"))
    ]

    target_ids = case_ids[:min(n_cases, 232)]
    print(f"[Download] Targeting {len(target_ids)} cases from first archive")

    try:
        extracted = _download_and_extract_archive(data_dir, FIRST_ARCHIVE, target_ids)
    except Exception as e:
        print(f"[ERROR] Archive download/extract failed: {e}")
        sys.exit(1)

    meta_lookup = meta_df.set_index("BDMAP ID")
    cases = []
    for case_id in target_ids:
        nifti_path = extracted.get(case_id)
        if not nifti_path or not os.path.exists(nifti_path):
            continue

        row = meta_lookup.loc[case_id]
        gt_narrative = str(row.get("narrative report", "")).strip()
        gt_structured = str(row.get("structured report", "")).strip()

        if gt_narrative and gt_narrative != "nan":
            cases.append({
                "case_id": case_id,
                "nifti_path": nifti_path,
                "gt_report": gt_narrative,
                "gt_report_structured": gt_structured if gt_structured != "nan" else "",
            })

    print(f"[Download] Ready: {len(cases)} cases with CT + ground truth")
    return cases


def load_existing_cases(data_dir: str, n_cases: int) -> list[dict]:
    """Load cases from already-downloaded AbdomenAtlas data."""
    cases = []
    data_path = Path(data_dir)
    if not data_path.exists():
        return cases

    csv_path = data_path / METADATA_FILENAME
    if not csv_path.exists():
        print(f"[Load] No metadata CSV found at {csv_path}")
        return _load_from_dirs(data_dir, n_cases)

    meta_df = pd.read_csv(csv_path)
    meta_lookup = meta_df.set_index("BDMAP ID")

    images_dir = data_path / "images"
    if not images_dir.exists():
        print(f"[Load] No images directory at {images_dir}")
        return _load_from_dirs(data_dir, n_cases)

    case_dirs = sorted([
        d for d in images_dir.iterdir()
        if d.is_dir() and d.name.startswith("BDMAP_")
    ])[:n_cases]

    for case_dir in case_dirs:
        case_id = case_dir.name
        nifti_files = list(case_dir.rglob("*.nii.gz"))
        if not nifti_files:
            continue

        if case_id not in meta_lookup.index:
            continue

        row = meta_lookup.loc[case_id]
        gt_narrative = str(row.get("narrative report", "")).strip()
        gt_structured = str(row.get("structured report", "")).strip()

        if gt_narrative and gt_narrative != "nan":
            cases.append({
                "case_id": case_id,
                "nifti_path": str(nifti_files[0]),
                "gt_report": gt_narrative,
                "gt_report_structured": gt_structured if gt_structured != "nan" else "",
            })

    print(f"[Load] Found {len(cases)} cases with CT + ground truth in {data_dir}")
    return cases


def _load_from_dirs(data_dir: str, n_cases: int) -> list[dict]:
    """Fallback: load from per-case directories with report text files."""
    cases = []
    data_path = Path(data_dir)

    case_dirs = sorted([
        d for d in data_path.iterdir()
        if d.is_dir() and (d.name.startswith("bdmap_") or d.name.startswith("BDMAP_"))
    ])[:n_cases]

    for case_dir in case_dirs:
        nifti_files = list(case_dir.rglob("*.nii.gz"))
        if not nifti_files:
            continue

        gt_narrative = ""
        for name in ["report_narrative.txt", "report.txt", "findings.txt"]:
            path = case_dir / name
            if path.exists():
                gt_narrative = path.read_text().strip()
                break

        gt_structured = ""
        for name in ["report_structured.txt", "structured_report.txt"]:
            path = case_dir / name
            if path.exists():
                gt_structured = path.read_text().strip()
                break

        if gt_narrative:
            cases.append({
                "case_id": case_dir.name,
                "nifti_path": str(nifti_files[0]),
                "gt_report": gt_narrative,
                "gt_report_structured": gt_structured,
            })

    print(f"[Load] Found {len(cases)} cases in {data_dir}")
    return cases


def filter_cases_by_pathology(cases: list[dict], pathology: str) -> list[dict]:
    """
    Filter cases by keyword matching on ground truth report text.

    Args:
        cases: list of case dicts (must have 'gt_report' key)
        pathology: 'all' | 'ovarian' | 'liver_nafld'

    Returns:
        Filtered list (or original if pathology='all')
    """
    if pathology == "all":
        return cases

    keywords = PATHOLOGY_KEYWORDS.get(pathology, [])
    if not keywords:
        print(f"[WARN] Unknown pathology filter '{pathology}', returning all cases")
        return cases

    filtered = []
    for case in cases:
        report_lower = case["gt_report"].lower()
        if any(kw.lower() in report_lower for kw in keywords):
            filtered.append(case)

    print(f"[Filter] {pathology}: {len(filtered)}/{len(cases)} cases match")
    return filtered
