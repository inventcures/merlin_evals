"""
DICOM to NIfTI conversion for TCIA-downloaded CT data.
Merlin expects NIfTI format (.nii.gz).
"""

import os
from pathlib import Path


def convert_dicom_to_nifti(dicom_dir: str, output_nifti_path: str) -> bool:
    """
    Convert a directory of DICOM slices to a single NIfTI file.

    Returns True on success.
    """
    try:
        import dicom2nifti
        import dicom2nifti.settings as settings
        settings.disable_validate_orthogonal()
        settings.disable_validate_slice_increment()
    except ImportError:
        raise ImportError("Run: uv pip install dicom2nifti")

    output_dir = os.path.dirname(output_nifti_path)
    os.makedirs(output_dir, exist_ok=True)

    try:
        dicom2nifti.convert_directory(
            dicom_dir, output_dir,
            compression=True, reorient=True,
        )
    except Exception as e:
        print(f"  [ERROR] Conversion failed for {dicom_dir}: {e}")
        return False

    nifti_files = list(Path(output_dir).glob("*.nii.gz"))
    if not nifti_files:
        print(f"  [ERROR] No NIfTI produced from {dicom_dir}")
        return False

    nifti_files[0].rename(output_nifti_path)
    print(f"  NIfTI saved: {output_nifti_path}")
    return True


def convert_all_datasets(dataset_dirs: dict, base_dir: str = "./ct_data") -> dict:
    """
    Convert all downloaded DICOM datasets to NIfTI.

    Args:
        dataset_dirs: {dataset_name: {"dirs": [patient_dirs], "indication": str}}

    Returns:
        {dataset_name: [{"patient_id", "nifti_path", "indication", "dataset"}, ...]}
    """
    nifti_map = {}
    for dataset_name, info in dataset_dirs.items():
        nifti_map[dataset_name] = []
        for patient_dir in info["dirs"]:
            patient_id = os.path.basename(patient_dir)
            nifti_path = os.path.join(
                base_dir, "nifti", dataset_name, f"{patient_id}.nii.gz"
            )

            print(f"Converting {dataset_name}/{patient_id}...")
            if convert_dicom_to_nifti(patient_dir, nifti_path):
                nifti_map[dataset_name].append({
                    "patient_id": patient_id,
                    "nifti_path": nifti_path,
                    "indication": info["indication"],
                    "dataset": dataset_name,
                })

    return nifti_map
