"""
Extract downsampled NIfTI volumes + thumbnail PNGs for the web viewer.

Runs on Modal CPU — no GPU needed. Reads full-resolution NIfTIs from
the merlin-data volume, downsamples to ~256x256xD (int16), and saves
compressed .nii.gz files suitable for browser-side NiiVue rendering.

Usage:
    modal run scripts/extract_web_data.py
    modal volume get merlin-data web/ ./merlin_eval_results_modal/web/
"""

import os

import modal

CACHE_DIR = "/cache"
DATA_DIR = "/data"

CASES = [
    {"case_id": "BDMAP_00000547", "pathology": "ovarian"},
    {"case_id": "BDMAP_00003256", "pathology": "ovarian"},
    {"case_id": "BDMAP_00003432", "pathology": "ovarian"},
    {"case_id": "BDMAP_00003687", "pathology": "ovarian"},
    {"case_id": "BDMAP_00004336", "pathology": "ovarian"},
    {"case_id": "BDMAP_00008992", "pathology": "ovarian"},
    {"case_id": "BDMAP_00000479", "pathology": "liver_nafld"},
    {"case_id": "BDMAP_00003258", "pathology": "liver_nafld"},
    {"case_id": "BDMAP_00003489", "pathology": "liver_nafld"},
    {"case_id": "BDMAP_00004185", "pathology": "liver_nafld"},
]

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("nibabel", "numpy>=1.26.4", "scipy", "Pillow")
)

data_vol = modal.Volume.from_name("merlin-data", create_if_missing=True)
app = modal.App("merlin-web-extract", image=image)


@app.function(volumes={DATA_DIR: data_vol}, timeout=600)
def extract_web_data():
    """Downsample NIfTIs for web viewer + extract thumbnail PNGs."""
    import nibabel as nib
    import numpy as np
    from PIL import Image
    from scipy.ndimage import zoom

    web_dir = os.path.join(DATA_DIR, "web")
    os.makedirs(web_dir, exist_ok=True)

    for case in CASES:
        cid = case["case_id"]
        src = os.path.join(DATA_DIR, "images", cid, "ct.nii.gz")
        if not os.path.exists(src):
            print(f"[SKIP] {cid}: NIfTI not found at {src}")
            continue

        print(f"[{cid}] Loading...")
        nifti = nib.load(src)
        data = nifti.get_fdata()
        affine = nifti.affine
        print(f"  Original shape: {data.shape}, dtype: {data.dtype}")

        target = (256, 256, min(data.shape[2], 128))
        factors = tuple(t / s for t, s in zip(target, data.shape))
        print(f"  Downsampling by factors {tuple(f'{f:.3f}' for f in factors)}...")
        downsampled = zoom(data, factors, order=1)

        scale = np.diag([1 / f for f in factors] + [1.0])
        new_affine = affine @ scale

        downsampled_int16 = np.clip(downsampled, -32768, 32767).astype(np.int16)
        out_nifti = nib.Nifti1Image(downsampled_int16, new_affine)

        out_path = os.path.join(web_dir, f"{cid}.nii.gz")
        nib.save(out_nifti, out_path)
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"  Saved: {out_path} ({size_mb:.1f} MB, shape {downsampled_int16.shape})")

        mid_z = data.shape[2] // 2
        axial = data[:, :, mid_z]
        axial = np.clip(axial, -1000, 1000)
        axial = ((axial + 1000) / 2000 * 255).astype(np.uint8)
        thumb_path = os.path.join(web_dir, f"{cid}_thumb.png")
        Image.fromarray(axial).save(thumb_path)
        print(f"  Thumbnail: {thumb_path}")

    data_vol.commit()
    print("\n[Done] All web assets extracted.")


@app.local_entrypoint()
def main():
    extract_web_data.remote()
    print("\nTo download locally:")
    print("  modal volume get merlin-data web/ ./merlin_eval_results_modal/web/")
