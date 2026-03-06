"""
Merlin Evaluation on Modal (GPU Serverless).

Runs the full Merlin eval pipeline on an A10G GPU via Modal,
bypassing local OOM issues. Model runs in bfloat16 — no INT8
quantization needed with 24 GB VRAM.

Usage:
    # One-time setup
    pip install modal
    modal setup
    modal secret create huggingface-token HF_TOKEN=hf_xxx

    # Download data to Modal volume (first run only)
    modal run scripts/modal_eval.py::download_cases

    # Run full eval
    modal run scripts/modal_eval.py
"""

import json
import os
import sys

import modal

CACHE_DIR = "/cache"
DATA_DIR = "/data"

ARCHIVES = [
    "image_only/AbdomenAtlas3_images_BDMAP_BDMAP_00000465_BDMAP_00000696.tar.gz",
    "image_only/AbdomenAtlas3_images_BDMAP_BDMAP_00003249_BDMAP_00003480.tar.gz",
    "image_only/AbdomenAtlas3_images_BDMAP_BDMAP_00003481_BDMAP_00003712.tar.gz",
    "image_only/AbdomenAtlas3_images_BDMAP_BDMAP_00004177_BDMAP_00004408.tar.gz",
    "image_only/AbdomenAtlas3_images_BDMAP_BDMAP_00008817_BDMAP_00009048.tar.gz",
]

CASES = [
    {"case_id": "BDMAP_00000547", "pathology": "ovarian", "archive_idx": 0},
    {"case_id": "BDMAP_00003256", "pathology": "ovarian", "archive_idx": 1},
    {"case_id": "BDMAP_00003432", "pathology": "ovarian", "archive_idx": 1},
    {"case_id": "BDMAP_00003687", "pathology": "ovarian", "archive_idx": 2},
    {"case_id": "BDMAP_00004336", "pathology": "ovarian", "archive_idx": 3},
    {"case_id": "BDMAP_00008992", "pathology": "ovarian", "archive_idx": 4},
    {"case_id": "BDMAP_00000479", "pathology": "liver_nafld", "archive_idx": 0},
    {"case_id": "BDMAP_00003258", "pathology": "liver_nafld", "archive_idx": 1},
    {"case_id": "BDMAP_00003489", "pathology": "liver_nafld", "archive_idx": 2},
    {"case_id": "BDMAP_00004185", "pathology": "liver_nafld", "archive_idx": 3},
]

ABDOMENATLAS_REPO_ID = "AbdomenAtlas/AbdomenAtlas3.0"
METADATA_FILENAME = "AbdomenAtlas3.0MiniWithMeta.csv"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.1.2",
        "transformers>=4.38.2",
        "torchvision>=0.16.2",
        "peft>=0.10.0",
        "accelerate>=0.34.2",
        "monai>=1.3.0",
        "nibabel",
        "numpy>=1.26.4",
        "einops",
        "sentencepiece",
        "protobuf",
        "huggingface_hub",
        "pandas",
        "nltk",
        "rouge_score",
        "bert_score",
        "radgraph",
        "rich",
        "tqdm",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .run_commands("python -c \"import nltk; nltk.download('punkt_tab')\"")
    .env({
        "HF_HUB_CACHE": f"{CACHE_DIR}/huggingface",
        "PYTHONPATH": "/app",
    })
    .add_local_dir(
        "merlin", remote_path="/app/merlin",
        ignore=["models/checkpoints", "**/*.pt", "**/*.bin", "**/*.safetensors"],
    )
    .add_local_dir("scripts", remote_path="/app/scripts")
)

model_cache = modal.Volume.from_name("merlin-model-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("merlin-data", create_if_missing=True)
app = modal.App("merlin-eval", image=image)


def _log(msg: str):
    print(msg, flush=True)


@app.function(
    volumes={DATA_DIR: data_vol},
    secrets=[modal.Secret.from_name("huggingface-token")],
    timeout=1800,
)
def download_single_archive(archive_idx: int) -> list[str]:
    """Download one archive, extract its target cases, return list of extracted case_ids."""
    import tarfile
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    archive_name = ARCHIVES[archive_idx]
    case_ids_needed = {
        c["case_id"] for c in CASES if c["archive_idx"] == archive_idx
    }

    images_dir = os.path.join(DATA_DIR, "images")
    os.makedirs(images_dir, exist_ok=True)

    already = set()
    for cid in case_ids_needed:
        if os.path.exists(os.path.join(images_dir, cid, "ct.nii.gz")):
            already.add(cid)

    if already == case_ids_needed:
        _log(f"[Archive {archive_idx}] All cases already extracted: {already}")
        return list(already)

    remaining = case_ids_needed - already
    _log(f"[Archive {archive_idx}] Downloading {archive_name}...")
    _log(f"[Archive {archive_idx}] Need: {remaining}")

    archive_path = os.path.join(DATA_DIR, archive_name)
    if not os.path.exists(archive_path):
        os.makedirs(os.path.dirname(archive_path), exist_ok=True)
        hf_hub_download(
            repo_id=ABDOMENATLAS_REPO_ID,
            filename=archive_name,
            repo_type="dataset",
            local_dir=DATA_DIR,
        )
    _log(f"[Archive {archive_idx}] Download complete, extracting...")

    still_need = set(remaining)
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar:
            if not still_need:
                break
            if not member.name.endswith(".nii.gz"):
                continue

            parts = Path(member.name).parts
            matched_id = None
            for part in parts:
                if part in still_need:
                    matched_id = part
                    break
            if matched_id is None:
                continue

            dest_dir = os.path.join(images_dir, matched_id)
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, "ct.nii.gz")

            fileobj = tar.extractfile(member)
            if fileobj is not None:
                with open(dest_path, "wb") as f:
                    f.write(fileobj.read())
                still_need.discard(matched_id)
                _log(f"  Extracted: {matched_id}")

    if still_need:
        _log(f"[Archive {archive_idx}] WARN: could not find: {still_need}")

    os.remove(archive_path)
    _log(f"[Archive {archive_idx}] Cleaned up archive")

    data_vol.commit()
    extracted = list(case_ids_needed - still_need)
    _log(f"[Archive {archive_idx}] Done: {extracted}")
    return extracted


@app.function(
    volumes={DATA_DIR: data_vol},
    secrets=[modal.Secret.from_name("huggingface-token")],
    timeout=300,
)
def download_metadata():
    """Download the metadata CSV with ground truth reports."""
    from huggingface_hub import hf_hub_download

    csv_path = os.path.join(DATA_DIR, METADATA_FILENAME)
    if not os.path.exists(csv_path):
        _log("[Metadata] Downloading CSV...")
        hf_hub_download(
            repo_id=ABDOMENATLAS_REPO_ID,
            filename=METADATA_FILENAME,
            repo_type="dataset",
            local_dir=DATA_DIR,
        )
        data_vol.commit()
        _log("[Metadata] Done")
    else:
        _log("[Metadata] Already exists")


@app.function(
    volumes={DATA_DIR: data_vol},
    secrets=[modal.Secret.from_name("huggingface-token")],
    timeout=3600,
)
def download_all_data():
    """Download all archives sequentially + metadata CSV. Runs entirely server-side."""
    import tarfile
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    images_dir = os.path.join(DATA_DIR, "images")
    os.makedirs(images_dir, exist_ok=True)

    csv_path = os.path.join(DATA_DIR, METADATA_FILENAME)
    if not os.path.exists(csv_path):
        _log("[Metadata] Downloading CSV...")
        hf_hub_download(
            repo_id=ABDOMENATLAS_REPO_ID,
            filename=METADATA_FILENAME,
            repo_type="dataset",
            local_dir=DATA_DIR,
        )
        _log("[Metadata] Done")

    archive_indices = sorted({c["archive_idx"] for c in CASES})
    for idx in archive_indices:
        archive_name = ARCHIVES[idx]
        case_ids_needed = {c["case_id"] for c in CASES if c["archive_idx"] == idx}

        already = {
            cid for cid in case_ids_needed
            if os.path.exists(os.path.join(images_dir, cid, "ct.nii.gz"))
        }
        if already == case_ids_needed:
            _log(f"[Archive {idx}] Already have: {already}")
            continue

        remaining = case_ids_needed - already
        _log(f"[Archive {idx}] Downloading {archive_name} for {remaining}...")

        archive_path = os.path.join(DATA_DIR, archive_name)
        if not os.path.exists(archive_path):
            os.makedirs(os.path.dirname(archive_path), exist_ok=True)
            hf_hub_download(
                repo_id=ABDOMENATLAS_REPO_ID,
                filename=archive_name,
                repo_type="dataset",
                local_dir=DATA_DIR,
            )
        _log(f"[Archive {idx}] Download complete, extracting...")

        still_need = set(remaining)
        with tarfile.open(archive_path, "r:gz") as tar:
            for member in tar:
                if not still_need:
                    break
                if not member.name.endswith(".nii.gz"):
                    continue
                parts = Path(member.name).parts
                matched_id = None
                for part in parts:
                    if part in still_need:
                        matched_id = part
                        break
                if matched_id is None:
                    continue
                dest_dir = os.path.join(images_dir, matched_id)
                os.makedirs(dest_dir, exist_ok=True)
                fileobj = tar.extractfile(member)
                if fileobj is not None:
                    with open(os.path.join(dest_dir, "ct.nii.gz"), "wb") as f:
                        f.write(fileobj.read())
                    still_need.discard(matched_id)
                    _log(f"  Extracted: {matched_id}")

        if still_need:
            _log(f"[Archive {idx}] WARN: missing {still_need}")

        os.remove(archive_path)
        _log(f"[Archive {idx}] Cleaned up archive")
        data_vol.commit()

    _log("[Download] All data ready.")


@app.cls(
    gpu="A10G",
    volumes={CACHE_DIR: model_cache, DATA_DIR: data_vol},
    timeout=1800,
    scaledown_window=600,
)
class MerlinInference:
    @modal.enter()
    def load_model(self):
        import torch

        sys.path.insert(0, "/app")
        os.chdir("/app")

        ckpt_cache = os.path.join(CACHE_DIR, "checkpoints")
        ckpt_link = "/app/merlin/models/checkpoints"
        os.makedirs(ckpt_cache, exist_ok=True)
        os.makedirs(os.path.dirname(ckpt_link), exist_ok=True)
        if not os.path.exists(ckpt_link):
            os.symlink(ckpt_cache, ckpt_link)

        import scripts.configs as cfg
        cfg.DEVICE = "cuda"

        from merlin import Merlin

        _log("[Merlin] Loading report generation model...")
        self.report_model = Merlin(RadiologyReport=True)
        self.report_model.eval()
        self.report_model.to("cuda")

        td = self.report_model.model.decode_text
        td.text_decoder = td.text_decoder.merge_and_unload()
        td.text_decoder.gradient_checkpointing_disable()
        _log("[Merlin] Report model ready (GPU, LoRA merged)")

        _log("[Merlin] Loading 5-year prediction model...")
        self.survival_model = Merlin(FiveYearPred=True)
        self.survival_model.eval()
        self.survival_model.to("cuda")
        _log("[Merlin] Survival model ready")

        model_cache.commit()

    @modal.method()
    def generate_report(self, case_id: str, gen_config: dict | None = None) -> dict:
        """Generate radiology report + 5-year prediction for a single case.

        Args:
            case_id: AbdomenAtlas case identifier.
            gen_config: Generation config dict with keys: mode, do_sample,
                num_beams, temperature, top_p, top_k, repetition_penalty,
                max_new_tokens. If None, uses baseline per-organ greedy.
        """
        import time

        import torch
        from transformers import StoppingCriteria

        from scripts.configs import (
            ORGAN_SYSTEMS, FIVE_YEAR_DISEASES, WHOLE_REPORT_PROMPT,
        )

        if gen_config is None:
            gen_config = {
                "mode": "per_organ",
                "do_sample": False,
                "num_beams": 1,
                "repetition_penalty": 1.2,
                "max_new_tokens": 128,
            }

        class EosListStoppingCriteria(StoppingCriteria):
            def __init__(self, eos_sequence=None):
                self.eos_sequence = eos_sequence or [48134]

            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                last_ids = input_ids[:, -len(self.eos_sequence):].tolist()
                return self.eos_sequence in last_ids

        nifti_path = os.path.join(DATA_DIR, "images", case_id, "ct.nii.gz")
        if not os.path.exists(nifti_path):
            return {"case_id": case_id, "error": f"NIfTI not found: {nifti_path}"}

        from merlin.data import DataLoader

        datalist = [{"image": nifti_path}]
        dataloader = DataLoader(
            datalist=datalist,
            cache_dir="/tmp/merlin_cache",
            batchsize=1,
            shuffle=False,
            num_workers=0,
        )
        image = None
        for batch in dataloader:
            image = batch["image"].to("cuda")
            break

        if image is None:
            return {"case_id": case_id, "error": "Failed to load image"}

        mode = gen_config.get("mode", "per_organ")
        gen_kwargs = {
            k: v for k, v in gen_config.items()
            if k not in ("mode",)
        }
        gen_kwargs["stopping_criteria"] = [EosListStoppingCriteria()]

        t0 = time.time()

        if mode == "whole_report":
            with torch.no_grad():
                generations = self.report_model.generate(
                    image, [WHOLE_REPORT_PROMPT], **gen_kwargs,
                )
            pred_report = generations[0].split("###")[0].strip()
        else:
            report_parts = []
            for organ_system in ORGAN_SYSTEMS:
                prefix = f"Generate a radiology report for {organ_system}###\n"
                with torch.no_grad():
                    generations = self.report_model.generate(
                        image, [prefix], **gen_kwargs,
                    )
                text = generations[0].split("###")[0].strip()
                if text:
                    report_parts.append(text)
            pred_report = " ".join(report_parts)

        report_elapsed = time.time() - t0

        five_year = {}
        try:
            with torch.no_grad():
                logits = self.survival_model(image)
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            five_year = {d: float(p) for d, p in zip(FIVE_YEAR_DISEASES, probs)}
        except Exception as e:
            five_year = {"error": str(e)}

        return {
            "case_id": case_id,
            "pred_report": pred_report,
            "inference_time_s": round(report_elapsed, 1),
            "five_year_predictions": five_year,
            "gen_config_name": gen_config.get("_name", "baseline"),
        }


def _load_ground_truth() -> dict:
    """Load ground truth reports from metadata CSV. Runs inside Modal container."""
    import pandas as pd

    csv_path = os.path.join(DATA_DIR, METADATA_FILENAME)
    if not os.path.exists(csv_path):
        return {}

    df = pd.read_csv(csv_path)
    lookup = {}
    for _, row in df.iterrows():
        case_id = row.get("BDMAP ID", "")
        narrative = str(row.get("narrative report", "")).strip()
        if narrative and narrative != "nan":
            lookup[case_id] = narrative
    return lookup


@app.function(
    volumes={DATA_DIR: data_vol},
    secrets=[modal.Secret.from_name("huggingface-token")],
    timeout=600,
)
def compute_metrics_remote(results: list) -> dict:
    """Compute NLP metrics for all cases (runs on CPU, no GPU needed)."""
    sys.path.insert(0, "/app")
    os.chdir("/app")

    from scripts.metrics import compute_all_metrics, aggregate_metrics

    gt_lookup = _load_ground_truth()

    all_metrics = []
    for result in results:
        case_id = result["case_id"]
        pred = result.get("pred_report", "")
        gt = gt_lookup.get(case_id, "")

        if not pred or not gt:
            result["gt_report"] = gt
            result["metrics"] = {"error": "missing prediction or ground truth"}
            continue

        _log(f"[Metrics] Computing for {case_id}...")
        metrics = compute_all_metrics(pred, gt)
        result["gt_report"] = gt
        result["metrics"] = metrics
        all_metrics.append(metrics)

    summary = aggregate_metrics(all_metrics)
    return {"cases": results, "summary": summary}


@app.local_entrypoint()
def run_eval(output_dir: str = "./merlin_eval_results_modal"):
    """Run full eval pipeline: download -> inference -> metrics."""
    print("\n" + "=" * 60)
    print("  Merlin Evaluation Pipeline (Modal GPU)")
    print(f"  Cases: {len(CASES)} | GPU: A10G | Precision: native")
    print("=" * 60)

    print("\n[1/3] Ensuring data is available on Modal volume...")
    download_all_data.remote()

    print("\n[2/3] Running GPU inference...")
    inferrer = MerlinInference()
    results = []
    for i, case in enumerate(CASES):
        print(f"  [{i + 1}/{len(CASES)}] {case['case_id']} ({case['pathology']})...")
        result = inferrer.generate_report.remote(case["case_id"])
        if result.get("error"):
            print(f"    ERROR: {result['error']}")
        else:
            elapsed = result.get("inference_time_s", 0)
            report_preview = result.get("pred_report", "")[:150]
            print(f"    Done ({elapsed}s): {report_preview}...")
        result["pathology"] = case["pathology"]
        results.append(result)

    print("\n[3/3] Computing metrics...")
    scored = compute_metrics_remote.remote(results)

    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "merlin_eval_results.json")
    with open(json_path, "w") as f:
        json.dump(scored, f, indent=2)
    print(f"\n[Save] Full results: {json_path}")

    import csv
    csv_path = os.path.join(output_dir, "merlin_eval_summary.csv")
    cases_data = scored.get("cases", [])
    if cases_data:
        fieldnames = ["case_id", "pathology", "inference_time_s"]
        metric_keys = set()
        for c in cases_data:
            if isinstance(c.get("metrics"), dict):
                metric_keys.update(
                    k for k, v in c["metrics"].items()
                    if isinstance(v, (int, float))
                )
        fieldnames.extend(sorted(metric_keys))

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for c in cases_data:
                row = {
                    "case_id": c["case_id"],
                    "pathology": c.get("pathology", ""),
                    "inference_time_s": c.get("inference_time_s"),
                }
                if isinstance(c.get("metrics"), dict):
                    row.update({
                        k: v for k, v in c["metrics"].items()
                        if isinstance(v, (int, float))
                    })
                writer.writerow(row)
        print(f"[Save] Per-case CSV: {csv_path}")

    summary = scored.get("summary", {})
    if summary:
        from scripts.configs import EXPECTED_METRIC_RANGES
        print(f"\n{'=' * 60}")
        print(f"  AGGREGATE METRICS (n={len(cases_data)} cases)")
        print(f"{'=' * 60}")
        for metric, stats in summary.items():
            if isinstance(stats, dict) and "mean" in stats:
                lo, hi = EXPECTED_METRIC_RANGES.get(metric, (None, None))
                range_str = f"  (expected: {lo:.2f}-{hi:.2f})" if lo is not None else ""
                print(f"  {metric:20s}: {stats['mean']:.4f} +/- {stats['std']:.4f}  (n={stats['n']}){range_str}")

    print(f"\n{'=' * 60}")
    print(f"  Done. Results saved to {output_dir}/")
    print(f"{'=' * 60}")


@app.local_entrypoint()
def run_ablation(output_dir: str = "./merlin_eval_results_modal/ablation"):
    """Run ablation study: test all generation configs on all cases."""
    import csv

    from scripts.configs import GENERATION_CONFIGS

    print("\n" + "=" * 60)
    print("  Merlin Ablation Study (Modal GPU)")
    print(f"  Configs: {len(GENERATION_CONFIGS)} | Cases: {len(CASES)}")
    print("=" * 60)

    print("\n[0/3] Ensuring data is available...")
    download_all_data.remote()

    inferrer = MerlinInference()
    all_summaries = []

    for config_name, config in GENERATION_CONFIGS.items():
        config_with_name = {**config, "_name": config_name}
        config_dir = os.path.join(output_dir, config_name)
        os.makedirs(config_dir, exist_ok=True)

        print(f"\n[Config: {config_name}] mode={config['mode']}, "
              f"do_sample={config.get('do_sample', False)}")

        results = []
        for i, case in enumerate(CASES):
            cid = case["case_id"]
            print(f"  [{i + 1}/{len(CASES)}] {cid}...")
            result = inferrer.generate_report.remote(cid, config_with_name)
            if result.get("error"):
                print(f"    ERROR: {result['error']}")
            else:
                preview = result.get("pred_report", "")[:120]
                print(f"    ({result.get('inference_time_s', 0)}s): {preview}...")
            result["pathology"] = case["pathology"]
            results.append(result)

        scored = compute_metrics_remote.remote(results)

        json_path = os.path.join(config_dir, "results.json")
        with open(json_path, "w") as f:
            json.dump(scored, f, indent=2)

        summary = scored.get("summary", {})
        summary_row = {"config": config_name, "mode": config["mode"]}
        for metric, stats in summary.items():
            if isinstance(stats, dict) and "mean" in stats:
                summary_row[f"{metric}_mean"] = round(stats["mean"], 4)
                summary_row[f"{metric}_std"] = round(stats["std"], 4)
        all_summaries.append(summary_row)
        print(f"  [Saved] {json_path}")

    csv_path = os.path.join(output_dir, "ablation_summary.csv")
    if all_summaries:
        fieldnames = list(all_summaries[0].keys())
        for row in all_summaries[1:]:
            for k in row:
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_summaries)

    print(f"\n{'=' * 60}")
    print(f"  Ablation Summary")
    print(f"{'=' * 60}")
    for row in all_summaries:
        bleu = row.get("BLEU-4_mean", "N/A")
        rouge = row.get("ROUGE-L_mean", "N/A")
        print(f"  {row['config']:25s} BLEU-4={bleu}  ROUGE-L={rouge}")
    print(f"\n  Results: {output_dir}/")
    print(f"  Summary: {csv_path}")
    print(f"{'=' * 60}")
