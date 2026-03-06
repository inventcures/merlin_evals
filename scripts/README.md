# Merlin Evaluation & Inference Scripts

## Overview

Two pipelines for evaluating and running Merlin on abdominal CT data:

| Pipeline | Script | Dataset | Ground Truth? | Scoring? |
|----------|--------|---------|--------------|----------|
| **Scored eval** | `eval_pipeline.py` | AbdomenAtlas 3.0 | Yes | BLEU, ROUGE, BERTScore, RadGraph-F1 |
| **Inference-only** | `inference_pipeline.py` | TCIA (TCGA-OV, CPTAC-OV, CT-ORG) | No | Report generation only |

All inference runs on **CPU** (safe for X1 Carbon Gen13).

---

## Install Dependencies

```bash
# Inside activated .venv
# Eval metrics
uv pip install rouge_score bert_score nltk radgraph

# TCIA download + DICOM conversion (only needed for inference_pipeline)
uv pip install tcia_utils dicom2nifti pydicom SimpleITK
```

---

## Scored Evaluation (AbdomenAtlas 3.0)

```bash
# Run 5 cases with all pathologies
python scripts/eval_pipeline.py --n_cases 5

# Filter to liver/NAFLD-related cases
python scripts/eval_pipeline.py --n_cases 20 --pathology liver_nafld

# Filter to ovarian/pelvic cases
python scripts/eval_pipeline.py --n_cases 20 --pathology ovarian

# Re-run without re-downloading
python scripts/eval_pipeline.py --skip_download --data_dir ./atlas_data

# Skip 5-year prediction (faster)
python scripts/eval_pipeline.py --n_cases 5 --no_survival

# Show metric interpretation guide
python scripts/eval_pipeline.py --show_metric_guide
```

### Time Estimates (CPU)

| Cases | Time |
|-------|------|
| 1 | ~20 min |
| 5 | ~1.5-2 hours |
| 10 | ~3-4 hours |
| 20 | ~5-8 hours |

### Output

| File | Contents |
|------|----------|
| `merlin_eval_results.json` | Per-case: GT report, generated report, all metrics, 5-year predictions |
| `merlin_eval_summary.csv` | One row per case with metric values |

---

## Inference-Only (TCIA Datasets)

```bash
# All TCIA collections, 3 patients each
python scripts/inference_pipeline.py --dataset all --max_patients 3

# Ovarian cancer only
python scripts/inference_pipeline.py --dataset TCGA-OV --max_patients 5

# Liver CT proxy (NAFLD)
python scripts/inference_pipeline.py --dataset CT-ORG --max_patients 5

# Re-run on existing NIfTI
python scripts/inference_pipeline.py --skip_download --base_dir ./ct_data
```

### Output

`merlin_inference_results.json` — generated reports + 5-year predictions (no metrics).

---

## Metrics

| Metric | What it measures | Priority |
|--------|-----------------|----------|
| **RadGraph-F1** | Clinical entity overlap (findings, anatomy) | Highest |
| **BERTScore-F1** | Semantic similarity via BiomedBERT | High |
| ROUGE-L | Longest common subsequence | Medium |
| BLEU-4 | 4-gram exact overlap | Low |

### Expected Ranges on AbdomenAtlas 3.0

| Metric | Range |
|--------|-------|
| BLEU-4 | 0.08-0.18 |
| ROUGE-L | 0.15-0.25 |
| BERTScore-F1 | 0.82-0.88 |
| RadGraph-F1 | 0.20-0.30 |

Lower scores reflect domain/style shift between Merlin's Stanford training data and
AbdomenAtlas reporting style — not necessarily model failure.

---

## Module Structure

```
scripts/
├── configs.py                  # Constants, organ systems, disease labels
├── download_abdomenatlas.py    # AbdomenAtlas 3.0 download + pathology filtering
├── download_tcia.py            # TCIA collection download
├── convert_dicom.py            # DICOM → NIfTI conversion
├── inference.py                # Merlin report gen + 5-year prediction
├── metrics.py                  # NLP evaluation metrics
├── eval_pipeline.py            # Scored eval orchestrator
└── inference_pipeline.py       # Inference-only orchestrator
```
