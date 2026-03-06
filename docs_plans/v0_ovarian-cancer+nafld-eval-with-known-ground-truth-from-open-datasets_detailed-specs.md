# V0 Spec: Ovarian Cancer + NAFLD Evaluation with Ground Truth — Merlin VLM

## 1. Problem Statement

We need to evaluate Merlin's radiology report generation quality on abdominal CT scans
for two clinical domains:

1. **Ovarian cancer** — pelvic/abdominal CT with relevant findings
2. **MASH/NASH/NAFLD** — liver-focused CT with steatosis/fibrosis indicators

The evaluation requires **ground truth radiology reports** to score Merlin's output
against expert reference text using NLP metrics. All inference must run on **CPU**
(ThinkPad X1 Carbon Gen13, no discrete GPU).

### Why this matters

Merlin (Stanford MIMI) was trained on 15,331 internal Stanford abdominal CTs. Before
using it in any clinical or research pipeline, we need empirical evidence of its report
generation quality on external data — especially for specific pathologies.

---

## 2. Dataset Strategy

### 2.1 Scored Evaluation: AbdomenAtlas 3.0

The **only** public abdominal CT dataset with paired expert-reviewed radiology reports.

| Property | Value |
|----------|-------|
| Source | HuggingFace: `AbdomenAtlas/AbdomenAtlas3.0` |
| Size | 9,262 abdominal CT volumes (NIfTI) |
| Reports | Narrative + structured (per organ system) |
| Annotations | Per-voxel tumor masks (liver, kidney, pancreas) |
| Institutions | 138 medical centers, 17 source datasets |
| Paper | https://arxiv.org/abs/2501.04678 |

Each case provides:
- `{case_id}.nii.gz` — CT volume
- `report_narrative.txt` — free-text clinical description
- `report_structured.txt` — findings organized by organ

**Pathology filtering**: Since there is no ovarian-cancer-only or NAFLD-only subset,
we filter AbdomenAtlas cases by keyword matching on ground truth reports:

```python
PATHOLOGY_KEYWORDS = {
    "ovarian": [
        "ovary", "ovarian", "adnexal", "pelvic mass", "oophor",
        "fallopian", "peritoneal carcinomatosis", "omental cake",
        "ascites",  # common in advanced ovarian cancer
    ],
    "liver_nafld": [
        "steatosis", "fatty liver", "nafld", "nash", "mash",
        "hepatic steatosis", "fatty infiltration", "cirrhosis",
        "fibrosis", "hepatomegaly", "liver lesion", "hepatic lesion",
        "hepatocellular", "liver metast",
    ],
}
```

This provides an approximation — not a curated disease-specific cohort, but the best
available with paired ground truth.

### 2.2 Inference-Only: TCIA Collections

These datasets have CT images but **no paired radiology reports**. We generate Merlin
reports for these but cannot score them against ground truth.

| Collection | Indication | Modality | Format | Access |
|-----------|------------|----------|--------|--------|
| TCGA-OV | High-grade serous ovarian cancer | CT | DICOM | TCIA (public) |
| CPTAC-OV | Ovarian serous cystadenocarcinoma | CT | DICOM | TCIA (public) |
| CT-ORG | Abdominal CT with liver lesions | CT | DICOM | TCIA (public) |

TCIA data requires:
1. Download via `tcia_utils` Python API (no login for public collections)
2. DICOM → NIfTI conversion via `dicom2nifti`

### 2.3 What Doesn't Exist

There is **no publicly available CT dataset specifically labeled for NAFLD/MASH staging**.
NAFLD is staged by liver biopsy (NAS score), not CT. CT can show gross steatosis
(Hounsfield units < 40 HU in liver) but cannot distinguish NAFL vs NASH vs fibrosis
stages. For NAFLD-specific imaging research, consider:
- MRI-PDFF (proton density fat fraction) datasets — not CT
- NASH CRN (Clinical Research Network) — requires special access

---

## 3. Merlin API Reference

### 3.1 Correct API (from source code)

The Merlin class (`merlin/models/load.py`) exposes:

```python
from merlin import Merlin

# Report generation
model = Merlin(RadiologyReport=True)
model.eval()
model.to("cpu")

# CORRECT: model.generate(image_tensor, text_labels, **kwargs)
reports = model.generate(
    image_tensor,              # (B, 1, 224, 224, 160) float32
    ["prompt###\n"] * B,       # list[str], one prompt per image
    do_sample=False,
    num_beams=1,
    repetition_penalty=1.2,
    max_new_tokens=128,
    stopping_criteria=[EosListStoppingCriteria()],
)

# 5-year disease prediction
model = Merlin(FiveYearPred=True)
logits = model(image_tensor)  # (B, 6)
probs = torch.sigmoid(logits)
```

### 3.2 Bugs in Existing Draft Code

| Bug | Wrong | Correct |
|-----|-------|---------|
| Method name | `model.generate_report(img)` | `model.generate(img, text_labels, **kwargs)` |
| Dataset class | `from merlin.data import MerlinDataset` | `from merlin.data import DataLoader` |
| Missing prompt | `model.generate_report(img)` (no prompt) | `model.generate(img, ["Generate a radiology report for liver###\n"])` |
| Disease labels | Hypertension, Diabetes, AFib, Heart Failure, CKD, COPD | CVD, IHD, HTN, DM, CKD, CLD |
| Stopping criteria | Missing | `EosListStoppingCriteria(eos_sequence=[48134])` |

### 3.3 Data Loading

```python
from merlin.data import DataLoader

datalist = [{"image": "/path/to/scan.nii.gz"}]
dataloader = DataLoader(
    datalist=datalist,
    cache_dir="./cache",
    batchsize=1,       # CPU: keep at 1
    shuffle=False,
    num_workers=0,      # CPU: keep at 0
)

for batch in dataloader:
    image_tensor = batch["image"].to("cpu")  # (B, 1, 224, 224, 160)
```

Internal transforms (applied automatically by DataLoader):
- LoadImaged → EnsureChannelFirst → Orient(RAS) → Spacing(1.5, 1.5, 3mm)
- ScaleIntensityRange(-1000, 1000 → 0, 1) → SpatialPad(224,224,160)
- CenterSpatialCrop(224,224,160) → ToTensor

### 3.4 Organ-System Prompts

Merlin generates reports per organ system. The full report is a concatenation of
per-system outputs. From `documentation/report_generation_demo.py`:

```python
ORGAN_SYSTEMS = {
    "lower thorax":    "lower thorax|lower chest|lung bases",
    "liver":           "liver|liver and biliary tree|biliary system",
    "gallbladder":     "gallbladder",
    "spleen":          "spleen",
    "pancreas":        "pancreas",
    "adrenal glands":  "adrenal glands|adrenals",
    "kidneys":         "kidneys|kidneys and ureters|gu|kidneys, ureters",
    "bowel":           "bowel|gastrointestinal tract|gi|bowel/mesentery",
    "peritoneum":      "peritoneal space|peritoneal cavity|abdominal wall|peritoneum",
    "pelvic":          "pelvic organs|bladder|prostate and seminal vesicles|pelvis|uterus and ovaries",
    "circulatory":     "vasculature",
    "lymph nodes":     "lymph nodes",
    "musculoskeletal": "musculoskeletal|bones",
}

# Prompt format:
prompt = f"Generate a radiology report for {organ_system}###\n"
```

### 3.5 Five-Year Disease Labels

From `documentation/demo.py:118-125`:

```python
FIVE_YEAR_DISEASES = [
    "Cardiovascular Disease (CVD)",
    "Ischemic Heart Disease (IHD)",
    "Hypertension (HTN)",
    "Diabetes Mellitus (DM)",
    "Chronic Kidney Disease (CKD)",
    "Chronic Liver Disease (CLD)",
]
```

---

## 4. File Organization

All evaluation/inference scripts move from `merlin/utils/` to `scripts/` at repo root.

### 4.1 New Structure

```
scripts/
├── __init__.py                   # Package marker
├── configs.py                    # Constants, labels, organ systems, keywords
├── download_abdomenatlas.py      # AbdomenAtlas 3.0 download from HuggingFace
├── download_tcia.py              # TCIA collection download (TCGA-OV, CPTAC-OV, CT-ORG)
├── convert_dicom.py              # DICOM → NIfTI conversion
├── inference.py                  # Merlin inference (report gen + 5-year pred)
├── metrics.py                    # NLP evaluation metrics
├── eval_pipeline.py              # Scored eval orchestrator (AbdomenAtlas)
├── inference_pipeline.py         # Inference-only orchestrator (TCIA)
└── README.md                     # Setup, usage, metric interpretation
```

### 4.2 Files Removed

```
merlin/utils/merlin_pipeline.py              # DELETED (replaced by scripts/)
merlin/utils/files/merlin_eval_pipeline.py   # DELETED (replaced by scripts/)
merlin/utils/files/MERLIN_EVAL_README.md     # DELETED (replaced by scripts/README.md)
```

### 4.3 Rationale

- `merlin/` is the installable library package — user-facing evaluation scripts don't
  belong here
- `scripts/` is a standard convention for repo-level tooling
- Separate modules by concern (download, inference, metrics, orchestration) instead
  of monolithic 500-line files

---

## 5. Code Architecture

### 5.1 Module Dependency Graph

```
eval_pipeline.py ──────┬──→ download_abdomenatlas.py ──→ configs.py
                       ├──→ inference.py ──────────────→ configs.py
                       └──→ metrics.py

inference_pipeline.py ─┬──→ download_tcia.py ──────────→ configs.py
                       ├──→ convert_dicom.py
                       └──→ inference.py ──────────────→ configs.py
```

### 5.2 Module Specifications

#### `configs.py`

```python
DEVICE = "cpu"

ABDOMENATLAS_REPO_ID = "AbdomenAtlas/AbdomenAtlas3.0"

TCIA_COLLECTIONS = {
    "TCGA-OV":  {"indication": "High-grade serous ovarian cancer"},
    "CPTAC-OV": {"indication": "Ovarian serous cystadenocarcinoma"},
    "CT-ORG":   {"indication": "Abdominal CT with liver lesions (NAFLD proxy)"},
}

ORGAN_SYSTEMS = { ... }  # 13 systems from report_generation_demo.py

FIVE_YEAR_DISEASES = [ ... ]  # 6 diseases from demo.py

PATHOLOGY_KEYWORDS = { ... }  # ovarian + liver_nafld keyword lists

EXPECTED_METRIC_RANGES = {
    "BLEU-4":        (0.08, 0.18),
    "ROUGE-L":       (0.15, 0.25),
    "BERTScore-F1":  (0.82, 0.88),
    "RadGraph-F1":   (0.20, 0.30),
}
```

#### `download_abdomenatlas.py`

```python
def download_abdomenatlas_subset(data_dir: str, n_cases: int = 5) -> list[dict]:
    """Download N cases from AbdomenAtlas 3.0 via HuggingFace.
    Returns: [{"case_id", "nifti_path", "gt_report", "gt_report_structured"}, ...]
    """

def load_existing_cases(data_dir: str, n_cases: int) -> list[dict]:
    """Load already-downloaded AbdomenAtlas cases."""

def filter_cases_by_pathology(cases: list[dict], pathology: str) -> list[dict]:
    """Filter cases by keyword matching on gt_report.
    pathology: 'all' | 'ovarian' | 'liver_nafld'
    """
```

#### `download_tcia.py`

```python
def download_tcia_collection(
    collection_name: str, output_dir: str, max_patients: int = 3
) -> list[str]:
    """Download CT DICOM series from TCIA. Returns list of patient dirs."""

def download_all_tcia_datasets(base_dir: str, max_patients: int = 3) -> dict:
    """Download all configured TCIA collections."""
```

#### `convert_dicom.py`

```python
def convert_dicom_to_nifti(dicom_dir: str, output_nifti_path: str) -> bool:
    """Convert DICOM directory to NIfTI using dicom2nifti."""

def convert_all_datasets(dataset_dirs: dict, base_dir: str) -> dict:
    """Batch convert all downloaded DICOM datasets to NIfTI."""
```

#### `inference.py`

```python
def get_merlin_model(mode: str) -> nn.Module:
    """Load and cache Merlin model. mode: 'report' | 'survival'"""

def run_report_generation(nifti_path: str) -> tuple[str, float]:
    """Generate full radiology report (all organ systems).
    Returns: (report_text, inference_time_seconds)
    Uses model.generate() with organ-system prompts and EosListStoppingCriteria.
    """

def run_five_year_prediction(nifti_path: str) -> dict[str, float]:
    """Run 5-year disease prediction. Returns: {disease_name: probability}"""
```

#### `metrics.py`

```python
def compute_bleu(hypothesis: str, reference: str) -> dict
def compute_rouge(hypothesis: str, reference: str) -> dict
def compute_bertscore(hypothesis: str, reference: str) -> dict
def compute_radgraph_f1(hypothesis: str, reference: str) -> dict
def compute_all_metrics(hypothesis: str, reference: str) -> dict
def aggregate_metrics(all_metrics: list[dict]) -> dict
```

#### `eval_pipeline.py`

CLI entry point for scored evaluation:

```
python scripts/eval_pipeline.py \
    --n_cases 5 \
    --pathology all \
    --output_dir ./merlin_eval_results \
    --data_dir ./atlas_data \
    --skip_download \
    --no_survival \
    --show_metric_guide
```

Pipeline steps:
1. Download/load AbdomenAtlas subset
2. Filter by pathology (if specified)
3. For each case: run report generation → score vs ground truth
4. Optionally run 5-year prediction
5. Aggregate metrics
6. Save JSON + CSV

#### `inference_pipeline.py`

CLI entry point for inference-only (TCIA):

```
python scripts/inference_pipeline.py \
    --dataset TCGA-OV \
    --max_patients 3 \
    --base_dir ./ct_data \
    --skip_download \
    --no_survival
```

Pipeline steps:
1. Download TCIA collection
2. Convert DICOM → NIfTI
3. For each case: run report generation + 5-year prediction
4. Save JSON (reports only, no metrics)

---

## 6. Metrics Specification

### 6.1 Metrics Suite

| Metric | Library | What It Measures | Clinical Relevance |
|--------|---------|-----------------|-------------------|
| BLEU-1..4 | `nltk` | N-gram precision overlap | Low — penalizes paraphrasing |
| ROUGE-1,2,L | `rouge_score` | Recall-oriented n-gram overlap | Medium — ROUGE-L captures word order |
| BERTScore P/R/F1 | `bert_score` + BiomedBERT | Semantic embedding similarity | High — robust to clinical synonyms |
| RadGraph-F1 | `radgraph` | Clinical entity overlap (findings, anatomy) | Highest — measures clinical correctness |

### 6.2 Expected Ranges on AbdomenAtlas 3.0

Merlin was trained on Stanford internal data. AbdomenAtlas uses RadGPT-assisted
structured reports — different distribution. Expect modest scores reflecting
domain/style shift:

| Metric | Expected Range | Merlin Published (own test set) |
|--------|---------------|-------------------------------|
| BLEU-4 | 0.08–0.18 | ~0.15–0.20 |
| ROUGE-L | 0.15–0.25 | — |
| BERTScore-F1 | 0.82–0.88 | — |
| RadGraph-F1 | 0.20–0.30 | ~0.27 (partial) |

### 6.3 Interpreting Low Scores

Low scores on AbdomenAtlas ≠ Merlin is broken. They reflect:
1. **Domain shift** — Stanford training data vs multi-institution Atlas
2. **Style shift** — RadGPT-assisted structured reports vs narrative clinical reports
3. **Inherent variability** — two radiologists describe the same scan differently
   (inter-radiologist BERTScore-F1 ceiling: 0.88–0.92)

---

## 7. CPU Inference Constraints

### 7.1 Hardware Target

ThinkPad X1 Carbon Gen13:
- CPU: Intel Core Ultra (likely 16 cores)
- RAM: 16–32 GB
- No discrete GPU

### 7.2 Performance Estimates

| Task | Time per Case (CPU) |
|------|-------------------|
| Report generation (13 organ systems) | 10–20 min |
| 5-year prediction | 2–5 min |
| NLP metrics (BLEU+ROUGE+BERTScore+RadGraph) | 1–3 min |
| **Total per case** | **~15–25 min** |

| N Cases | Total Time |
|---------|-----------|
| 1 | ~20 min |
| 5 | ~1.5–2 hours |
| 10 | ~3–4 hours |
| 20 | ~5–8 hours |

### 7.3 CPU-Specific Settings

```python
DEVICE = "cpu"
BATCH_SIZE = 1        # prevent OOM
NUM_WORKERS = 0       # avoid multiprocessing overhead on CPU
torch.set_num_threads(8)  # use physical cores, not hyperthreads
```

---

## 8. Known Limitations

1. **No NAFLD-specific CT dataset exists publicly.** Best proxy: AbdomenAtlas liver
   findings or CT-ORG liver segmentation cases.

2. **TCIA datasets lack paired reports.** TCGA-OV, CPTAC-OV, CT-ORG provide CT images
   only — we can generate Merlin reports but cannot score them.

3. **AbdomenAtlas reports are AI-assisted.** Ground truth reports were written with
   RadGPT assistance (expert-reviewed). This creates a specific reporting style that
   differs from typical clinical reports Merlin was trained on.

4. **Pathology filtering is approximate.** Keyword matching on report text is not a
   clinical diagnosis. Cases tagged "ovarian" may include incidental findings rather
   than primary ovarian pathology.

5. **Merlin generates per-organ-system, not holistic reports.** The full report is a
   concatenation of 13 organ-system outputs. Ground truth reports may have different
   structure.

6. **CPU inference is slow.** 10–20 minutes per case for report generation. Plan
   overnight runs for 10+ cases.

---

## 9. Dependencies

### Already installed (merlin deps)

```
torch>=2.1.2, monai>=1.3.0, transformers>=4.38.2, huggingface_hub,
nibabel, pandas, numpy, tqdm, peft, accelerate, einops, nltk
```

### New — eval metrics

```bash
uv pip install rouge_score bert_score radgraph
```

### New — TCIA download + DICOM conversion

```bash
uv pip install tcia_utils dicom2nifti pydicom SimpleITK
```

### Optional but recommended

```bash
uv pip install rich  # for pretty terminal output (already used in demo.py)
```

---

## 10. Verification Steps

### 10.1 Import Checks

```bash
python -c "from scripts.configs import DEVICE, ORGAN_SYSTEMS, FIVE_YEAR_DISEASES; print('configs OK')"
python -c "from scripts.download_abdomenatlas import download_abdomenatlas_subset; print('download OK')"
python -c "from scripts.inference import run_report_generation; print('inference OK')"
python -c "from scripts.metrics import compute_all_metrics; print('metrics OK')"
```

### 10.2 Single-Case Eval (AbdomenAtlas)

```bash
python scripts/eval_pipeline.py --n_cases 1 --output_dir ./test_eval
# Expected: downloads 1 case, generates report, computes metrics, saves JSON + CSV
# Time: ~20 min on CPU
```

### 10.3 Metric Guide

```bash
python scripts/eval_pipeline.py --show_metric_guide
# Expected: prints metric interpretation guide, exits
```

### 10.4 Inference-Only (TCIA)

```bash
python scripts/inference_pipeline.py --dataset TCGA-OV --max_patients 1
# Expected: downloads 1 patient, converts DICOM→NIfTI, generates report, saves JSON
# Time: ~25 min (includes download + conversion + inference)
```

### 10.5 File Cleanup Verification

```bash
# These should NOT exist after cleanup:
test ! -f merlin/utils/merlin_pipeline.py && echo "PASS" || echo "FAIL"
test ! -f merlin/utils/files/merlin_eval_pipeline.py && echo "PASS" || echo "FAIL"
test ! -f merlin/utils/files/MERLIN_EVAL_README.md && echo "PASS" || echo "FAIL"
```
