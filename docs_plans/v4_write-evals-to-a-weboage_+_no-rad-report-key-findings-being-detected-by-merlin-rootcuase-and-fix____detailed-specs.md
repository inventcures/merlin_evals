# v4: Eval Results Webpage + Root Cause Fix for Boilerplate Reports

## Context

Merlin eval pipeline (v5) completed on Modal A10G GPU — 10 cases (6 ovarian, 4 liver/NAFLD) processed in ~20s each. Results reveal a critical problem: **Merlin generates near-identical "normal" reports for every case**, missing major pathology (10cm liver masses, 5cm colon masses, massively enlarged organs). BLEU-4=0.01, ROUGE-L=0.13 — well below expected ranges.

Two tasks:
1. Build a public eval results webpage with an interactive CT scan viewer
2. Root-cause the boilerplate report problem and implement fixes

---

## Part A: Eval Results Webpage

**Deploy on**: `inventcures.github.io/merlin_evals.html` (GitHub Pages, follows `asco2025.html` standalone pattern)

### UI Layout

```
┌──────────────────────────────────────────────────────────────────┐
│  Header: Merlin CT Report Generation Evaluation                  │
│  Subtitle / metrics summary cards (BLEU-4, ROUGE-L, ...)        │
├──────────────────────────────────────────────────────────────────┤
│  Case Selector: [BDMAP_00000547 ▼] [Ovarian] [BLEU-4: 0.006]   │
├─────────────────────────┬────────────────────────────────────────┤
│                         │  GT Report          │ Merlin Report    │
│   CT Scan Viewer        │                     │                  │
│   (NiiVue, WebGL)       │  "The colon         │ "liver: normal.  │
│                         │   contains a large  │  gallbladder:    │
│   ┌─────────────────┐   │   hyperattenuating  │  normal. spleen: │
│   │  Axial slice     │   │   mass measuring    │  normal..."      │
│   │  ← slider →      │   │   5.1 x 4.4 cm..." │                  │
│   │  [⛶ Fullscreen]  │   │                     │                  │
│   └─────────────────┘   │                     │                  │
│                         ├─────────────────────┴──────────────────┤
│   W/L: [-1000, 1000]   │  Key Findings MISSED by Merlin         │
│   View: Axial/Cor/Sag  │  • 5.1x4.4cm colon mass (HU 82.4)     │
│                         │  • Spleen 160.5cc (normal noted)       │
├─────────────────────────┴────────────────────────────────────────┤
│  Metrics Dashboard (D3.js bar chart — per-case BLEU/ROUGE)       │
├──────────────────────────────────────────────────────────────────┤
│  Root Cause Analysis (structured explanation)                    │
└──────────────────────────────────────────────────────────────────┘
```

**Left half**: CT scan viewer (NiiVue.js — WebGL-based, loads `.nii.gz` directly in browser, MIT license). Includes:
- Axial/coronal/sagittal view toggle
- Scroll through slices (mouse wheel or slider)
- Window/level adjustment for CT
- **Fullscreen button** via browser Fullscreen API (`element.requestFullscreen()`)

**Right half**: Two-column tabular layout
- Column 1: Ground truth report (from AbdomenAtlas 3.0 metadata CSV)
- Column 2: Merlin predicted report
- Below: Key findings missed by Merlin (highlighted in red)

### Data Pipeline

```
Modal Volume (10 NIfTI volumes, ~500MB each raw)
         ↓  [Modal function: downsample + compress]
Downsampled .nii.gz (~256x256x~100, ~0.5-2MB each)
         ↓  [modal volume get → local]
Local: ./merlin_eval_results_modal/nifti_web/
         ↓  [copy to GitHub Pages repo]
inventcures.github.io/data/merlin_evals/*.nii.gz (10 files, ~5-15MB total)
         ↓  [browser fetch]
NiiVue renders in WebGL canvas
```

### Files to Create/Modify

| File | Repo | Type | Purpose |
|------|------|------|---------|
| `scripts/extract_web_data.py` | Merlin | NEW | Modal script: downsample NIfTIs + extract thumbnails |
| `merlin_evals.html` | inventcures.github.io | NEW | Standalone eval page (HTML + inline CSS/JS) |
| `merlin_evals_data.js` | inventcures.github.io | NEW | Eval results as JS const (reports, metrics, missed findings) |
| `data/merlin_evals/*.nii.gz` | inventcures.github.io | NEW | 10 downsampled NIfTI volumes for NiiVue |
| `images/merlin_evals/*_thumb.png` | inventcures.github.io | NEW | 10 thumbnail PNGs for case selector |

### Implementation Details

**Step A1: `scripts/extract_web_data.py`** — Modal function (CPU, no GPU) that:
- Loads each NIfTI from `/data/images/{case_id}/ct.nii.gz`
- Downsamples to ~256x256xD using scipy.ndimage.zoom (order=1)
- Saves as int16 .nii.gz with adjusted affine
- Extracts middle axial slice as PNG thumbnail (windowed -1000 to 1000 HU)

**Step A2: `merlin_evals_data.js`** — JS const built from `merlin_eval_results.json`:
- Cases array with case_id, pathology, nifti_url, thumb_url, gt_report, pred_report, metrics, missed_findings, five_year
- Summary stats (BLEU-4, ROUGE-L mean/std/n)
- Manually curated missed findings per case

**Step A3: `merlin_evals.html`** — Self-contained HTML following `asco2025.html` pattern:
- CDN deps: d3.v7.min.js, niivue.min.js
- Sections: Header, metric cards, case selector, split-panel (NiiVue + reports), D3 charts, root cause analysis
- NiiVue initialization with radiological convention, fullscreen support

---

## Part B: Root Cause Analysis — Why Merlin Generates Boilerplate

### The Problem

ALL 10 cases produce near-identical "normal" reports. Critical pathology missed:

| Case | GT Finding | Merlin Says |
|------|-----------|-------------|
| BDMAP_00000547 | 5.1x4.4cm colon mass | "no bowel obstruction" |
| BDMAP_00003256 | 6 kidney masses (largest 4.6cm) | "kidneys: normal" |
| BDMAP_00003432 | 4 liver lesions + 3 kidney lesions | "liver: normal, kidneys: normal" |
| BDMAP_00003687 | 10.2x6.5cm liver mass | "liver: normal" |
| BDMAP_00004336 | Massively enlarged spleen (518.7cc) | "spleen: normal" |
| BDMAP_00008992 | Enlarged spleen, pancreas, kidneys | all "normal" |

### 6 Root Causes (ordered by impact)

**1. Per-organ fragmentation (CRITICAL)**
- 13 separate `generate()` calls with independent prompts: `"Generate a radiology report for {organ_system}###\n"`
- Each call: 490 image tokens + ~15 prompt tokens → independent generation
- No holistic view — the model can't say "given the liver mass, also check lymph nodes"
- Each organ in isolation is most likely "normal" (even in sick patients, most individual organs are fine)
- **File**: `scripts/modal_eval.py:379-393`, `scripts/configs.py:ORGAN_SYSTEMS` (13 entries)

**2. Greedy decoding locks in "normal" (CRITICAL)**
- `do_sample=False, num_beams=1` → always picks the single most probable token
- P("normal" | image, organ_prompt) > P(any_specific_finding) for most organs, most images
- No exploration of lower-probability but correct paths like "lesion", "mass", "enlarged"
- **File**: `scripts/modal_eval.py:385-389`

**3. Training data distribution bias (CRITICAL, NOT fixable at inference)**
- Trained on Stanford radiology reports where ~70-80% of per-organ findings are "normal/unremarkable"
- LoRA adapter (r=512 — extremely large) memorized this distribution
- The checkpoint `resnet_gpt2_best_stanford_report_generation_average.pt` — "average" suggests ensemble-averaged weights which further smooth out pathology-specific features
- **File**: `merlin/models/radiology_report_generation.py:89-93` (LoRA config)

**4. Naive image-text fusion (HIGH)**
- Simple concatenation `[490 image tokens, N text tokens]` → self-attention
- No explicit cross-attention module between vision and language
- The language model's strong "normal" prior overwhelms subtle visual abnormality signals
- **File**: `merlin/models/radiology_report_generation.py:160` (`torch.cat((image_embeds, input_embeds), dim=1)`)

**5. Domain mismatch in EVALUATION (HIGH, affects metrics not model)**
- GT reports from AbdomenAtlas 3.0 are **algorithmically generated** with volumetric measurements and HU values (e.g., "measuring 10.5 x 7.0 cm, volume 283.6 cc, HU 57.3 ± 24.0")
- Merlin trained on **human-written** Stanford radiology reports (qualitative, no HU values)
- Even a perfect model would score low on BLEU/ROUGE due to this style gap

**6. Evidence the image encoder DOES work**
- 5-year survival predictions vary across cases (CVD: 0.605-0.627) — image features carry signal
- Some findings ARE detected: "surgically absent with bilateral salpingo-oophorectomy", "degenerative spine changes", occasional cysts
- The image encoder extracts meaningful features; the text decoder fails to express pathology

### Architecture Reference

```
NIfTI → MONAI preprocessing (224×224×160, HU [-1000,1000] → [0,1])
      → ModifiedImageEncoder (3D Inflated ResNet152) → (B, 490, 2048)
      → Adapter (Linear 2048→4096) → (B, 490, 4096)
      → [concat with text prompt embeddings]
      → RadLLaMA-7B + LoRA(r=512) → autoregressive text generation
```

Key files:
- `merlin/models/radiology_report_generation.py` — `Clip3DForTextGeneration`, `TextDecoder.generate()` (L147-163)
- `merlin/models/load.py` — `Merlin` class, checkpoint loading
- `merlin/data/monai_transforms.py` — image preprocessing pipeline
- `scripts/configs.py` — `ORGAN_SYSTEMS`, `DEVICE`

---

## Part C: Inference Fixes (Ablation Study)

### Strategy

Run the same 10 cases with different generation configs. Compare metrics and qualitative output to isolate root causes.

### Generation Configs to Test

```python
GENERATION_CONFIGS = {
    "baseline": {
        "mode": "per_organ",
        "do_sample": False,
        "num_beams": 1,
        "repetition_penalty": 1.2,
        "max_new_tokens": 128,
    },
    "whole_report_greedy": {
        "mode": "whole_report",
        "do_sample": False,
        "num_beams": 1,
        "repetition_penalty": 1.2,
        "max_new_tokens": 500,
    },
    "whole_report_sampling": {
        "mode": "whole_report",
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.5,
        "max_new_tokens": 500,
    },
    "per_organ_sampling": {
        "mode": "per_organ",
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.5,
        "max_new_tokens": 128,
    },
    "per_organ_beam": {
        "mode": "per_organ",
        "do_sample": False,
        "num_beams": 4,
        "repetition_penalty": 1.5,
        "max_new_tokens": 128,
    },
}

WHOLE_REPORT_PROMPT = "Generate a comprehensive radiology report for this CT scan###\n"
```

### Changes to `scripts/modal_eval.py`

1. **`generate_report()` method** — add `gen_config: dict | None = None` parameter:
   - If `mode == "whole_report"`: single `generate()` call with `WHOLE_REPORT_PROMPT`, `max_new_tokens=500`
   - If `mode == "per_organ"`: existing 13-call loop
   - Pass through all other kwargs (`do_sample`, `temperature`, `num_beams`, etc.)

2. **`run_ablation` local entrypoint** — new:
   - Loop over all configs in `GENERATION_CONFIGS`
   - For each config: run 10 cases, compute metrics, save to `{output_dir}/{config_name}/`
   - Generate `{output_dir}/ablation_summary.csv` comparing all configs

### Expected Ablation Results

If `whole_report` modes produce more pathology-specific text → confirms per-organ fragmentation as root cause.
If `sampling` modes produce more diverse output → confirms greedy decoding as root cause.
If ALL modes still produce boilerplate → confirms training data bias as the fundamental issue (not fixable at inference time without retraining).

---

## Verification

**Webpage**:
1. Open `merlin_evals.html` locally in browser
2. Verify NiiVue loads and renders a CT volume
3. Verify fullscreen toggle works
4. Switch between all 10 cases — reports update, viewer loads new volume
5. D3 charts render with correct data
6. Deploy to GitHub Pages, confirm URL works

**Ablation**:
1. Run `whole_report_greedy` on 1 case first — verify output is different from baseline
2. Full ablation: verify all 5 configs × 10 cases complete without errors
3. Compare metrics: does any config improve BLEU-4 above 0.05?
4. Qualitative: does any config mention actual pathology (masses, enlargement, lesions)?
