import torch

DEVICE = "cpu"
torch.set_num_threads(8)

ABDOMENATLAS_REPO_ID = "AbdomenAtlas/AbdomenAtlas3.0"

TCIA_COLLECTIONS = {
    "TCGA-OV": {
        "indication": "High-grade serous ovarian cancer",
    },
    "CPTAC-OV": {
        "indication": "Ovarian serous cystadenocarcinoma",
    },
    "CT-ORG": {
        "indication": "Abdominal CT with liver lesions (NAFLD proxy)",
    },
}

ORGAN_SYSTEMS = {
    "lower thorax": "lower thorax|lower chest|lung bases",
    "liver": "liver|liver and biliary tree|biliary system",
    "gallbladder": "gallbladder",
    "spleen": "spleen",
    "pancreas": "pancreas",
    "adrenal glands": "adrenal glands|adrenals",
    "kidneys": "kidneys|kidneys and ureters|gu|kidneys, ureters",
    "bowel": "bowel|gastrointestinal tract|gi|bowel/mesentery",
    "peritoneum": "peritoneal space|peritoneal cavity|abdominal wall|peritoneum",
    "pelvic": "pelvic organs|bladder|prostate and seminal vesicles|pelvis|uterus and ovaries",
    "circulatory": "vasculature",
    "lymph nodes": "lymph nodes",
    "musculoskeletal": "musculoskeletal|bones",
}

FIVE_YEAR_DISEASES = [
    "Cardiovascular Disease (CVD)",
    "Ischemic Heart Disease (IHD)",
    "Hypertension (HTN)",
    "Diabetes Mellitus (DM)",
    "Chronic Kidney Disease (CKD)",
    "Chronic Liver Disease (CLD)",
]

PATHOLOGY_KEYWORDS = {
    "ovarian": [
        "ovary", "ovarian", "adnexal", "pelvic mass", "oophor",
        "fallopian", "peritoneal carcinomatosis", "omental cake",
        "ascites",
    ],
    "liver_nafld": [
        "steatosis", "fatty liver", "nafld", "nash", "mash",
        "hepatic steatosis", "fatty infiltration", "cirrhosis",
        "fibrosis", "hepatomegaly", "liver lesion", "hepatic lesion",
        "hepatocellular", "liver metast",
    ],
}

EXPECTED_METRIC_RANGES = {
    "BLEU-4": (0.08, 0.18),
    "ROUGE-L": (0.15, 0.25),
    "BERTScore-F1": (0.82, 0.88),
    "RadGraph-F1": (0.20, 0.30),
}
