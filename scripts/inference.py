"""
Merlin inference: report generation and 5-year disease prediction.
All inference runs on CPU.

Uses the correct Merlin API:
  - model.generate(image, text_labels, **kwargs)  for report generation
  - model(image)                                    for 5-year prediction
  - merlin.data.DataLoader                          for data loading
"""

import time
import warnings

import torch
import torch.quantization
from transformers import StoppingCriteria

from scripts.configs import DEVICE, ORGAN_SYSTEMS, FIVE_YEAR_DISEASES

warnings.filterwarnings("ignore")

_model_cache = {}


class EosListStoppingCriteria(StoppingCriteria):
    """Stop generation when EOS token is produced."""

    def __init__(self, eos_sequence=None):
        self.eos_sequence = eos_sequence or [48134]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[:, -len(self.eos_sequence):].tolist()
        return self.eos_sequence in last_ids


def get_merlin_model(mode: str):
    """
    Load and cache a Merlin model.

    Args:
        mode: 'report' or 'survival'
    """
    if mode in _model_cache:
        return _model_cache[mode]

    from merlin import Merlin

    print(f"[Merlin] Loading model (mode={mode}) on {DEVICE}...")
    if mode == "report":
        model = Merlin(RadiologyReport=True)
    elif mode == "survival":
        model = Merlin(FiveYearPred=True)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    model.eval()
    model.to(DEVICE)

    if mode == "report":
        td = model.model.decode_text
        td.text_decoder = td.text_decoder.merge_and_unload()
        td.text_decoder.gradient_checkpointing_disable()
        td.text_decoder = td.text_decoder.float()
        td.text_decoder = torch.quantization.quantize_dynamic(
            td.text_decoder, {torch.nn.Linear}, dtype=torch.qint8
        )
        print("[Merlin] Applied LoRA merge + gradient ckpt disable + fp32 cast + INT8 quantization")

    _model_cache[mode] = model
    print(f"[Merlin] Model loaded")
    return model


def _load_image_tensor(nifti_path: str) -> torch.Tensor:
    """Load a NIfTI file and return a preprocessed image tensor via Merlin's DataLoader."""
    from merlin.data import DataLoader

    datalist = [{"image": nifti_path}]
    dataloader = DataLoader(
        datalist=datalist,
        cache_dir="/tmp/merlin_cache",
        batchsize=1,
        shuffle=False,
        num_workers=0,
    )

    for batch in dataloader:
        return batch["image"].to(DEVICE)


def run_report_generation(nifti_path: str) -> tuple[str, float]:
    """
    Generate a full radiology report by iterating over all organ systems.

    Returns:
        (full_report_text, inference_time_seconds)
    """
    model = get_merlin_model("report")
    image = _load_image_tensor(nifti_path)

    t0 = time.time()

    report_parts = []
    for organ_system in ORGAN_SYSTEMS:
        prefix = f"Generate a radiology report for {organ_system}###\n"
        with torch.no_grad():
            generations = model.generate(
                image,
                [prefix],
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.2,
                max_new_tokens=128,
                stopping_criteria=[EosListStoppingCriteria()],
            )
        text = generations[0].split("###")[0].strip()
        if text:
            report_parts.append(text)

    elapsed = time.time() - t0
    full_report = " ".join(report_parts)
    return full_report, elapsed


def run_five_year_prediction(nifti_path: str) -> dict[str, float]:
    """
    Run 5-year chronic disease risk prediction.

    Returns:
        {disease_name: probability}
    """
    model = get_merlin_model("survival")
    image = _load_image_tensor(nifti_path)

    with torch.no_grad():
        logits = model(image)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    return {d: float(p) for d, p in zip(FIVE_YEAR_DISEASES, probs)}
