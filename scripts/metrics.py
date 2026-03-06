"""
NLP evaluation metrics for radiology report generation.
Computes BLEU, ROUGE, BERTScore, and RadGraph-F1.
"""

import numpy as np

from scripts.configs import DEVICE


def compute_bleu(hypothesis: str, reference: str) -> dict:
    """BLEU-1 through BLEU-4 using NLTK."""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    smoother = SmoothingFunction().method1

    scores = {}
    for n in range(1, 5):
        weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
        scores[f"BLEU-{n}"] = sentence_bleu(
            [ref_tokens], hyp_tokens,
            weights=weights, smoothing_function=smoother,
        )
    return scores


def compute_rouge(hypothesis: str, reference: str) -> dict:
    """ROUGE-1, ROUGE-2, ROUGE-L F1 scores."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True,
    )
    scores = scorer.score(reference, hypothesis)
    return {
        "ROUGE-1": scores["rouge1"].fmeasure,
        "ROUGE-2": scores["rouge2"].fmeasure,
        "ROUGE-L": scores["rougeL"].fmeasure,
    }


def compute_bertscore(hypothesis: str, reference: str) -> dict:
    """BERTScore using BiomedBERT for clinical text."""
    try:
        from bert_score import score as bert_score

        P, R, F1 = bert_score(
            [hypothesis], [reference],
            model_type="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
            lang="en",
            verbose=False,
            device=DEVICE,
        )
        return {
            "BERTScore-P": float(P[0]),
            "BERTScore-R": float(R[0]),
            "BERTScore-F1": float(F1[0]),
        }
    except Exception as e:
        print(f"  [WARN] BERTScore failed: {e}")
        return {"BERTScore-P": None, "BERTScore-R": None, "BERTScore-F1": None}


def compute_radgraph_f1(hypothesis: str, reference: str) -> dict:
    """
    RadGraph-F1: clinical entity overlap metric.
    Measures whether the same findings and anatomical locations appear
    in both hypothesis and reference reports.
    """
    try:
        from radgraph import F1RadGraph

        f1_radgraph = F1RadGraph(reward_level="partial")
        score, _, _ = f1_radgraph([hypothesis], [reference])
        return {"RadGraph-F1": float(score)}
    except ImportError:
        return {"RadGraph-F1": None, "RadGraph-note": "Install: uv pip install radgraph"}
    except Exception as e:
        return {"RadGraph-F1": None, "RadGraph-note": str(e)}


def compute_all_metrics(hypothesis: str, reference: str) -> dict:
    """Compute all NLP evaluation metrics for a single (hypothesis, reference) pair."""
    if not hypothesis or not reference:
        return {"error": "empty hypothesis or reference"}

    metrics = {}
    metrics.update(compute_bleu(hypothesis, reference))
    metrics.update(compute_rouge(hypothesis, reference))
    metrics.update(compute_bertscore(hypothesis, reference))
    metrics.update(compute_radgraph_f1(hypothesis, reference))
    return metrics


def aggregate_metrics(all_metrics: list[dict]) -> dict:
    """Compute mean +/- std across all cases for each numeric metric."""
    if not all_metrics:
        return {}

    numeric_keys = [
        k for k in all_metrics[0]
        if isinstance(all_metrics[0].get(k), (int, float)) and all_metrics[0][k] is not None
    ]

    summary = {}
    for key in numeric_keys:
        vals = [m[key] for m in all_metrics if m.get(key) is not None]
        if vals:
            summary[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "n": len(vals),
            }
    return summary
