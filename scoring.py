from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Any, Dict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class AnchorAxis:
    """
    Represents a 1D semantic axis: left <-> right.

    Example:
      left = ["planned", "structured"]
      right = ["spontaneous", "impulsive"]
    """
    name: str
    left_anchors: List[str]
    right_anchors: List[str]


def dedupe_preserve_order(items: List[str]) -> List[str]:
    """Remove duplicates (case-insensitive) while preserving order."""
    seen = set()
    out: List[str] = []
    for it in items:
        key = it.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(it.strip())
    return out



#for zscore
def zscore(x: np.ndarray) -> np.ndarray:
    mean = float(np.mean(x))
    std = float(np.std(x))
    if np.isclose(std, 0.0):
        return np.zeros_like(x)
    return (x - mean) / std



def compute_axis_scores(
    texts: List[str],
    axis: AnchorAxis,
    model: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute directional raw score per text:

        raw_score = mean_sim(right_anchors) - mean_sim(left_anchors)

    Returns:
      raw_scores: (N,)
      left_sims:  (N,) mean cosine sim to left anchors
      right_sims: (N,) mean cosine sim to right anchors
    """
    if not texts:
        return np.array([]), np.array([]), np.array([])

    left = dedupe_preserve_order(axis.left_anchors)
    right = dedupe_preserve_order(axis.right_anchors)

    # Encode texts + anchors into normalized embeddings.
    # normalize_embeddings=True => cosine similarity is just dot product.
    text_emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    left_emb = model.encode(left, normalize_embeddings=True, show_progress_bar=False)
    right_emb = model.encode(right, normalize_embeddings=True, show_progress_bar=False)

    # Cosine similarity matrices: (N x A)
    left_sim_matrix = cosine_similarity(text_emb, left_emb)
    right_sim_matrix = cosine_similarity(text_emb, right_emb)

    left_sims = left_sim_matrix.mean(axis=1)
    right_sims = right_sim_matrix.mean(axis=1)

    raw_scores = right_sims - left_sims
    return raw_scores, left_sims, right_sims


def score_texts_on_axis(
    texts: List[str],
    axis: AnchorAxis,
    model_name: str = "all-MiniLM-L6-v2",
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Convenience wrapper:
    - imports SentenceTransformer lazily (avoids import-time issues elsewhere)
    - loads the model
    - computes raw score and normalized score in [-1, 1]

    Returns dict with:
      raw_scores, score_01, score_m11, left_sims, right_sims,
      model_name, axis_name, left_anchors, right_anchors
    """
    print("[score_texts_on_axis] importing SentenceTransformer...", flush=True)
    from sentence_transformers import SentenceTransformer
    print("[score_texts_on_axis] imported SentenceTransformer", flush=True)

    print(f"[score_texts_on_axis] loading model: {model_name} on {device}...", flush=True)
    model = SentenceTransformer(model_name, device=device)
    print("[score_texts_on_axis] model loaded", flush=True)

    raw_scores, left_sims, right_sims = compute_axis_scores(texts, axis, model)

    score_z = zscore(raw_scores)


    return {
        "raw_scores": raw_scores,
        "score_z": score_z,
        "left_sims": left_sims,
        "right_sims": right_sims,
        "model_name": model_name,
        "axis_name": axis.name,
        "left_anchors": dedupe_preserve_order(axis.left_anchors),
        "right_anchors": dedupe_preserve_order(axis.right_anchors),
    }

