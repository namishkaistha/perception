from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer

@dataclass(frozen=True)
class AnchorAxis:
    """
    Represents a 1D semantic axis: left (planned) <-> right (spontaneous).
    """
    name: str
    left_anchors: List[str]
    right_anchors: List[str]


#remove duplicate anchor words, while keeping their original order
def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []

    for it in items:
        key = it.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(it.strip())
    return out 

#map any numeric vector to [0,1] using the dataset's min/max
# this normalizes values so that they are easily comparable
def minmax_to_unit_interval(x: np.ndarray) -> np.ndarray:
    xmin, xmax = float(np.min(x), float(np.max(x)))
    if np.isclose(xmin, xmax):
        return np.full_like(x, 0.5, dtype=float)
    return (x-xmin) / (xmax-xmin)

def unit_interval_to_minus1_plus1(x01: np.ndarray) -> np.ndarray:
    """Map [0, 1] -> [-1, 1]."""
    return 2.0 * x01 - 1.0

def compute_axis_scores(
        texts: List[str],
        axis: AnchorAxis,
        model: SentenceTransformer,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a raw directional score per text as:
        score = mean_sim(right_anchors) - mean_sim(left_anchors)

    Returns:
      raw_scores: (N,) float
      left_sims: (N,) mean cosine sim to left anchors
      right_sims: (N,) mean cosine sim to right anchors
    """
    if not texts:
        return np.array([]), np.array([]), np.array([])
    
    left = dedupe_preserve_order(axis.left_anchors)
    right = dedupe_preserve_order(axis.right_anchors)

    # Encode texts + anchors into normalized embeddings
    text_emb = model.encode(texts, normalize_embeddings=True)
    left_emb = model.encode(left, normalize_embeddings=True)
    right_emb = model.encode(right, normalize_embeddings=True)

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
) -> dict:
    """
    Convenience wrapper:
    - loads model
    - computes raw score
    - normalizes to [-1, 1] for plotting

    Returns a dict with:
      raw_scores, score_01, score_m11, left_sims, right_sims
    """
    model = SentenceTransformer(model_name)

    raw_scores, left_sims, right_sims = compute_axis_scores(
        texts=texts,
        axis=axis,
        model=model,
    )

    score_01 = minmax_to_unit_interval(raw_scores)
    score_m11 = unit_interval_to_minus1_plus1(score_01)

    return {
        "raw_scores": raw_scores,
        "score_01": score_01,
        "score_m11": score_m11,
        "left_sims": left_sims,
        "right_sims": right_sims,
        "model_name": model_name,
        "axis_name": axis.name,
        "left_anchors": dedupe_preserve_order(axis.left_anchors),
        "right_anchors": dedupe_preserve_order(axis.right_anchors),
    }