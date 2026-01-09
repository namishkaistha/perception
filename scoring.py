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