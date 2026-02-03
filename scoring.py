# scoring.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np


def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for it in items:
        key = str(it).strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(str(it).strip())
    return out


def zscore(x: np.ndarray) -> np.ndarray:
    mean = float(np.mean(x))
    std = float(np.std(x))
    if np.isclose(std, 0.0):
        return np.zeros_like(x, dtype=float)
    return (x - mean) / std


def load_sentence_transformer(model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
    print("[scoring] importing SentenceTransformer...", flush=True)
    from sentence_transformers import SentenceTransformer
    print("[scoring] imported SentenceTransformer", flush=True)

    print(f"[scoring] loading model: {model_name} on {device}...", flush=True)
    model = SentenceTransformer(model_name, device=device)
    print("[scoring] model loaded", flush=True)
    return model


def encode_texts(model: Any, texts: List[str]) -> np.ndarray:
    # normalized embeddings => cosine sim becomes dot product
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=True)


def encode_anchors(model: Any, anchors: List[str]) -> np.ndarray:
    return model.encode(anchors, normalize_embeddings=True, show_progress_bar=False)


def mean_sim(text_emb: np.ndarray, anchor_emb: np.ndarray) -> np.ndarray:
    """
    text_emb: (N, D), anchor_emb: (A, D), all normalized
    mean cosine sim = mean(dot(text, anchor))
    """
    if anchor_emb.shape[0] == 0:
        return np.zeros((text_emb.shape[0],), dtype=float)
    sims = text_emb @ anchor_emb.T  # (N, A)
    return sims.mean(axis=1)


def score_axis_from_embeddings(
    text_emb: np.ndarray,
    axis_name: str,
    left_anchors: List[str],
    right_anchors: List[str],
    model: Any,
) -> Dict[str, np.ndarray]:
    left = dedupe_preserve_order(left_anchors)
    right = dedupe_preserve_order(right_anchors)

    left_emb = encode_anchors(model, left)
    right_emb = encode_anchors(model, right)

    left_sims = mean_sim(text_emb, left_emb)
    right_sims = mean_sim(text_emb, right_emb)

    raw = right_sims - left_sims
    z = zscore(raw)

    return {
        f"{axis_name}_raw": raw,
        f"{axis_name}_z": z,
        f"{axis_name}_sim_left": left_sims,
        f"{axis_name}_sim_right": right_sims,
    }
