# main.py
from __future__ import annotations

import argparse
import ast
import pandas as pd

from axes import AXES
from scoring import load_sentence_transformer, encode_texts, score_axis_from_embeddings


CLEAN_CSV_PATH = "perception_step2_clean.csv"
OUT_PATH = "perception_scored.csv"


def count_traits(val) -> int:
    if pd.isna(val):
        return 0
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            return len(parsed)
    except Exception:
        pass
    return 0


def count_words(text) -> int:
    if not isinstance(text, str):
        return 0
    return len([w for w in text.split() if w.strip()])


def parse_axes_arg(axes_arg: str):
    axes_arg = axes_arg.strip().lower()
    if axes_arg == "all":
        return list(AXES.keys())

    requested = [a.strip().lower() for a in axes_arg.split(",") if a.strip()]
    unknown = [a for a in requested if a not in AXES]
    if unknown:
        raise ValueError(f"Unknown axes: {unknown}. Available: {list(AXES.keys())}")
    return requested


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=CLEAN_CSV_PATH)
    parser.add_argument("--out", default=OUT_PATH)
    parser.add_argument("--axes", default="spontaneity", help="comma-separated axis names or 'all'")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    text_col = "traits_text" if "traits_text" in df.columns else "traits_raw"
    df[text_col] = df[text_col].fillna("").astype(str).str.strip()
    df = df[df[text_col] != ""].copy()

    # confidence columns (reused for all axes)
    if "traits_list" in df.columns:
        df["num_traits"] = df["traits_list"].apply(count_traits)
    else:
        df["num_traits"] = df[text_col].str.split(",").apply(len)

    df["num_words"] = df[text_col].apply(count_words)
    df["low_confidence"] = (df["num_traits"] < 2) | (df["num_words"] < 2)

    axes_to_run = parse_axes_arg(args.axes)

    # load model once, embed once
    model = load_sentence_transformer(args.model, args.device)
    texts = df[text_col].tolist()
    text_emb = encode_texts(model, texts)

    # score each axis
    for axis_name in axes_to_run:
        axis = AXES[axis_name]

        if len(axis.left_anchors) == 0 or len(axis.right_anchors) == 0:
            print(f"[main] skipping '{axis_name}' (anchors empty)", flush=True)
            continue

        print(f"[main] scoring axis: {axis_name}", flush=True)
        result_cols = score_axis_from_embeddings(
            text_emb=text_emb,
            axis_name=axis_name,
            left_anchors=axis.left_anchors,
            right_anchors=axis.right_anchors,
            model=model,
        )
        for col, arr in result_cols.items():
            df[col] = arr

    df.to_csv(args.out, index=False)
    print(f"\nSaved scored data to: {args.out}")
    print(f"Axes requested: {axes_to_run}")


if __name__ == "__main__":
    main()
