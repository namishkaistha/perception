import pandas as pd

from scoring import AnchorAxis, score_texts_on_axis

CLEAN_CSV_PATH = "perception_step2_clean.csv"
OUT_PATH = "perception_with_spontaneity.csv"

# Your anchors
SPONTANEITY_AXIS = AnchorAxis(
    name="spontaneity",
    left_anchors=[
        "Rigid", "Structured", "Planned", "Rigid", "Methodical", "Structured", "Logical"
    ],
    right_anchors=[
        "Spontaneous", "Flexible", "Impulsive", "Adventurous", "Chaotic"
    ],
)


def main():
    df = pd.read_csv(CLEAN_CSV_PATH)

    # Prefer traits_text if it exists, else fall back to traits_raw
    text_col = "traits_text" if "traits_text" in df.columns else "traits_raw"

    df[text_col] = df[text_col].fillna("").astype(str).str.strip()
    df = df[df[text_col] != ""].copy()

    result = score_texts_on_axis(
        texts=df[text_col].tolist(),
        axis=SPONTANEITY_AXIS,
        model_name="all-MiniLM-L6-v2",
    )

    # Attach scores back to df
    df["spontaneity_raw"] = result["raw_scores"]
    df["sim_planned_mean"] = result["left_sims"]
    df["sim_spontaneous_mean"] = result["right_sims"]
    df["spontaneity_score"] = result["score_m11"]  # final: -1 planned ... +1 spontaneous

    # Sanity checks
    print("\nTop 5 most PLANNED (lowest spontaneity_score):")
    print(
        df.sort_values("spontaneity_score")
          .head(5)[["context", "years_bucket", text_col, "spontaneity_score"]]
          .to_string(index=False)
    )

    print("\nTop 5 most SPONTANEOUS (highest spontaneity_score):")
    print(
        df.sort_values("spontaneity_score", ascending=False)
          .head(5)[["context", "years_bucket", text_col, "spontaneity_score"]]
          .to_string(index=False)
    )

    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved scored data to: {OUT_PATH}")
    print("\nAnchors used (deduped):")
    print("Planned-side:", result["left_anchors"])
    print("Spontaneous-side:", result["right_anchors"])


if __name__ == "__main__":
    main()
