print("MAIN: starting", flush=True)

import pandas as pd
print("MAIN: imported pandas", flush=True)

from scoring import AnchorAxis, score_texts_on_axis
print("MAIN: imported scoring", flush=True)

import ast


CLEAN_CSV_PATH = "perception_step2_clean.csv"
OUT_PATH = "perception_with_spontaneity.csv"

# Your anchors
SPONTANEITY_AXIS = AnchorAxis(
    name="spontaneity",
    left_anchors=[
        "Rigid", "Structured", "Planned", "Methodical", "Scheduled", "Consistent", "Routine"
    ],
    right_anchors=[
        "Spontaneous", "Flexible", "Impulsive", "Adventurous", "Chaotic", "Free-spirited", "Easy-going", "Playful"
    ],
)


def count_traits(val):
    """
    traits_list column may be a string like "['Kind', 'Funny']"
    Safely parse and count items.
    """
    if pd.isna(val):
        return 0
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            return len(parsed)
    except Exception:
        pass
    return 0


def count_words(text):
    if not isinstance(text, str):
        return 0
    return len([w for w in text.split() if w.strip()])


def main():
    df = pd.read_csv(CLEAN_CSV_PATH)

    # Prefer traits_text if it exists, else fall back to traits_raw
    text_col = "traits_text" if "traits_text" in df.columns else "traits_raw"

    df[text_col] = df[text_col].fillna("").astype(str).str.strip()
    df = df[df[text_col] != ""].copy()
    
    print("Scoring texts on spontaneity axis...")
    result = score_texts_on_axis(
        texts=df[text_col].tolist(),
        axis=SPONTANEITY_AXIS,
        model_name="all-MiniLM-L6-v2",
    )
    print("Scoring complete.")

    # Attach scores back to df
    df["spontaneity_raw"] = result["raw_scores"]
    df["sim_planned_mean"] = result["left_sims"]
    df["sim_spontaneous_mean"] = result["right_sims"]
    df["spontaneity_score"] = result["score_z"] #using zscore here


    # Trait count (from traits_list if available)
    if "traits_list" in df.columns:
        df["num_traits"] = df["traits_list"].apply(count_traits)
    else:
        # fallback: count comma-separated traits
        df["num_traits"] = df[text_col].str.split(",").apply(len)

    # Word count
    df["num_words"] = df[text_col].apply(count_words)

    # Low-confidence flag
    df["low_confidence"] = (df["num_traits"] < 2) | (df["num_words"] < 2)


    # Sanity checks
    high_conf_df = df[~df["low_confidence"]]

    print("\nTop 5 most PLANNED (high confidence):")
    print(
        high_conf_df.sort_values("spontaneity_score")
        .head(5)[["context", "years_bucket", text_col, "spontaneity_score"]]
        .to_string(index=False)
    )

    print("\nTop 5 most SPONTANEOUS (high confidence):")
    print(
        high_conf_df.sort_values("spontaneity_score", ascending=False)
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
