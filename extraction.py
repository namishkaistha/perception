import re 
import pandas as pd

XLSX_PATH = "/Users/namishkaistha/Desktop/perception/SentimentAnalysisResponses.xlsx"
df_raw = pd.read_excel(XLSX_PATH)

def find_col(df, patterns):
    " return the first column whose name matches any regex in 'patterns'"

    for pat in patterns:
        for c in df.columns:
            if re.search(pat, str(c), flags=re.IGNORECASE):
                return c
    raise KeyError(f"Could not find a column matching any of: {patterns}")

col_timestamp = find_col(df_raw, [r"^timestamp$"])
col_context   = find_col(df_raw, [r"context.*know me from", r"what context"])
col_years     = find_col(df_raw, [r"how long.*known", r"how long have you known"])
col_traits    = find_col(df_raw, [r"3-5 words", r"words you would use to describe", r"describe me"])
col_relationship = find_col(df_raw, [r"closest point.*relationship", r"best describes our relationship"])

# --- 3) Build a clean working dataframe ---
df = df_raw[[col_timestamp, col_context, col_years, col_relationship, col_traits]].copy()

df = df.rename(columns={
    col_timestamp: "timestamp",
    col_context: "context_raw",
    col_years: "years_known_raw",
    col_relationship: "relationship_raw",
    col_traits: "traits_raw",
})

# --- 4) Normalize context into your buckets (Home / Northwestern / Wisconsin / Abroad) ---
def normalize_context(x: str) -> str:
    if pd.isna(x):
        return "Unknown"
    s = str(x).strip().lower()
    if "naperville" in s or s.startswith("home"):
        return "Home"
    if "wisconsin" in s:
        return "Wisconsin"
    if "northwestern" in s or s == "nu":
        return "Northwestern"
    if "abroad" in s:
        return "Abroad"
    return str(x).strip()

df["context"] = df["context_raw"].apply(normalize_context)

# --- 5) Normalize "years known" into ordered buckets ---
def normalize_years_bucket(x: str) -> str:
    if pd.isna(x):
        return "Unknown"
    s = str(x).strip().lower()

    # common variants
    if "<" in s and "year" in s:
        return "<1"
    if "1-3" in s or "1–3" in s:
        return "1-3"
    if "4-6" in s or "4–6" in s:
        return "4-6"
    if "7+" in s:
        return "7+"

    # fallback: parse first number and bucket
    m = re.search(r"(\d+)", s)
    if m:
        n = int(m.group(1))
        if n < 1:
            return "<1"
        if 1 <= n <= 3:
            return "1-3"
        if 4 <= n <= 6:
            return "4-6"
        return "7+"
    return str(x).strip()

df["years_bucket"] = df["years_known_raw"].apply(normalize_years_bucket)

# --- 6) Normalize relationship labels (optional but helpful) ---
def normalize_relationship(x: str) -> str:
    if pd.isna(x):
        return "Unknown"
    s = str(x).strip()
    s_low = s.lower()

    # standardize a few common patterns from your form
    if "close friend" in s_low:
        return "Close friends"
    if "acquaint" in s_low:
        return "Acquaintances"
    if "hung out" in s_low or "1–1" in s_low or "1-1" in s_low:
        return "Hung out 1–1 a decent amount"
    if "same large friend group" in s_low or "teammates" in s_low or "coworkers" in s_low:
        return "Same large friend group / teammates / coworkers"

    return s  # keep original if it doesn't match

df["relationship"] = df["relationship_raw"].apply(normalize_relationship)

# --- 7) Split the traits into a list of tokens/phrases ---
def split_traits(x: str):
    if pd.isna(x):
        return []
    s = str(x).strip()

    # split primarily on commas; fallback separators too
    parts = re.split(r"\s*,\s*|\s*;\s*|\s*\|\s*", s)

    out = []
    for p in parts:
        p = p.strip()
        p = re.sub(r"\s+", " ", p)   # collapse whitespace
        p = p.strip(" .!?\n\t\r")
        if p:
            out.append(p)
    return out

df["traits_list"] = df["traits_raw"].apply(split_traits)
df["traits_text"] = df["traits_list"].apply(lambda xs: ", ".join(xs))

# --- 8) Final "Step 2" output ---
df_step2 = df[
    ["timestamp", "context", "years_bucket", "relationship", "traits_raw", "traits_list", "traits_text"]
].copy()

print("Rows:", len(df_step2))
print(df_step2.head(10).to_string(index=False))

# --- 9) Save for downstream metric scoring/plotting ---
OUT_CSV = "/Users/namishkaistha/Desktop/perception/perception_step2_clean.csv"
df_step2.to_csv(OUT_CSV, index=False)
print(f"\nSaved cleaned dataset to: {OUT_CSV}")









