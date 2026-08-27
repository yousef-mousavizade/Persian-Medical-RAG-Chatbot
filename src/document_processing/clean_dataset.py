import re
import pandas as pd 
DATA_PATH = r"D:\Programing\RAG_Project's me\datasets\Dataset.xlsx"
OUTPUT_PATH = r"D:\Programing\RAG_Project's me\datasets\cleaned_dataset.csv"

ARABIC_TO_PERSIAN = {
    "ي": "ی",
    "ك": "ک",
    "ة": "ه",
    "ۀ": "ه",
    "أ": "ا",
    "إ": "ا",
    "ؤ": "و",
    "ئ": "ی",
}

# Arabic diacritics and formation signs that should be removed
ARABIC_DIACRITICS = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]"
)

# Emojis and pictographic symbols
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)


def normalize_text(text: str) -> str:
    """Clean and normalize a Persian text string."""
    if not isinstance(text, str):
        return ""

    # Remove remaining Excel line-break characters
    text = text.replace("_x000D_", " ")
    text = text.replace("\r", " ").replace("\n", " ")

    # Normalize Arabic characters to Persian characters
    for ar, fa in ARABIC_TO_PERSIAN.items():
        text = text.replace(ar, fa)

    # Remove Arabic diacritics
    text = ARABIC_DIACRITICS.sub("", text)

    # Remove emojis
    text = EMOJI_PATTERN.sub("", text)

    # Normalize whitespace (multiple spaces -> one space)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    print(f"Raw record count: {len(df)}")

    # 1. Remove records without a response
    df = df[df["response"].notna()].copy()
    print(f"Records with response: {len(df)}")

    # 2. Clean question and response text
    df["comment_clean"] = df["comment"].apply(normalize_text)
    df["response_clean"] = df["response"].apply(normalize_text)

    # 3. Remove records with very short responses (less than 10 characters)
    before = len(df)
    combined_len = df["comment_clean"].str.len() + df["response_clean"].str.len()
    df = df[combined_len >= 20].copy()
    print(f"Removed (combined question+answer too short/empty): {before - len(df)}")

    # 4. Create combined text for retrieval (this text will be embedded later)
    df["retrieval_text"] = (
        "دارو: " + df["drug_name"].astype(str) + " | سؤال: "
        + df["comment_clean"] + " | پاسخ: " + df["response_clean"]
    )

    # 5. Keep only the required final columns
    final_cols = [
        "commenter_id",
        "drug_name",
        "persian_martidale_category",
        "therapeutic_category",
        "comment_clean",
        "response_clean",
        "retrieval_text",
    ]

    df_final = df[final_cols].reset_index(drop=True)
    df_final.insert(0, "doc_id", range(1, len(df_final) + 1))

    return df_final


if __name__ == "__main__":
    import os

    os.makedirs("data", exist_ok=True)

    df_clean = load_and_clean(DATA_PATH)

    print("\n=== Sample final record ===")
    print(df_clean.iloc[0]["retrieval_text"])

    df_clean.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nSaved to: {OUTPUT_PATH} ({len(df_clean)} records)")