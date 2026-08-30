
import json
from pathlib import Path
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
IN_FILE = DATA_DIR / "cleaned_dataset.csv"
OUT_FILE = DATA_DIR / "chunks.jsonl"

# Records shorter than this are kept as a single chunk (no splitting needed).
# Most of our records are short QA pairs (comment + response), so this only
# affects a small number of long outliers.
LONG_RECORD_THRESHOLD = 1000

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    # Persian-aware separators, matching the instructor's pattern
    separators=["\n\n", "\n", " .", ". ", ".", ":", "؟", "?", "!"],
)


def build_chunk_record(doc_id, chunk_index, drug_name, therapeutic_category, text, was_split):
    return {
        "chunk_id": f"pharma-{doc_id}-{chunk_index:04d}",
        "doc_id": f"pharma-{doc_id}",
        "drug_name": drug_name,
        "therapeutic_category": therapeutic_category,
        "chunk_index": chunk_index,
        "text": text,
        "was_split": was_split,
    }


def chunk_dataset(df: pd.DataFrame, out_path: Path) -> tuple[int, int]:
    chunk_count = 0
    split_count = 0

    with out_path.open("w", encoding="utf-8") as f_out:
        for _, row in df.iterrows():
            text = row["retrieval_text"]
            doc_id = row["doc_id"]
            drug_name = row["drug_name"]
            therapeutic_category = row["therapeutic_category"]

            if len(text) < LONG_RECORD_THRESHOLD:
                # Short record: keep as a single chunk, no splitting
                rec = build_chunk_record(doc_id, 0, drug_name, therapeutic_category, text, False)
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                chunk_count += 1
            else:
                # Long record: split into smaller chunks
                pieces = text_splitter.split_text(text)
                for i, piece in enumerate(pieces):
                    if not piece.strip():
                        continue
                    rec = build_chunk_record(doc_id, i, drug_name, therapeutic_category, piece, True)
                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    chunk_count += 1
                    split_count += 1

    return chunk_count, split_count


def main():
    if not IN_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found at: {IN_FILE}\n"
            f"Run clean_dataset.py first to generate it."
        )

    df = pd.read_csv(IN_FILE)
    print(f"Input records: {len(df)}")

    long_count = (df["retrieval_text"].str.len() >= LONG_RECORD_THRESHOLD).sum()
    print(f"Records requiring split (>= {LONG_RECORD_THRESHOLD} chars): {long_count}")

    chunk_count, split_count = chunk_dataset(df, OUT_FILE)

    print(f"Output chunks: {chunk_count}")
    print(f"Chunks created by splitting: {split_count}")
    print(f"Saved: {OUT_FILE}")


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    main()