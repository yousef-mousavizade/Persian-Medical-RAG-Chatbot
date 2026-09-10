"""
THEORY NOTE

Goal: test whether chunk_size/overlap affects retrieval for the ~55 long
records (>1000 chars) that actually get split by the splitter. Short
records are never split, so they're excluded from this test.

Pitfall (fixed): using the user's question as the test query gave 100%
for every config, because the question always lands in chunk 0
regardless of chunk_size - not a real test. Fixed by testing with an
excerpt from deep inside the answer text instead, so retrieval success
actually depends on how the text was split.

Metric: Hit Rate@5 - is the correct drug found in the top-5 results.

Result: 250/25 > 400/60 > 1000/200 (71.9% vs 65.6% vs 62.5%). Smaller
chunks worked better since our "long" records are only moderately long
(~1300 chars avg), so large chunks added noise by merging question+answer
into one vector.
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path.cwd().parent
CLEANED_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_dataset.csv"

LONG_RECORD_THRESHOLD = 1000
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
TOP_K = 5

# The configurations to compare - add or edit freely
CONFIGS = [
    {"name": "1000/200 (original)", "chunk_size": 1000, "chunk_overlap": 200},
    {"name": "400/60 (proposed middle ground)", "chunk_size": 400, "chunk_overlap": 60},
    {"name": "250/25 (your latest change)", "chunk_size": 250, "chunk_overlap": 25},
]


def build_test_set(df: pd.DataFrame) -> list[dict]:
    """Build (query, expected_drug) pairs from the long records only.

    IMPORTANT: the query must NOT be a verbatim substring of the source
    document, or retrieval becomes trivial (near 100% for every config,
    regardless of chunk size - this was the bug in the first version of
    this script). Instead, we pull a short excerpt from deep inside the
    RESPONSE text (offset ~60% into it). This tests whether information
    located later in a long record survives being split into a separate
    chunk under different chunk_size settings - which is exactly the
    scenario where chunk_size actually matters.
    """
    long_df = df[df["retrieval_text"].str.len() >= LONG_RECORD_THRESHOLD]
    test_set = []
    for _, row in long_df.iterrows():
        response = row["response_clean"]
        if not isinstance(response, str) or len(response) < 60:
            continue
        # Take a ~40-char excerpt starting 60% into the response text
        start = int(len(response) * 0.6)
        excerpt = response[start:start + 40].strip()
        if len(excerpt) < 20:
            continue
        test_set.append({"query": excerpt, "expected_drug": row["drug_name"]})
    return test_set


def build_vectorstore_for_config(long_df: pd.DataFrame, embedding, chunk_size: int, chunk_overlap: int) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " .", ". ", ".", ":", "؟", "?", "!"],
    )

    docs = []
    for _, row in long_df.iterrows():
        pieces = splitter.split_text(row["retrieval_text"])
        for piece in pieces:
            docs.append(Document(
                page_content="passage: " + piece,
                metadata={"drug_name": row["drug_name"]},
            ))

    return FAISS.from_documents(docs, embedding)


def evaluate_config(vectorstore: FAISS, test_set: list[dict]) -> float:
    hits = 0
    for item in test_set:
        query = "query: " + item["query"]
        results = vectorstore.similarity_search(query, k=TOP_K)
        retrieved_drugs = {doc.metadata.get("drug_name") for doc in results}
        if item["expected_drug"] in retrieved_drugs:
            hits += 1
    return hits / len(test_set)


def main():
    df = pd.read_csv(CLEANED_PATH)
    long_df = df[df["retrieval_text"].str.len() >= LONG_RECORD_THRESHOLD].copy()
    print(f"Long records used for this comparison: {len(long_df)}")

    test_set = build_test_set(df)
    print(f"Test questions built: {len(test_set)}")

    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    print("\n=== Results (hit rate: correct drug found in top-5) ===")
    for config in CONFIGS:
        vectorstore = build_vectorstore_for_config(
            long_df, embedding, config["chunk_size"], config["chunk_overlap"]
        )
        accuracy = evaluate_config(vectorstore, test_set)
        print(f"{config['name']}: {accuracy:.1%}")


if __name__ == "__main__":
    main()
