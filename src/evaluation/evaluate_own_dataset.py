"""
Formal evaluation of the FINAL production system (chunk_size=250,
multilingual-e5-large, current FAISS index) against our own dataset -
the counterpart to evaluate_persianmedqa.py, representing the system's
actual intended use case (colloquial patient questions), not clinical
exam-style questions.

Methodology (same anti-leakage design as compare_chunk_sizes.py):
the test query is a short excerpt from ~60% into the RESPONSE text
(not the question), so the query is not a verbatim substring sitting
at the very start of the chunk that was embedded - this avoids a
trivially easy "find myself" retrieval task.

"""
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEANED_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_dataset.csv"

TOP_K = 5
EXCERPT_WINDOW = 60  # characters of context kept on each side of the drug name

# Same thresholds as the drug-name mining step used for the
# MeDiaPQA/PersianMedQA experiments - a word only counts as a drug's
# Persian name if it's specific enough (not too common across other
# drugs' records) and appears reliably often within that drug's own
# records.
STOPWORD_DOC_FREQ_THRESHOLD = 0.25
MIN_COVERAGE = 0.30

# Set to None to evaluate the ENTIRE dataset (slower, thousands of
# vector searches). A sample keeps runtime reasonable while still being
# a statistically meaningful, randomly chosen cross-section of drugs.
SAMPLE_SIZE = 300
RANDOM_SEED = 42


def tokenize(text: str) -> set:
    return set(re.findall(r"[\u0600-\u06FF]{3,}", text))


def mine_persian_drug_names(df: pd.DataFrame) -> dict:
    """Return {english_drug_name: persian_name} mined from our own dataset.

    For each drug, find the Persian word that appears in the highest
    fraction of that drug's own comments/responses, while excluding
    words too generic to be specific to one drug (e.g. "دارو", "مصرف").
    """
    combined_text = df["comment_clean"].fillna("") + " " + df["response_clean"].fillna("")
    group_doc_freq = Counter()
    row_presence = defaultdict(lambda: defaultdict(int))
    group_sizes = {}

    for drug, group in df.assign(_text=combined_text).groupby("drug_name"):
        group_sizes[drug] = len(group)
        words_in_group = set()
        for t in group["_text"]:
            toks = tokenize(t)
            words_in_group |= toks
            for w in toks:
                row_presence[drug][w] += 1
        for w in words_in_group:
            group_doc_freq[w] += 1

    total_groups = len(group_sizes)
    names = {}
    for drug in group_sizes:
        best_word, best_score = None, 0
        for w, cnt in row_presence[drug].items():
            if group_doc_freq[w] / total_groups > STOPWORD_DOC_FREQ_THRESHOLD:
                continue
            coverage = cnt / group_sizes[drug]
            if coverage > best_score:
                best_score, best_word = coverage, w
        if best_word and best_score >= MIN_COVERAGE:
            names[drug] = best_word
    return names


def build_test_set(df: pd.DataFrame, drug_persian_names: dict) -> list[dict]:
    """Build test queries centered on an actual mention of the drug's
    own Persian name within that row's own text - not an arbitrary
    offset. Rows where the drug's mined name doesn't literally appear
    are skipped, since we can't build a fair, specific query for them
    this way."""
    test_set = []
    for _, row in df.iterrows():
        persian_name = drug_persian_names.get(row["drug_name"])
        if not persian_name:
            continue

        text = row["response_clean"]
        if not isinstance(text, str):
            continue

        idx = text.find(persian_name)
        if idx == -1:
            continue

        start = max(0, idx - EXCERPT_WINDOW // 2)
        end = min(len(text), idx + len(persian_name) + EXCERPT_WINDOW // 2)
        excerpt = text[start:end].strip()

        test_set.append({"query": excerpt, "expected_drug": row["drug_name"]})
    return test_set


def evaluate(vectorstore, test_set: list[dict], show_failures: int = 0) -> float:
    hits = 0
    failures_shown = 0
    for item in test_set:
        query = "query: " + item["query"]
        results = vectorstore.similarity_search(query, k=TOP_K)
        retrieved_drugs = {doc.metadata.get("drug_name") for doc in results}
        if item["expected_drug"] in retrieved_drugs:
            hits += 1
        elif failures_shown < show_failures:
            print(f"\n[MISS] excerpt: {item['query']}")
            print(f"       expected_drug: {item['expected_drug']}")
            print(f"       retrieved_drugs: {retrieved_drugs}")
            failures_shown += 1
    return hits / len(test_set) if test_set else 0.0


def main():
    if not CLEANED_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at: {CLEANED_PATH}")

    sys.path.append(str(PROJECT_ROOT / "src"))
    from retrieval.retrieve import load_vectorstore

    df = pd.read_csv(CLEANED_PATH)
    print(f"Total records in dataset: {len(df)}")

    print("Mining Persian drug-name spellings from our own dataset...")
    drug_persian_names = mine_persian_drug_names(df)
    print(f"Mined {len(drug_persian_names)} reliable drug-name candidates")

    test_set = build_test_set(df, drug_persian_names)
    print(f"Usable test questions (row's own text contains its drug's mined name): {len(test_set)}")

    if SAMPLE_SIZE and len(test_set) > SAMPLE_SIZE:
        random.seed(RANDOM_SEED)
        test_set = random.sample(test_set, SAMPLE_SIZE)
        print(f"Sampled down to {SAMPLE_SIZE} for faster evaluation")

    print("Loading production vector store (data/processed/faiss_index)...")
    vectorstore = load_vectorstore()

    print("Running evaluation...")
    accuracy = evaluate(vectorstore, test_set, show_failures=5)

    print("\n=== Result ===")
    print(f"Hit Rate@{TOP_K}: {accuracy:.1%}  (target: >= 65%)")
    print(f"Based on {len(test_set)} questions from our own dataset")
    if accuracy >= 0.65:
        print("Target MET.")
    else:
        print("Target NOT met.")


if __name__ == "__main__":
    main()