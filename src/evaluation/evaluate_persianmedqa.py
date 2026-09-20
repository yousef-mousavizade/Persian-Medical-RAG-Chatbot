"""
Formal evaluation against PersianMedQA - checks whether the project's
stated goal (retrieval accuracy >= 65%, Hit Rate@5) is actually met.

Methodology:
PersianMedQA has no drug_name ground truth field (unlike our own
dataset), so we can't directly reuse the earlier chunk_size comparison
script as-is. However, many "Pharmacology" field questions are
multiple-choice among DRUG NAMES themselves (e.g. "which drug is used
for X?" with drug names as the 4 options). So:

  1. We mine Persian drug-name spellings from our OWN dataset (same
     technique used earlier for the MeDiaPQA enrichment attempt): for
     each drug_name, find the Persian word that appears in the highest
     fraction of that drug's comments/responses, while excluding words
     that are too generic (appear across many different drugs).
  2. We filter PersianMedQA to field == "فارماکولوژی" (Pharmacology).
  3. For each question, we check whether the CORRECT answer option text
     matches one of our mined Persian drug names. Only matched
     questions are usable for this evaluation - a question whose
     correct answer isn't a drug our own dataset covers can't fairly
     test our retrieval system.
  4. For each usable question, we run full retrieval (k=5, same
     pipeline as production) using the question text as the query, and
     check Hit Rate@5: is the correct drug_name present among the
     top-5 retrieved chunks' metadata?

Requires: pip install pandas
          (plus everything retrieve.py already needs)
"""
import ast
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEANED_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_dataset.csv"

# Adjust this path to wherever your PersianMedQA CSV actually lives.
PERSIANMEDQA_PATH = PROJECT_ROOT / "data" / "raw" / "PersianMedQA-train.csv"

TOP_K = 5
PHARMACOLOGY_FIELD = "فارماکولوژی"

# Same thresholds as the mining step used for the MeDiaPQA experiment.
STOPWORD_DOC_FREQ_THRESHOLD = 0.25
MIN_COVERAGE = 0.30

ARABIC_TO_PERSIAN = {"ي": "ی", "ك": "ک", "ة": "ه", "ۀ": "ه", "أ": "ا", "إ": "ا", "ؤ": "و", "ئ": "ی"}


def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("_x000D_", " ").replace("\r", " ").replace("\n", " ")
    for ar, fa in ARABIC_TO_PERSIAN.items():
        text = text.replace(ar, fa)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> set:
    return set(re.findall(r"[\u0600-\u06FF]{3,}", text))


def mine_persian_drug_names(cleaned_df: pd.DataFrame) -> dict:
    """Return {persian_name: english_drug_name} mined from our own dataset."""
    group_doc_freq = Counter()
    row_presence = defaultdict(lambda: defaultdict(int))
    group_sizes = {}

    combined_text = cleaned_df["comment_clean"].fillna("") + " " + cleaned_df["response_clean"].fillna("")

    for drug, group in cleaned_df.assign(_text=combined_text).groupby("drug_name"):
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
    candidates = {}
    for drug in group_sizes:
        best_word, best_score = None, 0
        for w, cnt in row_presence[drug].items():
            if group_doc_freq[w] / total_groups > STOPWORD_DOC_FREQ_THRESHOLD:
                continue
            coverage = cnt / group_sizes[drug]
            if coverage > best_score:
                best_score, best_word = coverage, w
        if best_word and best_score >= MIN_COVERAGE:
            candidates[best_word] = drug

    return candidates


def build_test_set(persianmedqa_df: pd.DataFrame, persian_to_english: dict) -> list[dict]:
    """Match PersianMedQA pharmacology questions to a known drug_name."""
    pharma_df = persianmedqa_df[persianmedqa_df["field"] == PHARMACOLOGY_FIELD]

    test_set = []
    for _, row in pharma_df.iterrows():
        try:
            options = ast.literal_eval(row["answer"])
            correct_text = options[str(row["correct answer"])].strip()
        except (ValueError, KeyError, SyntaxError):
            continue

        normalized_correct = normalize(correct_text)
        expected_drug = None
        for persian_name, english_name in persian_to_english.items():
            if persian_name in normalized_correct or normalized_correct in persian_name:
                expected_drug = english_name
                break

        if expected_drug:
            test_set.append({"query": row["question"], "expected_drug": expected_drug})

    return test_set


def evaluate(vectorstore, test_set: list[dict]) -> float:
    hits = 0
    for item in test_set:
        query = "query: " + item["query"]
        results = vectorstore.similarity_search(query, k=TOP_K)
        retrieved_drugs = {doc.metadata.get("drug_name") for doc in results}
        if item["expected_drug"] in retrieved_drugs:
            hits += 1
    return hits / len(test_set) if test_set else 0.0


def main():
    if not CLEANED_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at: {CLEANED_PATH}")
    if not PERSIANMEDQA_PATH.exists():
        raise FileNotFoundError(
            f"PersianMedQA CSV not found at: {PERSIANMEDQA_PATH}\n"
            f"Edit PERSIANMEDQA_PATH at the top of this script if it's stored elsewhere."
        )

    from retrieval.retrieve import load_vectorstore  # local import: needs sys.path set by caller

    cleaned_df = pd.read_csv(CLEANED_PATH)
    persianmedqa_df = pd.read_csv(PERSIANMEDQA_PATH)

    print("Mining Persian drug-name spellings from our own dataset...")
    persian_to_english = mine_persian_drug_names(cleaned_df)
    print(f"Mined {len(persian_to_english)} reliable drug-name candidates")

    test_set = build_test_set(persianmedqa_df, persian_to_english)
    print(f"Usable PersianMedQA pharmacology questions (answer matches a known drug): {len(test_set)}")

    if not test_set:
        print("No usable test questions found - check PERSIANMEDQA_PATH and field name.")
        return

    print("Loading vector store (this may take a moment)...")
    vectorstore = load_vectorstore()

    print("Running evaluation...")
    accuracy = evaluate(vectorstore, test_set)

    print("\n=== Result ===")
    print(f"Hit Rate@{TOP_K}: {accuracy:.1%}  (target: >= 65%)")
    print(f"Based on {len(test_set)} PersianMedQA pharmacology questions")
    if accuracy >= 0.65:
        print("Target MET.")
    else:
        print("Target NOT met - see project report for discussion.")


if __name__ == "__main__":
    import sys
    sys.path.append(str(PROJECT_ROOT / "src"))
    main()
