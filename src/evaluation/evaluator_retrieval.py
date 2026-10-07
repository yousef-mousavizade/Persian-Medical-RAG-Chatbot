
"""
Formal evaluation of the FINAL production retrieval system
against our own dataset.

Evaluation methodology:
- No Gold Dataset is used.
- No external ground truth is used.
- The existing anti-leakage test-set construction is preserved.
- Retrieval is evaluated only at Top-K = 5.
- Relevance is defined at the drug level:
    retrieved_drug == expected_drug

Metrics:
- Precision@5
- Recall@5
- Hit Rate@5
- False Positive Rate@5

Important:
- Recall@5 and Hit Rate@5 are numerically identical under
  the current methodology because both measure whether the
  expected drug appears in the Top-5 results.
- Precision@5 and False Positive Rate@5 are complementary
  under this drug-level definition.
- False Answer and Citation Accuracy are intentionally NOT
  calculated here. They belong to the generation evaluation
  and require the actual LLM/main pipeline.
"""

import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEANED_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_dataset.csv"

TOP_K = 5

EXCERPT_WINDOW = 60

# Same thresholds used by the original drug-name mining method.
STOPWORD_DOC_FREQ_THRESHOLD = 0.25
MIN_COVERAGE = 0.30

# Set to None to evaluate the complete test set.
# 300 provides a fixed, reproducible sample for faster evaluation.
SAMPLE_SIZE = 300
RANDOM_SEED = 42


# ============================================================
# Text processing
# ============================================================

def tokenize(text: str) -> set:
    """
    Extract Persian words with at least 3 characters.
    """
    return set(re.findall(r"[\u0600-\u06FF]{3,}", text))


# ============================================================
# Drug-name mining
# ============================================================

def mine_persian_drug_names(df: pd.DataFrame) -> dict:
    """
    Mine a reliable Persian drug-name candidate for each drug.

    For each drug, the Persian word with the highest coverage
    across that drug's own records is selected, while excluding
    words that are too common across different drugs.
    """

    combined_text = (
        df["comment_clean"].fillna("")
        + " "
        + df["response_clean"].fillna("")
    )

    group_doc_freq = Counter()
    row_presence = defaultdict(lambda: defaultdict(int))
    group_sizes = {}

    for drug, group in df.assign(_text=combined_text).groupby("drug_name"):
        group_sizes[drug] = len(group)
        words_in_group = set()

        for text in group["_text"]:
            tokens = tokenize(text)

            words_in_group |= tokens

            for word in tokens:
                row_presence[drug][word] += 1

        for word in words_in_group:
            group_doc_freq[word] += 1

    total_groups = len(group_sizes)

    names = {}

    for drug in group_sizes:
        best_word = None
        best_score = 0

        for word, count in row_presence[drug].items():

            # Ignore words appearing across too many drugs.
            if group_doc_freq[word] / total_groups > STOPWORD_DOC_FREQ_THRESHOLD:
                continue

            coverage = count / group_sizes[drug]

            if coverage > best_score:
                best_score = coverage
                best_word = word

        if best_word and best_score >= MIN_COVERAGE:
            names[drug] = best_word

    return names


# ============================================================
# Test-set construction
# ============================================================

def build_test_set(
    df: pd.DataFrame,
    drug_persian_names: dict,
) -> list[dict]:
    """
    Build evaluation queries using an actual occurrence of the
    mined Persian drug name inside the corresponding response.

    This preserves the existing anti-leakage methodology:
    the query is an excerpt around the drug mention rather than
    simply using the original question or an exact chunk.
    """

    test_set = []

    for _, row in df.iterrows():

        persian_name = drug_persian_names.get(row["drug_name"])

        if not persian_name:
            continue

        text = row["response_clean"]

        if not isinstance(text, str):
            continue

        index = text.find(persian_name)

        if index == -1:
            continue

        start = max(
            0,
            index - EXCERPT_WINDOW // 2,
        )

        end = min(
            len(text),
            index + len(persian_name) + EXCERPT_WINDOW // 2,
        )

        excerpt = text[start:end].strip()

        test_set.append(
            {
                "query": excerpt,
                "expected_drug": row["drug_name"],
            }
        )

    return test_set


# ============================================================
# Per-query retrieval metrics
# ============================================================

def calculate_metrics(
    results,
    expected_drug: str,
) -> dict:
    """
    Calculate Top-5 retrieval metrics for one query.

    Relevant chunk:
        retrieved_drug == expected_drug

    Metrics:
        Precision@5
        Recall@5
        Hit Rate@5
        False Positive Rate@5
    """

    retrieved_drugs = [
        doc.metadata.get("drug_name")
        for doc in results
    ]

    # Safety check: evaluation should receive exactly TOP_K results.
    retrieved_drugs = retrieved_drugs[:TOP_K]

    relevant_count = sum(
        drug == expected_drug
        for drug in retrieved_drugs
    )

    false_positive_count = sum(
        drug != expected_drug
        for drug in retrieved_drugs
    )

    # Precision@5:
    # Relevant retrieved chunks / total retrieved chunks.
    precision_at_5 = (
        relevant_count / TOP_K
        if TOP_K > 0
        else 0.0
    )

    # Recall@5:
    # Under the current drug-level methodology, there is one
    # relevant drug target. Therefore recall is 1 if the expected
    # drug appears anywhere in Top-5, otherwise 0.
    recall_at_5 = int(
        expected_drug in retrieved_drugs
    )

    # Hit Rate@5 uses the same definition as Recall@5 here.
    hit_rate_at_5 = int(
        expected_drug in retrieved_drugs
    )

    # False Positive Rate@5:
    # Wrong-drug chunks / total retrieved chunks.
    false_positive_rate_at_5 = (
        false_positive_count / TOP_K
        if TOP_K > 0
        else 0.0
    )

    return {
        "precision": precision_at_5,
        "recall": recall_at_5,
        "hit_rate": hit_rate_at_5,
        "false_positive_rate": false_positive_rate_at_5,
        "relevant_count": relevant_count,
        "false_positive_count": false_positive_count,
    }


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    vectorstore,
    test_set: list[dict],
    show_failures: int = 5,
) -> dict:
    """
    Run the complete Top-5 retrieval evaluation.
    """

    totals = {
        "precision": 0.0,
        "recall": 0.0,
        "hit_rate": 0.0,
        "false_positive_rate": 0.0,
        "relevant_count": 0,
        "false_positive_count": 0,
    }

    failures_shown = 0

    for item in test_set:

        query = "query: " + item["query"]
        expected_drug = item["expected_drug"]

        # One FAISS search per query.
        results = vectorstore.similarity_search(
            query,
            k=TOP_K,
        )

        metrics = calculate_metrics(
            results,
            expected_drug,
        )

        totals["precision"] += metrics["precision"]
        totals["recall"] += metrics["recall"]
        totals["hit_rate"] += metrics["hit_rate"]
        totals["false_positive_rate"] += (
            metrics["false_positive_rate"]
        )

        totals["relevant_count"] += (
            metrics["relevant_count"]
        )

        totals["false_positive_count"] += (
            metrics["false_positive_count"]
        )

        # Show a few complete Top-5 retrieval failures.
        if (
            metrics["hit_rate"] == 0
            and failures_shown < show_failures
        ):
            retrieved_drugs = [
                doc.metadata.get("drug_name")
                for doc in results
            ]

            print(
                f"\n[MISS] excerpt: {item['query']}"
            )
            print(
                f"       expected_drug: {expected_drug}"
            )
            print(
                f"       retrieved_drugs: {retrieved_drugs}"
            )

            failures_shown += 1

    n = len(test_set)

    if n == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "hit_rate": 0.0,
            "false_positive_rate": 0.0,
            "relevant_count": 0,
            "false_positive_count": 0,
        }

    # Mean per-query metrics.
    totals["precision"] /= n
    totals["recall"] /= n
    totals["hit_rate"] /= n
    totals["false_positive_rate"] /= n

    return totals


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not CLEANED_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at: {CLEANED_PATH}"
        )

    # --------------------------------------------------------
    # Import production retrieval
    # --------------------------------------------------------

    sys.path.append(str(PROJECT_ROOT / "src"))

    from retrieval.retrieve import load_vectorstore

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(CLEANED_PATH)

    print(
        f"Total records in dataset: {len(df)}"
    )

    # --------------------------------------------------------
    # Mine Persian drug names
    # --------------------------------------------------------

    print(
        "Mining Persian drug-name spellings "
        "from our own dataset..."
    )

    drug_persian_names = mine_persian_drug_names(df)

    print(
        f"Mined {len(drug_persian_names)} "
        "reliable drug-name candidates"
    )

    # --------------------------------------------------------
    # Build evaluation test set
    # --------------------------------------------------------

    test_set = build_test_set(
        df,
        drug_persian_names,
    )

    print(
        "Usable test questions "
        "(row's own text contains its drug's mined name): "
        f"{len(test_set)}"
    )

    # --------------------------------------------------------
    # Reproducible sampling
    # --------------------------------------------------------

    if SAMPLE_SIZE and len(test_set) > SAMPLE_SIZE:

        random.seed(RANDOM_SEED)

        test_set = random.sample(
            test_set,
            SAMPLE_SIZE,
        )

        print(
            f"Sampled down to {SAMPLE_SIZE} "
            "for faster evaluation"
        )

    # --------------------------------------------------------
    # Load production vector store
    # --------------------------------------------------------

    print(
        "Loading production vector store "
        "(data/processed/faiss_index)..."
    )

    vectorstore = load_vectorstore()

    # --------------------------------------------------------
    # Run evaluation
    # --------------------------------------------------------

    print(
        f"Running retrieval evaluation "
        f"with Top-K={TOP_K}..."
    )

    metrics = evaluate(
        vectorstore,
        test_set,
        show_failures=5,
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 55)
    print("Retrieval Evaluation")
    print("=" * 55)

    print(
        f"Precision@{TOP_K}: "
        f"{metrics['precision']:.4f} "
        f"({metrics['precision']:.1%})"
    )

    print(
        f"Recall@{TOP_K}: "
        f"{metrics['recall']:.4f} "
        f"({metrics['recall']:.1%})"
    )

    print(
        f"Hit Rate@{TOP_K}: "
        f"{metrics['hit_rate']:.4f} "
        f"({metrics['hit_rate']:.1%})"
    )

    print(
        f"False Positive Rate@{TOP_K}: "
        f"{metrics['false_positive_rate']:.4f} "
        f"({metrics['false_positive_rate']:.1%})"
    )

    print("\n" + "-" * 55)

    print(
        f"Relevant chunks retrieved: "
        f"{metrics['relevant_count']}"
    )

    print(
        f"False-positive chunks: "
        f"{metrics['false_positive_count']}"
    )

    print(
        f"Total retrieved chunks: "
        f"{len(test_set) * TOP_K}"
    )

    # --------------------------------------------------------
    # Generation metrics
    # --------------------------------------------------------

    print("\n" + "=" * 55)
    print("Generation Evaluation")
    print("=" * 55)

    print(
        "False Answer Rate: N/A"
    )

    print(
        "Reason: generation evaluation is handled separately "
        "using the actual LLM/main pipeline."
    )

    print(
        "Citation Accuracy: N/A"
    )

    print(
        "Reason: citation evaluation is handled separately "
        "using the actual LLM/main pipeline."
    )

    # --------------------------------------------------------
    # Methodology notes
    # --------------------------------------------------------

    print("\n" + "=" * 55)
    print("Evaluation Notes")
    print("=" * 55)

    print(
        f"Top-K: {TOP_K}"
    )

    print(
        "Recall@5 and Hit Rate@5 are numerically identical "
        "under the current methodology because both measure "
        "whether expected_drug appears in Top-5."
    )

    print(
        "Precision@5 and False Positive Rate@5 are "
        "drug-level retrieval metrics."
    )

    print(
        "They do not represent full semantic chunk relevance."
    )

    print(
        "No Gold Dataset or external Ground Truth was used."
    )

    print(
        f"\nBased on {len(test_set)} test cases."
    )


if __name__ == "__main__":
    main()