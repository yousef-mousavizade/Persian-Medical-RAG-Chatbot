"""
Generation-level evaluation for the FINAL production pipeline.

Important:
- No Gold Dataset is used.
- No external Ground Truth is used.
- Test cases are constructed with the SAME anti-leakage methodology
  as evaluate_own_data.py.
- The real FastAPI /ask endpoint is executed, so retrieval + generation
  are evaluated together.
- Citation Accuracy is reported as N/A because the current API does not
  expose explicit citation markers in the generated answer. It only
  returns source metadata separately.
- A full clinical False Answer Rate cannot be established without
  ground-truth answers. Therefore this script reports a conservative
  deterministic "False Answer Rate (drug attribution)" only for cases
  where the answer explicitly names a drug.

Run from the project root:
    python evaluate_generation.py
"""

import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
import uuid
import pandas as pd
from fastapi.testclient import TestClient


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent[2]
CLEANED_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_dataset.csv"

TOP_K = 5
EXCERPT_WINDOW = 60

STOPWORD_DOC_FREQ_THRESHOLD = 0.25
MIN_COVERAGE = 0.30

SAMPLE_SIZE = 300
RANDOM_SEED = 42

NO_CONFIDENT_MATCH_MESSAGE = (
    "اطلاعات کافی و مرتبطی در منابع موجود برای پاسخ به این سؤال پیدا نشد."
)


# ============================================================
# Text processing / drug-name mining
# ============================================================

def tokenize(text: str) -> set:
    """Extract Persian words with at least 3 characters."""
    return set(re.findall(r"[\u0600-\u06FF]{3,}", text))


def mine_persian_drug_names(df: pd.DataFrame) -> dict:
    """
    Same drug-name mining logic used by evaluate_own_data.py.
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

            if (
                group_doc_freq[word] / total_groups
                > STOPWORD_DOC_FREQ_THRESHOLD
            ):
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
    Same anti-leakage test-set construction used by
    evaluate_own_data.py.
    """

    test_set = []

    for _, row in df.iterrows():

        persian_name = drug_persian_names.get(
            row["drug_name"]
        )

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
            index
            + len(persian_name)
            + EXCERPT_WINDOW // 2,
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
# Generation evaluation
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize common Persian/Arabic character variants.
    """

    return (
        str(text)
        .replace("ي", "ی")
        .replace("ى", "ی")
        .replace("ك", "ک")
        .replace("\u200c", " ")
        .strip()
    )


def contains_name(
    text: str,
    name: str,
) -> bool:

    text = normalize_text(text).lower()
    name = normalize_text(name).lower()

    if not name:
        return False

    return name in text


def classify_drug_attribution(
    answer: str,
    expected_drug: str,
    drug_names: dict,
) -> str:

    """
    Conservative deterministic classification.

    CORRECT:
        Expected drug is explicitly mentioned and no other
        known drug is mentioned.

    FALSE:
        A known drug other than the expected drug is mentioned,
        while the expected drug is not mentioned.

    AMBIGUOUS:
        Expected drug and at least one other known drug
        are both mentioned.

    UNDETERMINED:
        No known drug is explicitly mentioned.

    This is NOT a medical correctness judge.
    It only evaluates explicit drug attribution.
    """

    answer_norm = normalize_text(answer)

    expected_mentioned = contains_name(
        answer_norm,
        expected_drug,
    )

    mentioned_drugs = []

    for drug, persian_name in drug_names.items():

        if (
            contains_name(
                answer_norm,
                persian_name,
            )
            or contains_name(
                answer_norm,
                drug,
            )
        ):
            mentioned_drugs.append(drug)

    mentioned_drugs = list(
        dict.fromkeys(mentioned_drugs)
    )

    other_drugs = [
        drug
        for drug in mentioned_drugs
        if drug != expected_drug
    ]

    if expected_mentioned and not other_drugs:
        return "CORRECT"

    if not expected_mentioned and other_drugs:
        return "FALSE"

    if expected_mentioned and other_drugs:
        return "AMBIGUOUS"

    return "UNDETERMINED"


# ============================================================
# Run one production request
# ============================================================

def evaluate_case(
    client,
    item,
    drug_names,
):

    question = item["query"]
    expected_drug = item["expected_drug"]

    start = time.perf_counter()

    response = client.post(
        "/ask",
        json={
            "question": question,
            "session_id": str(uuid.uuid4()),
        },
    )

    latency = time.perf_counter() - start

    if response.status_code != 200:

        return {
            "status": "HTTP_ERROR",
            "latency": latency,
            "http_status": response.status_code,
            "expected_drug": expected_drug,
            "answer": "",
            "sources": [],
            "classification": "UNDETERMINED",
        }

    payload = response.json()

    answer = payload.get(
        "answer",
        "",
    )

    sources = payload.get(
        "sources",
        [],
    )

    is_abstained = (
        not sources
        and NO_CONFIDENT_MATCH_MESSAGE in answer
    )

    if is_abstained:

        classification = "ABSTAINED"

    else:

        classification = classify_drug_attribution(
            answer,
            expected_drug,
            drug_names,
        )

    return {
        "status": "OK",
        "latency": latency,
        "http_status": response.status_code,
        "expected_drug": expected_drug,
        "answer": answer,
        "sources": sources,
        "classification": classification,
    }


# ============================================================
# Main
# ============================================================

def main():

    if not CLEANED_PATH.exists():

        raise FileNotFoundError(
            f"Cleaned dataset not found at: {CLEANED_PATH}"
        )

    # --------------------------------------------------------
    # Make src importable
    # --------------------------------------------------------

    sys.path.insert(
        0,
        str(PROJECT_ROOT / "src"),
    )

    from fastAPI.main import app

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print(
        f"Dataset: {CLEANED_PATH}"
    )

    df = pd.read_csv(
        CLEANED_PATH
    )

    print(
        f"Total records in dataset: {len(df)}"
    )

    # --------------------------------------------------------
    # Mine drug names
    # --------------------------------------------------------

    print(
        "Mining Persian drug names..."
    )

    drug_names = mine_persian_drug_names(
        df
    )

    print(
        f"Mined {len(drug_names)} "
        "reliable drug-name candidates."
    )

    # --------------------------------------------------------
    # Build same test set
    # --------------------------------------------------------

    print(
        "Building test set with the existing "
        "anti-leakage methodology..."
    )

    test_set = build_test_set(
        df,
        drug_names,
    )

    print(
        f"Usable test cases: {len(test_set)}"
    )

    # --------------------------------------------------------
    # Sampling
    # --------------------------------------------------------

    if (
        SAMPLE_SIZE
        and len(test_set) > SAMPLE_SIZE
    ):

        random.seed(
            RANDOM_SEED
        )

        test_set = random.sample(
            test_set,
            SAMPLE_SIZE,
        )

    print(
        f"Evaluation cases after sampling: "
        f"{len(test_set)} "
        f"(seed={RANDOM_SEED})"
    )

    # --------------------------------------------------------
    # Production FastAPI
    # --------------------------------------------------------

    print(
        "\nStarting production FastAPI application..."
    )

    print(
        "WARNING: this evaluates the real /ask pipeline "
        "and therefore runs the local LLM for every test case."
    )

    results = []

    # FastAPI lifespan loads the production vector store.
    with TestClient(app) as client:

        for i, item in enumerate(
            test_set,
            start=1,
        ):

            result = evaluate_case(
                client,
                item,
                drug_names,
            )

            results.append(
                result
            )

            if (
                i % 10 == 0
                or i == len(test_set)
            ):

                print(
                    f"Progress: "
                    f"{i}/{len(test_set)}"
                )

    # ========================================================
    # Aggregate
    # ========================================================

    total = len(results)

    ok_results = [
        r
        for r in results
        if r["status"] == "OK"
    ]

    http_errors = [
        r
        for r in results
        if r["status"] != "OK"
    ]

    abstained = [
        r
        for r in ok_results
        if r["classification"] == "ABSTAINED"
    ]

    correct = [
        r
        for r in ok_results
        if r["classification"] == "CORRECT"
    ]

    false_answers = [
        r
        for r in ok_results
        if r["classification"] == "FALSE"
    ]

    ambiguous = [
        r
        for r in ok_results
        if r["classification"] == "AMBIGUOUS"
    ]

    undetermined = [
        r
        for r in ok_results
        if r["classification"] == "UNDETERMINED"
    ]

    evaluable_attribution = (
        correct
        + false_answers
    )

    if evaluable_attribution:

        false_answer_rate = (
            len(false_answers)
            / len(evaluable_attribution)
        )

    else:

        false_answer_rate = None

    if ok_results:

        avg_latency = (
            sum(
                r["latency"]
                for r in ok_results
            )
            / len(ok_results)
        )

    else:

        avg_latency = None

    # ========================================================
    # Results
    # ========================================================

    print("\n" + "=" * 65)
    print("Generation Evaluation")
    print("=" * 65)

    print(
        "False Answer Rate "
        "(full medical/clinical correctness): N/A"
    )

    print(
        "Reason: no Gold Dataset or independent "
        "ground-truth answer is available."
    )

    print(
        "\nFalse Answer Rate "
        "(deterministic drug attribution): "
        + (
            f"{false_answer_rate:.4f} "
            f"({false_answer_rate:.1%})"
            if false_answer_rate is not None
            else "N/A"
        )
    )

    print(
        f"  Evaluated cases: "
        f"{len(evaluable_attribution)}"
    )

    print(
        f"  Correct drug attribution: "
        f"{len(correct)}"
    )

    print(
        f"  False drug attribution: "
        f"{len(false_answers)}"
    )

    print(
        f"  Ambiguous: "
        f"{len(ambiguous)}"
    )

    print(
        f"  Undetermined: "
        f"{len(undetermined)}"
    )

    # --------------------------------------------------------
    # Citation Accuracy
    # --------------------------------------------------------

    print(
        "\nCitation Accuracy: N/A"
    )

    print(
        "Reason: the current generator does not emit "
        "explicit citation markers in the final answer."
    )

    print(
        "The API returns source metadata separately, "
        "but that is not equivalent to a citation "
        "inside the generated answer."
    )

    # --------------------------------------------------------
    # Abstention
    # --------------------------------------------------------

    if total:

        abstention_rate = (
            len(abstained)
            / total
        )

        print(
            f"\nAbstention Rate: "
            f"{abstention_rate:.4f} "
            f"({abstention_rate:.1%})"
        )

    else:

        print(
            "\nAbstention Rate: N/A"
        )

    # --------------------------------------------------------
    # Errors / latency
    # --------------------------------------------------------

    print(
        f"\nHTTP Errors: "
        f"{len(http_errors)}"
    )

    print(
        "Average end-to-end latency: "
        + (
            f"{avg_latency:.2f} seconds"
            if avg_latency is not None
            else "N/A"
        )
    )

    # ========================================================
    # Methodology
    # ========================================================

    print(
        "\n" + "=" * 65
    )

    print(
        "Methodology Notes"
    )

    print(
        "=" * 65
    )

    print(
        "1. No Gold Dataset was used."
    )

    print(
        "2. No external Ground Truth was used."
    )

    print(
        "3. Test queries were constructed using the "
        "same anti-leakage methodology as evaluate_own_data.py."
    )

    print(
        "4. The actual production /ask endpoint was evaluated."
    )

    print(
        "5. Full medical correctness cannot be established "
        "from the current data alone."
    )

    print(
        "6. The deterministic drug-attribution metric only "
        "checks explicit drug-name attribution and must not "
        "be described as clinical answer accuracy."
    )

    print(
        "7. Citation Accuracy is N/A because the current "
        "generated answer contains no explicit citation mechanism."
    )

    print(
        f"\nTotal test cases: "
        f"{total}"
    )


if __name__ == "__main__":
    main()