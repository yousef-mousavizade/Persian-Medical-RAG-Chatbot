from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FAISS_DIR = PROJECT_ROOT / "data" / "processed" / "faiss_index"

# Must match the exact model used in build_index.py - mixing embedding
# models between indexing and querying produces meaningless results.
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# Number of chunks to retrieve per query (k=5 per the project proposal)
TOP_K = 5

def load_vectorstore() -> FAISS:
    if not FAISS_DIR.exists():
        raise FileNotFoundError(
            f"FAISS index not found at: {FAISS_DIR}\n"
            f"Run build_index.py first to generate it."
        )

    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    return FAISS.load_local(
        str(FAISS_DIR),
        embedding,
        allow_dangerous_deserialization=True,
    )


def search(vectorstore: FAISS, query: str, k: int = TOP_K):
    """Return the top-k most relevant chunks for a given query.

    Note: multilingual-e5 models require a "query: " prefix on search
    queries (documents were indexed with a "passage: " prefix in
    build_index.py). This is a requirement of the model itself.
    """
    prefixed_query = "query: " + query
    results = vectorstore.similarity_search_with_score(prefixed_query, k=k)
    return results


def print_results(query: str, results) -> None:
    print(f"\nQuery: {query}")
    print(f"Top {len(results)} results:\n")
    for rank, (doc, score) in enumerate(results, start=1):
        print(f"[{rank}] score={score:.4f} | drug={doc.metadata.get('drug_name')}")
        print(f"    {doc.page_content}")
        print()


if __name__ == "__main__":
    vectorstore = load_vectorstore()

    # A few sample Persian queries to sanity-check retrieval quality
    sample_queries = [
        "آسپرین برای سردرد خوبه؟",
        "مصرف قرص فلوکستین چه عوارضی داره؟",
        "آیا شربت لاکتولوز برای یبوست کودکان مناسب است؟",
    ]

    for q in sample_queries:
        results = search(vectorstore, q)
        print_results(q, results)