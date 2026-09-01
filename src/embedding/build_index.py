import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNK_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
FAISS_DIR = PROJECT_ROOT / "data" / "processed" / "faiss_index"

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

def load_documents(chunk_file: Path) -> list[Document]:
    """Load chunked records and convert them to LangChain Document objects."""
    docs = []
    with chunk_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            docs.append(Document(
                page_content="passage: " + item["text"],
                metadata={
                    "chunk_id": item["chunk_id"],
                    "doc_id": item["doc_id"],
                    "drug_name": item.get("drug_name"),
                    "therapeutic_category": item.get("therapeutic_category"),
                    "chunk_index": item.get("chunk_index"),
                },
            ))
    return docs


def main():
    if not CHUNK_FILE.exists():
        raise FileNotFoundError(
            f"Chunk file not found at: {CHUNK_FILE}\n"
            f"Run chunk_dataset.py first to generate it."
        )

    docs = load_documents(CHUNK_FILE)
    print(f"Loaded {len(docs)} chunks as documents")

    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Building FAISS index (this may take a while on first run)...")
    vectorstore = FAISS.from_documents(docs, embedding)

    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_DIR))

    print(f"Done! Index saved to: {FAISS_DIR}")


if __name__ == "__main__":
    main()