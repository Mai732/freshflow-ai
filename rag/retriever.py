from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parents[1]

CHROMA_DIR = (
    BASE_DIR
    / "data"
    / "knowledge_base"
    / "chroma_db"
)

COLLECTION_NAME = "fda_food_guidelines"

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


_embedding_model = None
_collection = None


def get_embedding_model():
    global _embedding_model

    if _embedding_model is None:
        _embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

    return _embedding_model


def get_collection():
    global _collection

    if _collection is not None:
        return _collection

    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            "\nChroma database does not exist.\n"
            "Run first:\n"
            "python rag/build_index.py"
        )

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    _collection = client.get_collection(
        COLLECTION_NAME
    )

    return _collection


def retrieve_guidance(
    query,
    top_k=5,
):
    model = get_embedding_model()
    collection = get_collection()

    query_embedding = (
        model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        .tolist()
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    documents = results.get(
        "documents",
        [[]],
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]],
    )[0]

    distances = results.get(
        "distances",
        [[]],
    )[0]

    retrieved = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        retrieved.append(
            {
                "text": document,
                "source": metadata.get("source"),
                "page": metadata.get("page"),
                "distance": float(distance),
            }
        )

    return retrieved


def format_guidance(retrieved_chunks):
    if not retrieved_chunks:
        return (
            "No relevant FDA guidance "
            "was retrieved."
        )

    sections = []

    for index, item in enumerate(
        retrieved_chunks,
        start=1,
    ):
        sections.append(
            f"[FDA SOURCE {index}]\n"
            f"Source: {item['source']}\n"
            f"Page: {item['page']}\n"
            f"Text: {item['text']}"
        )

    return "\n\n".join(sections)


if __name__ == "__main__":
    query = input(
        "Enter RAG query: "
    )

    results = retrieve_guidance(
        query,
        top_k=5,
    )

    print(
        "\n======================================"
    )

    print(
        "RETRIEVED FDA GUIDANCE"
    )

    print(
        format_guidance(
            results
        )
    )