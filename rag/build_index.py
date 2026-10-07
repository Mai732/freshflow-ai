from pathlib import Path
import hashlib

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parents[1]

PDF_DIR = (
    BASE_DIR
    / "data"
    / "official_guidelines"
)

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

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def find_pdf():
    pdf_files = list(
        PDF_DIR.glob("*.pdf")
    )

    if not pdf_files:
        raise FileNotFoundError(
            "\nNo PDF found inside:\n"
            f"{PDF_DIR}"
        )

    return pdf_files[0]


def extract_pages(pdf_path):
    reader = PdfReader(
        str(pdf_path)
    )

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        text = page.extract_text()

        if not text:
            continue

        text = (
            text
            .replace("\n", " ")
            .replace("\r", " ")
            .strip()
        )

        if len(text) < 50:
            continue

        pages.append(
            {
                "page": page_number,
                "text": text,
            }
        )

    return pages


def chunk_text(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP,
):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(
            start + chunk_size,
            text_length,
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - overlap

    return chunks


def create_documents(
    pages,
    source_name,
):
    documents = []
    metadatas = []
    ids = []

    for page in pages:
        page_number = page["page"]
        text = page["text"]

        chunks = chunk_text(text)

        for chunk_index, chunk in enumerate(chunks):
            raw_id = (
                f"{source_name}-"
                f"{page_number}-"
                f"{chunk_index}"
            )

            chunk_id = hashlib.md5(
                raw_id.encode("utf-8")
            ).hexdigest()

            documents.append(chunk)

            metadatas.append(
                {
                    "source": source_name,
                    "page": page_number,
                    "chunk_index": chunk_index,
                }
            )

            ids.append(chunk_id)

    return (
        documents,
        metadatas,
        ids,
    )


def main():
    pdf_path = find_pdf()

    print("\nUsing PDF:")
    print(pdf_path.name)

    print("\nLoading embedding model...")

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    print("\nExtracting PDF pages...")

    pages = extract_pages(
        pdf_path
    )

    print(
        f"Pages extracted: "
        f"{len(pages)}"
    )

    documents, metadatas, ids = create_documents(
        pages,
        pdf_path.name,
    )

    print(
        f"Text chunks created: "
        f"{len(documents)}"
    )

    print("\nCreating embeddings...")

    embeddings = embedding_model.encode(
        documents,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = embeddings.tolist()

    print("\nCreating Chroma database...")

    CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    # Rebuild the collection from scratch
    try:
        client.delete_collection(
            COLLECTION_NAME
        )
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine"
        },
    )

    batch_size = 500

    for start in range(
        0,
        len(documents),
        batch_size,
    ):
        end = min(
            start + batch_size,
            len(documents),
        )

        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end],
        )

    print(
        "\n======================================"
    )
    print("RAG INDEX CREATED")

    print(
        f"\nStored chunks: "
        f"{collection.count()}"
    )

    print(
        f"\nDatabase saved to:\n"
        f"{CHROMA_DIR}"
    )


if __name__ == "__main__":
    main()