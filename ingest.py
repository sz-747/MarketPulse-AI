"""
Ingest pipeline to load a local PDF, split it into chunks, embed with a
Hugging Face model (384-dim), and push to Pinecone index `marketpulse`.
"""

from pathlib import Path
from typing import List, Sequence
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter



def find_first_pdf(data_dir: Path) -> Path:
    """Return the first PDF in the data directory."""
    for pdf_path in sorted(data_dir.glob("*.pdf")):
        return pdf_path
    raise FileNotFoundError(f"No PDF files found in {data_dir.resolve()}")


def load_pdf(pdf_path: Path) -> List[Document]:
    """Load PDF pages into a list of LangChain Documents."""
    try:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()  # List[Document] (each Document is a page)
        return pages
    except Exception as exc:
        raise RuntimeError(f"Failed to load PDF {pdf_path}") from exc


def split_docs(
    docs: Sequence[Document], *, chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[Document]:
    """
    Split documents into overlapping chunks.

    chunk_overlap preserves a bit of context across chunk boundaries so
    sentences/paragraphs that span a cut still have enough surrounding text
    for better retrieval accuracy.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(list(docs))


def embed_and_upsert(
    chunks: Sequence[Document], index_name: str, batch_size: int = 32
) -> int:
    """
    Embed chunks and upsert into Pinecone with progress logging.

    Returns the number of chunks attempted for upload.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    total = len(chunks)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = list(chunks[start:end])
        PineconeVectorStore.from_documents(
            documents=batch,
            embedding=embeddings,
            index_name=index_name,
        )
        if end % 10 == 0 or end == total:
            print(f"Uploaded {end}/{total} chunks...")

    return total


def main() -> None:
    data_dir = Path("data")
    pdf_path = find_first_pdf(data_dir)

    print(f"Loading PDF: {pdf_path.name}")
    pages = load_pdf(pdf_path)  # List[Document] (one per page)
    print(f"Pages loaded: {len(pages)}")

    chunks = split_docs(pages, chunk_size=1000, chunk_overlap=200)
    print(f"Chunks prepared: {len(chunks)}")

    index_name = "marketpulse"
    # Require API key via env var for safety; prevents hardcoding secrets.
    if not os.getenv("PINECONE_API_KEY"):
        raise EnvironmentError("PINECONE_API_KEY not set in environment")

    uploaded = embed_and_upsert(chunks, index_name=index_name)
    print(f"Success: uploaded {uploaded} chunks to Pinecone index '{index_name}'.")


if __name__ == "__main__":
    main()

