"""
Quick RAG playground for experimenting with PDF loading and chunking.

Loads the first PDF found in the `data/` folder, splits it into overlapping
chunks, prints basic stats, and shows sample chunks for inspection.
"""

from pathlib import Path
import random
from typing import List, Sequence

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def find_first_pdf(data_dir: Path) -> Path:
    """Return the first PDF file found in the given directory."""
    for pdf_path in sorted(data_dir.glob("*.pdf")):
        return pdf_path
    raise FileNotFoundError(f"No PDF files found in {data_dir.resolve()}")


def load_pdf_documents(pdf_path: Path) -> List[Document]:
    """
    Load all pages from a PDF into LangChain Document objects.

    Returns a list of Documents (like index cards) where each Document holds
    page_content and metadata for one page.
    """
    try:
        loader = PyPDFLoader(str(pdf_path))
        return loader.load()
    except Exception as exc:  # Catch loader errors (bad file, permissions, etc.)
        raise RuntimeError(f"Failed to load PDF {pdf_path}") from exc


def split_documents(
    docs: Sequence[Document], *, chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[Document]:
    """
    Split Documents into overlapping chunks to preserve context across boundaries.

    chunk_overlap lets consecutive chunks share some text so we don't lose
    sentence/paragraph continuity at split points (useful for retrieval quality).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(list(docs))


def total_characters(docs: Sequence[Document]) -> int:
    """Count total characters across all Document page_content fields."""
    return sum(len(doc.page_content) for doc in docs)


def print_samples(chunks: Sequence[Document]) -> None:
    """Print the first two chunks and one random middle chunk for inspection."""
    if not chunks:
        print("No chunks to display.")
        return

    print("\n--- First chunk ---")
    print(chunks[0].page_content)

    if len(chunks) > 1:
        print("\n--- Second chunk ---")
        print(chunks[1].page_content)

    # Choose a chunk from the middle range for a representative sample.
    middle_start = max(0, len(chunks) // 3)
    middle_end = max(middle_start + 1, (2 * len(chunks)) // 3)
    random_index = random.randint(middle_start, middle_end - 1)

    print(f"\n--- Random middle chunk (index {random_index}) ---")
    print(chunks[random_index].page_content)


def main() -> None:
    data_dir = Path("data")
    pdf_path = find_first_pdf(data_dir)

    pages = load_pdf_documents(pdf_path)  # List of Documents (pages)
    print(f"Loaded PDF: {pdf_path.name}")
    print(f"Pages loaded: {len(pages)}")

    chunks = split_documents(pages, chunk_size=1000, chunk_overlap=200)

    print("\nStats:")
    print(f"- Total number of characters in document: {total_characters(pages)}")
    print(f"- Total number of chunks created: {len(chunks)}")

    print_samples(chunks)


if __name__ == "__main__":
    # Set a seed so the "random middle chunk" is reproducible across runs.
    random.seed(42)
    main()

