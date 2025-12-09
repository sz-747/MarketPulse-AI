"""
State-machine ingestion to keep speaker attribution intact.
Parses the PDF line-by-line using timestamp cues, builds speaker blocks,
then chunks with overlap and uploads to Pinecone.
"""

import os
import re
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

TIMESTAMP_RE = re.compile(r"^\d{1,2}:\d{2}:\d{2}$")


def load_pdf_lines(pdf_path: Path) -> List[str]:
    """Extract text from all pages and split into lines."""
    reader = PdfReader(str(pdf_path))
    lines: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        lines.extend(text.split("\n"))
    return lines


def clean_speaker(line: Optional[str]) -> str:
    if not line:
        return "Unknown"
    line = line.strip()
    return line if line else "Unknown"


def build_speaker_blocks(lines: List[str]) -> List[Document]:
    """
    State machine:
    - On timestamp line, flush current buffer as a speaker block.
    - Speaker is assumed to be two lines above the timestamp.
    """
    current_speaker = "Unknown"
    current_buffer: List[str] = []
    blocks: List[Document] = []

    def flush_buffer():
        nonlocal current_buffer, current_speaker, blocks
        if not current_buffer:
            return
        text = "\n".join(current_buffer).strip()
        if not text:
            current_buffer = []
            return
        doc = Document(
            page_content=f"Speaker {current_speaker} says: {text}",
            metadata={"speaker": current_speaker},
        )
        blocks.append(doc)
        current_buffer = []

    for idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        if TIMESTAMP_RE.match(line):
            # Flush previous speaker text
            flush_buffer()
            # Determine new speaker from two lines back, if available
            guess = lines[idx - 2].strip() if idx >= 2 else ""
            current_speaker = clean_speaker(guess)
            continue
        else:
            current_buffer.append(raw_line)

    # Flush any remaining text
    flush_buffer()
    return blocks


def chunk_blocks(blocks: List[Document]) -> List[Document]:
    """
    Split large speaker blocks, ensuring prefix stays on every chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
    )
    final_chunks: List[Document] = []
    for block in blocks:
        speaker = block.metadata.get("speaker", "Unknown")
        # Split without prefix, then add prefix to every chunk
        temp = Document(page_content=block.page_content)
        splits = splitter.split_documents([temp])
        for s in splits:
            content = s.page_content.strip()
            chunk = Document(
                page_content=f"Speaker {speaker} says: {content}",
                metadata={"speaker": speaker},
            )
            final_chunks.append(chunk)
    return final_chunks


def embed_and_upsert(chunks: List[Document], index_name: str, batch_size: int = 32) -> int:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    total = len(chunks)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = chunks[start:end]
        PineconeVectorStore.from_documents(
            documents=batch,
            embedding=embeddings,
            index_name=index_name,
        )
        if end % 10 == 0 or end == total:
            print(f"Uploaded {end}/{total} chunks...")
    return total


def main() -> None:
    print("Starting ingestion (state-machine, speaker-aware, 2000/200)...")
    data_dir = Path("data")
    pdf_path = next(data_dir.glob("*.pdf"), None)
    if not pdf_path:
        raise FileNotFoundError("No PDF in data/")

    print(f"Loading PDF: {pdf_path.name}")
    lines = load_pdf_lines(pdf_path)
    print(f"Total lines extracted: {len(lines)}")

    blocks = build_speaker_blocks(lines)
    print(f"Speaker blocks found: {len(blocks)}")

    chunks = chunk_blocks(blocks)
    print(f"Chunks prepared after splitting: {len(chunks)}")

    if not os.getenv("PINECONE_API_KEY"):
        raise EnvironmentError("PINECONE_API_KEY not set")

    uploaded = embed_and_upsert(chunks, index_name="marketpulse")
    print(f"Success: uploaded {uploaded} chunks to Pinecone index 'marketpulse'.")


if __name__ == "__main__":
    main()

