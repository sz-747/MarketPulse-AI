"""
State-machine ingestion (v4) to keep speaker attribution intact and debug capture of "$65 billion".

Logic:
- Load PDF, split into lines.
- State machine: track current speaker; update on timestamp lines (^\d{1,2}:\d{2}:\d{2}$); accumulate text lines into speaker blocks.
- After building speaker blocks, split each block with RecursiveCharacterTextSplitter.
- For each small chunk, prepend "Speaker {name} says: " (only on the small chunk, not the big block).
- If a chunk contains "$65 billion", print it with its speaker.
- Upload chunks to Pinecone index "marketpulse".
"""

import os
import re
from pathlib import Path
from typing import List, Tuple

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


def build_speaker_blocks(lines: List[str]) -> List[Tuple[str, str]]:
    """
    Build (speaker, text) blocks using a state machine.
    - On timestamp line, update current speaker.
    - Text lines accumulate into the current block.
    """
    current_speaker = "Unknown"
    current_block: List[str] = []
    blocks: List[Tuple[str, str]] = []

    def flush():
        nonlocal current_block, current_speaker, blocks
        if not current_block:
            return
        text = "\n".join(current_block).strip()
        if text:
            blocks.append((current_speaker, text))
        current_block = []

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if TIMESTAMP_RE.match(stripped):
            flush()
            # Look 2 lines back for the name, fallback to 1 line back
            possible_name = lines[idx - 2].strip() if idx >= 2 else ""
            if possible_name and len(possible_name) > 3:
                current_speaker = possible_name
            else:
                current_speaker = lines[idx - 1].strip() if idx >= 1 else current_speaker
            continue
        elif stripped:
            current_block.append(stripped)
    flush()
    return blocks


def chunk_blocks(blocks: List[Tuple[str, str]], chunk_size: int = 2000, chunk_overlap: int = 200) -> List[Document]:
    """Split blocks and prepend speaker prefix on each chunk; debug-print chunks containing '$65 billion'."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks: List[Document] = []
    for speaker, text in blocks:
        doc = Document(page_content=text)
        splits = splitter.split_documents([doc])
        for s in splits:
            content = s.page_content.strip()
            final_text = f"Speaker {speaker} says: {content}"
            # Debug: print any chunk containing "$65 billion"
            if "$65 billion" in content:
                print("\n[DEBUG] Found '$65 billion' chunk")
                print(f"Speaker: {speaker}")
                print(final_text)
                print("-" * 60)
            chunks.append(Document(page_content=final_text, metadata={"speaker": speaker}))
    return chunks


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
    print("Starting ingestion v4 (state-machine, 2000/200)...")
    data_dir = Path("data")
    pdf_path = next(data_dir.glob("*.pdf"), None)
    if not pdf_path:
        raise FileNotFoundError("No PDF in data/")

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

