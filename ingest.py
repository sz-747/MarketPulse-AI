"""
Advanced ingest pipeline with speaker-aware chunking.
Parses PDF by speaker sections using timestamp regex patterns.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple, Optional
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


def load_and_join_pdf(pdf_path: Path) -> str:
    """Load all PDF pages and join into one big string."""
    try:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        # Join all page content with double newline separator
        full_text = "\n\n".join([page.page_content for page in pages])
        return full_text
    except Exception as exc:
        raise RuntimeError(f"Failed to load PDF {pdf_path}") from exc


def parse_speaker_sections(text: str) -> List[Tuple[str, str]]:
    """
    Parse text into (speaker, speech) tuples using timestamp pattern.
    
    Regex pattern explanation:
    - \d{1,2}:\d{2}:\d{2} matches timestamps like 0:22:29 or 10:15:43
    - We split on these timestamps
    - The lines before each timestamp usually contain the speaker name
    
    Returns:
        List of (speaker_name, speech_text) tuples
    """
    # Regex pattern to match timestamps (e.g., 0:22:29, 1:15:43, 10:30:12)
    # Pattern: one or two digits, colon, two digits, colon, two digits
    timestamp_pattern = r'\b(\d{1,2}:\d{2}:\d{2})\b'
    
    sections = []
    
    # Find all timestamp matches with their positions
    matches = list(re.finditer(timestamp_pattern, text))
    
    if not matches:
        # No timestamps found - treat entire text as one section with unknown speaker
        return [("Unknown", text)]
    
    # Process each section between timestamps
    for i, match in enumerate(matches):
        timestamp_pos = match.start()
        
        # Determine section boundaries
        # Start: after previous timestamp (or beginning of text)
        if i == 0:
            section_start = 0
        else:
            section_start = matches[i - 1].end()
        
        # End: current timestamp position
        section_end = timestamp_pos
        
        # Extract section text (before current timestamp)
        section_text = text[section_start:section_end].strip()
        
        if not section_text:
            continue
        
        # Extract speaker name from the header lines
        # Typically the speaker name appears in the last 2-3 lines before timestamp
        lines = section_text.split('\n')
        speaker = "Unknown"
        speech_start_idx = 0
        
        # Look for speaker name in the last few lines before timestamp
        # Common patterns: "Colette Kress", "Jensen Huang", "Operator"
        # Often preceded by role like "EVP and CFO"
        for idx in range(max(0, len(lines) - 5), len(lines)):
            line = lines[idx].strip()
            # Check if line looks like a speaker name (capitalized words, not too long)
            if line and len(line) < 50 and line[0].isupper():
                # Check for known speaker patterns
                if any(name in line.lower() for name in ['kress', 'huang', 'operator', 'analyst', 'hari']):
                    speaker = line
                    speech_start_idx = idx + 1
                    break
        
        # Extract the actual speech text (after speaker name)
        speech_lines = lines[speech_start_idx:]
        speech_text = '\n'.join(speech_lines).strip()
        
        if speech_text:
            sections.append((speaker, speech_text))
    
    # Handle text after the last timestamp
    if matches:
        last_timestamp_end = matches[-1].end()
        remaining_text = text[last_timestamp_end:].strip()
        if remaining_text:
            # Try to extract speaker from remaining text
            lines = remaining_text.split('\n')
            speaker = "Unknown"
            speech_start = 0
            for idx, line in enumerate(lines[:5]):
                line = line.strip()
                if line and len(line) < 50 and line[0].isupper():
                    if any(name in line.lower() for name in ['kress', 'huang', 'operator', 'analyst']):
                        speaker = line
                        speech_start = idx + 1
                        break
            speech_text = '\n'.join(lines[speech_start:]).strip()
            if speech_text:
                sections.append((speaker, speech_text))
    
    return sections


def create_smart_chunks(
    sections: List[Tuple[str, str]],
    chunk_size: int = 3000,
    chunk_overlap: int = 500
) -> List[Document]:
    """
    Create LangChain Documents from speaker sections.
    If a section is too large, split it while preserving speaker metadata.
    
    Args:
        sections: List of (speaker_name, speech_text) tuples
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of Document objects with speaker metadata
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    documents = []
    
    for speaker, speech in sections:
        # Clean up speaker name
        speaker = speaker.strip()
        
        # If speech is short enough, create single document
        if len(speech) <= chunk_size:
            doc = Document(
                page_content=speech,
                metadata={"speaker": speaker}
            )
            documents.append(doc)
        else:
            # Speech is too long - split it but preserve speaker metadata
            # Create temporary document for splitting
            temp_doc = Document(page_content=speech)
            split_docs = splitter.split_documents([temp_doc])
            
            # Add speaker metadata to every split chunk
            for split_doc in split_docs:
                split_doc.metadata["speaker"] = speaker
                documents.append(split_doc)
    
    return documents


def log_chunk_metadata(chunks: List[Document], preview_chars: int = 120) -> None:
    """Print chunk metadata for QA."""
    should_log = os.getenv("INGEST_LOG_CHUNKS", "true").lower() in {"1", "true", "yes", "on"}
    if not should_log:
        return

    print("\n[Ingest] Chunk metadata:")
    for idx, doc in enumerate(chunks):
        speaker = doc.metadata.get("speaker", "unknown")
        length = len(doc.page_content)
        preview = doc.page_content[:preview_chars].replace("\n", " ").strip()
        print(
            f"  #{idx:03d} | speaker={speaker} | len={length} | preview=\"{preview}\""
        )
    print("[Ingest] End of chunk metadata\n")


def embed_and_upsert(
    chunks: List[Document], index_name: str, batch_size: int = 32
) -> int:
    """
    Embed chunks and upsert into Pinecone with progress logging.
    """
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
    print("Starting ingestion with Speaker-Aware Parsing (3000/500)...")
    
    data_dir = Path("data")
    pdf_path = find_first_pdf(data_dir)

    print(f"Loading PDF: {pdf_path.name}")
    full_text = load_and_join_pdf(pdf_path)
    print(f"Total characters: {len(full_text)}")

    print("Parsing speaker sections...")
    sections = parse_speaker_sections(full_text)
    print(f"Found {len(sections)} speaker sections")

    print("Creating smart chunks...")
    chunks = create_smart_chunks(sections, chunk_size=3000, chunk_overlap=500)
    print(f"Chunks prepared: {len(chunks)}")

    log_chunk_metadata(chunks)

    index_name = "marketpulse"
    if not os.getenv("PINECONE_API_KEY"):
        raise EnvironmentError("PINECONE_API_KEY not set in environment")

    uploaded = embed_and_upsert(chunks, index_name=index_name)
    print(f"Success: uploaded {uploaded} chunks to Pinecone index '{index_name}'.")


if __name__ == "__main__":
    main()
