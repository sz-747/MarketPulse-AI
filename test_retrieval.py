"""
Simple retrieval test against the Pinecone index `marketpulse`.

It uses the same embedding model as ingest (`sentence-transformers/all-MiniLM-L6-v2`)
to ensure vector compatibility.
"""

import os
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore


def get_vector_store(index_name: str) -> PineconeVectorStore:
    """Initialize the Pinecone vector store using the shared embedding model."""
    if not os.getenv("PINECONE_API_KEY"):
        raise EnvironmentError("PINECONE_API_KEY not set in environment")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
    )


def run_query(store: PineconeVectorStore, query: str, k: int = 5) -> List[Tuple[float, str, dict]]:
    """
    Run similarity search and return (score, text, metadata) for top results.
    """
    results = store.similarity_search_with_score(query, k=k)
    return [(score, doc.page_content, doc.metadata) for doc, score in results]


def main() -> None:
    load_dotenv()

    query = "What is the revenue guidance for Q4?"
    index_name = "marketpulse"

    store = get_vector_store(index_name)
    hits = run_query(store, query, k=5)

    print(f"Query: {query}")
    print("-" * 50)
    if not hits:
        print("No results found.")
        return

    print("\nTop results:")
    for idx, (score, text, meta) in enumerate(hits, start=1):
        loc = meta.get("page", "unknown page") if isinstance(meta, dict) else "unknown"
        print(f"\nResult {idx} | score={score:.4f} | page={loc}")
        print(text)
        print("-" * 50)


if __name__ == "__main__":
    main()

