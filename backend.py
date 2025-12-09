"""
MarketPulse AI - Backend Logic
Handles data fetching and processing for stock news attribution.
"""

import os
from datetime import datetime, timedelta
from typing import List, Dict

import yfinance as yf
from dotenv import load_dotenv
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore


def get_fresh_news(ticker: str) -> List[Dict[str, str]]:
    """
    Fetches news articles for a stock ticker that were published within the last 24 hours.
    
    Args:
        ticker: Stock symbol (e.g., 'NVDA', 'AAPL')
    
    Returns:
        A List of Dictionaries, where each Dictionary contains:
        - 'title': Article headline
        - 'link': URL to the article
        - 'published': Formatted date string (YYYY-MM-DD HH:MM)
    """
    # Step 1: Get the Ticker object from yfinance
    stock = yf.Ticker(ticker)
    
    # Step 2: Create an empty List (Shopping Cart) to hold our filtered articles
    fresh_articles = []  # List of Dictionaries
    
    # Step 3: Calculate the cutoff time (24 hours ago from now)
    now = datetime.now()
    twenty_four_hours_ago = now - timedelta(hours=24)
    cutoff_timestamp = int(twenty_four_hours_ago.timestamp())  # Convert to Unix timestamp
    
    # Step 4: Loop through all news articles
    try:
        news_list = stock.news  # This is a List of Dictionaries from yfinance
        
        for article in news_list:  # Loop through each article Dictionary
            # The actual article data is nested inside 'content'!
            content = article.get('content', {})
            if not content:
                continue  # Skip if no content
            
            # Extract title
            title = content.get('title', '')
            if not title:
                continue  # Skip if no title
            
            # Extract pubDate - it's a string in ISO format (e.g., '2025-12-07T13:30:23Z')
            pub_date_str = content.get('pubDate', '')
            if not pub_date_str:
                continue  # Skip if no date
            
            # Parse the ISO date string to datetime object
            try:
                pub_date_str_clean = pub_date_str.replace('Z', '+00:00')
                pub_datetime = datetime.fromisoformat(pub_date_str_clean)
                publish_time = pub_datetime.timestamp()  # Convert to Unix timestamp
            except Exception:
                continue  # Skip if date parsing fails
            
            # Step 5: Check if article is within 24 hours (Boolean Condition)
            # IF the article is older than 24 hours, skip it (continue to next iteration)
            if publish_time < cutoff_timestamp:
                continue  # Skip this article, move to the next one
            
            # Step 6: Article is fresh! Create a Dictionary (ID Card) for it
            formatted_date = pub_datetime.strftime('%Y-%m-%d %H:%M')
            
            # Extract link - the URL might be nested in a 'url' key within canonicalUrl dict
            link = ''  # Start with empty string
            
            # First check canonicalUrl
            canonical_url = content.get('canonicalUrl')
            if isinstance(canonical_url, dict):
                link = canonical_url.get('url', '')
            elif isinstance(canonical_url, str) and canonical_url.strip():
                link = canonical_url
            
            # If still no link, try clickThroughUrl
            if not link:
                click_url = content.get('clickThroughUrl')
                if isinstance(click_url, dict):
                    link = click_url.get('url', '')
                elif isinstance(click_url, str) and click_url.strip():
                    link = click_url
            
            article_dict = {
                'title': title,
                'link': link,
                'published': formatted_date
            }
            
            # Step 7: Add this Dictionary to our results List
            fresh_articles.append(article_dict)
    
    except Exception as e:
        # Error Handling: Catch any issues (network problems, invalid ticker, etc.)
        print(f"Error fetching news for {ticker}: {e}")
        return []  # Return empty list if something goes wrong
    
    # Step 8: Return the filtered list
    return fresh_articles


# Load environment variables from .env
load_dotenv()

# Initialize Groq client (requires GROQ_API_KEY in environment)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Initialize embeddings/vector store for document QA (single global for reuse)
_pc_index_name = "marketpulse"
_pc_api_key = os.environ.get("PINECONE_API_KEY")
if not _pc_api_key:
    raise EnvironmentError("PINECONE_API_KEY not set in environment")

_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = PineconeVectorStore.from_existing_index(
    index_name=_pc_index_name,
    embedding=_embeddings,
)


def summarize_news(news_list: List[Dict[str, str]]) -> str:
    """
    Use Groq LLM to summarize news headlines into a single reason the stock moved.
    
    Args:
        news_list: List of article dictionaries with at least a 'title' field.
    
    Returns:
        A single-sentence explanation from the LLM.
    """
    # Build a simple list of headlines for the prompt
    headlines = [item.get("title", "") for item in news_list if item.get("title")]

    # Safety: if no headlines, return early
    if not headlines:
        return "No headlines available to summarize."

    system_prompt = (
        "You are a Financial Analyst. You are given a list of news headlines. "
        "Your goal is to identify the SINGLE most likely reason the stock moved today. "
        "Ignore generic articles (like 'Tesla vs Amazon'). Write a 1-sentence explanation."
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "\n".join(headlines)},
            ],
            temperature=0.3,
        )
        # Return the assistant message content
        return response.choices[0].message.content if response.choices else ""
    except Exception as e:
        # Catch API/connection errors
        return f"Error from Groq API: {e}"


def ask_document(question: str) -> str:
    """
    Retrieve top chunks from Pinecone and answer using Groq LLM.
    """
    try:
        results = vector_store.similarity_search_with_score(question, k=15)
        docs = [doc for doc, _ in results]
    except Exception as exc:
        return f"Error during retrieval: {exc}"

    if not docs:
        return "I cannot find that information in the document."

    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = (
        "You are a relentless financial analyst. The user is asking about a specific topic "
        "(e.g., Supply). You must scan the context for ANY related concepts (e.g., capacity, "
        "constraints, sold out, inventory, backlog). Connect the dots: if the text says 'sold out', "
        "that IS an answer about supply. Quote the specific sentence that supports your answer.\n\n"
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        'If the answer is not in the context, say "I cannot find that information in the document."'
    )

    debug = os.getenv("RETRIEVAL_DEBUG", "false").lower() in {"1", "true", "yes", "on"}
    if debug:
        print(f"[Retrieval Debug] query='{question}' | k=15")
        for idx, (doc, score) in enumerate(results, start=1):
            page = doc.metadata.get("page", "unknown")
            speaker = doc.metadata.get("speaker") or doc.metadata.get("Speaker") or "unknown"
            preview = " ".join(doc.page_content.split())[:300]
            print(
                f"  {idx}. score={score:.4f} | page={page} | speaker={speaker}\n"
                f"     {preview}"
            )

    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content if resp.choices else ""
    except Exception as exc:
        return f"Error from Groq API: {exc}"


if __name__ == '__main__':
    # Test block for RAG retrieval QA
    test_q = "What is the revenue guidance for Q4?"
    print(f"\n[TEST] Asking document: {test_q}")
    answer = ask_document(test_q)
    print(f"\nAnswer:\n{answer}")

