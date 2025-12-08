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


if __name__ == '__main__':
    # Test block: Only runs when we execute this file directly
    print("Fetching most recent news for NVDA...")
    print("-" * 60)
    
    try:
        # Get all news (no time restriction)
        stock = yf.Ticker('NVDA')
        all_news = stock.news  # List of Dictionaries
        
        # Debug: Check what we got
        print(f"Debug - Type of all_news: {type(all_news)}")
        print(f"Debug - Length of all_news: {len(all_news) if all_news else 'None/Empty'}")
        
        if not all_news:
            print("No news articles found from yfinance.")
            print("This could be a network issue or yfinance API change.")
            exit()
        
        # Debug: Show first article structure
        print(f"\nDebug - First article type: {type(all_news[0])}")
        print(f"Debug - First article keys: {list(all_news[0].keys()) if isinstance(all_news[0], dict) else 'Not a dict'}")
        if isinstance(all_news[0], dict):
            print(f"Debug - First article sample (first 500 chars): {str(all_news[0])[:500]}\n")
        
        # Process all articles into our format
        all_articles = []  # List to hold processed articles
        for idx, article in enumerate(all_news):
            # Debug first article processing
            if idx == 0:
                print(f"Debug - Processing article {idx}:")
                print(f"  Article type: {type(article)}")
                if isinstance(article, dict):
                    print(f"  Available keys: {list(article.keys())}")
            
            # Check if article is a dictionary
            if not isinstance(article, dict):
                print(f"Warning: Article {idx} is not a dictionary, skipping...")
                continue
            
            # The actual article data is nested inside 'content'!
            content = article.get('content', {})
            if not content:
                if idx < 3:
                    print(f"Debug - Article {idx} has no 'content' field, skipping...")
                continue
            
            # Extract title from nested content
            title = content.get('title', '')
            if not title:
                if idx < 3:
                    print(f"Debug - Article {idx} has no title in content")
                continue
            
            # Extract pubDate - it's a string in ISO format (e.g., '2025-12-07T13:30:23Z')
            pub_date_str = content.get('pubDate', '')
            if not pub_date_str:
                if idx < 3:
                    print(f"Debug - Article {idx} has no pubDate")
                continue
            
            # Parse the ISO date string to datetime object
            try:
                # Remove 'Z' and parse ISO format: '2025-12-07T13:30:23Z'
                pub_date_str_clean = pub_date_str.replace('Z', '+00:00')
                pub_datetime = datetime.fromisoformat(pub_date_str_clean)
                publish_time = pub_datetime.timestamp()  # Convert to Unix timestamp for sorting
                formatted_date = pub_datetime.strftime('%Y-%m-%d %H:%M')
            except Exception as e:
                if idx < 3:
                    print(f"Debug - Article {idx} date parsing error: {e}")
                continue
            
            # Debug: Show all content keys for first article
            if idx == 0:
                print(f"  All content keys: {list(content.keys())}")
                print(f"  Content sample: {str(content)[:800]}\n")
            
            # Debug: Show what canonicalUrl and clickThroughUrl actually contain
            if idx == 0:
                canonical_raw = content.get('canonicalUrl')
                click_raw = content.get('clickThroughUrl')
                print(f"  canonicalUrl type: {type(canonical_raw)}, value: {canonical_raw}")
                print(f"  clickThroughUrl type: {type(click_raw)}, value: {click_raw}")
            
            # Extract link - try various possible field names
            # The URL might be nested in a 'url' key within canonicalUrl dict
            link = ''  # Start with empty string
            
            # First check if canonicalUrl is a dict with a nested 'url' key
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
            
            # Debug first article extraction
            if idx == 0:
                print(f"  Extracted title: {title[:50]}...")
                print(f"  Final link extracted: {link[:100] if link else 'None'}...")
                print(f"  Date string: {pub_date_str} -> {formatted_date}\n")
            
            # Create article dictionary
            article_dict = {
                'title': title,
                'link': link,
                'published': formatted_date,
                'timestamp': publish_time  # Keep timestamp for sorting
            }
            all_articles.append(article_dict)
        
        # Sort by timestamp (most recent first) - highest number = most recent
        all_articles.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Get top 3 most recent
        top_3 = all_articles[:3]
        
        if top_3:
            print(f"Top 3 Most Recent Articles:\n")
            for i, article in enumerate(top_3, 1):
                print(f"{i}. {article['title']}")
                print(f"   Published: {article['published']}")
                print(f"   Link: {article['link']}")
                print()
        else:
            print("No articles found.")

        # Summarize with Groq (only if we have articles)
        if all_articles:
            print("Summary from Groq:")
            summary = summarize_news(all_articles)
            print(summary or "No summary generated.")
        else:
            print("No articles to summarize.")
            
    except Exception as e:
        print(f"Error: {e}")

