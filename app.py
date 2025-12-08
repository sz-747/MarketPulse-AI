"""Streamlit frontend for MarketPulse AI."""

from typing import List, Dict

import streamlit as st

from backend import get_fresh_news, summarize_news


def render_articles(articles: List[Dict[str, str]]) -> None:
    """Render a list of news articles in Streamlit."""
    for article in articles:
        title = article.get("title", "Untitled")
        link = article.get("link", "")
        published = article.get("published", "")

        st.markdown(f"### {title}")
        if link:
            st.markdown(f"[Read article]({link})")
        else:
            st.write("Link unavailable")

        if published:
            st.caption(f"Published: {published}")

        st.markdown("---")


def main() -> None:
    """Main entrypoint for the Streamlit app."""
    st.set_page_config(page_title="MarketPulse AI")
    st.title("MarketPulse AI 📈")

    ticker_input = st.text_input("Ticker Symbol", value="AAPL").strip().upper()

    if st.button("Get News"):
        if not ticker_input:
            st.warning("Please enter a ticker symbol.")
            return

        with st.spinner("Fetching news..."):
            articles = get_fresh_news(ticker_input)

        if articles:
            # AI summary
            summary = summarize_news(articles)
            if summary:
                st.success(f"AI Analysis: {summary}")

            st.success(f"Found {len(articles)} article(s) in the last 24 hours.")
            render_articles(articles)
        else:
            st.info("No fresh articles found. Try another ticker or check back later.")


if __name__ == "__main__":
    main()

