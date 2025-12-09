"""
Debug script to find specific text in the PDF.
Searches for '65 billion' and shows surrounding context.
"""

from pathlib import Path
from pypdf import PdfReader


def find_text_in_pdf(pdf_path: Path, search_term: str) -> None:
    """
    Search for a substring in all PDF pages and show surrounding context.
    """
    print(f"Searching for '{search_term}' in {pdf_path.name}...\n")
    
    reader = PdfReader(pdf_path)
    found = False
    
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        
        # Search for the term (case-insensitive)
        search_lower = search_term.lower()
        text_lower = text.lower()
        
        if search_lower in text_lower:
            found = True
            # Find all occurrences on this page
            start_idx = 0
            occurrence = 1
            while True:
                idx = text_lower.find(search_lower, start_idx)
                if idx == -1:
                    break
                
                # Extract 500 chars of context (250 before, 250 after)
                context_start = max(0, idx - 250)
                context_end = min(len(text), idx + len(search_term) + 250)
                context = text[context_start:context_end]
                
                print(f"Page {page_num} (Occurrence {occurrence}):")
                print("-" * 60)
                print(context)
                print("-" * 60)
                print()
                
                start_idx = idx + 1
                occurrence += 1
    
    if not found:
        print(f"The text '{search_term}' was not found in the PDF extraction.")


def main() -> None:
    data_dir = Path("data")
    # Find first PDF
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in data/ directory.")
        return
    
    pdf_path = pdf_files[0]
    find_text_in_pdf(pdf_path, "65 billion")


if __name__ == "__main__":
    main()

