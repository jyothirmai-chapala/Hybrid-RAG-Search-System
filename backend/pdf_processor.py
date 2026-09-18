import pymupdf
from pathlib import Path


# Folder containing our PDF documents
PDF_FOLDER = Path("data/documents")


def extract_text_from_pdf(pdf_path):
    """Extract text from all pages of a PDF."""

    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text()

        pages.append({
            "page": page_number,
            "text": text
        })

    document.close()

    return pages


def main():

    # Find all PDF files
    pdf_files = list(PDF_FOLDER.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found.")
        print(f"Please add PDFs to: {PDF_FOLDER}")
        return

    print(f"Found {len(pdf_files)} PDF file(s).\n")

    for pdf_path in pdf_files:

        print("=" * 60)
        print(f"DOCUMENT: {pdf_path.name}")
        print("=" * 60)

        pages = extract_text_from_pdf(pdf_path)

        print(f"Number of pages: {len(pages)}\n")

        # Display text from the first page
        if pages:
            print("FIRST PAGE TEXT:")
            print("-" * 60)
            print(pages[0]["text"][:2000])


if __name__ == "__main__":
    main()