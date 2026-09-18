import pymupdf
import json
from pathlib import Path


PDF_FOLDER = Path("data/documents")
CHUNKS_FOLDER = Path("data/chunks")
CHUNKS_FILE = CHUNKS_FOLDER / "chunks.json"


def extract_pages_from_pdf(pdf_path):
    """Extract text from every page of a PDF."""

    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text().strip()

        if text:
            pages.append({
                "document": pdf_path.name,
                "page": page_number,
                "text": text
            })

    document.close()

    return pages


def create_chunks(pages, chunk_size=1000, overlap=200):
    """Split page text into overlapping chunks."""

    chunks = []

    for page in pages:

        text = page["text"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end]

            chunks.append({
                "document": page["document"],
                "page": page["page"],
                "text": chunk_text
            })

            start = end - overlap

    return chunks


def main():

    pdf_files = list(PDF_FOLDER.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found.")
        return

    all_chunks = []

    for pdf_path in pdf_files:

        print(f"\nProcessing: {pdf_path.name}")

        pages = extract_pages_from_pdf(pdf_path)

        print(f"Pages with text: {len(pages)}")

        chunks = create_chunks(pages)

        print(f"Chunks created: {len(chunks)}")

        all_chunks.extend(chunks)

    # Create chunks folder if it doesn't exist
    CHUNKS_FOLDER.mkdir(parents=True, exist_ok=True)

    # Save chunks to JSON
    with open(CHUNKS_FILE, "w", encoding="utf-8") as file:
        json.dump(all_chunks, file, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("TOTAL CHUNKS:", len(all_chunks))
    print("=" * 60)

    print(f"\nChunks saved to:")
    print(CHUNKS_FILE)

    # Show first 3 chunks
    for i, chunk in enumerate(all_chunks[:3], start=1):

        print(f"\nCHUNK {i}")
        print("-" * 60)
        print("Document:", chunk["document"])
        print("Page:", chunk["page"])
        print("Text:")
        print(chunk["text"][:500])


if __name__ == "__main__":
    main()