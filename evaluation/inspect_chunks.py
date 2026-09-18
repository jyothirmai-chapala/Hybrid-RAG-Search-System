import json

# Load chunks
with open("data/chunks/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Total chunks: {len(chunks)}")

print("\nEnter a keyword to search inside chunks.")
keyword = input("Keyword: ").lower()

print("\n" + "=" * 80)
print("MATCHING CHUNKS")
print("=" * 80)

count = 0

for i, chunk in enumerate(chunks):

    if keyword in chunk["text"].lower():

        print("\n" + "-" * 80)

        print(f"Chunk ID : {i}")
        print(f"Document : {chunk['document']}")
        print(f"Page     : {chunk['page']}")

        print("\nText:")
        print(chunk["text"][:1000])

        count += 1

        if count >= 10:
            break

print("\n" + "=" * 80)
print(f"Displayed {count} matching chunks")
print("=" * 80)