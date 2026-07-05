import fitz

def split_into_chunks(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    return chunks

# extract text from your pdf
doc = fitz.open("SYALLABUS FIRST YEAR.pdf")
full_text = ""
for page in doc:
    full_text += page.get_text()

# chunk it
chunks = split_into_chunks(full_text)

print(f"Total chunks: {len(chunks)}")
print(f"\nChunk 1:\n{chunks[0]}")
print(f"\nChunk 2:\n{chunks[1]}")