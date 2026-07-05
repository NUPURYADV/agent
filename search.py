import fitz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Step 1 - extract text
doc = fitz.open("SYALLABUS FIRST YEAR.pdf")
full_text = ""
for page in doc:
    full_text += page.get_text()

# Step 2 - chunk it
def split_into_chunks(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

chunks = split_into_chunks(full_text)

# Step 3 - embed all chunks
model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embeddings = model.encode(chunks)

# Step 4 - search
question = "list of subjects in first semester"
question_embedding = model.encode([question])

similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]
top_3_indices = np.argsort(similarities)[-3:][::-1]

print(f"Top 3 relevant chunks for: '{question}'\n")
for i in top_3_indices:
    print(f"Score: {similarities[i]:.2f}")
    print(f"Chunk: {chunks[i][:200]}")
    print("---")