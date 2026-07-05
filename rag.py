import fitz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

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

# Step 4 - get question and find relevant chunks
question = "What are the course outcomes for BEC 101?"
question_embedding = model.encode([question])
similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]
top_indices = np.argsort(similarities)[-3:][::-1]
relevant_chunks = "\n\n".join([chunks[i] for i in top_indices])

# Step 5 - send to LLM
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": "You are a helpful assistant. Answer the user's question using only the context provided below. If the answer is not in the context, say 'I don't know'.\n\nContext:\n" + relevant_chunks},
        {"role": "user", "content": question}
    ]
)

print(f"Question: {question}")
print(f"\nAnswer: {response.choices[0].message.content}")