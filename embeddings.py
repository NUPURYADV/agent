from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "What subjects are in first year?",
    "First year course structure and syllabus",
    "Recipe for making pasta",
]

embeddings = model.encode(sentences)

print(f"Shape: {embeddings.shape}")
print(f"First embedding (first 5 numbers): {embeddings[0][:5]}")
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(embeddings)
print(f"\nSimilarity matrix:\n{similarity.round(2)}")
