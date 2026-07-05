import streamlit as st
import fitz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="Ask your PDF", page_icon="📄")
st.title("📄 Ask your PDF")
st.write("Upload a PDF and ask any question about it!")

def split_into_chunks(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
question = st.text_input("Ask a question about your PDF")

if uploaded_file and question:
    with st.spinner("Reading your PDF..."):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        chunks = split_into_chunks(full_text)
        chunk_embeddings = model.encode(chunks)

    with st.spinner("Finding relevant sections..."):
        question_embedding = model.encode([question])
        similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]
        top_indices = np.argsort(similarities)[-3:][::-1]
        relevant_chunks = "\n\n".join([chunks[i] for i in top_indices])

    with st.spinner("Generating answer..."):
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Answer using only the context below. If answer not found, say 'I don't know'.\n\nContext:\n" + relevant_chunks},
                {"role": "user", "content": question}
            ]
        )

    st.success("Done!")
    st.write("### 💬 Answer:")
    st.write(response.choices[0].message.content)

    with st.expander("📚 View source chunks"):
        for i, idx in enumerate(top_indices):
            st.markdown(f"**Chunk {i+1}** (similarity: {similarities[idx]:.2f})")
            st.write(chunks[idx])
            st.divider()