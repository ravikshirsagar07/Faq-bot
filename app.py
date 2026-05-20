import streamlit as st
from sentence_transformers import SentenceTransformer

st.title("FAQ Bot")

@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-MiniLM-L3-v2")

model = load_model()

query = st.text_input("Ask something")

if query:
    embedding = model.encode(query)
    st.success("Embedding generated successfully!")
    st.write(embedding[:10])
