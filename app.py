# app.py

import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="FAQ Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Title
st.title("🤖 AI FAQ Chatbot")
st.write("Ask a question and get the most relevant FAQ answer.")

# Load Model
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L3-v2')

model = load_model()

# FAQs
faq_questions = [
    "What is AI?",
    "What is Machine Learning?",
    "How does Deep Learning work?",
    "What is Python used for?"
]

faq_answers = [
    "AI enables machines to mimic human intelligence.",
    "Machine Learning allows systems to learn from data.",
    "Deep Learning uses neural networks with many layers.",
    "Python is widely used in AI, ML, and web development."
]

# Precompute FAQ Embeddings
faq_embeddings = model.encode(faq_questions)

# User Input
query = st.text_input("Enter your question:")

# Button
if st.button("Get Answer"):

    if query.strip() == "":
        st.warning("Please enter a question.")
    else:
        # Query Embedding
        query_embedding = model.encode([query])

        # Similarity Calculation
        scores = cosine_similarity(
            query_embedding,
            faq_embeddings
        )

        # Best Match
        best_index = np.argmax(scores)
        best_score = scores[0][best_index]

        # Display Results
        st.subheader("Most Relevant FAQ")
        st.success(faq_questions[best_index])

        st.subheader("Answer")
        st.info(faq_answers[best_index])

        st.subheader("Similarity Score")
        st.write(f"{best_score:.4f}")

# Sidebar
st.sidebar.title("About")
st.sidebar.write(
    """
    This Streamlit app uses:
    - Sentence Transformers
    - Cosine Similarity
    - Semantic Search
    """
)
