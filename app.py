import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.set_page_config(page_title="FAQ Assistant", page_icon="🤖")
st.title("🤖 Intelligent FAQ Bot")

# Cache the model resource globally so it is only loaded once into RAM
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

with st.spinner("Loading Transformer model..."):
    model = load_embedding_model()

# FAQ Knowledge Base Data
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

# Cache computed data vectors so they don't regenerate every rerun
@st.cache_data
def get_cached_embeddings():
    return model.encode(faq_questions)

faq_embeddings = get_cached_embeddings()

# Input box interface
user_query = st.text_input("Ask a question:", placeholder="e.g., What is deep learning?")

if user_query.strip():
    query_vector = model.encode([user_query])
    scores = cosine_similarity(query_vector, faq_embeddings)
    best_match_idx = np.argmax(scores)
    
    st.markdown("---")
    if scores[best_match_idx] > 0.38:
        st.info(f"**Matched FAQ:** {faq_questions[best_match_idx]}")
        st.success(f"**Answer:** {faq_answers[best_match_idx]}")
        st.caption(f"Match confidence score: {scores[best_match_idx]:.4f}")
    else:
        st.warning("Could not find a highly matching answer. Please try rephrasing your question!")
