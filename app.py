import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="FAQ Bot",
    page_icon="🤖"
)

st.title("🤖 Intelligent FAQ Bot")

# =====================================================
# FAQ DATABASE
# =====================================================
faq_questions = [
    "What is AI?",
    "What is Machine Learning?",
    "How does Deep Learning work?",
    "What is Python used for?",
    "What are embeddings?",
    "What is NLP?"
]

faq_answers = [
    "AI enables machines to mimic human intelligence.",
    "Machine Learning allows systems to learn from data.",
    "Deep Learning uses neural networks with many layers.",
    "Python is widely used in AI, ML, and web development.",
    "Embeddings are vector representations of text.",
    "NLP stands for Natural Language Processing."
]

# =====================================================
# TF-IDF MODEL
# =====================================================
vectorizer = TfidfVectorizer()

faq_vectors = vectorizer.fit_transform(faq_questions)

# =====================================================
# USER INPUT
# =====================================================
user_query = st.text_input(
    "Ask your question:",
    placeholder="Example: Explain deep learning"
)

# =====================================================
# SEARCH
# =====================================================
if user_query:

    user_vector = vectorizer.transform([user_query])

    similarity = cosine_similarity(
        user_vector,
        faq_vectors
    )

    best_match = np.argmax(similarity)

    score = similarity[0][best_match]

    st.divider()

    if score > 0.2:

        st.success(faq_answers[best_match])

        with st.expander("Match Details"):
            st.write("Matched Question:")
            st.info(faq_questions[best_match])

            st.write("Confidence Score:")
            st.write(round(float(score), 4))

    else:
        st.warning(
            "No matching FAQ found. Try rephrasing."
        )

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:

    st.header("📚 FAQs")

    for q in faq_questions:
        st.write(f"• {q}")

    st.divider()

    st.markdown("### ⚡ Tech Used")
    st.write("- Streamlit")
    st.write("- TF-IDF")
    st.write("- Cosine Similarity")
