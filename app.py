import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Intelligent FAQ Bot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Intelligent FAQ Bot")
st.caption("Semantic Search Powered by Sentence Transformers")

# =========================================================
# FAQ DATABASE
# =========================================================
FAQ_DATA = {
    "What is AI?":
        "AI enables machines to mimic human intelligence.",

    "What is Machine Learning?":
        "Machine Learning allows systems to learn from data.",

    "How does Deep Learning work?":
        "Deep Learning uses neural networks with many layers.",

    "What is Python used for?":
        "Python is widely used in AI, ML, automation, and web development.",

    "What are embeddings?":
        "Embeddings are numerical vector representations of text.",

    "What is NLP?":
        "NLP stands for Natural Language Processing."
}

questions = list(FAQ_DATA.keys())
answers = list(FAQ_DATA.values())

# =========================================================
# LOAD MODEL (CACHED)
# =========================================================
@st.cache_resource(show_spinner=False)
def load_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

model = load_model()

# =========================================================
# CREATE EMBEDDINGS (CACHED)
# =========================================================
@st.cache_data(show_spinner=False)
def create_embeddings(texts):
    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

faq_embeddings = create_embeddings(questions)

# =========================================================
# SEARCH FUNCTION
# =========================================================
def get_best_match(user_input):

    query_embedding = model.encode(
        [user_input],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    similarity_scores = cosine_similarity(
        query_embedding,
        faq_embeddings
    )[0]

    best_index = np.argmax(similarity_scores)

    return (
        questions[best_index],
        answers[best_index],
        similarity_scores[best_index]
    )

# =========================================================
# USER INPUT
# =========================================================
user_query = st.text_input(
    "Ask your question",
    placeholder="Example: Explain deep learning"
)

# =========================================================
# PROCESS QUERY
# =========================================================
if user_query:

    with st.spinner("Finding best answer..."):

        matched_question, answer, confidence = get_best_match(user_query)

        st.divider()

        # Confidence Threshold
        if confidence >= 0.45:

            st.success(answer)

            with st.expander("View Match Details"):
                st.write(f"**Matched FAQ:** {matched_question}")
                st.write(f"**Confidence Score:** {confidence:.4f}")

        else:
            st.warning(
                "Sorry, I could not find a good matching answer."
            )

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.header("📚 Available FAQs")

    for q in questions:
        st.write(f"• {q}")

    st.divider()

    st.markdown("### ⚡ Tech Stack")
    st.write("- Streamlit")
    st.write("- Sentence Transformers")
    st.write("- Cosine Similarity")
    st.write("- Semantic Search")
