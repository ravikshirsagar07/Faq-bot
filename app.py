import streamlit as st
from difflib import SequenceMatcher

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="FAQ Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI FAQ Chatbot")
st.write("Ask a question and get the closest FAQ answer.")

# =====================================================
# FAQ DATA
# =====================================================
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

# =====================================================
# MATCH FUNCTION
# =====================================================
def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

# =====================================================
# USER INPUT
# =====================================================
query = st.text_input("Enter your question:")

# =====================================================
# BUTTON
# =====================================================
if st.button("Get Answer"):

    if query.strip() == "":
        st.warning("Please enter a question.")

    else:

        scores = []

        for question in faq_questions:
            score = similarity(query, question)
            scores.append(score)

        best_index = scores.index(max(scores))
        best_score = scores[best_index]

        st.subheader("Most Relevant FAQ")
        st.success(faq_questions[best_index])

        st.subheader("Answer")
        st.info(faq_answers[best_index])

        st.subheader("Similarity Score")
        st.write(f"{best_score:.4f}")

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("About")

st.sidebar.write("""
This chatbot uses:

- Streamlit
- String Similarity Matching
- Lightweight FAQ Search
""")
