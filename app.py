import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Semantic FAQ Bot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# App Header
st.title("🤖 Intelligent FAQ Assistant")
st.markdown("---")

# 1. Optimizing Resource Loading (Cached globally)
@st.cache_resource
def initialize_transformer():
    """Loads and caches the heavy embedding model in memory."""
    return SentenceTransformer('all-MiniLM-L6-v2')

with st.spinner("Initializing neural network..."):
    model = initialize_transformer()

# 2. Knowledge Base Data
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

# 3. Optimizing Data Vectors (Cached to prevent recalculation)
@st.cache_data
def compute_faq_embeddings():
    """Computes coordinate vectors for the stable FAQ bank."""
    return model.encode(faq_questions)

faq_embeddings = compute_faq_embeddings()

# Sidebar Helper Info
with st.sidebar:
    st.header("About the App")
    st.write("This application utilizes a lightweight Transformer model to calculate **Cosine Similarity** between your question and the database.")
    st.caption("Model: `all-MiniLM-L6-v2`")

# 4. User Interface Container
with st.container():
    user_query = st.text_input(
        "Ask a question about our services:", 
        placeholder="e.g., Can you tell me what Python programming does?"
    )

# 5. Semantic Search Execution
if user_query.strip():
    # Encode user intent into vector space
    query_embedding = model.encode([user_query])
    
    # Calculate geometric cosine scores
    similarity_scores = cosine_similarity(query_embedding, faq_embeddings)
    
    # Identify index of top choice
    best_match_idx = np.argmax(similarity_scores)
    confidence_score = similarity_scores[best_match_idx]
    
    st.write("### Search Results")
    
    # Set a robust baseline threshold to block unrelated context noise
    if confidence_score >= 0.38:
        # Layout metrics alongside answers
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.markdown(f"**Matched Question:** *{faq_questions[best_match_idx]}*")
            st.success(faq_answers[best_match_idx])
            
        with col2:
            # Displays confidence score in a visual widget
            st.metric(label="Match Score", value=f"{confidence_score * 100:.1f}%")
            
    else:
        st.warning("⚠️ No matching FAQ found. We could not find a clear match with a high enough confidence score. Please try rephrasing your query.")