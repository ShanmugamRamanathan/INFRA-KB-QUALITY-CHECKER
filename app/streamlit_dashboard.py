import sys
import os 
import streamlit as st
import ollama
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import torch
import json
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.evaluation_metrics import evaluate_kb_quality_with_metrics

# ---------- Config ----------
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "infra_kb"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
OLLAMA_MODEL = "llama3.2"

# Page config
st.set_page_config(
    page_title="Infra KB Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Load models (cached)
@st.cache_resource
def load_embedding_model():
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(QDRANT_HOST, port=QDRANT_PORT)

embedding_model = load_embedding_model()
qdrant_client = get_qdrant_client()

# Header
st.title("🤖 Infrastructure Knowledge Base Assistant")
st.markdown("Ask any question about SCOM, SCCM, or Windows infrastructure troubleshooting!")

# Sidebar - Configuration
st.sidebar.header("⚙️ Settings")
show_metrics = st.sidebar.checkbox("Show Quality Metrics", value=True)
show_sources = st.sidebar.checkbox("Show Source Documents", value=True)
num_sources = st.sidebar.slider("Number of sources to retrieve", 1, 5, 3)

st.sidebar.divider()
st.sidebar.subheader("📊 Session Stats")
if st.session_state.chat_history:
    st.sidebar.metric("Questions Asked", len(st.session_state.chat_history))
    avg_score = sum([item['metrics']['overall_score'] for item in st.session_state.chat_history]) / len(st.session_state.chat_history)
    st.sidebar.metric("Avg Quality Score", f"{avg_score:.3f}")

# Main Functions
def retrieve_documents(question: str, top_k: int = 3):
    """Retrieve relevant KB documents."""
    q_vec = embedding_model.encode([question])[0].tolist()
    
    search_result = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=q_vec,
        limit=top_k,
    )
    
    results = []
    for point in search_result.points:
        results.append({
            'text': point.payload['text'],
            'score': point.score
        })
    
    return results

def generate_answer(question: str, retrieved_docs: list):
    """Generate answer using LLM based on retrieved documents."""
    context = "\n\n".join([f"[Source {i+1}]\n{doc['text']}" 
                           for i, doc in enumerate(retrieved_docs)])
    
    prompt = f"""You are an IT infrastructure support assistant. Answer the user's question based ONLY on the provided knowledge base articles.

QUESTION: {question}

KNOWLEDGE BASE ARTICLES:
{context}

INSTRUCTIONS:
1. Provide a clear, step-by-step answer based on the KB articles
2. If the KB doesn't fully answer the question, say so clearly
3. Reference which source(s) you used (e.g., "According to Source 1...")
4. Be concise but complete
5. Do not hallucinate

ANSWER:"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{'role': 'user', 'content': prompt}]
    )
    
    return response['message']['content']

def get_score_color(score: float) -> str:
    """Return color based on score."""
    if score >= 0.8:
        return "🟢"
    elif score >= 0.6:
        return "🟡"
    elif score >= 0.4:
        return "🟠"
    else:
        return "🔴"

# Main Chat Interface
st.divider()

# Display chat history
for i, item in enumerate(st.session_state.chat_history):
    with st.container():
        # User question
        st.markdown(f"**🙋 You asked:** {item['question']}")
        
        # Answer
        st.markdown(f"**🤖 Answer:**")
        st.info(item['answer'])
        
        # Quality metrics (if enabled)
        if show_metrics:
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                score = item['metrics']['overall_score']
                st.metric("Overall", f"{score:.2f}", delta=None, delta_color="off")
            with col2:
                score = item['metrics']['context_relevancy']
                st.metric("Relevancy", f"{score:.2f} {get_score_color(score)}")
            with col3:
                score = item['metrics']['answer_completeness']
                st.metric("Complete", f"{score:.2f} {get_score_color(score)}")
            with col4:
                score = item['metrics']['faithfulness']
                st.metric("Faithful", f"{score:.2f} {get_score_color(score)}")
            with col5:
                score = item['metrics']['precision_at_3']
                st.metric("Precision", f"{score:.2f} {get_score_color(score)}")
        
        # Source documents (if enabled)
        if show_sources:
            with st.expander(f"📚 View {len(item['sources'])} Source Documents"):
                for j, doc in enumerate(item['sources'], 1):
                    st.markdown(f"**Source {j}** (Similarity: {doc['score']:.3f})")
                    st.text(doc['text'][:300] + "...")
                    st.divider()
        
        st.divider()

# Question input form
with st.form("question_form", clear_on_submit=True):
    user_question = st.text_input(
        "Ask your question:",
        placeholder="e.g., Why is my SCOM agent not sending heartbeats?",
        label_visibility="collapsed"
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        submit_button = st.form_submit_button("🔍 Ask", use_container_width=True)
    with col2:
        clear_button = st.form_submit_button("🗑️ Clear History", use_container_width=True)

# Handle form submission
if submit_button and user_question:
    with st.spinner("🔍 Searching knowledge base..."):
        # Step 1: Retrieve documents
        retrieved_docs = retrieve_documents(user_question, top_k=num_sources)
        
    with st.spinner("🤖 Generating answer..."):
        # Step 2: Generate answer
        answer = generate_answer(user_question, retrieved_docs)
        
    with st.spinner("📊 Evaluating quality..."):
        # Step 3: Calculate metrics
        metrics = evaluate_kb_quality_with_metrics(
            user_question, 
            retrieved_docs, 
            answer
        )
        
        # Calculate overall score
        overall_score = (
            metrics['context_relevancy'] * 0.25 +
            metrics['answer_completeness'] * 0.35 +
            metrics['faithfulness'] * 0.20 +
            metrics['precision_at_3'] * 0.20
        )
        
        metrics['overall_score'] = round(overall_score, 3)
    
    # Save to session
    st.session_state.chat_history.append({
        'question': user_question,
        'answer': answer,
        'sources': retrieved_docs,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    })
    
    # Rerun to display new message
    st.rerun()

if clear_button:
    st.session_state.chat_history = []
    st.rerun()

# Export functionality
if st.session_state.chat_history:
    st.sidebar.divider()
    if st.sidebar.button("💾 Export Session"):
        export_data = {
            'session_start': st.session_state.chat_history[0]['timestamp'],
            'total_questions': len(st.session_state.chat_history),
            'conversations': st.session_state.chat_history
        }
        
        filename = f"session_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        st.sidebar.download_button(
            label="📥 Download JSON",
            data=json.dumps(export_data, indent=2),
            file_name=filename,
            mime="application/json"
        )

# Footer
st.sidebar.divider()
st.sidebar.markdown("""
---
**Infrastructure KB Assistant**  
Powered by: Qdrant + Ollama + BGE Embeddings
""")