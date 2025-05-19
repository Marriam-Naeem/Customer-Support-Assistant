import os
import requests
import json
import uuid
import time
from typing import Dict, List, Any, Optional
import streamlit as st
from streamlit_chat import message
import pandas as pd

# API configuration
API_URL = "http://localhost:8000"  # Change as needed

# App title and configuration
st.set_page_config(
    page_title="NUST Bank Customer Service",
    page_icon="🏦",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #f5f5f5;
}
.chat-message {
    padding: 1.5rem;
    border-radius: 0.8rem;
    margin-bottom: 1rem;
    display: flex;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}
.chat-message.user {
    background-color: #E0F7FA;
    color: #00838F;
    border-bottom-right-radius: 0.2rem;
}
.chat-message.bot {
    background-color: #FFF;
    color: #00838F;
    border-bottom-left-radius: 0.2rem;
}
.chat-message .avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    object-fit: cover;
    margin-right: 1rem;
}
.chat-message .message {
    flex-grow: 1;
}
.stButton>button {
    background-color: #00838F;
    color: white;
    border-radius: 0.5rem;
    padding: 0.5rem 1rem;
    border: none;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}
.source-box {
    background-color: #F9FBE7;
    border-radius: 0.5rem;
    padding: 0.8rem;
    margin-top: 0.5rem;
    font-size: 0.9rem;
    border-left: 3px solid #C0CA33;
}
.file-uploader {
    padding: 1.5rem;
    border: 2px dashed #BDBDBD;
    border-radius: 0.5rem;
    background-color: #FAFAFA;
    text-align: center;
}
h1, h2, h3 {
    color: #00838F;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 2rem;
}
.stTabs [data-baseweb="tab"] {
    height: 4rem;
    white-space: pre-wrap;
    background-color: white;
    border-radius: 4px 4px 0 0;
    padding: 0.5rem 1rem;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background-color: #00838F !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# Helper functions
def generate_user_id():
    """Generate a random user ID if not already set."""
    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    return st.session_state.user_id

def initialize_chat():
    """Initialize chat session state variables."""
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "sources" not in st.session_state:
        st.session_state.sources = {}
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = False

def query_api(query: str):
    """Send a query to the API and get the response."""
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={
                "query": query,
                "user_id": generate_user_id(),
                "conversation_id": st.session_state.conversation_id
            }
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with API: {str(e)}")
        return {
            "response": "I'm sorry, I'm having trouble connecting to the server right now. Please try again later.",
            "conversation_id": st.session_state.conversation_id,
            "sources": []
        }

def upload_document(file):
    """Upload a document to the API."""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        response = requests.post(
            f"{API_URL}/upload",
            files=files
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error uploading document: {str(e)}")
        return {"success": False, "message": str(e)}

def display_chat_message(message_obj, is_user):
    """Display a chat message with proper styling."""
    avatar = "👤" if is_user else "🏦"
    background_color = "#E0F7FA" if is_user else "#FFF"
    alignment = "flex-end" if is_user else "flex-start"
    
    st.markdown(f"""
    <div class="chat-message {'user' if is_user else 'bot'}" style="align-self: {alignment}">
        <div class="avatar">{avatar}</div>
        <div class="message">{message_obj}</div>
    </div>
    """, unsafe_allow_html=True)

def display_sources(sources):
    """Display sources for the bot's response."""
    if not sources or not st.session_state.show_sources:
        return
    
    st.markdown("### Sources")
    for i, source in enumerate(sources):
        with st.expander(f"Source {i+1}: {source.get('source', 'Unknown')}"):
            st.markdown(f"""
            <div class="source-box">
                {source.get('content', 'No content available')}
            </div>
            """, unsafe_allow_html=True)

def clear_chat():
    """Clear the current conversation."""
    st.session_state.messages = []
    st.session_state.sources = {}
    st.session_state.conversation_id = str(uuid.uuid4())
    
    # Also clear on the server
    try:
        requests.delete(f"{API_URL}/conversations/{st.session_state.conversation_id}")
    except:
        pass

def get_system_stats():
    """Get system statistics from the API."""
    try:
        response = requests.get(f"{API_URL}/stats")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return {
            "vector_db": {"status": "unknown"},
            "model": {"name": "unknown", "type": "unknown"}
        }

# Initialize chat session
initialize_chat()

# Create main layout with tabs
tab1, tab2, tab3 = st.tabs(["Chat", "Upload Documents", "System Info"])

# Tab 1: Chat Interface
with tab1:
    st.markdown("# 🏦 NUST Bank Customer Service")
    st.markdown("Welcome to NUST Bank's virtual assistant. How can I help you today?")
    
    # Chat container
    chat_container = st.container()
    
    # Display chat messages
    with chat_container:
        for i, (msg, is_user) in enumerate(st.session_state.messages):
            display_chat_message(msg, is_user)
            
            # Display sources after bot responses (if available)
            if not is_user and i in st.session_state.sources:
                display_sources(st.session_state.sources[i])
    
    # User input and settings
    with st.container():
        # Sources display toggle
        col1, col2 = st.columns([4, 1])
        with col2:
            st.session_state.show_sources = st.checkbox("Show Sources", value=st.session_state.show_sources)
        
        # User input
        user_input = st.text_input("Your question:", key="user_question")
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            send_button = st.button("Send")
        with col2:
            clear_button = st.button("New Chat")
    
    # Process user input
    if send_button and user_input:
        # Add user message to chat
        st.session_state.messages.append((user_input, True))
        
        # Call API
        response_data = query_api(user_input)
        
        # Add bot response to chat
        response_index = len(st.session_state.messages)
        st.session_state.messages.append((response_data["response"], False))
        
        # Store sources
        if "sources" in response_data and response_data["sources"]:
            st.session_state.sources[response_index] = response_data["sources"]
        
        # Reset input
        st.session_state.user_question = ""
        
        # Force refresh
        st.experimental_rerun()
    
    # Clear chat if requested
    if clear_button:
        clear_chat()
        st.experimental_rerun()

# Tab 2: Document Upload
with tab2:
    st.markdown("# 📄 Upload Documents")
    st.markdown("""
    Upload new documents to enhance the knowledge of our virtual assistant. 
    Supported formats: PDF, CSV, JSON, TXT.
    """)
    
    # File uploader
    st.markdown('<div class="file-uploader">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a file", 
                                    type=["pdf", "csv", "json", "txt"],
                                    help="Supported formats: PDF, CSV, JSON, TXT")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Upload button
    if uploaded_file is not None:
        if st.button("Upload Document"):
            with st.spinner("Uploading and processing document..."):
                result = upload_document(uploaded_file)
                
                if result.get("success", False):
                    st.success(f"Document uploaded successfully! {result.get('message', '')}")
                    # Display document ID
                    st.info(f"Document ID: {result.get('document_id', 'Unknown')}")
                else:
                    st.error(f"Failed to upload document: {result.get('message', 'Unknown error')}")
    
    # Recently uploaded documents
    st.markdown("### Tips for Effective Document Uploads")
    st.markdown("""
    - **Clean Content**: Ensure documents are well-formatted and contain relevant information
    - **Context Matters**: Documents with clear context help the assistant provide better answers
    - **File Size**: Keep files under 10MB for optimal performance
    - **Processing Time**: Complex documents may take a few minutes to process
    """)

# Tab 3: System Information
with tab3:
    st.markdown("# ⚙️ System Information")
    
    # Get system stats
    stats = get_system_stats()
    
    # Display system information
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Vector Database")
        vector_status = stats.get("vector_db", {}).get("status", "unknown")
        doc_count = stats.get("vector_db", {}).get("document_count", 0)
        
        status_color = "green" if vector_status == "active" else "red"
        st.markdown(f"Status: <span style='color:{status_color};font-weight:bold;'>{vector_status}</span>", unsafe_allow_html=True)
        st.markdown(f"Document Count: {doc_count}")
        
        if "embedding_model" in stats.get("vector_db", {}):
            st.markdown(f"Embedding Model: `{stats['vector_db']['embedding_model']}`")
    
    with col2:
        st.markdown("### LLM Model")
        model_name = stats.get("model", {}).get("name", "unknown")
        model_type = stats.get("model", {}).get("type", "unknown")
        
        st.markdown(f"Model: `{model_name}`")
        st.markdown(f"Type: `{model_type}`")
    
    # Model explanation and justification
    st.markdown("### Model Selection")
    st.markdown("""
    Our chatbot uses **Google's Flan-T5** model for sequence-to-sequence language tasks. 
    This model was chosen for the following reasons:
    
    1. **Strong RAG capabilities**: Flan-T5 is fine-tuned for following instructions and generating coherent responses based on provided context
    2. **Efficiency**: Smaller than full-scale LLMs while maintaining high-quality outputs
    3. **Banking domain adaptation**: Easily fine-tuned on banking data with parameter-efficient techniques
    4. **Multilingual support**: Can handle requests in multiple languages, important for our diverse customer base
    
    The model is further enhanced with:
    - Parameter-efficient fine-tuning (LoRA)
    - Quantization for reduced memory usage
    - Banking domain-specific training examples
    """)
    
    # Advanced technical information
    with st.expander("Technical Details"):
        st.markdown("""
        ### Technical Implementation
        
        **Data Processing Pipeline:**
        - Text chunking with 300 token chunks and 50 token overlap
        - Enhanced anonymization for sensitive financial information
        - Metadata preservation for better source attribution
        
        **Vector Database:**
        - ChromaDB for vector storage
        - All-MiniLM-L6-v2 embeddings (384 dimensions)
        - Approximate nearest neighbor search with cosine similarity
        
        **RAG Implementation:**
        - Top-k retrieval (k=5) with similarity threshold
        - Context formatting with source attribution
        - Conversation history incorporation for coherent multi-turn interactions
        
        **Guardrails:**
        - Banking-specific content filtering
        - Prompt injection detection
        - Sensitive information redaction
        - Out-of-domain topic detection
        """)

# Main page footer
st.markdown("---")
st.markdown("© 2025 NUST Bank. Virtual Assistant Version 1.0")

# Example usage suggestions (at the bottom of every tab)
with st.sidebar:
    st.markdown("### Example Questions")
    example_questions = [
        "What is the current daily ATM withdrawal limit?",
        "How can I update my mobile number in the app?",
        "Is biometric login supported in the mobile app?",
        "How do I add new beneficiaries for funds transfer?",
        "What is RAAST and how do I use it?"
    ]
    
    for q in example_questions:
        if st.button(q):
            # Set the question in the input box
            st.session_state.user_question = q
            # Force refresh
            st.experimental_rerun()