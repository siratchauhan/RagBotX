# app.py
import os
import streamlit as st
import requests
import json
import re
import tempfile
import shutil
from typing import List, Dict
from doc_handler import process_uploaded_files
from utils.retriever_pipeline import get_relevant_documents
from utils.build_graph import build_knowledge_graph, visualize_graph
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(page_title="Advanced RAG Assistant", page_icon="🤖", layout="wide")
st.title("RagBotX 🚀")
st.write("Upload your documents and start asking questions!")

# Custom CSS
st.markdown("""
<style>
.stChatMessage {padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; border-left: 4px solid;}
.stChatMessage.user {background-color: #f0f8ff; border-color: #0066cc;}
.stChatMessage.assistant {background-color: #f0fff0; border-color: #00cc66;}
.sidebar .sidebar-content {background-color: #f8f9fa;}
</style>
""", unsafe_allow_html=True)

# ---------------- Session State Initialization ----------------
if "vector_store" not in st.session_state: 
    st.session_state.vector_store = None
if "messages" not in st.session_state: 
    st.session_state.messages = []
if "chat_mode" not in st.session_state: 
    st.session_state.chat_mode = "rag"
if "enable_hyde" not in st.session_state: 
    st.session_state.enable_hyde = os.getenv("ENABLE_HYDE", "true").lower() == "true"
if "enable_reranking" not in st.session_state: 
    st.session_state.enable_reranking = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
if "enable_graph_rag" not in st.session_state: 
    st.session_state.enable_graph_rag = os.getenv("ENABLE_GRAPH_RAG", "true").lower() == "true"
if "temperature" not in st.session_state: 
    st.session_state.temperature = float(os.getenv("TEMPERATURE", "0.7"))
if "max_contexts" not in st.session_state: 
    st.session_state.max_contexts = int(os.getenv("MAX_CONTEXTS", "3"))
if "knowledge_graph" not in st.session_state: 
    st.session_state.knowledge_graph = None
if "processed_documents" not in st.session_state: 
    st.session_state.processed_documents = None
if "temp_dir" not in st.session_state:
    st.session_state.temp_dir = None

# ---------------- Helper Functions ----------------
def summarize_conversation(messages: List[Dict], model: str) -> str:
    if len(messages) <= 8:
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    try:
        recent_messages = messages[-4:]
        conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages[:-4]])
        summary_prompt = f"Summarize this conversation briefly while preserving key facts:\n{conversation_text}\nSummary:"
        response = requests.post(
            os.getenv("LLM_API_URL", "http://localhost:11434/api/generate"),
            json={"model": model, "prompt": summary_prompt, "stream": False}
        )
        response.raise_for_status()
        summary = response.json().get("response", "").strip()
        return f"Conversation summary: {summary}\n" + "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_messages])
    except Exception:
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages[-6:]])

def extract_key_entities(text: str) -> List[str]:
    entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    return list(set(entities))

def build_contextual_prompt(messages: List[Dict], current_query: str, context: str, model: str, use_rag: bool) -> str:
    conversation_history = summarize_conversation(messages, model)
    all_text = current_query + " " + " ".join([msg['content'] for msg in messages])
    key_entities = extract_key_entities(all_text)
    if use_rag and context:
        return f"""Use the conversation history and document context to answer.

Conversation History:
{conversation_history}

Key Entities: {', '.join(key_entities)}

Document Context:
{context}

Current Question: {current_query}

Answer thoughtfully:"""
    else:
        return f"""Continue the conversation naturally.

Conversation History:
{conversation_history}

Key Entities: {', '.join(key_entities)}

Current Message: {current_query}

Respond naturally:"""

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Advanced Settings")
    available_models = ["qwen2:0.5b", "llama3", "mistral", "gemma"]
    selected_model = st.selectbox(
        "🧠 AI Model",
        available_models,
        index=available_models.index(os.getenv("DEFAULT_MODEL", "qwen2:0.5b"))
    )
    st.session_state.enable_hyde = st.checkbox("Enable HyDE", value=st.session_state.enable_hyde)
    st.session_state.enable_reranking = st.checkbox("Enable Reranking", value=st.session_state.enable_reranking)
    st.session_state.enable_graph_rag = st.checkbox("Enable GraphRAG", value=st.session_state.enable_graph_rag)
    st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.temperature, 0.1)
    st.session_state.max_contexts = st.slider("Max Contexts", 1, 10, st.session_state.max_contexts)
    st.session_state.chat_mode = st.radio("Choose Chat Mode:", ["RAG Mode 🤓", "Chat Mode 💬"], index=0 if st.session_state.chat_mode=="rag" else 1)
    use_rag = st.session_state.chat_mode == "RAG Mode 🤓"
    uploaded_files = st.file_uploader("Choose PDF, DOCX, or TXT files", type=['pdf','docx','txt'], accept_multiple_files=True, disabled=not use_rag)
    process_button = st.button("Process Documents", type="primary", disabled=not use_rag)

# ---------------- Document Processing ----------------
if process_button:
    if not uploaded_files:
        st.warning("Please upload at least one file before processing.")
    elif use_rag:
        with st.spinner("Processing documents..."):
            processing_result = process_uploaded_files(uploaded_files)
            if processing_result:
                st.session_state.vector_store = processing_result["vectorstore"]
                st.session_state.knowledge_graph = processing_result["knowledge_graph"]
                st.session_state.processed_documents = processing_result["documents"]
                st.session_state.messages = []
                st.success("✅ Documents processed successfully!")

# ---------------- Conversation Tools ----------------
if st.session_state.messages:
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📋 Export Chat"):
            chat_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
            st.download_button("Download", chat_text, "conversation.txt")
    with col2:
        if st.button("🧹 Clear Memory"):
            st.session_state.messages = []
            st.rerun()
    with col3:
        if st.button("📊 Stats"):
            st.info(f"Messages: {len(st.session_state.messages)}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------- Chat Input ----------------
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build context
    context_text = ""
    if use_rag and st.session_state.vector_store:
        with st.spinner("🔍 Searching documents..."):
            relevant_docs = get_relevant_documents(
                query=prompt,
                vector_store=st.session_state.vector_store,
                knowledge_graph=st.session_state.knowledge_graph,
                use_hyde=st.session_state.enable_hyde,
                use_reranking=st.session_state.enable_reranking,
                use_graphrag=st.session_state.enable_graph_rag,
                max_results=st.session_state.max_contexts
            )
        if relevant_docs:
            context_text = "\n\n".join([f"Source {i+1}:\n{doc.page_content}" for i, doc in enumerate(relevant_docs)])
            with st.expander("📚 View Sources", expanded=False):
                for i, doc in enumerate(relevant_docs):
                    st.write(f"**Source {i+1}:**")
                    st.caption(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                    st.write("---")
        if st.session_state.enable_graph_rag and st.session_state.knowledge_graph:
            with st.expander("🌐 Knowledge Graph", expanded=False):
                visualize_graph(st.session_state.knowledge_graph)

    # Build system prompt
    system_prompt = build_contextual_prompt(
        messages=st.session_state.messages[:-1],
        current_query=prompt,
        context=context_text,
        model=selected_model,
        use_rag=use_rag
    )

    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            response = requests.post(
                os.getenv("LLM_API_URL", "http://localhost:11434/api/generate"),
                json={
                    "model": selected_model,
                    "prompt": system_prompt,
                    "stream": True,
                    "options": {"temperature": st.session_state.temperature, "num_ctx": 4096}
                },
                stream=True
            )
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    try:
                        decoded_line = json.loads(line.decode('utf-8'))
                        token = decoded_line.get("response", "")
                        full_response += token
                        message_placeholder.markdown(full_response + "▌")
                        if decoded_line.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
            message_placeholder.markdown(full_response)
        except requests.exceptions.ConnectionError:
            error_msg = "**Error:** Could not connect to the LLM API. Make sure Ollama is running."
            message_placeholder.markdown(error_msg)
            full_response = error_msg
        except Exception as e:
            error_msg = f"**Error:** {str(e)}"
            message_placeholder.markdown(error_msg)
            full_response = error_msg

        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Cleanup on app close
if st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
    shutil.rmtree(st.session_state.temp_dir)