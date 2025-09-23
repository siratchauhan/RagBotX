# doc_handler.py
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import os
import tempfile
from utils.build_graph import build_knowledge_graph

def process_uploaded_files(uploaded_files):
    """
    Takes uploaded files, loads text, splits it into chunks,
    creates vector store and knowledge graph.
    """
    if not uploaded_files:
        st.error("No files uploaded!")
        return None

    all_docs = []
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    for uploaded_file in uploaded_files:
        st.write(f"📄 Processing: {uploaded_file.name}...")
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)

        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            if uploaded_file.name.lower().endswith('.pdf'):
                loader = PyPDFLoader(temp_file_path)
            elif uploaded_file.name.lower().endswith('.docx'):
                loader = Docx2txtLoader(temp_file_path)
            elif uploaded_file.name.lower().endswith('.txt'):
                loader = TextLoader(temp_file_path)
            else:
                # Fallback for other file types
                loader = UnstructuredFileLoader(temp_file_path)
                st.warning(f"Using unstructured loader for: {uploaded_file.name}")

            documents = loader.load()
            all_docs.extend(documents)

        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {e}")
        finally:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    # Clean up temp directory
    if os.path.exists(temp_dir):
        os.rmdir(temp_dir)

    if not all_docs:
        st.error("No documents were successfully loaded.")
        return None

    st.success(f" Loaded {len(all_docs)} document pages.")

    # Split into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separator="\n")
    chunked_documents = text_splitter.split_documents(all_docs)
    st.write(f" Split into {len(chunked_documents)} chunks.")

    if chunked_documents:
        with st.spinner("Creating knowledge base..."):
            try:
                embeddings = OllamaEmbeddings(model="nomic-embed-text")
                vectorstore = FAISS.from_documents(chunked_documents, embeddings)

                knowledge_graph = None
                if st.session_state.get('enable_graph_rag', False):
                    with st.spinner("Building knowledge graph..."):
                        knowledge_graph = build_knowledge_graph(chunked_documents)
                        if knowledge_graph:
                            st.success(f" Knowledge graph built with {len(knowledge_graph.nodes)} nodes and {len(knowledge_graph.edges)} relationships")
                        else:
                            st.warning("Knowledge graph construction failed")

                st.success(" Vector database created!")
                return {
                    "vectorstore": vectorstore,
                    "knowledge_graph": knowledge_graph,
                    "documents": chunked_documents
                }
            except Exception as e:
                st.error(f"Error creating vector store: {e}")
                return None
    else:
        st.error("No chunks created. Cannot build knowledge base.")
        return None