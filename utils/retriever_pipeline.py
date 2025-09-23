# utils/retriever_pipeline.py
import os
import requests
import streamlit as st
from rank_bm25 import BM25Okapi
from typing import List, Optional, Any
from langchain.schema import Document

def _unique_docs(docs: List[Any]) -> List[Any]:
    seen = set()
    out = []
    for d in docs:
        # Handle both Document objects and other types
        if hasattr(d, 'page_content') and hasattr(d, 'metadata'):
            key = str(d.metadata.get("source", "")) + "|" + d.page_content[:200]
        else:
            # Fallback for other types
            key = str(d)[:200]
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out

def _bm25_rerank(query: str, candidates: List[Any], k: int) -> List[Any]:
    if not candidates:
        return []
    
    # Extract page_content for BM25
    corpus = []
    valid_candidates = []
    for c in candidates:
        if hasattr(c, 'page_content'):
            corpus.append(c.page_content)
            valid_candidates.append(c)
        else:
            # Skip candidates without page_content
            continue
    
    if not corpus:
        return []
    
    tokenized_corpus = [c.split() for c in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    scored = sorted(zip(scores, valid_candidates), key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored[:k]]

def _call_ollama_generate(prompt: str, model: str = None, ollama_url: Optional[str] = None) -> Optional[str]:
    ollama_url = ollama_url or os.getenv("OLLAMA_API_URL", "http://localhost:11434")
    model = model or os.getenv("DEFAULT_MODEL", "qwen2:0.5b")
    try:
        resp = requests.post(
            f"{ollama_url.rstrip('/')}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=10
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        st.warning(f"HyDE generation failed: {e}")
        return None

def get_relevant_documents(
    query: str,
    vector_store,
    knowledge_graph=None,
    use_hyde: bool = False,
    use_reranking: bool = True,
    use_graphrag: bool = False,
    max_results: int = 5,
    ollama_model: Optional[str] = None,
    ollama_url: Optional[str] = None
) -> List[Any]:
    """
    Combined retriever:
      - dense search (vector_store.similarity_search)
      - optional HyDE augmentation
      - optional GraphRAG augmentation
      - optional BM25 reranking
    """
    if vector_store is None:
        st.error("No documents processed yet. Upload files first.")
        return []

    candidates = []

    # Dense retrieval
    try:
        primary_candidates = vector_store.similarity_search(query, k=max_results * 2)
        candidates.extend(primary_candidates)
    except Exception as e:
        st.warning(f"Vector search failed: {e}")
        try:
            # Fallback to smaller k
            primary_candidates = vector_store.similarity_search(query, k=max_results)
            candidates.extend(primary_candidates)
        except Exception:
            st.error("Vector search completely failed")
            return []

    # HyDE augmentation
    if use_hyde:
        hyde_prompt = f"Generate a concise, plausible answer to improve retrieval.\nQuestion: {query}\nAnswer:"
        hyde_text = _call_ollama_generate(hyde_prompt, model=ollama_model, ollama_url=ollama_url)
        if hyde_text:
            try:
                hyde_docs = vector_store.similarity_search(hyde_text, k=max_results)
                candidates.extend(hyde_docs)
            except Exception as e:
                st.warning(f"HyDE retrieval failed: {e}")

    # GraphRAG augmentation
    if use_graphrag and knowledge_graph is not None:
        try:
            from utils.build_graph import retrieve_from_graph
            graph_entities = retrieve_from_graph(query, knowledge_graph, top_k=5)
            for ent in graph_entities:
                try:
                    ent_docs = vector_store.similarity_search(ent, k=2)
                    candidates.extend(ent_docs)
                except Exception as e:
                    continue
        except Exception as e:
            st.warning(f"GraphRAG retrieval failed: {e}")

    # Dedupe
    candidates = _unique_docs(candidates)

    # Rerank
    if use_reranking and candidates:
        try:
            ranked = _bm25_rerank(query, candidates, k=max_results)
            return ranked
        except Exception as e:
            st.warning(f"BM25 reranking failed: {e}")
            return candidates[:max_results]

    return candidates[:max_results]