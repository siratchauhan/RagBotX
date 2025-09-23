# utils/build_graph.py
import networkx as nx
import re
import streamlit as st
from pyvis.network import Network
import streamlit.components.v1 as components

def build_knowledge_graph(documents):
    """
    Build a knowledge graph from document chunks by extracting entities and relationships.
    """
    G = nx.Graph()
    
    for doc in documents:
        text = doc.page_content
        entities = re.findall(r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b', text)
        
        for entity in entities:
            if entity not in G.nodes:
                G.add_node(entity, type='entity')
        
        if len(entities) > 1:
            for i in range(len(entities) - 1):
                for j in range(i + 1, len(entities)):
                    if G.has_edge(entities[i], entities[j]):
                        G[entities[i]][entities[j]]['weight'] += 1
                    else:
                        G.add_edge(entities[i], entities[j], weight=1)
    
    return G

def retrieve_from_graph(query, knowledge_graph, top_k=5):
    """
    Retrieve related concepts from the knowledge graph based on query.
    """
    if knowledge_graph is None:
        return []
    
    query_words = query.lower().split()
    matched_nodes = []
    
    for node in knowledge_graph.nodes():
        node_lower = node.lower()
        if any(word in node_lower for word in query_words):
            matched_nodes.append(node)
    
    related_nodes = []
    for node in matched_nodes:
        if node in knowledge_graph:
            related_nodes.extend(list(knowledge_graph.neighbors(node)))
    
    unique_related = list(set(related_nodes))
    return unique_related[:top_k]

import plotly.graph_objects as go

def visualize_graph(knowledge_graph):
    if knowledge_graph and len(knowledge_graph.nodes) > 0:
        st.write(f"**Knowledge Graph:** {len(knowledge_graph.nodes)} nodes, {len(knowledge_graph.edges)} relationships")

        pos = nx.spring_layout(knowledge_graph, k=0.15, iterations=20)

        edge_x, edge_y = [], []
        for edge in knowledge_graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(x=edge_x, y=edge_y,
                                line=dict(width=0.5, color='#888'),
                                hoverinfo='none',
                                mode='lines')

        node_x, node_y, text = [], [], []
        for node in knowledge_graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            text.append(node)

        node_trace = go.Scatter(
            x=node_x, y=node_y, text=text,
            mode='markers+text',
            textposition="top center",
            hoverinfo='text',
            marker=dict(size=10, color='skyblue')
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=0, l=0, r=0, t=0)
                        ))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No knowledge graph available. Enable GraphRAG in settings.")