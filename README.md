🤖 RagBotX – Advanced RAG Chatbot with GraphRAG & Chat Memory

100% Local • Private • Ollama-powered 🚀

Upload PDFs, DOCX, or TXT files and get fast, contextual answers with RAG, HyDE, Neural Reranking, GraphRAG, and chat memory.
Runs fully offline with Ollama
 — no internet or external APIs needed.

✨ Features

📂 Upload PDF, DOCX, or TXT documents

🔎 Hybrid Retrieval with FAISS + reranking

🌐 GraphRAG – builds a Knowledge Graph for deeper context

🧠 Chat Memory – remembers past conversation turns

🎯 HyDE Query Expansion – better document recall

🔄 Switch between RAG Mode (contextual Q&A) and Chat Mode (general AI)

🎛️ Customizable settings (model, temperature, max contexts)

📦 Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/siratchauhan/RagBotX.git
cd RagBotX

2️⃣ Create a Virtual Environment
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

3️⃣ Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt

4️⃣ Install Ollama & Pull Models

Download Ollama → https://ollama.ai

Then pull your models:

ollama pull qwen2:0.5b
ollama pull nomic-embed-text


(You can swap models later in .env)

5️⃣ Setup Environment Variables

Create a .env file in the project root:

LLM_API_URL=http://localhost:11434

DEFAULT_MODEL=qwen2:0.5b

EMBEDDINGS_MODEL=nomic-embed-text

ENABLE_HYDE=true

ENABLE_RERANKING=true

ENABLE_GRAPH_RAG=true

TEMPERATURE=0.7

MAX_CONTEXTS=3

6️⃣ Run Ollama
ollama serve

7️⃣ Launch the Chatbot
streamlit run app.py


Open 👉 http://localhost:8501

🖥️ How It Works

Upload documents in the sidebar

RagBotX splits & indexes them into a vector database

Queries are expanded (HyDE), matched (FAISS), reranked, and enriched with GraphRAG

Ollama generates final answers, with context + memory

📌 Roadmap

 Suggested follow-up questions in chat

 Support for ChromaDB / Pinecone / Weaviate backends

 Advanced RAG pipelines (basic vs graph-enhanced)

 Export knowledge graphs as JSON

🤝 Contributing

Pull requests, feature ideas, and bug reports are welcome!

⚡ Demo Preview



🔗 Repo: https://github.com/siratchauhan/RagBotX

💬 Built with ❤️ using Python, Streamlit, FAISS, LangChain, and Ollama.
