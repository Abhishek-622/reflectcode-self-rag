# 🤖 ReflectAI: Self-Reflective Retrieval-Augmented Generation (Self-RAG)

[![Hugging Face Space](https://huggingface.co/spaces/abhikur420/reflectai-self-rag)

**ReflectAI** is an open-source Self-RAG application that builds a reflective AI agent for code debugging, technical explanations, and interview preparation. Using a multi-step pipeline, it retrieves relevant context from a local knowledge base, generates responses with Groq's Llama 3.3, *critiques its own output* for accuracy and relevance, and iteratively refines until optimal.

Perfect for developers, recruiters, and learners—watch the AI "think aloud" in real-time!

## 🚀 Live Demo
Try it instantly:  
[ReflectAI on Hugging Face Spaces](https://huggingface.co/spaces/abhikur420/reflectai-self-rag)

## 🧩 Features
- **Self-Reflective Loop**: Retrieval → Generation → Critique → Refinement (up to 2 iterations).
- **Dual Modes**:
  - **Dev Mode**: Checks for hallucinations, factual accuracy, and technical depth.
  - **Recruiter Mode**: Evaluates responses for interview readiness with scores (1-10), strengths, and weaknesses.
- **Custom Knowledge Base**: Ingest text files, PDFs, or notes into a Chroma vector store.
- **Code Snippet Support**: Paste buggy code directly into queries for targeted reviews.
- **Exportable PDF Reviews**: In Recruiter mode, download polished summaries.
- **Fast & Local**: Powered by Groq for inference; embeddings via Hugging Face.

## 🛠️ Tech Stack
- **LLM**: Groq (Llama 3.3 70B Versatile)
- **RAG Framework**: LangChain (core, Groq, Chroma, Hugging Face integrations)
- **Vector Store**: ChromaDB
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **UI**: Gradio
- **Utilities**: ReportLab (PDF gen), Colorama (CLI colors)

## 📁 Project Structure
reflectai-self-rag/
├── app.py                  # Gradio web interface (main entry point)
├── self_rag.py             # Core Self-RAG pipeline logic
├── ingest.py               # Script to build vector store from data/
├── data/                   # Your input files (txt/pdf) – add your own!
├── chroma_db/              # Persistent vector store (auto-created)
├── requirements.txt        # Dependencies


## 🏗️ Quick Start (Local Setup)

1. **Clone the Repo**:
   ```bash
   git clone https://github.com/Abhishek-622/reflectai-self-rag.git
   cd reflectai-self-rag
2. Set Up Environment:
 # Create virtual env (recommended)
python -m venv rag_env
source rag_env/bin/activate  # On Windows: rag_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

3. Add Your Groq API Key: Copy .env.example to .env and add:
GROQ_API_KEY=your_groq_api_key_here

4. Build Knowledge Base (Optional but Recommended): Add your docs (txt/pdf) to data/, then run:
python ingest.py

5.Launch the App:
python app.py

🎯 Usage Examples

Dev Mode Query: "Debug this overfitting code: def train_model(X, y): model.fit(X, y) # No validation?"
Recruiter Mode: Target role "Senior ML Engineer" + query above → Gets scored review + PDF export.

🔍 Pipeline Breakdown

Retrieval: Fetches top-5 chunks from vector store.
Generation: Initial answer using context.
Critique: JSON eval (relevance score, issues like "hallucinated fact").
Refine: If needed, re-retrieve or polish output.
Output: Step-by-step Markdown + optional PDF.

See self_rag.py for prompts/chains.

🙏 Acknowledgments

LangChain for the RAG magic.
Groq for speedy inference.
Inspired by Self-RAG paper: arXiv.

Built with ❤️ by Abhishek. Questions? Ping me!

