# 🏛️ Vietnamese Legal RAG Chatbot

A Retrieval-Augmented Generation (RAG) system designed for answering Vietnamese legal queries with accurate references and zero hallucination.

## 📌 Features
- **Retrieval-Augmented Generation (RAG)**: Combines dense vector retrieval with LLM text generation.
- **Fast Similarity Search**: Powered by **FAISS** and `paraphrase-multilingual-MiniLM-L12-v2`.
- **Fact-Based Generation**: Leverages `Qwen2-1.5B-Instruct` constrained strictly to retrieved context.
- **Hardware-Adaptive**: Auto-detects CUDA GPU vs. CPU for optimized inference (`float16`/`float32`).
- **Interactive UI**: Built with **Gradio** for seamless web-based legal Q&A.

---

## 📂 Project Structure
```text
laws/
├── app/
│   ├── GUI.py           # Gradio Web Interface
│   └── rag_engine.py    # Core RAG Pipeline (Retrieval & LLM Generation)
├── artifacts/
│   ├── config.json      # Model configurations
│   ├── faiss.index      # FAISS Vector Index
│   └── docstore.pkl     # Legal text & citation mappings
├── data/                # Raw datasets & benchmarks
├── requirements.txt     # Python dependencies
└── README.md
```

---

## 🛠️ Quick Start

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/Ein05/vietnam_law_chatbot.git
cd vietnam_law_chatbot
pip install -r requirements.txt
```

### 2. Run the Application
Launch the Gradio Web UI:
```bash
python app/GUI.py
```
Open your browser at `http://127.0.0.1:7860`.

---

## 📊 Tech Stack & Dataset
- **Embedding**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **LLM**: `Qwen/Qwen2-1.5B-Instruct`
- **Vector DB**: `FAISS`
- **Interface**: `Gradio`
- **Dataset**: [Vietnamese Legal Dataset (Kaggle)](https://www.kaggle.com/datasets/quangbut/vietnamese-legal)