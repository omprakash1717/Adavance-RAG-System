# NeuralDoc — Intelligent Document Assistant

NeuralDoc is a premium, next-generation Retrieval-Augmented Generation (RAG) platform. It allows users to upload PDFs and structured data (CSV/Excel) to instantly extract insights, perform complex calculations, and summarize contents using **Google Gemini AI** and the **Endee Cloud Vector Database**.

## ✨ Premium Features
- **Neural UI/UX**: A luxury dark-themed interface featuring an animated Aurora background, Syne & DM Sans typography, and interactive 3D card tilt effects.
- **Multi-Modal Analysis**:
    - **PDF Reader**: Semantic search, chunking, and vector indexing for deep document interrogation.
    - **Data Interrogator**: Analytical processing for CSV/XLSX files, enabling trend analysis and automated calculations.
- **Endee Cloud Core**: Blazingly fast vector storage and similarity search.
- **Gemini Intelligence**: Powered by `gemini-2.5-flash` for high-accuracy, context-aware responses with citations.

---

## 🏗️ Technical Architecture
- **Backend**: Flask (Python 3.9+)
- **LLM**: Google Gemini (`langchain-google-genai`)
- **Vector DB**: Endee Cloud (`endee`)
- **Embeddings**: HuggingFace (`sentence-transformers`)
- **Frontend**: Vanilla JS (ES6+), Modern CSS3 (Neural Aesthetics), Feather Icons.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirement.txt
```

### 2. Configure Environment (`.env`)
Create a `.env` file in the root directory:
```ini
# --- HuggingFace (Local Embeddings)
HUGGINGFACE_API_TOKEN=your_token

# --- Google Gemini
GEMINI_API_KEY=your_key

# --- Endee Cloud
ENDEE_API_KEY=project_id:secret:region
ENDEE_BASE_URL=https://dev.endee.io/api/v1
ENDEE_COLLECTION=NeuralDoc_Storage
```

### 3. Launch Platform
```bash
python app.py
```
Access the dashboard at **[http://localhost:5000](http://localhost:5000)**.

---

## 🛠️ Performance & Limits
- **File Size**: Supports uploads up to 50MB.
- **Lazy Loading**: AI models are loaded on-demand to ensure lightning-fast server startup.
- **Auto-Formatting**: Integrated `marked.js` for professional Markdown rendering of AI responses.

---

## 🛡️ Troubleshooting
- **Logo/Icons Not Loading**: Ensure you have an active internet connection for Feather Icons CDN.
- **PDF Analysis Hangs**: Large documents (100+ pages) may take 30-60 seconds for initial vector indexing.
---

## 📂 Project Structure
```text
├── app.py                # Main Flask entry point (UI & Routing)
├── rag_engine.py         # Core RAG logic (Gemini & Endee integration)
├── index.py              # Script for indexing and vector storage
├── retrieval.py          # Script for testing semantic search & retrieval
├── csv_engine.py         # Specialized analysis for CSV/Structured data
├── roadmap.md            # Vision for future feature set
├── requirement.txt       # Project dependencies
└── static/               # Premium Neural UI CSS & JS assets
```

## 🗺️ Roadmap
Check out [roadmap.md](roadmap.md) for details on:
- 🌐 Website Content Scraper
- 📺 YouTube Video Summarizer
- 📊 Advanced CSV Interrogator

---

## ⚖️ License
Distributed under the MIT License. See `LICENSE` for more information.

