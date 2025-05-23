
# Backend - PDF Question Answering App (FastAPI)

This is the **backend server** for the Fullstack Internship Assignment — a PDF Question-Answering system using **FastAPI**, **LangChain**, **GEMINI API**, and **Vector Database**.

---

## 📚 Features

- Upload PDF files
- Background PDF text extraction, chunking, and embedding
- Semantic search on extracted content
- Query answering using GEMINI 
- Clean API endpoints with proper error handling

---

## ⚙️ Tech Stack

- **FastAPI** - Modern Python web framework
- **PyMuPDF** - PDF text extraction
- **LangChain** - RAG pipeline construction
- **GEMINI API** - LLM inference
- **ChromaDB / FAISS** - Vector database for embeddings
- **BackgroundTasks** - Asynchronous PDF processing
- **Cloudinary** - PDF storage
---

## 🚀 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Tarunnagpal7/ai-planet-langchain.git
cd backend
```
### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Configure Environment Variables
```bash
GOOGLE_API_KEY =
CLOUDINARY_API_KEY =
CLOUDINARY_API_SECRET =
CLOUDINARY_CLOUD_NAME =
```
### 5. Run the FastAPI Server
```bash
uvicorn main:app --reload
```
