#  RAG Support Chatbot (Free & Local Version)

A fully local Retrieval-Augmented Generation (RAG) chatbot that answers user queries using customer support documents — **no OpenAI API key or payment required**.

---

## ✅ Features

- 💬 Answers questions based on uploaded PDFs & DOCX documents
- ❓ Replies with "I don't know" for out-of-scope queries
- 🔍 Uses **local embeddings** from `sentence-transformers`
- 💾 Stores documents in **FAISS vector database**
- 🖥️ Clean and simple chat UI using **Streamlit**

---

## 🛠️ Tech Stack

| Layer       | Tool / Library                          |
|-------------|------------------------------------------|
| UI          | Streamlit                                |
| Embeddings  | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB   | FAISS                                    |
| NLP Engine  | Basic retrieval (RAG-lite)               |

---

## 📦 Installation

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/rag_chatbot.git
cd rag_chatbot
```
### 2. Install Requirements
```bash
pip install -r requirements.txt
```
If not already installed:
```bash
pip install streamlit langchain langchain-community faiss-cpu sentence-transformers python-docx pdfplumber
```
📄 How It Works
### 1. Document Ingestion (Local Embeddings)
```bash
python ingest.py
```
Loads all files from data/
Splits into chunks
Generates sentence-transformer embeddings
Stores in FAISS

### 2. Launch the Chatbot
```bash
streamlit run app.py
```
