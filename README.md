# Travel Assistant – Document Q&A

Travel Assistant is a document-based question-answering system designed for travel-related use cases. It supports PDF, image (PNG/JPG), and text files using OCR and a Retrieval-Augmented Generation (RAG) architecture.

## Features
- Upload documents in PDF, PNG, JPG, or TXT format
- Natural language Q&A based on document content
- OCR support for scanned images and image-based PDFs
- Context-aware answers with RAG (FAISS + LLM)
- Source context visibility
- Optional answer evaluation via ROUGE
- FastAPI backend
- Streamlit frontend

## Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start backend (FastAPI)
```bash
uvicorn app.main:app --reload
```

### 3. Start frontend (Streamlit)
```bash
streamlit run gui/app.py
```

## Example Use Cases
- Ask: "What time is the flight?" → PDF or image with itinerary
- Ask: "Where is the hotel?" → JPG scan of confirmation email
- Evaluate: Compare generated answer to expected answer using ROUGE

## Tech Stack
- LLM: google/flan-t5-base via Hugging Face
- Embedding: sentence-transformers (MiniLM)
- Retrieval: FAISS
- OCR: EasyOCR + OpenCV
- PDF Parsing: PyMuPDF
- Backend: FastAPI
- Frontend: Streamlit

## Project Structure
```
travel-assistant/
├── app/               # FastAPI backend + RAG logic
│   ├── main.py
│   └── rag_pipeline.py
├── gui/               # Streamlit UI
│   └── app.py
├── requirements.txt
├── README.md
└── .gitignore
```


Author: Ismail Eren
