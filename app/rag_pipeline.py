import os
import tempfile
import faiss
import fitz  # PyMuPDF
import cv2
from PIL import Image
import numpy as np
import easyocr
import filetype
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from rouge_score import rouge_scorer

retriever_model = SentenceTransformer("all-MiniLM-L6-v2")
generator = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=100)
reader = easyocr.Reader(['en'], gpu=False)

def extract_text_from_bytes(file_bytes, filename):
    kind = filetype.guess(file_bytes)
    ext = kind.extension if kind else (filename.lower() if filename else "")

    # PDF: Try PyMuPDF for text, fallback to OCR from rendered page image
    if ext == "pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        doc = fitz.open(tmp_path)
        text = "\n".join([page.get_text() for page in doc])

        if not text.strip() or len(text.strip()) < 10:
            ocr_texts = []
            for page in doc:
                pix = page.get_pixmap(dpi=300)
                img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                if pix.n == 4:
                    img_data = cv2.cvtColor(img_data, cv2.COLOR_BGRA2BGR)
                gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
                ocr_text = reader.readtext(thresh, detail=0)
                ocr_texts.append(" ".join(ocr_text))
            text = "\n".join(ocr_texts)

        doc.close()
        os.unlink(tmp_path)
        return text

    # Image files (JPG/PNG)
    elif ext in ["png", "jpg", "jpeg"]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        img = cv2.imread(tmp_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        text = " ".join(reader.readtext(thresh, detail=0))
        os.unlink(tmp_path)
        return text

    # Plain text fallback
    else:
        return file_bytes.decode("utf-8", errors="ignore")

def compute_rouge_score(pred, ref):
    if not ref:
        return None
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = scorer.score(ref, pred)
    return {k: round(v.fmeasure, 3) for k, v in scores.items()}

def answer_question_from_docs(file_bytes, question, filename, reference_answer=None):
    text = extract_text_from_bytes(file_bytes, filename)
    chunks = [chunk.strip() for chunk in text.split("\n") if len(chunk.strip()) > 20]
    embeddings = retriever_model.encode(chunks)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    D, I = index.search(retriever_model.encode([question]), k=3)

    relevant_chunks = [chunks[i] for i in I[0]]
    context = "\n".join(relevant_chunks)
    prompt = f"Answer the question based on the context:\n{context}\n\nQuestion: {question}\nAnswer:"

    response = generator(prompt)[0]["generated_text"]
    final_answer = response.strip()
    rouge_scores = compute_rouge_score(final_answer, reference_answer)

    return {
        "answer": final_answer,
        "context": context,
        "rouge_scores": rouge_scores
    }

