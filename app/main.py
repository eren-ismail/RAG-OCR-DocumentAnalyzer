from fastapi import FastAPI, UploadFile, File, Form
from app.rag_pipeline import answer_question_from_docs

app = FastAPI()

@app.post("/query")
async def query_endpoint(
    question: str = Form(...),
    file: UploadFile = File(...),
    reference_answer: str = Form(None)
):
    contents = await file.read()
    result = answer_question_from_docs(contents, question, file.filename, reference_answer)
    return result
