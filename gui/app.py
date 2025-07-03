import streamlit as st
import requests

st.set_page_config(page_title="Travel Assistant Q&A", layout="wide")

st.title("Travel Assistant — Document Q&A")

uploaded_file = st.file_uploader("Upload a travel document (.txt, .pdf, .png, .jpg)")
question = st.text_input("Ask a question about the document")
reference = st.text_input("Optional: Provide reference answer to compute ROUGE score")

if st.button("Get Answer") and uploaded_file and question:
    response = requests.post(
        "http://localhost:8000/query",
        files={"file": uploaded_file.getvalue()},
        data={"question": question, "reference_answer": reference}
    )
    result = response.json()
    st.success("Answer generated successfully!")
    st.write("Answer:")
    st.write(result["answer"])

    st.write("Context Passages Used:")
    context_text = result["context"].encode("latin1", "ignore").decode("utf-8", "ignore")
    st.code(context_text)

    if result.get("rouge_scores"):
        st.write("ROUGE Scores:")
        st.json(result["rouge_scores"])
