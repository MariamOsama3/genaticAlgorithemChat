import streamlit as st
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
import pdfplumber
import os
import tempfile

# -- SETTINGS --
embedding_model_name = "mxbai-embed-large"
llm_model_name = "llama3.2"
persist_dir = "./chroma_db"

# -- PDF Chunking --
def extract_pdf_chunks(file_path, source_name):
    docs = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and len(text.strip()) > 100:  # Skip empty/short pages
                docs.append(Document(
                    page_content=text.strip(),
                    metadata={"source": source_name, "page": i + 1},
                    id=f"{source_name}-{i}"
                ))
    return docs

# -- UI Starts Here --
st.title("📚 PDF Q&A using Ollama + LangChain")

# Upload
uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_docs = []
    for uploaded_file in uploaded_files:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Extract chunks
        file_docs = extract_pdf_chunks(tmp_path, uploaded_file.name)
        all_docs.extend(file_docs)
        st.success(f"📄 Processed {len(file_docs)} pages from {uploaded_file.name}")

    # Embedding
    embedding = OllamaEmbeddings(model=embedding_model_name)
    vector_store = Chroma(
        collection_name="pdf_qa",
        persist_directory=persist_dir,
        embedding_function=embedding
    )

    # Add only once
    if not os.path.exists(persist_dir) or len(vector_store.get()["ids"]) == 0:
        vector_store.add_documents(all_docs, ids=[doc.id for doc in all_docs])
        st.success("✅ Embeddings created and stored.")

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = OllamaLLM(model=llm_model_name)

    # Prompt Template
    template = """
You are an expert in answering questions about genetic algorithms.

Here is the content from the papers:
{paper}

Question:
{question}

If the question is unrelated or cannot be answered from the papers, respond:
"I'm sorry, I don't have enough information from the papers to answer that."
"""
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm

    # Chat Interface
    st.header("🔎 Ask a Question")
    user_question = st.text_input("Enter your question")

    if st.button("Get Answer") and user_question:
        paper_docs = retriever.invoke(user_question)
        if not paper_docs:
            st.warning("🤷 No relevant information found in papers.")
        else:
            result = chain.invoke({
                "paper": paper_docs,
                "question": user_question
            })
            st.success("✅ Answer:")
            st.write(result)
