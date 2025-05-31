from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import pandas as pd
import pdfplumber

def extract_pdf_chunks(pdf_path, source_name):
    docs = []
    # Initialize text splitter with optimal parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                # Split page text into manageable chunks
                chunks = text_splitter.split_text(text)
                for j, chunk in enumerate(chunks):
                    docs.append(Document(
                        page_content=chunk.strip(),
                        metadata={"source": source_name, "page": i + 1, "chunk": j+1},
                        id=f"{source_name}-{i}-{j}"
                    ))
    return docs


# Extract docs from two PDFs
docs_pdf1 = extract_pdf_chunks("An_introduction_to_genetic_algorithms.pdf", "paper1")
docs_pdf2 = extract_pdf_chunks("Genetic_Algorithms.pdf", "paper2")

# Combine all documents
all_docs = docs_pdf1 + docs_pdf2


#embeddings
embedding = OllamaEmbeddings(model = "mxbai-embed-large")
db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

vector_store = Chroma(
    collection_name="papers_pdfs",
    persist_directory=db_location,
    embedding_function=embedding
)

if add_documents:
    vector_store.add_documents(documents=all_docs, ids=[doc.id for doc in all_docs])

retriever = vector_store.as_retriever(search_kwargs={"k": 3})
