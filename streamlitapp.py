import streamlit as st
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import pdfplumber
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Set page config first
st.set_page_config(
    page_title="Genetic Algorithm Expert",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Cache expensive operations
@st.cache_resource(show_spinner=False)
def setup_embeddings():
    return OllamaEmbeddings(model="mxbai-embed-large")

@st.cache_resource(show_spinner=False)
def create_vector_store(_embedding):
    return Chroma(
        collection_name="papers_pdfs",
        persist_directory="./chrome_langchain_db",
        embedding_function=_embedding
    )

@st.cache_data(show_spinner="Processing PDFs...")
def extract_pdf_chunks(pdf_path, source_name):
    docs = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
    )
    
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                chunks = text_splitter.split_text(text)
                for j, chunk in enumerate(chunks):
                    docs.append(Document(
                        page_content=chunk.strip(),
                        metadata={"source": source_name, "page": i + 1},
                        id=f"{source_name}-{i}-{j}"
                    ))
    return docs

# Initialize once
embedding = setup_embeddings()
vector_store = create_vector_store(embedding)

# Check if database needs initialization
if not os.path.exists("./chrome_langchain_db"):
    with st.spinner("Initializing knowledge base..."):
        docs_pdf1 = extract_pdf_chunks("An_introduction_to_genetic_algorithms.pdf", "paper1")
        docs_pdf2 = extract_pdf_chunks("Genetic_Algorithms.pdf", "paper2")
        all_docs = docs_pdf1 + docs_pdf2
        vector_store.add_documents(documents=all_docs, ids=[doc.id for doc in all_docs])

retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# UI Components
st.title("🧬 Genetic Algorithm Expert")
st.caption("Ask questions about genetic algorithms based on research papers")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Ask me about genetic algorithms!"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Your question about genetic algorithms"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("🔍 Searching papers..."):
            try:
                results = retriever.invoke(prompt)
                paper = "\n\n".join(
                    [f"📄 **Source: {doc.metadata['source']} (Page {doc.metadata['page']})**\n{doc.page_content}" 
                    for doc in results]
                )
            except Exception as e:
                st.error(f"Retrieval error: {str(e)}")
                st.stop()
        
        
        model = OllamaLLM(model="llama3.2", temperature=0.3)
        template = """You are a genetic algorithm expert. Answer concisely using ONLY this context:
        
        Context:
        {paper}
        
        Question: {question}
        
        If context doesn't contain answer, say "I couldn't find relevant information in the papers."
        """
        prompt_template = ChatPromptTemplate.from_template(template)
        chain = prompt_template | model
        
        with st.spinner("💡 Generating answer..."):
            try:
                # Stream the response
                for chunk in chain.stream({"paper": paper, "question": prompt}):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            except Exception as e:
                st.error(f"Generation error: {str(e)}")
                full_response = "Sorry, I encountered an error processing your request."
                message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Show sources
    with st.expander("📚 Sources used"):
        for doc in results:
            st.caption(f"**{doc.metadata['source']}** (Page {doc.metadata['page']})")
            st.text(doc.page_content[:300] + "...")

# Add performance notes to sidebar
with st.sidebar:
    st.header("Performance Notes")
    st.markdown("""
    - Using efficient llama3.2 model to reduce heat
    - Smaller text chunks (800 characters)
    - Only 2 documents retrieved per query
    - Database persists between sessions
    - Heavy operations cached
    """)
    if st.button("Clear Chat History"):
        st.session_state.messages = [{"role": "assistant", "content": "Ask me about genetic algorithms!"}]
        st.experimental_rerun()

# Add cooling recommendation
st.sidebar.info("💡 **Laptop Cooling Tip**:\nPlace your laptop on a hard, flat surface for better airflow during extended use.")