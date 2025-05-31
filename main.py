from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector_app import retriever

model = OllamaLLM(model="llama3.2")
templete = """you are an expert in answering questions about genetic algorithem 
here is the papers that you will use {paper}
and here is the questions that you should answer{question}

"""
prompt = ChatPromptTemplate.from_template(templete)
chain = prompt | model

while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break
    
    results = retriever.invoke(question)
    paper = "\n\n".join([doc.page_content for doc in results])
    result = chain.invoke({"paper": paper, "question": question})
    print(result)