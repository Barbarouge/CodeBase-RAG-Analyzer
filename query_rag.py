import time
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

VECTOR_DB_PATH = "./chroma_db"

print("1. Loading Vector Database and Local LLM (Qwen2.5-Coder)...")
# Load Embeddings and Local Vector DB
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)

# Configure the Retriever to fetch the top 3 most relevant AST blocks
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Load the Local LLM via Ollama
llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0)

# Define the System Prompt for Architectural Understanding
template = """You are an expert software architect assistant.
Use the following pieces of retrieved code context to answer the user's question.
If you don't know the answer, just say that you don't know.
Keep your answer concise, highly technical, and explicitly reference the file names,
functions, or classes provided in the context.

Context: {context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(f"Source: {doc.metadata.get('source')} | Code:\n{doc.page_content}" for doc in docs)

# Create the modern LCEL (LangChain Expression Language) pipeline
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Ground-Truth Test Query (Q2)
query = "What is the core method that acts as the WSGI application interface in the Flask framework?"

print(f"\n2. Executing Architectural Query: '{query}'")
start_time = time.time()

# 1. First fetch the documents to log our retrieval metrics for the thesis
retrieved_docs = retriever.invoke(query)

# 2. Then generate the answer using the LCEL chain
answer = rag_chain.invoke(query)
inference_time = time.time() - start_time

print("\n--- RAG SYSTEM GENERATED RESPONSE ---")
print(answer)
print("-------------------------------------")

print("\n3. Retrieved Source Documents (AST Blocks):")
for i, doc in enumerate(retrieved_docs):
    print(f" - Match {i+1}: {doc.metadata.get('source')} (Type: {doc.metadata.get('type')}, Name: {doc.metadata.get('name')})")

print(f"\n[METRICS] Total Inference Time: {inference_time:.2f} seconds")