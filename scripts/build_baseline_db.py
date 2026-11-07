import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

TARGET_DIR = "./flask/src/flask"
BASELINE_DB_PATH = "./chroma_db_baseline"

def build_baseline():
    print("Building Baseline Vector DB (Naive Recursive Text Splitter)...")
    documents = []
    # Standard fixed-size chunking used in traditional RAG systems
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    for root, _, files in os.walk(TARGET_DIR):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                with open(filepath, "r", encoding="utf-8") as f:
                    source = f.read()
                chunks = text_splitter.split_text(source)
                for chunk in chunks:
                    documents.append(Document(
                        page_content=chunk,
                        metadata={"source": file, "method": "baseline_naive"}
                    ))
                    
    print(f"Extracted {len(documents)} naive chunks. Vectorizing now...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=BASELINE_DB_PATH)
    print("Success! Baseline Vector DB saved to disk.")

if __name__ == "__main__":
    build_baseline()
