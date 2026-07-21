import ast
import os
import time
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Target directory containing the Flask source code
TARGET_DIR = "./flask/src/flask"
VECTOR_DB_PATH = "./chroma_db"

def get_ast_chunks(directory):
    """
    Reads Python files using the 'ast' module, extracting 
    functions and classes as structurally intact Documents.
    Uses a fallback splitter for extremely large AST nodes.
    """
    documents = []
    # Fallback splitter for massive chunks that exceed Nomic's context window
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                with open(filepath, "r", encoding="utf-8") as f:
                    source = f.read()
                
                try:
                    # Parse the source code into an Abstract Syntax Tree (AST)
                    tree = ast.parse(source)
                    for node in tree.body:
                        # Capture only structural entities: Classes and Functions
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                            chunk_code = ast.get_source_segment(source, node)
                            if chunk_code:
                                # Safe-guard: Sub-chunk if the AST block is too large for the embedding model
                                if len(chunk_code) > 4000:
                                    sub_chunks = text_splitter.split_text(chunk_code)
                                    for i, sub_chunk in enumerate(sub_chunks):
                                        documents.append(Document(
                                            page_content=sub_chunk,
                                            metadata={"source": file, "type": type(node).__name__, "name": f"{node.name}_part{i+1}"}
                                        ))
                                else:
                                    documents.append(Document(
                                        page_content=chunk_code,
                                        metadata={"source": file, "type": type(node).__name__, "name": node.name}
                                    ))
                except Exception as e:
                    print(f"Error processing file {file}: {e}")
    return documents

print("1. Starting AST Parsing process...")
start_time = time.time()
ast_documents = get_ast_chunks(TARGET_DIR)
ast_time = time.time() - start_time
print(f"Success! Extracted {len(ast_documents)} structural AST blocks. (Time: {ast_time:.2f} seconds)")

print("\n2. Starting Vectorization and saving to ChromaDB (Nomic-embed-text)...")
embed_start_time = time.time()
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Initialize and persist the vector database
vectorstore = Chroma.from_documents(
    documents=ast_documents,
    embedding=embeddings,
    persist_directory=VECTOR_DB_PATH
)
embed_time = time.time() - embed_start_time
print(f"Success! Vectors saved to local ChromaDB. (Time: {embed_time:.2f} seconds)")