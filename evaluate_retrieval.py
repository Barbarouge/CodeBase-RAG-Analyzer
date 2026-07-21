import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# 1. Initialize Vector DB
VECTOR_DB_PATH = "./chroma_db"
embeddings = OllamaEmbeddings(model="nomic-embed-text")

print("Initializing Vector Database for evaluation...")
try:
    vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
except Exception as e:
    print(f"Error loading ChromaDB: {e}")
    exit()

# 2. Ground Truth Dataset (Test Soruları ve Beklenen Doğru Dosyalar)
qa_pairs = [
    {"question": "What is the core method that acts as the WSGI application interface?", "expected_source": "app.py"},
    {"question": "Which file contains the implementation of the application context and request context globals?", "expected_source": "globals.py"},
    {"question": "Where is the Blueprint class defined for modular applications?", "expected_source": "blueprints.py"},
    {"question": "Which module handles the command-line interface (CLI) commands for Flask?", "expected_source": "cli.py"},
    {"question": "How does Flask manage the testing client setup?", "expected_source": "testing.py"},
    {"question": "What file defines the Request and Response objects specifically for Flask?", "expected_source": "wrappers.py"},
    {"question": "Where are the session management interfaces (like SecureCookieSessionInterface) defined?", "expected_source": "sessions.py"}
]

# 3. Metrics Calculation Variables
k = 3
hits = 0
precision_sum = 0
mrr_sum = 0

print("\nStarting Quantitative Evaluation (Retrieval Metrics)...\n" + "-"*50)

# 4. Evaluation Loop
for idx, item in enumerate(qa_pairs):
    query = item["question"]
    expected = item["expected_source"]
    
    # Retrieve top k documents
    docs = retriever.invoke(query)
    retrieved_sources = [doc.metadata.get('source') for doc in docs]
    
    # Calculate Hit Rate (Is the expected file in the top k?)
    is_hit = expected in retrieved_sources
    if is_hit:
        hits += 1
        
    # Calculate Precision@K
    relevant_count = retrieved_sources.count(expected)
    precision_sum += relevant_count / k
    
    # Calculate MRR (Mean Reciprocal Rank)
    rank = 0
    for i, source in enumerate(retrieved_sources):
        if source == expected:
            rank = i + 1
            break
            
    if rank > 0:
        mrr_sum += 1.0 / rank
        
    print(f"Q{idx+1}: {query}")
    print(f"  Expected: {expected} | Retrieved: {retrieved_sources}")
    print(f"  Hit: {'YES' if is_hit else 'NO'}\n")

# 5. Final Calculations
total_q = len(qa_pairs)
hit_rate = (hits / total_q) * 100
avg_precision = (precision_sum / total_q) * 100
mrr = mrr_sum / total_q

print("-" * 50)
print("FINAL EVALUATION METRICS")
print(f"Total Queries Tested : {total_q}")
print(f"Hit Rate@{k}           : {hit_rate:.2f}%")
print(f"Precision@{k}          : {avg_precision:.2f}%")
print(f"Mean Reciprocal Rank : {mrr:.3f}")
print("-" * 50)